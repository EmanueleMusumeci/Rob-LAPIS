"""
AlfWorld game file loader for LAPIS benchmark.

Extracts from .tw-pddl + traj_data.json:
  - Human-annotated NL task description
  - GT PDDL domain + problem (embedded in .tw-pddl)
  - Full NL scene description (formatted from PDDL init state)
  - symbol_table: clean_name → GT coordinate-encoded ID (for name resolution)
"""

import json
import os
import re
import random
import glob
from pathlib import Path
from typing import Optional
from dataclasses import dataclass, field


CACHE_DIR = os.path.expanduser("~/.cache/alfworld/json_2.1.1/train")

TASK_TEMPLATES = {
    "pick_and_place_simple":         "Pick up the {object} and place it in the {receptacle}.",
    "look_at_obj_in_light":          "Find the {object} and examine it under the {toggle}.",
    "pick_clean_then_place_in_recep":"Clean the {object} and then place it in the {receptacle}.",
    "pick_heat_then_place_in_recep": "Heat the {object} and then place it in the {receptacle}.",
    "pick_cool_then_place_in_recep": "Cool the {object} and then place it in the {receptacle}.",
    "pick_two_obj_and_place":        "Pick up two {object}s and place them in the {receptacle}.",
}


@dataclass
class AlfWorldTask:
    game_file: str
    task_id: str
    task_type: str
    nl_task: str          # human-annotated NL task description
    nl_scene: str         # full scene state as NL
    nl_combined: str      # nl_task + nl_scene — LAPIS input
    pddl_domain: str      # alfred.pddl domain text
    pddl_problem: str     # GT PDDL problem text
    pddl_params: dict     = field(default_factory=dict)
    symbol_table: dict    = field(default_factory=dict)   # clean_name → GT coord ID


def _camel_to_words(name: str) -> str:
    return re.sub(r'(?<!^)(?=[A-Z])', ' ', name).lower()


def _strip_coords(raw_id: str) -> str:
    """'SoapBar_bar__minus_00_dot_18...' → 'SoapBar'"""
    return raw_id.split("_bar__")[0]


def _build_maps(pddl_problem: str) -> tuple[dict[str, str], dict[str, str]]:
    """
    Build two maps from the GT PDDL problem:
      raw_to_clean: GT coord ID  → clean semantic name  (e.g. SoapBar_bar__... → soap_bar_1)
      clean_to_raw: clean name   → GT coord ID          (symbol_table for name resolution)

    Only covers objects and receptacles (objectType / receptacleType predicates).
    Location IDs (loc_bar_...) keep their raw form in the plan and are not remapped.
    """
    counters: dict[str, int] = {}
    raw_to_clean: dict[str, str] = {}

    # Restrict to :init section only — goal has existential vars like ?o that must not leak in
    init_match = re.search(r'\(:init(.*?)\(:goal', pddl_problem, re.DOTALL)
    init_text = init_match.group(1) if init_match else pddl_problem

    for line in init_text.split("\n"):
        line = line.strip()
        for pat in [
            r'\(objectType\s+(\S+)\s+\S+\)',
            r'\(receptacleType\s+(\S+)\s+\S+\)',
        ]:
            m = re.match(pat, line)
            if m:
                raw = m.group(1)
                if raw in raw_to_clean:
                    continue
                base = _camel_to_words(_strip_coords(raw)).replace(" ", "_")
                counters[base] = counters.get(base, 0) + 1
                raw_to_clean[raw] = f"{base}_{counters[base]}"

    clean_to_raw = {v: k for k, v in raw_to_clean.items()}
    return raw_to_clean, clean_to_raw


def _clean(raw: str, raw_to_clean: dict[str, str]) -> str:
    return raw_to_clean.get(raw, _strip_coords(raw))


def _format_init_as_nl(pddl_problem: str) -> str:
    """Convert PDDL :init section predicates to readable NL using clean names."""
    m = re.search(r'\(:init(.*?)\(:goal', pddl_problem, re.DOTALL)
    if not m:
        return ""
    init_text = m.group(1)
    raw_to_clean, _ = _build_maps(pddl_problem)

    lines = []
    for raw in init_text.split("\n"):
        raw = raw.strip()

        m1 = re.match(r'\(atLocation\s+(\S+)\s+(\S+)\)', raw)
        if m1:
            lines.append(f"Agent is at location {m1.group(2)}.")
            continue

        m2 = re.match(r'\(objectAtLocation\s+(\S+)\s+(\S+)\)', raw)
        if m2:
            lines.append(f"{_clean(m2.group(1), raw_to_clean)} is at location {m2.group(2)}.")
            continue

        m3 = re.match(r'\(receptacleAtLocation\s+(\S+)\s+(\S+)\)', raw)
        if m3:
            lines.append(f"{_clean(m3.group(1), raw_to_clean)} is at location {m3.group(2)}.")
            continue

        m4 = re.match(r'\(inReceptacle\s+(\S+)\s+(\S+)\)', raw)
        if m4 and "?" not in m4.group(1):
            lines.append(f"{_clean(m4.group(1), raw_to_clean)} is inside {_clean(m4.group(2), raw_to_clean)}.")
            continue

        m5 = re.match(r'\(openable\s+(\S+)\)', raw)
        if m5:
            lines.append(f"{_clean(m5.group(1), raw_to_clean)} can be opened.")
            continue

        m6 = re.match(r'\(cleanable\s+(\S+)\)', raw)
        if m6:
            lines.append(f"{_clean(m6.group(1), raw_to_clean)} can be cleaned in a sink.")
            continue

        m7 = re.match(r'\(heatable\s+(\S+)\)', raw)
        if m7:
            lines.append(f"{_clean(m7.group(1), raw_to_clean)} can be heated in a microwave.")
            continue

        m8 = re.match(r'\(coolable\s+(\S+)\)', raw)
        if m8:
            lines.append(f"{_clean(m8.group(1), raw_to_clean)} can be cooled in a fridge.")
            continue

        m9 = re.match(r'\(pickupable\s+(\S+)\)', raw)
        if m9:
            lines.append(f"{_clean(m9.group(1), raw_to_clean)} can be picked up.")
            continue

    return "\n".join(lines)


def load_game(game_file: str) -> Optional[AlfWorldTask]:
    game_dir = Path(game_file).parent
    traj_file = game_dir / "traj_data.json"
    if not traj_file.exists():
        return None

    with open(traj_file) as f:
        traj = json.load(f)
    with open(game_file) as f:
        tw = json.load(f)

    task_type = traj.get("task_type", "")
    params = traj.get("pddl_params", {})
    pddl_domain = tw.get("pddl_domain", "")
    pddl_problem = tw.get("pddl_problem", "")

    # Human NL annotation (first annotator), fallback to template
    anns = traj.get("turk_annotations", {}).get("anns", [])
    if anns and anns[0].get("task_desc"):
        nl_task = re.sub(r'\.?k\s*$', '.', anns[0]["task_desc"].strip())
    else:
        obj = _camel_to_words(params.get("object_target", "object"))
        rec = _camel_to_words(params.get("parent_target") or params.get("toggle_target", "receptacle"))
        nl_task = TASK_TEMPLATES.get(task_type, "Complete the task.").format(
            object=obj, receptacle=rec, toggle=rec
        )

    nl_scene = _format_init_as_nl(pddl_problem)
    nl_combined = f"Task: {nl_task}\n\nScene state:\n{nl_scene}"

    _, symbol_table = _build_maps(pddl_problem)

    return AlfWorldTask(
        game_file=game_file,
        task_id=traj.get("task_id", Path(game_file).parent.name),
        task_type=task_type,
        nl_task=nl_task,
        nl_scene=nl_scene,
        nl_combined=nl_combined,
        pddl_domain=pddl_domain,
        pddl_problem=pddl_problem,
        pddl_params=params,
        symbol_table=symbol_table,
    )


def sample_games(
    n_per_type: int = 5,
    task_types: Optional[list] = None,
    cache_dir: str = CACHE_DIR,
    seed: int = 42,
) -> list[AlfWorldTask]:
    """Sample N games per task type with valid .tw-pddl + traj_data.json."""
    rng = random.Random(seed)
    all_tw = glob.glob(os.path.join(cache_dir, "**", "*.tw-pddl"), recursive=True)

    by_type: dict[str, list] = {}
    for tw in all_tw:
        tt = Path(tw).parent.parent.name.split("-")[0]
        if task_types and tt not in task_types:
            continue
        by_type.setdefault(tt, []).append(tw)

    tasks = []
    for tt, files in sorted(by_type.items()):
        rng.shuffle(files)
        count = 0
        for f in files:
            if count >= n_per_type:
                break
            task = load_game(f)
            if task and task.pddl_problem:
                tasks.append(task)
                count += 1

    return tasks


if __name__ == "__main__":
    tasks = sample_games(n_per_type=1)
    for t in tasks:
        print(f"\n{'='*60}")
        print(f"Type : {t.task_type}")
        print(f"NL   : {t.nl_task}")
        print(f"Scene (first 300 chars):\n{t.nl_scene[:300]}")
        print(f"Symbol table ({len(t.symbol_table)} entries)")
