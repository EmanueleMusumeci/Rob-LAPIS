"""
VirtualHome task loader for LAPIS benchmark.

Reads pre-processed GT PDDL problems from final_results/vh2_gen_domain/
(one per task type, from the EAI NeurIPS 2024 benchmark).

NL scene is derived from the GT PDDL :init section predicates.
NL task descriptions are curated per task type.
"""

import re
from pathlib import Path
from dataclasses import dataclass, field
from typing import Optional

# Source of GT PDDL problems — pre-processed from EAI benchmark
_DATA_ROOT = Path(__file__).parent.parent.parent / "final_results" / "vh2_gen_domain"

NL_TASKS = {
    "Browse_internet":              "Sit at the computer and browse the internet.",
    "Brush_teeth":                  "Go to the bathroom and brush your teeth.",
    "Change_TV_channel":            "Turn on the television and change the channel.",
    "Cook_some_food":               "Cook food using the kitchen appliance.",
    "Drink":                        "Pick up a drink and consume it.",
    "Go_to_sleep":                  "Go to the bedroom and lie down on the bed to sleep.",
    "Go_to_toilet":                 "Go to the bathroom and use the toilet.",
    "Listen_to_music":              "Turn on the music player and listen to music.",
    "Make_coffee":                  "Make coffee using the coffee maker.",
    "Pick_up_phone":                "Find the phone and pick it up.",
    "Put_groceries_in_Fridge":      "Put the groceries inside the fridge.",
    "Read_book":                    "Find the book and hold it to read.",
    "Relax_on_sofa":                "Sit on the sofa and relax.",
    "Set_up_table":                 "Set up the table with plates and utensils.",
    "Turn_on_light":                "Find the lamp and turn it on.",
    "Wash_clothes":                 "Load the washing machine with clothes and start it.",
    "Wash_dishes_with_dishwasher":  "Load the dishwasher and run it to wash the dishes.",
    "Wash_hands":                   "Go to the bathroom and wash your hands.",
    "Watch_TV":                     "Turn on the television and watch it.",
    "Work":                         "Go to the home office and turn on the computer to work.",
}


@dataclass
class VHTask:
    task_id: str
    task_type: str
    nl_task: str
    nl_scene: str
    pddl_problem: str


def _pddl_init_to_nl(pddl_problem: str) -> str:
    """Convert PDDL :init predicates to readable NL sentences."""
    m = re.search(r'\(:init(.*?)\(:goal', pddl_problem, re.DOTALL)
    if not m:
        # Try without goal
        m = re.search(r'\(:init(.*?)\)', pddl_problem, re.DOTALL)
    if not m:
        return ""

    init_text = m.group(1)
    lines = []

    for raw in init_text.splitlines():
        raw = raw.strip()

        m1 = re.match(r'\(inside\s+(\S+)\s+(\S+)\)', raw)
        if m1:
            lines.append(f"The {m1.group(1)} is inside the {m1.group(2).replace('_', ' ')}.")
            continue

        m2 = re.match(r'\(inside_room\s+(\S+)\s+(\S+)\)', raw)
        if m2:
            lines.append(f"The {m2.group(1).replace('_', ' ')} is in the {m2.group(2).replace('_', ' ')}.")
            continue

        m3 = re.match(r'\(obj_inside\s+(\S+)\s+(\S+)\)', raw)
        if m3:
            lines.append(f"The {m3.group(1).replace('_', ' ')} is inside the {m3.group(2).replace('_', ' ')}.")
            continue

        m4 = re.match(r'\(obj_ontop\s+(\S+)\s+(\S+)\)', raw)
        if m4:
            lines.append(f"The {m4.group(1).replace('_', ' ')} is on top of the {m4.group(2).replace('_', ' ')}.")
            continue

        m5 = re.match(r'\(obj_next_to\s+(\S+)\s+(\S+)\)', raw)
        if m5:
            lines.append(f"The {m5.group(1).replace('_', ' ')} is next to the {m5.group(2).replace('_', ' ')}.")
            continue

        m6 = re.match(r'\(on\s+(\S+)\s+(\S+)\)', raw)
        if m6:
            lines.append(f"The {m6.group(1).replace('_', ' ')} is on the {m6.group(2).replace('_', ' ')}.")
            continue

        for pred in ['plugged_in', 'plugged_out', 'switched_on', 'switched_off',
                     'open', 'closed', 'sitting', 'lying', 'clean', 'dirty',
                     'movable', 'surfaces', 'has_plug', 'readable', 'grabbable']:
            mp = re.match(rf'\({pred}\s+(\S+)\)', raw)
            if mp:
                obj = mp.group(1).replace('_', ' ')
                state = pred.replace('_', ' ')
                lines.append(f"The {obj} is {state}.")
                break

    return "\n".join(lines) if lines else pddl_problem


def load_tasks(
    task_types: Optional[list] = None,
    data_root: Path = _DATA_ROOT,
) -> list:
    """Load VHTask objects from pre-processed GT PDDL problems."""
    tasks = []
    for task_dir in sorted(data_root.iterdir()):
        if task_dir.name == "summary.json" or not task_dir.is_dir():
            continue

        task_id = task_dir.name
        # task_id format: TaskType_sceneNum_instance (e.g. Browse_internet_384_1)
        # Strip trailing _<digit> and _<digit> to recover task_type
        parts = task_id.rsplit("_", 2)
        if len(parts) == 3 and parts[1].isdigit() and parts[2].isdigit():
            task_type = parts[0]
        elif len(parts) >= 2 and parts[-1].isdigit():
            task_type = "_".join(parts[:-1])
        else:
            task_type = task_id

        if task_types and task_type not in task_types:
            continue

        gt_problem_path = task_dir / "gt_problem.pddl"
        if not gt_problem_path.exists():
            continue

        pddl_problem = gt_problem_path.read_text()
        nl_scene = _pddl_init_to_nl(pddl_problem)
        nl_task = NL_TASKS.get(task_type, f"Complete the {task_type.replace('_', ' ').lower()} task.")

        tasks.append(VHTask(
            task_id=task_id,
            task_type=task_type,
            nl_task=nl_task,
            nl_scene=nl_scene,
            pddl_problem=pddl_problem,
        ))

    return tasks


if __name__ == "__main__":
    tasks = load_tasks()
    print(f"Loaded {len(tasks)} VH tasks:")
    for t in tasks:
        print(f"  {t.task_id}: {t.nl_task}")
        print(f"    scene ({len(t.nl_scene.splitlines())} lines): {t.nl_scene[:80]}...")
