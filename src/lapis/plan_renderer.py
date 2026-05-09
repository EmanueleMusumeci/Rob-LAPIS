"""
plan_renderer.py — Step a PDDL plan through UP SequentialSimulator and optionally
render per-step frames to a GIF.

Supports:
  - Any PDDL domain/problem pair (UP sim verification)
  - Graphical GIF output for Blocksworld (via BlocksworldSimulator.get_image())
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Optional


# ---------------------------------------------------------------------------
# Plan file parsing
# ---------------------------------------------------------------------------

def parse_plan_file(plan_path: str) -> list[str]:
    """
    Parse a plan_0.out file (UP/FD format) into PDDL-style action strings.

    Input line:  grasp(left, shaker1)
    Output item: (grasp left shaker1)

    Also handles already-PDDL lines like:  (unstack b2 b3)
    Empty plans (empty file) return [].
    """
    path = Path(plan_path)
    if not path.exists() or path.stat().st_size == 0:
        return []

    actions = []
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith(";"):
            continue
        # Strip leading "N: " prefix from translated_plan format
        line = re.sub(r"^\d+:\s*", "", line)
        actions.append(_to_pddl_action(line))
    return actions


def _to_pddl_action(line: str) -> str:
    """Convert  grasp(left, shaker1)  →  (grasp left shaker1)."""
    line = line.strip()
    # UP format: action_name(arg1, arg2, ...)
    m = re.match(r"^([\w][\w-]*)\(([^)]*)\)$", line)
    if m:
        name = m.group(1)
        args = [a.strip() for a in m.group(2).split(",") if a.strip()]
        return "({})".format(" ".join([name] + args))
    # Already space-separated format: (unstack b1 b3) or unstack b1 b3
    # Strip outer parens if present
    line = line.lstrip("(").rstrip(")")
    return "({})".format(line) if not line.startswith("(") else line


# ---------------------------------------------------------------------------
# UP simulator verification (domain-agnostic)
# ---------------------------------------------------------------------------

def simulate_plan(
    domain_file: str,
    problem_file: str,
    plan_path: str,
) -> tuple[bool, Optional[str], list[str]]:
    """
    Step through the plan with UPSequentialSimulator against the given problem.

    Returns:
        (goal_reached, failure_message, action_strings)
        goal_reached:    True if all actions applied and goal satisfied
        failure_message: description of first failure, or None
        action_strings:  parsed PDDL-style actions (useful for rendering)
    """
    action_strs = parse_plan_file(plan_path)

    try:
        from unified_planning.io import PDDLReader
        from unified_planning.shortcuts import SequentialSimulator, get_environment
        from unified_planning.plans import ActionInstance

        # Preprocess for UP-specific incompatibilities (action/predicate name collisions,
        # type unions in tyreworld/storage/floortile). Only needed for UP's SequentialSimulator.
        from .utils.pddl_preprocessor import preprocess_pddl_for_up
        try:
            domain_file, problem_file = preprocess_pddl_for_up(domain_file, problem_file)
        except Exception:
            pass  # Fall back to original files

        reader = PDDLReader()
        problem = reader.parse_problem(domain_file, problem_file)
        action_map = {a.name: a for a in problem.actions}
        env = get_environment()

        sim = SequentialSimulator(problem)
        state = sim.get_initial_state()

        for a_str in action_strs:
            clean = a_str.strip().lstrip("(").rstrip(")")
            parts = clean.split()
            if not parts:
                continue
            a_name = parts[0]
            param_names = parts[1:]
            if a_name not in action_map:
                return False, f"Action not in domain: {a_name}", action_strs
            action = action_map[a_name]
            params = []
            for p in param_names:
                obj = next((o for o in problem.all_objects if o.name == p), None)
                if obj is None:
                    return False, f"Object not found: {p}", action_strs
                params.append(env.expression_manager.ObjectExp(obj))
            ai = ActionInstance(action, tuple(params))
            if not sim.is_applicable(state, ai):
                unsatisfied = sim.get_unsatisfied_conditions(state, ai)
                return False, f"Not applicable: {clean} | unsatisfied: {unsatisfied}", action_strs
            state = sim.apply(state, ai)

        goal_reached = sim.is_goal(state)
        if not goal_reached:
            unsatisfied_goals = sim.get_unsatisfied_goals(state)
            return False, f"Goal not reached: {unsatisfied_goals}", action_strs
        return True, None, action_strs

    except Exception as e:
        return False, f"UP sim error: {e}", action_strs


# ---------------------------------------------------------------------------
# Graphical rendering — Blocksworld GIF
# ---------------------------------------------------------------------------

def render_blocksworld_gif(
    domain_file: str,
    problem_file: str,
    action_strs: list[str],
    output_path: str,
    fps: int = 2,
) -> bool:
    """
    Render a Blocksworld plan as an animated GIF.

    Requires: PIL (Pillow), unified_planning, BlocksworldSimulator.

    Returns True if GIF was saved, False otherwise.
    """
    try:
        from PIL import Image
        from unified_planning.io import PDDLReader
        from unified_planning.shortcuts import SequentialSimulator, get_environment
        from unified_planning.plans import ActionInstance
        # Import directly from the module file to avoid triggering __init__.py
        # (which pulls in BabyAI/AI2Thor simulators that may not be installed)
        import importlib.util, sys
        _spec = importlib.util.spec_from_file_location(
            "blocksworld_simulator",
            str(Path(__file__).parent / "simulators" / "blocksworld_simulator.py"),
        )
        _mod = importlib.util.module_from_spec(_spec)
        _spec.loader.exec_module(_mod)
        BlocksworldSimulator = _mod.BlocksworldSimulator
    except Exception:
        return False

    try:
        # Load GT problem
        reader = PDDLReader()
        problem = reader.parse_problem(domain_file, problem_file)

        # Build simulator backed by GT problem
        sim_wrapper = BlocksworldSimulator.__new__(BlocksworldSimulator)
        sim_wrapper.problem = problem
        sim_wrapper.simulator = SequentialSimulator(problem)
        sim_wrapper.current_state = sim_wrapper.simulator.get_initial_state()

        frames: list[Image.Image] = []

        # Initial state frame
        img = sim_wrapper.get_image(action_text="[Initial state]")
        if img:
            frames.append(img)

        # Build action map for step-by-step execution
        action_map = {a.name: a for a in problem.actions}
        env = get_environment()
        state = sim_wrapper.current_state

        up_sim = SequentialSimulator(problem)
        state = up_sim.get_initial_state()
        for a_str in action_strs:
            clean = a_str.strip().lstrip("(").rstrip(")")
            parts = clean.split()
            if not parts:
                continue
            a_name = parts[0]
            param_names = parts[1:]
            if a_name not in action_map:
                break
            action = action_map[a_name]
            params = []
            ok = True
            for p in param_names:
                obj = next((o for o in problem.all_objects if o.name == p), None)
                if obj is None:
                    ok = False
                    break
                params.append(env.expression_manager.ObjectExp(obj))
            if not ok:
                break
            ai = ActionInstance(action, tuple(params))
            if up_sim.is_applicable(state, ai):
                state = up_sim.apply(state, ai)
                sim_wrapper.current_state = state
                img = sim_wrapper.get_image(action_text=clean)
                if img:
                    frames.append(img)

        # Final state — longer pause
        if frames:
            frames.extend([frames[-1]] * fps)

        if not frames:
            return False

        duration = int(1000 / fps)
        frames[0].save(
            output_path,
            save_all=True,
            append_images=frames[1:],
            loop=0,
            duration=duration,
        )
        return True

    except Exception:
        return False


# ---------------------------------------------------------------------------
# Domain dispatcher
# ---------------------------------------------------------------------------

RENDERABLE_DOMAINS = {
    "blocksworld", "barman", "floortile", "grippers",
    "storage", "termes", "tyreworld"
}

# Map domain names to their simulator module and class
_DOMAIN_SIMULATOR_MAP = {
    "grippers":  ("src.lapis.simulators.grippers_simulator",  "GrippersSimulator"),
    "barman":    ("src.lapis.simulators.barman_simulator",    "BarmanSimulator"),
    "floortile": ("src.lapis.simulators.floortile_simulator", "FloortileSimulator"),
    "storage":   ("src.lapis.simulators.storage_simulator",   "StorageSimulator"),
    "termes":    ("src.lapis.simulators.termes_simulator",    "TermesSimulator"),
    "tyreworld": ("src.lapis.simulators.tyreworld_simulator", "TyreworldSimulator"),
}


def render_plan_gif(
    domain_name: str,
    domain_file: str,
    problem_file: str,
    plan_path: str,
    output_path: str,
) -> bool:
    """Render plan as GIF if a graphical simulator is available for this domain."""
    action_strs = parse_plan_file(plan_path)
    if not action_strs:
        return False

    norm = domain_name.lower()
    if "blocksworld" in norm:
        return render_blocksworld_gif(domain_file, problem_file, action_strs, output_path)

    # All other domains use the generic renderer
    for key, (mod_path, cls_name) in _DOMAIN_SIMULATOR_MAP.items():
        if key in norm:
            return _render_generic_gif(
                cls_name, mod_path,
                domain_file, problem_file, action_strs, output_path
            )

    return False


def _render_generic_gif(
    cls_name: str,
    mod_path: str,
    domain_file: str,
    problem_file: str,
    action_strs: list[str],
    output_path: str,
    fps: int = 2,
) -> bool:
    """
    Generic GIF renderer using a PDDLSimulator subclass.

    Instantiates the named simulator class, steps through the plan,
    collects PIL frames from get_image(), and saves as animated GIF.
    """
    try:
        from PIL import Image
        import importlib
        from unified_planning.io import PDDLReader
        from unified_planning.shortcuts import SequentialSimulator, get_environment
        from unified_planning.plans import ActionInstance

        mod = importlib.import_module(mod_path)
        SimClass = getattr(mod, cls_name)

        # Preprocess for UP-specific incompatibilities. Only needed for UP's SequentialSimulator.
        from .utils.pddl_preprocessor import preprocess_pddl_for_up
        try:
            domain_file, problem_file = preprocess_pddl_for_up(domain_file, problem_file)
        except Exception:
            pass

        sim = SimClass()
        if not sim.setup(domain_file, problem_file):
            return False

        # Re-parse problem for action stepping (use preprocessed files)
        reader = PDDLReader()
        problem = reader.parse_problem(domain_file, problem_file)
        action_map = {a.name: a for a in problem.actions}
        env = get_environment()
        up_sim = SequentialSimulator(problem)
        state = up_sim.get_initial_state()
        sim.current_state = state

        frames: list[Image.Image] = []
        img0 = sim.get_image(action_text="[Initial state]")
        if img0:
            frames.append(img0)

        for a_str in action_strs:
            clean = a_str.strip().lstrip("(").rstrip(")")
            parts = clean.split()
            if not parts:
                continue
            a_name = parts[0]
            if a_name not in action_map:
                a_name = a_name.replace("_", "-")
            if a_name not in action_map:
                continue

            action = action_map[a_name]
            params = []
            ok = True
            for p in parts[1:]:
                obj = next((o for o in problem.all_objects if o.name == p), None)
                if obj is None:
                    ok = False
                    break
                params.append(env.expression_manager.ObjectExp(obj))
            if not ok:
                continue

            ai = ActionInstance(action, tuple(params))
            if up_sim.is_applicable(state, ai):
                state = up_sim.apply(state, ai)
                sim.current_state = state
                img = sim.get_image(action_text=clean)
                if img:
                    frames.append(img)

        if frames:
            frames.extend([frames[-1]] * fps)
            duration = int(1000 / fps)
            frames[0].save(
                output_path,
                save_all=True,
                append_images=frames[1:],
                loop=0,
                duration=duration,
            )
            return True

    except Exception:
        pass
    return False
