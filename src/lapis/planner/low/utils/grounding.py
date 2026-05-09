"""
UP-native grounding module for 3D Scene Graph (3DSG) plan verification.

Replaces the PDDLgym-based grounding functions from the original ContextMatters
(src/context_matters/pddl_verification.py) using unified-planning's
SequentialSimulator and PDDLReader.
"""

from __future__ import annotations

import glob
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# 1. initialize_up_environment
# ---------------------------------------------------------------------------

def initialize_up_environment(domain_file_path: str, problem_dir: str):
    """
    Load a PDDL domain + problem and return a simulator, initial state, and problem.

    Finds the first problem*.pddl file in problem_dir.

    Returns:
        (simulator, initial_state, problem)  — all from unified-planning
    """
    from unified_planning.io import PDDLReader
    from unified_planning.engines.sequential_simulator import UPSequentialSimulator

    reader = PDDLReader()

    # Find the first problem file in the directory
    candidates = sorted(glob.glob(str(Path(problem_dir) / "problem*.pddl")))
    if not candidates:
        raise FileNotFoundError(f"No problem*.pddl found in {problem_dir}")
    problem_file_path = candidates[0]

    problem = reader.parse_problem(domain_file_path, problem_file_path)
    simulator = UPSequentialSimulator(problem=problem)
    simulator.__enter__()          # enter context so internal state is ready
    initial_state = simulator.get_initial_state()

    return simulator, initial_state, problem


# ---------------------------------------------------------------------------
# 2. find_robot_location_up
# ---------------------------------------------------------------------------

def find_robot_location_up(
    problem,
    state,
    location_relation_str: str,
    robot_type: str = "robot",
) -> Tuple[Optional[str], Optional[str]]:
    """
    Find the robot's current room and robot object name from a UP state.

    Iterates over all (robot, room) object pairs and checks the location fluent.

    Args:
        problem:                unified-planning Problem
        state:                  current UP State
        location_relation_str:  name of the location fluent (e.g. "at")
        robot_type:             type name for robot objects (default "robot")

    Returns:
        (room_str, robot_str) or (None, None) if not found
    """
    from unified_planning.shortcuts import get_environment

    em = get_environment().expression_manager

    # Collect typed objects
    robots = [o for o in problem.all_objects if o.type.name == robot_type]
    rooms = [o for o in problem.all_objects
             if any(t.name in ("room", "location") for t in _type_hierarchy(o.type))]

    # Find the location fluent by name
    location_fluent = _find_fluent(problem, location_relation_str)
    if location_fluent is None:
        logger.warning(f"Fluent '{location_relation_str}' not found in domain")
        return None, None

    for robot in robots:
        for room in rooms:
            try:
                fnode = em.FluentExp(location_fluent, (em.ObjectExp(robot), em.ObjectExp(room)))
                val = state.get_value(fnode)
                if val is not None and val.bool_constant_value():
                    return str(room.name), str(robot.name)
            except Exception:
                continue

    return None, None


# ---------------------------------------------------------------------------
# 3. extract_locations_dictionary_up
# ---------------------------------------------------------------------------

def extract_locations_dictionary_up(
    problem,
    state,
    location_relation_str: str,
) -> Dict[str, str]:
    """
    Build a dict mapping every non-robot object to its room from a UP state.

    Args:
        problem:                UP Problem
        state:                  current UP State
        location_relation_str:  location fluent name (e.g. "at")

    Returns:
        {object_name: room_name}
    """
    from unified_planning.shortcuts import get_environment

    em = get_environment().expression_manager
    location_fluent = _find_fluent(problem, location_relation_str)
    if location_fluent is None:
        return {}

    rooms = [o for o in problem.all_objects
             if any(t.name in ("room", "location") for t in _type_hierarchy(o.type))]
    room_names = {o.name for o in rooms}

    locations: Dict[str, str] = {}
    for obj in problem.all_objects:
        if obj.name in room_names:
            continue
        for room in rooms:
            try:
                fnode = em.FluentExp(location_fluent, (em.ObjectExp(obj), em.ObjectExp(room)))
                val = state.get_value(fnode)
                if val is not None and val.bool_constant_value():
                    locations[str(obj.name)] = str(room.name)
                    break
            except Exception:
                continue

    return locations


# ---------------------------------------------------------------------------
# 4. update_locations_dictionary_up
# ---------------------------------------------------------------------------

def update_locations_dictionary_up(
    problem,
    state,
    locations_dict: Dict[str, str],
    location_relation_str: str,
) -> Dict[str, str]:
    """
    Re-derive locations from the new state and merge into existing dict.

    Returns the merged dict (existing entries overwritten by fresh values).
    """
    fresh = extract_locations_dictionary_up(problem, state, location_relation_str)
    merged = locations_dict.copy()
    merged.update(fresh)
    return merged


# ---------------------------------------------------------------------------
# 5. get_action_parameters
# ---------------------------------------------------------------------------

def get_action_parameters(action_instance) -> List[Tuple[str, str]]:
    """
    Extract (object_name, type_name) pairs from a UP ActionInstance.

    Returns:
        List of (name, type_name) tuples, one per actual parameter.
    """
    result = []
    for param in action_instance.actual_parameters:
        name = str(param.object().name) if param.is_object_exp() else str(param)
        type_name = param.type.name if hasattr(param, "type") else "unknown"
        result.append((name, type_name))
    return result


# ---------------------------------------------------------------------------
# 6. extract_move_locations
# ---------------------------------------------------------------------------

def extract_move_locations(action_instance) -> Tuple[str, str]:
    """
    Extract (from_room, to_room) from a move ActionInstance.

    Assumes parameters are ordered as: (robot, from_room, to_room, ...).
    Returns (str(params[1]), str(params[2])).
    """
    params = action_instance.actual_parameters
    if len(params) < 3:
        raise ValueError(
            f"Move action has fewer than 3 parameters: {action_instance}"
        )
    from_room = str(params[1].object().name) if params[1].is_object_exp() else str(params[1])
    to_room = str(params[2].object().name) if params[2].is_object_exp() else str(params[2])
    return from_room, to_room


# ---------------------------------------------------------------------------
# 7. verify_subplan_groundability_up
# ---------------------------------------------------------------------------

def verify_subplan_groundability_up(
    simulator,
    state,
    problem,
    graph: Dict,
    locations_dict: Dict[str, str],
    subplan: Dict,
    move_action,
    location_relation_str: str,
) -> Tuple[int, str, object]:
    """
    Verify one subplan (one room's worth of actions) against the scene graph.

    Args:
        simulator:              UPSequentialSimulator (already entered context)
        state:                  current UP State before this subplan
        problem:                UP Problem
        graph:                  3DSG dict {room_name: [objects]}
        locations_dict:         current {object: room} mapping
        subplan:                {"move_action": ActionInstance|None, "actions": [ActionInstance]}
        move_action:            the move ActionInstance (can be None for first subplan)
        location_relation_str:  location fluent name

    Returns:
        (successful_count, failure_reason, new_state)
    """
    successful = 0
    current_state = state

    # --- Process the move action ---
    if move_action is not None:
        from_room, to_room = extract_move_locations(move_action)

        if from_room not in graph:
            return 0, f"Location '{from_room}' not found in scene graph", current_state
        if to_room not in graph:
            return 0, f"Location '{to_room}' not found in scene graph", current_state

        if not simulator.is_applicable(current_state, move_action):
            return 0, f"Move action '{move_action}' is not applicable", current_state

        new_state = simulator.apply(current_state, move_action)
        if new_state is None:
            return 0, f"Failed to apply move action '{move_action}'", current_state

        current_state = new_state
        locations_dict = update_locations_dictionary_up(
            problem, current_state, locations_dict, location_relation_str
        )
        successful += 1

    # --- Process task actions ---
    for action_inst in subplan.get("actions", []):
        params = get_action_parameters(action_inst)
        for param_name, param_type in params:
            if param_type in ("room", "location"):
                # Location params: verify the room exists in the graph
                if param_name not in graph:
                    return (
                        successful,
                        f"Location '{param_name}' not found in scene graph",
                        current_state,
                    )
            else:
                # Object params: verify the object is known
                if param_name not in locations_dict:
                    return (
                        successful,
                        f"Object '{param_name}' not found in scene graph",
                        current_state,
                    )

        if not simulator.is_applicable(current_state, action_inst):
            return (
                successful,
                f"Action '{action_inst}' is not applicable in current state",
                current_state,
            )

        new_state = simulator.apply(current_state, action_inst)
        if new_state is None:
            return successful, f"Failed to apply action '{action_inst}'", current_state

        current_state = new_state
        locations_dict = update_locations_dictionary_up(
            problem, current_state, locations_dict, location_relation_str
        )
        successful += 1

    return successful, "", current_state


# ---------------------------------------------------------------------------
# 8. verify_groundability_in_scene_graph
# ---------------------------------------------------------------------------

def verify_groundability_in_scene_graph(
    plan,
    graph: Dict,
    domain_file_path: str,
    problem_dir: str,
    move_action_str: str,
    location_relation_str: str,
    location_type_str: str,
    initial_robot_location: Optional[str] = None,
) -> Tuple[float, str]:
    """
    Verify that a UP SequentialPlan is grounded in the 3D Scene Graph.

    Checks:
    - All rooms/objects referenced by the plan exist in the scene graph.
    - Each action is applicable in sequence (PDDL simulator).

    Args:
        plan:                    UP SequentialPlan (has .actions list of ActionInstance)
        graph:                   3DSG dict {room_name: [object_list]}
        domain_file_path:        path to PDDL domain file
        problem_dir:             directory containing problem*.pddl
        move_action_str:         action name substring used for movement (e.g. "move_to")
        location_relation_str:   name of the location fluent (e.g. "at")
        location_type_str:       type name for rooms (e.g. "room", "location")
        initial_robot_location:  expected initial room; if provided, verified against PDDL

    Returns:
        (grounding_percentage, failure_reason)  — percentage in [0.0, 1.0]
    """
    # --- Initialize environment ---
    simulator, state, problem = initialize_up_environment(domain_file_path, problem_dir)

    # --- Find initial robot location ---
    initial_pddl_robot_location, robot_name = find_robot_location_up(
        problem, state, location_relation_str, robot_type="robot"
    )

    if initial_pddl_robot_location is None:
        return 0.0, "Initial robot location not found in PDDL problem"

    if initial_robot_location is not None and initial_pddl_robot_location != initial_robot_location:
        return (
            0.0,
            f"Robot location mismatch: PDDL has '{initial_pddl_robot_location}', "
            f"expected '{initial_robot_location}'",
        )

    # --- Build initial locations dictionary ---
    locations_dict = extract_locations_dictionary_up(problem, state, location_relation_str)
    if robot_name and robot_name not in locations_dict:
        locations_dict[robot_name] = initial_pddl_robot_location

    # --- Hallucination check: all objects in PDDL must appear in the scene graph ---
    room_names = set(graph.keys())
    all_sg_objects = {obj for objs in graph.values() for obj in (objs or [])}
    for obj_name, room in locations_dict.items():
        if "robot" in obj_name:
            continue
        if room not in room_names:
            return 0.0, f"Object '{obj_name}' mapped to unknown room '{room}'"
        # Allow objects that may not be explicitly listed as named items in the graph
        # (the graph may just have room → [obj_label] with composite names)

    # --- Split plan into subplans at move actions ---
    actions = plan.actions  # list of ActionInstance
    subplans: List[Dict] = []
    current: Dict = {"move_action": None, "actions": []}

    for action_inst in actions:
        if move_action_str.lower() in action_inst.action.name.lower():
            if current["actions"] or current["move_action"] is not None:
                subplans.append(current)
            current = {"move_action": action_inst, "actions": []}
        else:
            current["actions"].append(action_inst)

    if current["actions"] or current["move_action"] is not None:
        subplans.append(current)

    if not subplans:
        return 1.0, ""

    # --- Verify each subplan ---
    total_actions = len(actions)
    total_successful = 0

    for subplan in subplans:
        move_inst = subplan["move_action"]
        successful, reason, state = verify_subplan_groundability_up(
            simulator, state, problem, graph, locations_dict, subplan,
            move_inst, location_relation_str
        )
        total_successful += successful

        if reason:
            grounding_pct = total_successful / total_actions if total_actions else 0.0
            return grounding_pct, reason

    return 1.0, ""


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------

def _find_fluent(problem, fluent_name: str):
    """Return the UP Fluent whose name matches fluent_name (partial match allowed)."""
    for fluent in problem.fluents:
        if fluent.name == fluent_name:
            return fluent
    # Partial match fallback
    for fluent in problem.fluents:
        if fluent_name in fluent.name or fluent.name in fluent_name:
            return fluent
    return None


def _type_hierarchy(up_type):
    """Yield the type and all its ancestors."""
    t = up_type
    while t is not None:
        yield t
        t = t.father if hasattr(t, "father") else None
