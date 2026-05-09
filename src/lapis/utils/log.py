import json
import os
import traceback

def copy_file(src: str, dst: str):
    """
    Copy a file from src to dst

    :param src: Source file path
    :param dst: Destination file path
    """
    with open(src, "r") as f:
        data = f.read()

    with open(dst, "w") as f:
        f.write(data)


def save_file(data: str, path: str):
    """
    Save a string to a file

    :param data: String to save
    :param path: Path to the file
    """
    with open(path, "w") as f:
        f.write(data)


def save_log_file(log_content: str, log_type: str, logs_dir: str) -> str:
    """
    Save a log file to the logs directory with a standardized naming convention.
    
    :param log_content: The content of the log to save
    :param log_type: Type of log (e.g., 'VAL_validation', 'VAL_grounding', 'planning', 'grounding')
    :param logs_dir: Directory where logs should be saved
    :return: Path to the saved log file
    """
    if not os.path.exists(logs_dir):
        os.makedirs(logs_dir)
    
    log_filename = f"{log_type}.log"
    log_path = os.path.join(logs_dir, log_filename)
    
    with open(log_path, "w") as f:
        f.write(str(log_content))
    
    return log_path


def save_statistics(
    phase,
    dir,
    workflow_iteration,
    subgoal_id=None,
    pddl_refinement_iteration=None,
    plan_successful=None,
    pddlenv_error_log=None,
    planner_error_log=None,
    planner_statistics=None,
    VAL_validation_log=None,
    VAL_grounding_log=None,
    scene_graph_grounding_log=None,
    grounding_success_percentage=None,
    db_entries=None,
    exception=None,
    domain_generation_time=None,
    scene_pruning_time=None,
    problem_generation_time=None,
    problem_decomposition_time=None,
    total_inference_time=None,
    VAL_validation_log_path=None,
    VAL_grounding_log_path=None,
    planning_log_path=None,
    grounding_log_path=None,
):

    data = {
        "subgoal_id": subgoal_id,
        "pddl_refinement_iteration": pddl_refinement_iteration,
        "plan_successful": plan_successful,
        "pddlenv_error_log": pddlenv_error_log,
        "planner_error_log": planner_error_log,
        "planner_statistics": planner_statistics,
        "VAL_validation_log": VAL_validation_log,
        "VAL_grounding_log": VAL_grounding_log,
        "scene_graph_grounding_log": scene_graph_grounding_log,
        "grounding_success_percentage": grounding_success_percentage,
        "db_entries": db_entries,
        "domain_generation_time": domain_generation_time,
        "scene_pruning_time": scene_pruning_time,
        "problem_generation_time": problem_generation_time,
        "problem_decomposition_time": problem_decomposition_time,
        "total_inference_time": total_inference_time,
        "VAL_validation_log_path": VAL_validation_log_path,
        "VAL_grounding_log_path": VAL_grounding_log_path,
        "planning_log_path": planning_log_path,
        "grounding_log_path": grounding_log_path,
    }

    if not os.path.exists(dir):
        os.makedirs(dir)

    stats_file = os.path.join(dir, "statistics.json")

    statistics = {}

    if not os.path.exists(stats_file):
        if phase is not None:
            statistics["statistics"] = {"0": {phase: {}}}

            if phase == "PDDL_REFINEMENT":
                statistics["statistics"]["0"][phase] = [data]
            else:
                statistics["statistics"]["0"][phase] = data
    else:
        with open(stats_file, "r") as f:
            statistics = json.load(f)

        if str(workflow_iteration) not in statistics["statistics"]:
            statistics["statistics"][str(workflow_iteration)] = {}

        target_stats = statistics["statistics"][str(workflow_iteration)]
        if subgoal_id is not None:
             subgoal_key = f"subgoal_{subgoal_id}"
             if subgoal_key not in target_stats:
                 target_stats[subgoal_key] = {}
             target_stats = target_stats[subgoal_key]

        if phase == "PDDL_REFINEMENT":
            if phase not in target_stats:
                target_stats[phase] = []
            target_stats[phase].append(data)
        else:
            target_stats[phase] = data

    if exception is not None:
        statistics["exception"] = {
            "reason": str(exception),
            "traceback": "".join(traceback.format_tb(exception.__traceback__)),
            "workflow_iteration": workflow_iteration,
            "refinement_iteration": pddl_refinement_iteration,
            "phase": phase,
        }

    with open(stats_file, "w") as f:
        json.dump(statistics, f, indent=4)

def dump_sayplan_plan(response, path):
    """
    Dump the SayPlan plan to a file.

    :param response: The SayPlan response to parse and dump
    :param path: The path to the file
    """
    start_idx = response.find("{")
    end_idx = response.find("}")
    response_dict = eval(response[start_idx:end_idx+1])
    plan = response_dict["plan"]
    if "```" in plan:
        plan = plan.split("```")[1]
    if "\\n" in plan:
        plan = plan.replace("\\n", "\n")
    response_lines = plan.split("\n")
    plan_list = [line[line.find("("):]
                 for line in response_lines if line.strip() != ""]
    plan_length = len(plan_list)
    with open(path, "w") as f:
        f.write("\n".join(plan_list))
        
    return plan_list, plan_length