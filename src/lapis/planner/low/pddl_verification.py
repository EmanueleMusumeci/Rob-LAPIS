import os 
from pathlib import Path
import subprocess
import re
import logging
from src.lapis.logger_cfg import logger
from src.lapis.validators import VerificationService

def translate_plan(input_arg, output_path=None):
    """
    Utility for backward compatibility and VAL formatting. 
    1) If input_arg is a list (actions), convert to strings.
    2) If input_arg is a path and output_path is provided, copy/format file.
    """
    if isinstance(input_arg, list):
        if not input_arg:
            return []
        return [str(a) for a in input_arg]
    
    # File-to-file translation for VAL
    if output_path:
        with open(input_arg, 'r') as f:
            lines = f.readlines()
        
        with open(output_path, 'w') as f:
            for i, line in enumerate(lines):
                line = line.strip()
                if not line: continue
                # VAL expects: 0: (action)
                if not line.startswith('('):
                    line = f"({line})"
                f.write(f"{i}: {line}\n")

def VAL_validate(
    domain_file_path,
    problem_file_path=None,
    plan_file_path=None,
    run_semantic=False,
    verification_service=None,
):
    BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))
    validate_path = os.path.join(BASE_DIR, "third-party", "VAL", "build", "linux64", "Release", "bin", "Validate")
    command = [validate_path, domain_file_path]
    if problem_file_path:
        command.append(problem_file_path)
    if plan_file_path:
        command.append(plan_file_path)
    
    result = subprocess.run(command, capture_output=True, text=True)
    output_text = result.stdout
    semantic_is_valid = True

    if run_semantic and problem_file_path:
        try:
            service = verification_service or VerificationService()
            domain_text = Path(domain_file_path).read_text()
            problem_text = Path(problem_file_path).read_text()
            semantic_result = service.verify(domain_text, problem_text, run_asp=True)
            semantic_is_valid = semantic_result.valid
            output_text += "\n\n[Semantic Verification]\n" + semantic_result.to_text()
        except Exception as exc:
            semantic_is_valid = False
            output_text += f"\n\n[Semantic Verification]\nfailed: {exc}"

    # VAL output usually contains "Errors: 0, warnings: 5"
    # We must check if the number of errors is actually > 0.
    error_match = re.search(r"Errors:\s*(\d+)", output_text)
    if error_match:
        num_errors = int(error_match.group(1))
        if num_errors > 0:
            return False, output_text
        return semantic_is_valid, output_text
    
    # Fallback status checks
    if "Bad pddl" in output_text or "Check failed" in output_text:
        return False, output_text
    
    return semantic_is_valid, output_text

def VAL_ground(domain_file_path, problem_file_path=None):
    BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))
    validate_path = os.path.join(BASE_DIR, "third-party", "VAL", "build", "linux64", "Release", "bin", "Instantiate")
    command = [validate_path, domain_file_path]
    if problem_file_path:
        command.append(problem_file_path)
    
    result = subprocess.run(command, capture_output=True, text=True)
    output_text = result.stdout

    if "Errors Encountered" in output_text:
        return False, output_text
    else:
        return True, output_text

def verify_plan_with_up_simulator(domain_file_path, problem_file_path, plan_action_strs):
    try:
        from unified_planning.io import PDDLReader
        from unified_planning.plans import SequentialPlan, ActionInstance
        from unified_planning.shortcuts import get_environment
        from unified_planning.engines.sequential_simulator import UPSequentialSimulator

        reader = PDDLReader()
        problem = reader.parse_problem(domain_file_path, problem_file_path)

        action_map = {a.name: a for a in problem.actions}

        with UPSequentialSimulator(problem=problem) as sim:
            state = sim.get_initial_state()
            for a_str in plan_action_strs:
                a_str = a_str.strip().lstrip('(').rstrip(')')
                parts = a_str.split()
                action_name = parts[0]
                param_names = parts[1:]
                if action_name not in action_map:
                    return False, f"Action not in domain: {action_name}"
                action = action_map[action_name]
                
                # Ground the action
                params = []
                for p_name in param_names:
                    # Search for object by name
                    obj = None
                    # First check objects
                    for o in problem.all_objects:
                        if o.name == p_name:
                            obj = o
                            break
                    if obj is None:
                        return False, f"Object {p_name} not found in problem."
                    params.append(get_environment().expression_manager.ObjectExp(obj))
                
                action_instance = ActionInstance(action, tuple(params))
                
                if not sim.is_applicable(state, action_instance):
                    return False, f"Action {a_str} is not applicable in current state."
                
                state = sim.apply(state, action_instance)
                if state is None:
                    return False, f"Failed to apply action {a_str}."

            return sim.is_goal(state), None
    except Exception as e:
        return False, f"UP Simulation Error: {str(e)}"

class VALVerifier:
    def verify(self, domain_path, problem_path=None, plan_path=None):
        return VAL_validate(domain_path, problem_path, plan_path)

    def parse_output(self, output):
        issues = []
        if not output:
             return issues
        
        # Simple parsing for VAL errors
        lines = output.split('\n')
        for line in lines:
            if "Error:" in line or "Warning:" in line:
                issues.append(line.strip())
        return issues