import subprocess
import os
from pathlib import Path
from src.lapis.logger_cfg import logger

def verify_plan_with_compiled_val(compiled_domain_path, compiled_problem_path, plan_path):
    """
    Verifies a generated plan against the strictly compiled PDDL problem (where constraints are baked in).
    Returns (success: bool, val_output: str)
    """
    # Adjust this path based on where 'src/lapis/utils' is relative to 'third-party'
    # Assuming repository root is Parent of src
    repo_root = Path(__file__).parent.parent.parent.parent
    val_executable = str(repo_root / "third-party" / "VAL" / "build" / "linux64" / "Release" / "bin" / "Validate")
    
    if not os.path.exists(val_executable):
        logger.error(f"VAL executable not found at {val_executable}")
        return False, f"VAL executable not found at {val_executable}"

    if not os.path.exists(compiled_domain_path) or not os.path.exists(compiled_problem_path):
        logger.warning(f"Compiled PDDL files not found. Cannot verify ground truth against {compiled_domain_path} or {compiled_problem_path}")
        return False, "Compiled PDDLs missing"

    if not os.path.exists(plan_path):
        logger.warning(f"Plan file not found: {plan_path}")
        return False, "Plan file missing"

    try:
        # Run VAL: Validate domain.pddl problem.pddl plan.txt
        result = subprocess.run(
            [val_executable, "-v", str(compiled_domain_path), str(compiled_problem_path), str(plan_path)],
            capture_output=True,
            text=True,
            check=False
        )
        output = result.stdout + "\n" + result.stderr
        
        # According to VAL, success usually corresponds to return_code 0 
        success = (result.returncode == 0)
        
        reason = "Success"
        if not success:
            reason = _parse_val_failure_reason(output)
            
        return success, reason
    except Exception as e:
        logger.error(f"Error running VAL compiled check: {e}")
        return False, str(e)

def _parse_val_failure_reason(val_output):
    """Parses VAL output to extract a human-readable, brief failure reason."""
    lines = val_output.splitlines()
    for i, line in enumerate(lines):
        line = line.strip()
        if "Plan failed because of unsatisfied precondition" in line:
            return "Unsatisfied precondition at time step"
        if "has an unsatisfied precondition at time" in line:
            return line
        if "Error:" in line or "error:" in line.lower():
            return line
        if line.startswith("Suspect action"):
            return line
        if "Goal not satisfied" in line or "Plan failed to execute" in line:
             return "Plan failed to reach the goal or execute fully"
        if "Syntax error" in line:
            return "Syntax error in plan"
    
    # Fallback to returning the first 100 characters of stderr/stdout if no match found
    return (val_output[:100] + '...') if len(val_output) > 100 else (val_output or "Unknown Failure")
