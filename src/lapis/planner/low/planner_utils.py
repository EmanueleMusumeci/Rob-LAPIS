# Copied from ContextMatters (repository link (https://github.com/Lab-RoCoCo-Sapienza/context-matters))
import os

# Add third-party/symk_wrapper to path if not present
import sys
symk_wrapper_path = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))), "third-party", "symk_wrapper")
if symk_wrapper_path not in sys.path:
    sys.path.append(symk_wrapper_path)
    
try:
    from symk_wrapper.symk import SymK
except ImportError:
    print(f"Warning: Could not import SymK from {symk_wrapper_path}")
    SymK = None

import traceback
import unified_planning as up
from unified_planning.shortcuts import OneshotPlanner
from unified_planning.io import PDDLReader
from unified_planning.environment import get_environment

# Disable strictly naming conflict check (for Floortile "up" and "down" labels)
get_environment().error_used_name = False

import re

def _preprocess_pddl(pddl_str):
    """
    Last-resort cleanup of non-standard PDDL constructs that break the UP parser.
    
    UP does not support `(either type1 type2)` union types anywhere.
    If the LLM still generates them despite prompt instructions, we replace
    with the first listed type as a safe fallback.
    
    NOTE: We do NOT strip `?var - type` from :parameters as that breaks PDDL.
    """
    def replace_either(m):
        inner = m.group(1)
        first_type = inner.strip().split()[0]
        return f'- {first_type}'
    pddl_str = re.sub(r'-\s*\(\s*either\s+([^)]+)\)', replace_either, pddl_str)
    
    # Strip (:objects ...) from domain files if the LLM hallucinated it
    if '(define (domain' in pddl_str:
        pddl_str = re.sub(r'\(\s*:objects[^)]+\)', '', pddl_str, flags=re.IGNORECASE)
        
    return pddl_str



def run_planner_UP(domain_file_path, problem_dir, planner_name="up_fd", timeout=180):
    """
    Robust planning using Unified Planning (UP).
    Handles non-standard PDDL and provides clear error logs.
    """
    try:
        reader = PDDLReader()
        problem_file_path = os.path.join(problem_dir, "problem.pddl")
        
        # Parse PDDL using UP's robust reader
        # Preprocess both domain and problem to remove problematic '(either ...)' syntax
        with open(domain_file_path, 'r') as f:
            domain_str = f.read()
        with open(problem_file_path, 'r') as f:
            problem_str = f.read()
            
        domain_str = _preprocess_pddl(domain_str)
        problem_str = _preprocess_pddl(problem_str)
        
        # Log preprocessed versions for debugging if reader fails
        try:
            problem = reader.parse_problem_string(domain_str, problem_str)
        except Exception as parse_e:
            print(f"DEBUG: PDDL PREPROCESS FAILED. Domain snippet:\n{domain_str[:500]}")
            print(f"DEBUG: Problem snippet:\n{problem_str[:500]}")
            raise parse_e
        
        # Use specifying planner if provided (e.g. 'up_fast_downward')
        # Map common names to UP names
        up_planner_name = "fast-downward" if "fd" in planner_name.lower() else "pyperplan"
        
        with OneshotPlanner(name=up_planner_name) as planner:
            result = planner.solve(problem, timeout=timeout)
            
            from unified_planning.engines import PlanGenerationResultStatus
            if result.status in [PlanGenerationResultStatus.SOLVED_SATISFICING, PlanGenerationResultStatus.SOLVED_OPTIMALLY]:
                # Convert UP plan to string representation for compatibility with pipeline logic
                # The pipeline expects a list of actions as strings
                plan = [str(action) for action in result.plan.actions]
                return plan, None, None, None
            else:
                return None, None, f"UP Planner failed with status: {result.status}", None
                
    except Exception as e:
        error_msg = f"Exception in UP parsing/planning: {str(e)}\n{traceback.format_exc()}"
        print(error_msg)
        return None, error_msg, None, None


def run_planner_native_fd(domain_file_path, problem_dir, timeout=180):
    """
    Call FastDownward directly (bypassing unified-planning's PDDLReader).
    Required for ADL domains like alfred.pddl that UP's wrapper cannot parse/solve.
    Uses FD's translate.py then the downward binary.
    """
    import subprocess, tempfile, shutil

    translate_py = os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        "../../../../.venv/lib/python3.12/site-packages/up_fast_downward/downward/builds/release/bin/translate/translate.py",
    )
    fd_bin = os.path.join(os.path.dirname(translate_py), "..", "downward")
    translate_py = os.path.normpath(translate_py)
    fd_bin = os.path.normpath(fd_bin)

    problem_file = os.path.join(problem_dir, "problem.pddl")
    tmpdir = tempfile.mkdtemp()
    sas_file = os.path.join(tmpdir, "output.sas")
    plan_out = os.path.join(tmpdir, "sas_plan")

    try:
        # Step 1: translate PDDL → SAS+
        tr = subprocess.run(
            [sys.executable, translate_py, domain_file_path, problem_file, "--sas-file", sas_file],
            capture_output=True, text=True, timeout=timeout,
        )
        if tr.returncode != 0 or not os.path.exists(sas_file):
            err = f"FD translate failed (rc={tr.returncode}):\n{(tr.stdout + tr.stderr)[-2000:]}"
            print(err)
            return None, err, None, None

        # Step 2: search (FD reads SAS+ from stdin, writes plan to sas_plan in cwd)
        plan_out = os.path.join(tmpdir, "sas_plan")
        with open(sas_file) as sas_in:
            fd = subprocess.run(
                [fd_bin, "--search", "astar(add())"],
                stdin=sas_in, capture_output=True, text=True, timeout=timeout, cwd=tmpdir,
            )
        if not os.path.exists(plan_out):
            err = f"FD search failed (rc={fd.returncode}):\n{fd.stderr[-2000:]}\n{fd.stdout[-1000:]}"
            print(err)
            return None, None, err, None

        with open(plan_out) as f:
            lines = [l.strip() for l in f if l.strip() and not l.startswith(";")]
        return lines, None, None, None

    except subprocess.TimeoutExpired:
        return None, None, f"FD timed out after {timeout}s", None
    except Exception as e:
        return None, None, f"FD exception: {e}\n{traceback.format_exc()}", None
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)


def run_planner_symk(domain_file_path, problem_dir, alias="seq-sat-lama-2011", timeout=180):
    """
    Call SymK directly via its fast-downward.py driver (bypassing UP).
    SymK binary must be compiled at third-party/symk_wrapper/SymK/builds/release/bin/.
    Default alias: seq-sat-lama-2011 (satisficing, same quality class as native FD astar(add())).
    """
    import subprocess, tempfile, shutil

    symk_driver = os.path.normpath(os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        "../../../../third-party/symk_wrapper/SymK/fast-downward.py",
    ))
    if not os.path.exists(symk_driver):
        return None, None, f"SymK driver not found at {symk_driver}", None

    problem_file = os.path.join(problem_dir, "problem.pddl")
    tmpdir = tempfile.mkdtemp()
    plan_file = os.path.join(tmpdir, "sas_plan")

    try:
        result = subprocess.run(
            [sys.executable, symk_driver,
             "--alias", alias,
             "--plan-file", plan_file,
             domain_file_path, problem_file],
            capture_output=True, text=True, timeout=timeout, cwd=tmpdir,
        )
        if not os.path.exists(plan_file):
            err = f"SymK failed (rc={result.returncode}):\n{result.stderr[-2000:]}\n{result.stdout[-1000:]}"
            print(err)
            return None, None, err, None

        with open(plan_file) as f:
            lines = [l.strip() for l in f if l.strip() and not l.startswith(";")]
        return lines, None, None, None

    except subprocess.TimeoutExpired:
        return None, None, f"SymK timed out after {timeout}s", None
    except Exception as e:
        return None, None, f"SymK exception: {e}\n{traceback.format_exc()}", None
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)


def plan_with_output(domain_file_path, problem_dir, plan_file_path, env=None, planner_name="fd", search_flag=None, timeout=180):

    # PLANNING #

    print("\n\n\tPerforming planning...")
    print(domain_file_path)
    print(os.path.join(problem_dir, "problem.pddl"))

    if planner_name.lower() == "native_fd":
        plan, pddlenv_error_log, planner_error_log, statistics = run_planner_native_fd(domain_file_path, problem_dir, timeout)
    elif planner_name.lower() == "symk":
        plan, pddlenv_error_log, planner_error_log, statistics = run_planner_symk(domain_file_path, problem_dir, timeout=timeout)
    elif planner_name.lower().startswith("up") or planner_name.lower() in ["fd", "pyperplan"]:
        plan, pddlenv_error_log, planner_error_log, statistics = run_planner_UP(domain_file_path, problem_dir, planner_name, timeout)

    # Save planner output
    with open(plan_file_path, "w") as file:
        if plan is not None:    
            if isinstance(plan, list):
                file.write("\n".join(plan) + "\n")
            else:
                file.write(str(plan) + "\n")
        elif pddlenv_error_log is not None:
            file.write(pddlenv_error_log)
        elif planner_error_log is not None:
            file.write(planner_error_log)

    return plan, pddlenv_error_log, planner_error_log, statistics