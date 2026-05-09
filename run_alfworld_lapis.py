#!/usr/bin/env python3
"""
LAPIS benchmark on AlfWorld household tasks.

For each sampled AlfWorld game:
  1. Load task NL (human annotation) + full scene NL (formatted from GT PDDL init)
  2. Run LAPIS grounded_planning with alfred.pddl as GT domain
  3. VAL-validate generated plan against GT PDDL problem
  4. Optionally execute plan in AlfredTWEnv and record task success

Usage:
    python run_alfworld_lapis.py                          # 5 games per task type, no execution
    python run_alfworld_lapis.py --n 3 --execute          # 3 per type, execute in env
    python run_alfworld_lapis.py --task_types pick_and_place_simple pick_clean_then_place_in_recep
    python run_alfworld_lapis.py --model claude-sonnet-4-6 --iterations 3
"""

import argparse
import json
import os
import sys
import time
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from dotenv import load_dotenv
load_dotenv()

from data.alfworld.name_remapper import resolve_plan, objects_block

ALFRED_PDDL = Path(__file__).parent / "data/alfworld/alfred_patched.pddl"
RESULTS_DIR = Path(__file__).parent / "results_alfworld"

ALL_TASK_TYPES = [
    "pick_and_place_simple",
    "look_at_obj_in_light",
    "pick_clean_then_place_in_recep",
    "pick_heat_then_place_in_recep",
    "pick_cool_then_place_in_recep",
    "pick_two_obj_and_place",
]


def build_agent(model: str):
    if "claude" in model:
        from src.lapis.agents.claude import ClaudeAgent
        return ClaudeAgent(model=model)
    elif "gemini" in model:
        from src.lapis.agents.gemini import GeminiAgent
        return GeminiAgent(model=model)
    else:
        from src.lapis.agents.gpt import GPTAgent
        return GPTAgent(model=model)


def run_lapis_on_task(task, pipeline, results_dir: Path, domain_file_path=None, domain_description=None, check_adequacy: bool = False) -> dict:
    """Run LAPIS on one AlfWorldTask. Returns a result dict."""
    problem_dir = results_dir / task.task_id.replace("/", "_")
    problem_dir.mkdir(parents=True, exist_ok=True)

    # Save GT domain and problem for reference
    gt_domain_path = problem_dir / "gt_domain.pddl"
    gt_problem_path = problem_dir / "gt_problem.pddl"
    gt_domain_path.write_text(task.pddl_domain)
    gt_problem_path.write_text(task.pddl_problem)

    pipeline.agent.reset_token_counts()
    t0 = time.time()
    try:
        results = pipeline.grounded_planning(
            current_goal_text=task.nl_task,
            extracted_sg_str=task.nl_scene,
            domain_file_path=domain_file_path,
            domain_description=domain_description,
            results_dir=str(problem_dir),
            inject_domain_schema=True,
            check_adequacy=check_adequacy,
            objects_list=objects_block(task.symbol_table),
        )
    except Exception as e:
        elapsed = time.time() - t0
        return {
            "task_id": task.task_id,
            "task_type": task.task_type,
            "nl_task": task.nl_task,
            "planning_successful": False,
            "val_valid_self": False,
            "val_valid_gt": False,
            "plan_length": 0,
            "refinements": [],
            "elapsed_s": elapsed,
            "failure_stage": "exception",
            "failure_reason": str(e),
        }

    (
        final_problem_path,
        final_plan_path,
        planning_successful,
        grounding_successful,
        _task_possible,
        _explanation,
        refinements_per_iteration,
        _domain_gen_time,
        problem_gen_time,
        refinement_time,
        total_llm_time,
        failure_stage,
        failure_reason,
    ) = results

    elapsed = time.time() - t0

    # Self-consistent VAL validation (generated domain + generated problem)
    val_valid_self = False
    val_valid_gt = False
    plan_length = 0

    if planning_successful and final_plan_path and os.path.exists(final_plan_path):
        from src.lapis.planner.low.pddl_verification import VAL_validate

        # Self-consistent: validate against the generated problem
        gen_domain = problem_dir / "iteration_0" / "generated_domain.pddl"
        if gen_domain.exists() and final_problem_path and os.path.exists(final_problem_path):
            val_valid_self, _ = VAL_validate(str(gen_domain), final_problem_path, final_plan_path)
        elif final_problem_path and os.path.exists(final_problem_path):
            # GT domain mode: validate against alfred.pddl + generated problem
            val_valid_self, _ = VAL_validate(str(ALFRED_PDDL), final_problem_path, final_plan_path)

        # GT validation: resolve lifted symbols → GT IDs, then validate
        raw_plan = Path(final_plan_path).read_text()
        resolved_plan = resolve_plan(raw_plan, task.symbol_table)
        resolved_plan_path = str(final_plan_path).replace(".out", "_resolved.out")
        Path(resolved_plan_path).write_text(resolved_plan)
        val_valid_gt, _ = VAL_validate(str(ALFRED_PDDL), str(gt_problem_path), resolved_plan_path)

        try:
            plan_text = Path(final_plan_path).read_text().strip()
            plan_length = sum(1 for l in plan_text.splitlines() if l.strip() and not l.startswith(";"))
        except Exception:
            pass

    taxonomy_history = []
    sidecar = problem_dir / "taxonomy_history.json"
    if sidecar.exists():
        try:
            taxonomy_history = json.loads(sidecar.read_text())
        except Exception:
            pass

    result = {
        "task_id": task.task_id,
        "task_type": task.task_type,
        "nl_task": task.nl_task,
        "planning_successful": planning_successful,
        "val_valid_self": val_valid_self,
        "val_valid_gt": val_valid_gt,
        "plan_length": plan_length,
        "refinements": refinements_per_iteration,
        "problem_gen_s": problem_gen_time,
        "total_llm_s": total_llm_time,
        "elapsed_s": elapsed,
        "failure_stage": failure_stage,
        "failure_reason": failure_reason,
        "refinement_history": taxonomy_history,
        **pipeline.agent.token_stats(),
    }

    (problem_dir / "manifold.json").write_text(json.dumps(result, indent=2))
    return result


def execute_in_env(task, plan_path: str) -> dict:
    """Execute a LAPIS plan in AlfredTWEnv. Returns execution result."""
    try:
        import yaml
        from alfworld.agents.environment.alfred_tw_env import AlfredTWEnv
    except ImportError:
        return {"executed": False, "reason": "alfworld not importable"}

    alfworld_data = os.path.expanduser("~/.cache/alfworld")
    alfred_pddl_logic = str(ALFRED_PDDL)
    alfred_twl2 = str(ALFRED_PDDL).replace("alfred.pddl", "alfred.twl2")

    config = {
        "dataset": {
            "data_path": os.path.join(alfworld_data, "json_2.1.1/train"),
            "eval_id_data_path": None,
            "eval_ood_data_path": None,
            "num_train_games": -1,
            "num_eval_games": -1,
        },
        "logic": {
            "domain": alfred_pddl_logic,
            "grammar": alfred_twl2,
        },
        "env": {
            "type": "AlfredTWEnv",
            "domain_randomization": False,
            "task_types": [1, 2, 3, 4, 5, 6],
            "expert_timeout_steps": 150,
            "expert_type": "handcoded",
            "goal_desc_human_anns_prob": 1.0,
            "hybrid": {"start_eps": 100000, "thor_prob": 0.5, "eval_mode": "tw"},
            "thor": {"screen_width": 300, "screen_height": 300, "smooth_nav": False,
                     "save_frames_to_disk": False, "save_frames_path": "./videos/"},
        },
        "controller": {"type": "oracle", "debug": False, "load_receps": True},
        "general": {"training_method": "dagger"},
        "dagger": {"training": {"max_nb_steps_per_episode": 50}},
        "mask_rcnn": {"pretrained_model_path": ""},
    }

    try:
        from src.lapis.simulators.alfworld_simulator import AlfWorldSimulator
        alfred_env = AlfredTWEnv(config)
        env = alfred_env.init_env(batch_size=1)
        # Load specific game file
        env.load(task.game_file)
        obs, info = env.reset()

        plan_lines = [l.strip() for l in Path(plan_path).read_text().splitlines()
                      if l.strip() and not l.startswith(";")]

        reward = 0
        done = False
        steps = 0
        for line in plan_lines:
            if done:
                break
            # Parse PDDL action line: (action_name param1 param2 ...)
            m = __import__("re").match(r'\((\S+)\s*(.*?)\)', line)
            if not m:
                continue
            action_name = m.group(1).lower()
            params = m.group(2).split()

            class FakeAction:
                def __init__(self, name, actual_parameters):
                    self.action = type("A", (), {"name": name})()
                    self.actual_parameters = actual_parameters

            try:
                sim = AlfWorldSimulator.__new__(AlfWorldSimulator)
                text_cmd = sim.map_pddl2simulator(FakeAction(action_name, params))
                obs, reward, done, info = env.step([text_cmd])
                steps += 1
            except Exception as e:
                pass

        env.close()
        return {"executed": True, "task_success": bool(reward and reward[0] > 0), "steps": steps}
    except Exception as e:
        return {"executed": False, "reason": str(e)}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=5, help="Games per task type")
    parser.add_argument("--task_types", nargs="+", default=None, help="Subset of task types")
    parser.add_argument("--trials", nargs="+", default=None, help="Specific trial IDs to run (overrides --n/--task_types)")
    parser.add_argument("--model", default="claude-sonnet-4-6")
    parser.add_argument("--iterations", type=int, default=3, help="LAPIS refinement iterations")
    parser.add_argument("--execute", action="store_true", help="Execute plans in AlfredTWEnv")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--results_dir", default=str(RESULTS_DIR))
    parser.add_argument("--generate_domain", action="store_true", help="Generate PDDL domain from NL (gen-domain mode)")
    parser.add_argument("--domain_nl", default=str(Path(__file__).parent / "data/alfworld/domain.nl"), help="Path to domain NL description (used when --generate_domain)")
    parser.add_argument("--planner", default="native_fd", choices=["native_fd", "up_fd", "pyperplan", "symk"], help="Planner backend")
    parser.add_argument("--use_taxonomy", default=True, type=lambda x: x.lower() != "false", help="Enable taxonomy-driven error correction (default: true)")
    parser.add_argument("--use_vector_db", default=True, type=lambda x: x.lower() != "false", help="Enable per-task vector DB for in-task learning across refinements (default: true)")
    parser.add_argument("--bundle_by_error_class", default=True, type=lambda x: x.lower() != "false", help="Bundle issues sharing a priority bucket (P0/P1/P2/P3) into a single LLM correction call per refinement iteration (default: true)")
    parser.add_argument("--check_adequacy", default=False, type=lambda x: x.lower() != "false", help="Run CoT adequacy check after domain/problem generation (default: false)")
    args = parser.parse_args()

    from data.alfworld.game_loader import sample_games, load_game
    import glob as _glob

    if args.trials:
        print(f"Loading {len(args.trials)} specific trials...")
        trial_set = set(args.trials)
        all_tw = _glob.glob(os.path.join(os.path.expanduser("~/.cache/alfworld/json_2.1.1/train"), "**", "*.tw-pddl"), recursive=True)
        tasks = []
        for tw in all_tw:
            from pathlib import Path as _Path
            tid = None
            traj = _Path(tw).parent / "traj_data.json"
            if traj.exists():
                import json as _json
                tid = _json.load(open(traj)).get("task_id", "")
            if tid in trial_set:
                t = load_game(tw)
                if t:
                    tasks.append(t)
        found = {t.task_id for t in tasks}
        missing = trial_set - found
        if missing:
            print(f"WARNING: could not find {len(missing)} trial(s): {missing}")
        print(f"Loaded {len(tasks)} tasks")
    else:
        print(f"Sampling {args.n} games per task type...")
        tasks = sample_games(n_per_type=args.n, task_types=args.task_types, seed=args.seed)
        print(f"Loaded {len(tasks)} tasks")

    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    run_tag = "lapis_gen" if args.generate_domain else "lapis_gt"
    run_dir = Path(args.results_dir) / f"{run_tag}_{args.model.replace('-','_')}_{timestamp}"
    run_dir.mkdir(parents=True, exist_ok=True)

    agent = build_agent(args.model)

    domain_description = None
    domain_file_path_for_pipeline = str(ALFRED_PDDL)
    if args.generate_domain:
        domain_description = Path(args.domain_nl).read_text()
        domain_file_path_for_pipeline = None

    from src.lapis.pipelines.lapis_low_level import LAPISLowLevelPipeline

    pipeline = LAPISLowLevelPipeline(
        domain_name="alfworld",
        batch_id="",
        llmpp_source_dir=None,
        base_dir=str(Path(__file__).parent),
        data_dir=str(Path(__file__).parent / "data"),
        results_dir=str(run_dir),
        splits=[],
        agent=agent,
        generate_domain=args.generate_domain,
        determine_possibility=False,
        prevent_impossibility=False,
        pddl_gen_iterations=args.iterations,
        planner_timeout=180,
        ground_in_sg=True,
        planner_name=args.planner,
        use_taxonomy=args.use_taxonomy,
        use_vector_db=args.use_vector_db,
        # vector_db_path=None → refine_problem auto-resolves to per-task path
        # (<task_dir>/vector_db.json) so positives/negatives never bleed across tasks.
        vector_db_path=None,
        bundle_by_error_class=args.bundle_by_error_class,
    )

    all_results = []
    for i, task in enumerate(tasks):
        print(f"\n[{i+1}/{len(tasks)}] {task.task_type} — {task.nl_task}")
        result = run_lapis_on_task(task, pipeline, run_dir,
                                    domain_file_path=domain_file_path_for_pipeline,
                                    domain_description=domain_description,
                                    check_adequacy=args.check_adequacy)

        if args.execute and result["planning_successful"]:
            plan_path = run_dir / task.task_id.replace("/", "_") / "iteration_0" / "refinement_0" / "plan_0.out"
            if plan_path.exists():
                exec_result = execute_in_env(task, str(plan_path))
                result.update(exec_result)

        all_results.append(result)
        status = "✓ VAL" if result["val_valid_self"] else ("planned" if result["planning_successful"] else "✗")
        gt_status = " | GT✓" if result.get("val_valid_gt") else ""
        print(f"  → {status}{gt_status} | plan_len={result['plan_length']} | {result['elapsed_s']:.1f}s")

    # Summary
    total = len(all_results)
    planned = sum(1 for r in all_results if r["planning_successful"])
    val_self = sum(1 for r in all_results if r["val_valid_self"])
    val_gt = sum(1 for r in all_results if r.get("val_valid_gt"))

    by_type: dict = {}
    for r in all_results:
        tt = r["task_type"]
        by_type.setdefault(tt, {"total": 0, "planned": 0, "val_self": 0, "val_gt": 0})
        by_type[tt]["total"] += 1
        by_type[tt]["planned"] += int(r["planning_successful"])
        by_type[tt]["val_self"] += int(r["val_valid_self"])
        by_type[tt]["val_gt"] += int(r.get("val_valid_gt", False))

    # Aggregate taxonomy correction counts across all refinement steps
    taxonomy_counts: dict = {}
    for r in all_results:
        for step in r.get("refinement_history", []):
            for m in step.get("taxonomy_matches", []):
                key = f"{m['error_class']}/{m['subtype']}"
                taxonomy_counts[key] = taxonomy_counts.get(key, 0) + 1

    agg_tokens = {
        "llm_calls": sum(r.get("llm_calls", 0) for r in all_results),
        "tokens_in": sum(r.get("tokens_in", 0) for r in all_results),
        "tokens_out": sum(r.get("tokens_out", 0) for r in all_results),
        "tokens_thinking": sum(r.get("tokens_thinking", 0) for r in all_results),
        "tokens_total": sum(r.get("tokens_total", 0) for r in all_results),
    }

    summary = {
        "model": args.model,
        "planner": args.planner,
        "use_taxonomy": args.use_taxonomy,
        "use_vector_db": args.use_vector_db,
        "check_adequacy": args.check_adequacy,
        "bundle_by_error_class": args.bundle_by_error_class,
        "iterations": args.iterations,
        "total": total,
        "planned": planned,
        "val_valid_self": val_self,
        "val_valid_gt": val_gt,
        "by_task_type": by_type,
        "taxonomy_correction_counts": taxonomy_counts,
        "token_totals": agg_tokens,
        "results": all_results,
    }

    summary_path = run_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2))

    print(f"\n{'='*50}")
    print(f"Results: {val_self}/{total} VAL-valid (self) | {val_gt}/{total} VAL-valid (GT)")
    print(f"By task type:")
    for tt, s in sorted(by_type.items()):
        print(f"  {tt:45s}: {s['val_self']}/{s['total']} self | {s['val_gt']}/{s['total']} GT")
    print(f"Tokens: {agg_tokens['tokens_total']:,} total  ({agg_tokens['tokens_in']:,} in / {agg_tokens['tokens_out']:,} out / {agg_tokens['tokens_thinking']:,} thinking)  calls={agg_tokens['llm_calls']}")
    print(f"\nSaved to: {run_dir}")


if __name__ == "__main__":
    main()
