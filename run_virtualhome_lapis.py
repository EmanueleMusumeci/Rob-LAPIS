#!/usr/bin/env python3
"""
LAPIS benchmark on VirtualHome household tasks.

Uses real problem PDDL files from the Embodied Agent Interface (EAI) benchmark
(NeurIPS 2024) as ground truth. For each task:
  1. Load task NL (from curated descriptions) + NL scene (from PDDL init)
  2. [Optional] Rewrite NL scene through prose barrier (breaks bijective encoding)
  3. Run LAPIS grounded_planning:
       - generate_domain=True (default): LLM synthesises domain from domain.nl
       - generate_domain=False: use pre-built virtualhome.pddl
  4. VAL-validate generated plan:
       - self-consistent: generated domain + generated problem
       - GT-valid: GT domain + GT problem (with optional plan aliasing)

Usage:
    python run_virtualhome_lapis.py
    python run_virtualhome_lapis.py --model claude-sonnet-4-6 --iterations 3
    python run_virtualhome_lapis.py --task_types Read_book Watch_TV Go_to_sleep
    python run_virtualhome_lapis.py --generate_domain false --prose_barrier false  # ablation
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

VH_DOMAIN     = Path(__file__).parent / "data/virtualhome/virtualhome.pddl"
VH_DOMAIN_NL  = Path(__file__).parent / "data/virtualhome/domain.nl"
RESULTS_DIR   = Path(__file__).parent / "results_virtualhome"


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


def _str_bool(v: str) -> bool:
    return v.lower() not in {"false", "0", "no", "off"}


def run_lapis_on_task(
    task,
    pipeline,
    agent,
    results_dir: Path,
    generate_domain: bool,
    prose_barrier: bool,
    use_aliasing: bool,
    domain_nl_text: str,
) -> dict:
    """Run LAPIS on one VHTask. Returns a result dict."""
    problem_dir = results_dir / task.task_id
    problem_dir.mkdir(parents=True, exist_ok=True)

    # Save GT problem for reference (disk only — never passed to LLM)
    gt_problem_path = problem_dir / "gt_problem.pddl"
    gt_problem_path.write_text(task.pddl_problem)

    agent.reset_token_counts()

    # --- NL barrier: rewrite bijective scene NL into genuine prose ---
    if prose_barrier:
        from data.virtualhome.nl_barrier import rewrite_as_prose
        nl_scene_input = rewrite_as_prose(task.nl_scene, agent, enabled=True)
        (problem_dir / "nl_scene_prose.txt").write_text(nl_scene_input)
    else:
        nl_scene_input = task.nl_scene

    # --- Build grounded_planning kwargs ---
    gp_kwargs = dict(
        current_goal_text=task.nl_task,
        extracted_sg_str=nl_scene_input,
        results_dir=str(problem_dir),
        inject_domain_schema=True,
        check_adequacy=False,
    )
    if generate_domain:
        gp_kwargs["domain_description"] = domain_nl_text
        gp_kwargs["domain_file_path"] = None   # LLM generates domain; GT domain NOT seen by LLM
    else:
        gp_kwargs["domain_file_path"] = str(VH_DOMAIN)

    t0 = time.time()
    try:
        results = pipeline.grounded_planning(**gp_kwargs)
    except Exception as e:
        elapsed = time.time() - t0
        return {
            "task_id": task.task_id,
            "task_type": task.task_type,
            "nl_task": task.nl_task,
            "generate_domain": generate_domain,
            "prose_barrier": prose_barrier,
            "use_aliasing": use_aliasing,
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

    val_valid_self = False
    val_valid_gt = False
    plan_length = 0

    if planning_successful and final_plan_path and os.path.exists(final_plan_path):
        from src.lapis.planner.low.pddl_verification import VAL_validate

        # Determine which domain to use for self-consistent check
        # When generate_domain=True, look for the generated domain file
        if generate_domain:
            # LAPIS writes generated domain to iteration_0/generated_domain.pddl
            gen_domain_candidates = sorted(
                (problem_dir / "iteration_0").glob("generated_domain.pddl")
            ) if (problem_dir / "iteration_0").exists() else []
            if not gen_domain_candidates:
                gen_domain_candidates = sorted(problem_dir.glob("**/generated_domain.pddl"))
            self_domain = str(gen_domain_candidates[0]) if gen_domain_candidates else str(VH_DOMAIN)
        else:
            self_domain = str(VH_DOMAIN)

        if final_problem_path and os.path.exists(final_problem_path):
            val_valid_self, _ = VAL_validate(self_domain, final_problem_path, final_plan_path)

        # GT validation: always uses GT domain + GT problem
        # Optionally alias generated action names → GT canonical names first
        plan_for_gt = final_plan_path
        if use_aliasing:
            from data.virtualhome.plan_aliaser import alias_vh_plan
            plan_text = Path(final_plan_path).read_text()
            aliased_text = alias_vh_plan(plan_text, enabled=True)
            aliased_path = final_plan_path.replace(".out", "_aliased.out")
            Path(aliased_path).write_text(aliased_text)
            plan_for_gt = aliased_path

        val_valid_gt, _ = VAL_validate(str(VH_DOMAIN), str(gt_problem_path), plan_for_gt)

        try:
            plan_text = Path(final_plan_path).read_text().strip()
            plan_length = sum(
                1 for l in plan_text.splitlines() if l.strip() and not l.startswith(";")
            )
        except Exception:
            pass

    result = {
        "task_id": task.task_id,
        "task_type": task.task_type,
        "nl_task": task.nl_task,
        "generate_domain": generate_domain,
        "prose_barrier": prose_barrier,
        "use_aliasing": use_aliasing,
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
        **agent.token_stats(),
    }

    (problem_dir / "manifold.json").write_text(json.dumps(result, indent=2))
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--task_types", nargs="+", default=None)
    parser.add_argument("--model", default="claude-sonnet-4-6")
    parser.add_argument("--iterations", type=int, default=3)
    parser.add_argument("--results_dir", default=str(RESULTS_DIR))
    # NL pipeline flags (all default True — ablate with false)
    parser.add_argument("--generate_domain", type=_str_bool, default=True,
                        metavar="BOOL",
                        help="Generate PDDL domain from domain.nl (default: true)")
    parser.add_argument("--prose_barrier", type=_str_bool, default=True,
                        metavar="BOOL",
                        help="Rewrite NL scene through prose LLM call (default: true)")
    parser.add_argument("--use_aliasing", type=_str_bool, default=True,
                        metavar="BOOL",
                        help="Alias generated action names to GT names for GT-valid check (default: true)")
    parser.add_argument("--use_taxonomy", type=_str_bool, default=True,
                        metavar="BOOL",
                        help="Enable taxonomy-driven error correction (default: true)")
    parser.add_argument("--use_vector_db", type=_str_bool, default=False,
                        metavar="BOOL",
                        help="Enable vector DB cross-task learning (default: false)")
    # Unity execution flags
    parser.add_argument("--execute", action="store_true", help="Execute plans in Unity and record video")
    parser.add_argument("--port", default="8080", help="Unity comm port")
    parser.add_argument("--display", default="1", help="X display number")
    args = parser.parse_args()

    from data.virtualhome.vh_loader import load_tasks

    tasks = load_tasks(task_types=args.task_types)
    print(f"Loaded {len(tasks)} VirtualHome tasks")
    print(f"  generate_domain={args.generate_domain} | prose_barrier={args.prose_barrier} | use_aliasing={args.use_aliasing}")

    # Load domain NL description once
    domain_nl_text = VH_DOMAIN_NL.read_text() if args.generate_domain else ""

    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    suffix = "gen" if args.generate_domain else "gt"
    run_dir = Path(args.results_dir) / f"lapis_{suffix}_{args.model.replace('-','_')}_{timestamp}"
    run_dir.mkdir(parents=True, exist_ok=True)

    agent = build_agent(args.model)

    from src.lapis.pipelines.lapis_low_level import LAPISLowLevelPipeline

    pipeline = LAPISLowLevelPipeline(
        domain_name="virtualhome",
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
        planner_name="native_fd",
        ground_in_sg=True,
        use_taxonomy=args.use_taxonomy,
        use_vector_db=args.use_vector_db,
        vector_db_path=str(run_dir / "vector_db.json") if args.use_vector_db else None,
    )

    if args.execute:
        from src.lapis.simulators.vh_executor import run_plan_in_unity
        videos_dir = run_dir / "videos"
        videos_dir.mkdir(exist_ok=True)

    all_results = []
    for i, task in enumerate(tasks):
        print(f"\n[{i+1}/{len(tasks)}] {task.task_type} — {task.nl_task}")
        result = run_lapis_on_task(
            task, pipeline, agent, run_dir,
            generate_domain=args.generate_domain,
            prose_barrier=args.prose_barrier,
            use_aliasing=args.use_aliasing,
            domain_nl_text=domain_nl_text,
        )
        all_results.append(result)
        status = "✓ VAL" if result["val_valid_self"] else ("planned" if result["planning_successful"] else "✗")
        gt_status = " | GT✓" if result.get("val_valid_gt") else ""
        print(f"  → {status}{gt_status} | plan_len={result['plan_length']} | {result['elapsed_s']:.1f}s")

        if args.execute and result["planning_successful"] and result.get("plan_length", 0) > 0:
            task_dir = run_dir / task.task_id
            plan_files = sorted(task_dir.glob("iteration_*/refinement_*/plan_*.out"))
            plan_path = None
            for pf in reversed(plan_files):
                if pf.stat().st_size > 0:
                    plan_path = str(pf)
                    break
            if plan_path:
                scene_id = int(task.task_id.rsplit("_", 1)[-1])
                success, video_path = run_plan_in_unity(
                    plan_path, scene_id, str(videos_dir),
                    port=args.port, x_display=args.display,
                )
                result["executed"] = success
                result["video_path"] = video_path or ""
                exec_status = f" | exec={'✓' if success else '✗'}"
                if video_path:
                    exec_status += f" | video={Path(video_path).name}"
                print(f"  {exec_status}")

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
        "iterations": args.iterations,
        "generate_domain": args.generate_domain,
        "prose_barrier": args.prose_barrier,
        "use_aliasing": args.use_aliasing,
        "use_taxonomy": args.use_taxonomy,
        "total": total,
        "planned": planned,
        "val_valid_self": val_self,
        "val_valid_gt": val_gt,
        "by_task_type": by_type,
        "taxonomy_correction_counts": taxonomy_counts,
        "token_totals": agg_tokens,
        "results": all_results,
    }
    (run_dir / "summary.json").write_text(json.dumps(summary, indent=2))

    print(f"\n{'='*50}")
    print(f"Results: {val_self}/{total} VAL-valid (self) | {val_gt}/{total} VAL-valid (GT)")
    for tt, s in sorted(by_type.items()):
        print(f"  {tt:40s}: {s['val_self']}/{s['total']} self | {s['val_gt']}/{s['total']} GT")
    print(f"Tokens: {agg_tokens['tokens_total']:,} total  ({agg_tokens['tokens_in']:,} in / {agg_tokens['tokens_out']:,} out / {agg_tokens['tokens_thinking']:,} thinking)  calls={agg_tokens['llm_calls']}")
    print(f"\nSaved to: {run_dir}")


if __name__ == "__main__":
    main()
