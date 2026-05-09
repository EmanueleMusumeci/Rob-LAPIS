#!/usr/bin/env python3
"""
run_benchmark.py — LAPIS benchmark on LLM+P (IPC) domains.

Usage:
    # Single method runs:
    python run_benchmark.py --domain blocksworld --method lapis
    python run_benchmark.py --domain blocksworld --method llmpp
    python run_benchmark.py --domain barman --method lapis --generate_domain

    # Side-by-side comparison (runs both methods, prints table):
    python run_benchmark.py --domain blocksworld --method compare
    python run_benchmark.py --domain blocksworld --method compare --generate_domain

Methods:
    llmpp  — LLM+P baseline: generate problem once, plan, no refinement (pddl_gen_iterations=0)
    lapis  — LAPIS:          generate problem + iterative VAL-guided refinement (pddl_gen_iterations=3)
    compare — run both llmpp and lapis, print side-by-side comparison table

Domain generation (--generate_domain):
    When set, both methods also generate the PDDL domain from domain.nl.
    Reads from third-party/llm-pddl/domains/{domain}/domain.nl and pXX.nl.
    Validation uses generated domain + generated problem (GT predicates differ).

Ablation study (--ablation):
    Controls which Approach A prompt improvements are active (for domain-generation mode):
    baseline        — legacy prompts: hardcoded navigation hints in domain, no schema in problem
    clean_domain    — clean domain prompt only (no hardcoded hints); baseline problem prompt
    schema_problem  — baseline domain prompt; schema-injected problem prompt
    full            — clean domain + schema injection (default, recommended)
    full_adequacy   — full + 3-step CoT adequacy checks on domain and problem (most capable)

Data:
    Reads from:  data/llmpp/{domain}/{problem_id}/nl  (domain.nl + pXX.nl concatenated)
                 data/llmpp/{domain}/{problem_id}/domain.pddl
                 data/llmpp/{domain}/{problem_id}/problem.pddl
    Prepare with:  python prepare_llmpp_data.py

Results:
    Written to:  results_llmpp/benchmark_{domain}_{method}_{model}/{timestamp}/
"""

import argparse
import json
import os
import sys
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

sys.path.insert(0, str(Path(__file__).parent))

LLMPP_DOMAINS = ["barman", "blocksworld", "floortile", "grippers", "storage", "termes", "tyreworld"]

# LAPIS: number of VAL-guided refinement iterations after initial generation
LAPIS_REFINEMENTS = 3
# LLM+P: no refinement
LLMPP_REFINEMENTS = 0


def discover_problems(data_dir: str, domain: str) -> list:
    domain_path = Path(data_dir) / domain
    if not domain_path.exists():
        raise FileNotFoundError(f"Domain directory not found: {domain_path}")
    problems = sorted(
        d.name for d in domain_path.iterdir()
        if d.is_dir() and (d / "nl").exists()
    )
    return problems


def build_summary(results_dir: str, domain: str, method: str) -> dict:
    results_path = Path(results_dir)
    manifests = list(results_path.rglob("manifold.json"))

    total = len(manifests)
    planned = sum(1 for m in manifests if json.loads(m.read_text()).get("planning_successful"))
    valid = sum(1 for m in manifests if json.loads(m.read_text()).get("val_valid"))

    times = [json.loads(m.read_text())["timing"]["total_llm_s"] for m in manifests
             if "timing" in json.loads(m.read_text())]
    avg_time = round(sum(times) / len(times), 2) if times else 0.0

    plan_lens = [json.loads(m.read_text()).get("plan_length", 0) for m in manifests
                 if json.loads(m.read_text()).get("val_valid")]
    avg_plan_len = round(sum(plan_lens) / len(plan_lens), 1) if plan_lens else 0.0

    refinements = [json.loads(m.read_text()).get("pddl_refinements", 0) for m in manifests]
    avg_refinements = round(sum(refinements) / len(refinements), 2) if refinements else 0.0

    return {
        "domain": domain,
        "method": method,
        "benchmark": "llmpp",
        "total_problems": total,
        "planning_success": planned,
        "val_success": valid,
        "success_rate": round(valid / total, 3) if total else 0.0,
        "avg_llm_time_s": avg_time,
        "avg_plan_length": avg_plan_len,
        "avg_refinements": avg_refinements,
    }


ABLATION_FLAGS = {
    #                           clean_domain_prompt  inject_domain_schema  check_adequacy
    "baseline":                (False,              False,                False),
    "clean_domain":            (True,               False,                False),
    "schema_problem":          (False,              True,                 False),
    "full":                    (True,               True,                 False),
    "full_adequacy":           (True,               True,                 True),
}


def run_method(domain, method, problems, agent, args, generate_domain):
    from src.lapis.pipelines.lapis_low_level import LAPISLowLevelPipeline

    ablation = getattr(args, "ablation", "full")
    clean_domain_prompt, inject_domain_schema, check_adequacy = ABLATION_FLAGS.get(ablation, (True, True, False))

    pddl_gen_iterations = LAPIS_REFINEMENTS if method == "lapis" else LLMPP_REFINEMENTS
    gen_suffix = "_domgen" if generate_domain else ""
    abl_suffix = f"_{ablation}" if ablation != "full" else ""

    semantic_checks = getattr(args, "semantic_checks", False)
    refine_domain = getattr(args, "refine_domain", False)
    extractor_type = getattr(args, "extractor_type", "auto")

    # Add refine_domain suffix to experiment name
    refine_suffix = "_refdom" if refine_domain else ""
    experiment_name = f"benchmark_llmpp_{domain}_{method}{gen_suffix}{abl_suffix}{refine_suffix}_{args.model.replace('-', '_')}"

    # Pipeline parameters
    pipeline_params = {
        "domain_name": domain,
        "batch_id": "",
        "llmpp_source_dir": str(Path(__file__).parent / "third-party" / "llm-pddl" / "domains"),
        "determine_possibility": False,
        "prevent_impossibility": False,
        "pddl_gen_iterations": pddl_gen_iterations,
        "agent": agent,
        "base_dir": str(Path(__file__).parent),
        "data_dir": args.data_dir,
        "results_dir": args.results_dir,
        "splits": problems,
        "generate_domain": generate_domain,
        "ground_in_sg": False,
        "clean_domain_prompt": clean_domain_prompt,
        "inject_domain_schema": inject_domain_schema,
        "check_adequacy": check_adequacy,
        "planner_timeout": args.planner_timeout,
        "semantic_checks": semantic_checks,
        "refine_domain": refine_domain,
        "extractor_type": extractor_type,
    }

    pipeline = LAPISLowLevelPipeline(**pipeline_params)
    pipeline.experiment_name = experiment_name
    pipeline.run()

    run_results_dir = os.path.join(args.results_dir, experiment_name, pipeline.timestamp)
    summary = build_summary(run_results_dir, domain, method)

    summary_path = Path(run_results_dir) / "summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)

    return summary, run_results_dir


def print_comparison_table(summaries_by_domain: dict, generate_domain: bool):
    """Print a formatted side-by-side comparison table."""
    domain_gen_note = " (domain generated from NL)" if generate_domain else " (GT domain provided)"

    print(f"\n{'='*72}")
    print(f"  LAPIS vs LLM+P Comparison{domain_gen_note}")
    print(f"{'='*72}")

    header = f"{'Domain':<14} {'Method':<8} {'Plan%':>6} {'VAL%':>6} {'AvgLen':>7} {'AvgRef':>7} {'AvgTime':>8}"
    print(header)
    print("-" * 72)

    for domain, summaries in summaries_by_domain.items():
        for i, s in enumerate(summaries):
            total = s["total_problems"]
            plan_pct = f"{s['planning_success']}/{total}"
            val_pct = f"{s['val_success']}/{total} ({s['success_rate']*100:.0f}%)"
            dom_col = domain if i == 0 else ""
            print(
                f"{dom_col:<14} {s['method'].upper():<8} "
                f"{plan_pct:>6} {val_pct:>13} "
                f"{s['avg_plan_length']:>7} {s['avg_refinements']:>7} {s['avg_llm_time_s']:>7}s"
            )
        if len(summaries_by_domain) > 1:
            print()

    print("=" * 72)
    print("Plan%  = problems where planner found a plan")
    print("VAL%   = plans valid under VAL" + (" (generated domain+problem)" if generate_domain else " (GT domain+problem)"))
    print("AvgLen = avg plan length (successful plans only)")
    print("AvgRef = avg PDDL refinement iterations (LAPIS only)")


def main():
    parser = argparse.ArgumentParser(description="LAPIS vs LLM+P benchmark on IPC domains")
    parser.add_argument("--domain", default="blocksworld", choices=LLMPP_DOMAINS + ["all"])
    parser.add_argument("--method", default="compare", choices=["lapis", "llmpp", "compare"],
                        help="lapis=iterative refinement, llmpp=single-shot, compare=both")
    parser.add_argument("--generate_domain", action="store_true",
                        help="Generate PDDL domain from NL (reads domain.nl from llm-pddl source)")
    parser.add_argument("--ablation", default="full",
                        choices=["baseline", "clean_domain", "schema_problem", "full", "full_adequacy"],
                        help="Approach A ablation: which prompt improvements to enable")
    parser.add_argument("--model", default="claude-sonnet-4-6")
    parser.add_argument("--problems", nargs="*", default=None)
    parser.add_argument("--pddl_gen_iterations", type=int, default=None,
                        help="Override PDDL refinement iterations (default: 3 for lapis, 0 for llmpp)")
    parser.add_argument("--planner_timeout", type=int, default=180,
                        help="Timeout for symbolic planner in seconds (default: 180)")
    parser.add_argument("--semantic_checks", action="store_true",
                        help="Enable semantic verification (predicate coverage, action reachability)")
    parser.add_argument("--refine_domain", action="store_true",
                        help="Enable iterative domain refinement when semantic/adequacy checks identify domain-level issues")
    parser.add_argument("--extractor_type", default="auto",
                        choices=["regex", "up", "auto"],
                        help="PDDL extraction backend for semantic checks (default: auto)")
    parser.add_argument("--data_dir", default="data/llm-pddl")
    parser.add_argument("--results_dir", default="results_llmpp")
    args = parser.parse_args()

    # Override defaults if explicitly set
    global LAPIS_REFINEMENTS
    if args.pddl_gen_iterations is not None:
        LAPIS_REFINEMENTS = args.pddl_gen_iterations

    domains = LLMPP_DOMAINS if args.domain == "all" else [args.domain]
    methods = ["llmpp", "lapis"] if args.method == "compare" else [args.method]

    from src.lapis.agents.gpt import GPTAgent
    from src.lapis.agents.claude import ClaudeAgent

    if args.model.startswith("claude"):
        agent = ClaudeAgent(model=args.model)
    else:
        agent = GPTAgent(model=args.model)

    all_summaries: dict = {}

    for domain in domains:
        problems = args.problems or discover_problems(args.data_dir, domain)
        if not problems:
            print(f"No problems found in {args.data_dir}/{domain}/")
            continue

        gen_label = " [domain generated]" if args.generate_domain else " [GT domain]"
        print(f"\nDomain: {domain}{gen_label} | model={args.model} | {len(problems)} problems | methods={methods}")

        domain_summaries = []
        for method in methods:
            print(f"\n--- Running {method.upper()} on {domain} ---")
            summary, run_dir = run_method(domain, method, problems, agent, args, args.generate_domain)
            domain_summaries.append(summary)

            print(f"  Planned:  {summary['planning_success']}/{summary['total_problems']}")
            print(f"  VAL:      {summary['val_success']}/{summary['total_problems']} ({summary['success_rate']*100:.1f}%)")
            print(f"  Avg time: {summary['avg_llm_time_s']}s | Results: {run_dir}")

        all_summaries[domain] = domain_summaries

    if args.method == "compare" and all_summaries:
        print_comparison_table(all_summaries, args.generate_domain)


if __name__ == "__main__":
    main()
