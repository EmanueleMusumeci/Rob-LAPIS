"""
LAPISLowLevelPipeline — LAPIS low-level planner evaluated on Lexicon benchmark problems.

Reads the NL task description from the `nl` file in each problem folder, uses the
pre-existing `domain.pddl`, runs the LAPIS problem-generation + planning + refinement
loop, then validates the produced plan against the ground-truth `problem.pddl`.

When generate_domain=True (domain generation mode), the pipeline reads separate
domain.nl and pXX.nl files from llmpp_source_dir (third-party/llm-pddl/domains),
generates the PDDL domain from NL, and validates using generated artifacts.
"""

import csv
import json
import os
import time
from pathlib import Path
from typing import List, Optional

from src.lapis.pipelines.low_level_planning import LowLevelPlanningPipeline
from src.lapis.planner.low.pddl_verification import VAL_validate
from src.lapis.plan_renderer import simulate_plan, render_plan_gif
from src.lapis.logger_cfg import logger


class LAPISLowLevelPipeline(LowLevelPlanningPipeline):
    """
    Subclass of LowLevelPlanningPipeline for Lexicon benchmark evaluation.

    Each `task_name` in `splits` is a problem ID (e.g. "100", "102").
    Data layout expected:
        {data_dir}/{domain_name}/{batch_id}/{problem_id}/
            nl                    <- NL domain+problem description
            domain.pddl           <- base PDDL domain (unconstrained)
            problem.pddl          <- ground-truth PDDL problem
            unconstrained_plan    <- ground-truth plan (for reference)

    When generate_domain=True:
        llmpp_source_dir must point to third-party/llm-pddl/domains.
        Domain is generated from domain.nl; problem is generated from pXX.nl.
        Validation uses generated domain + generated problem (not GT).
    """

    def __init__(
        self,
        domain_name: str,
        batch_id: str = "data_2",
        llmpp_source_dir: str = None,
        clean_domain_prompt: bool = True,
        inject_domain_schema: bool = True,
        check_adequacy: bool = False,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.domain_name = domain_name
        self.batch_id = batch_id
        self.llmpp_source_dir = llmpp_source_dir
        self.clean_domain_prompt = clean_domain_prompt
        self.inject_domain_schema = inject_domain_schema
        self.check_adequacy = check_adequacy

    # ------------------------------------------------------------------
    # BasePipeline abstract methods
    # ------------------------------------------------------------------

    def _initialize_csv(self, csv_filepath: str):
        header = [
            "problem_id",
            "planning_successful",
            "val_valid",
            "up_sim_valid",
            "plan_length",
            "pddl_refinements",
            "problem_gen_time_s",
            "refinement_time_s",
            "total_llm_time_s",
            "failure_stage",
        ]
        with open(csv_filepath, "w", newline="") as f:
            csv.writer(f, delimiter="|").writerow(header)

    def _process_task(self, problem_id: str, results_dir: str):
        problem_dir = Path(self.data_dir) / self.domain_name / self.batch_id / problem_id
        results_problem_dir = Path(results_dir) / problem_id
        results_problem_dir.mkdir(parents=True, exist_ok=True)

        nl_file = problem_dir / "nl"
        domain_file = problem_dir / "domain.pddl"
        gt_problem_file = problem_dir / "problem.pddl"

        if not nl_file.exists():
            logger.error(f"NL file missing: {nl_file}")
            return

        nl_description = nl_file.read_text()
        logger.info(f"--- Problem {problem_id} ({self.domain_name}/{self.batch_id}) ---")

        if self.generate_domain and self.llmpp_source_dir:
            # Domain generation mode: read domain.nl and pXX.nl from source repo
            src_dir = Path(self.llmpp_source_dir) / self.domain_name
            domain_nl_file = src_dir / "domain.nl"
            prob_nl_file = src_dir / f"{problem_id}.nl"

            domain_description = domain_nl_file.read_text() if domain_nl_file.exists() else None
            if domain_description is None:
                logger.error(f"domain.nl missing in {src_dir}")
                return
            problem_nl = prob_nl_file.read_text() if prob_nl_file.exists() else nl_description

            results = self.grounded_planning(
                current_goal_text=problem_nl,
                domain_description=domain_description,
                domain_file_path=None,  # generated at iteration_0/generated_domain.pddl
                results_dir=str(results_problem_dir),
                clean_domain_prompt=self.clean_domain_prompt,
                inject_domain_schema=self.inject_domain_schema,
                check_adequacy=self.check_adequacy,
            )
        else:
            if not domain_file.exists():
                logger.error(f"Domain PDDL missing: {domain_file}")
                return
            # Run LAPIS low-level planning loop
            # domain_description=None → skip LLM domain generation (use existing domain.pddl)
            results = self.grounded_planning(
                current_goal_text=nl_description,
                domain_file_path=str(domain_file),
                results_dir=str(results_problem_dir),
                inject_domain_schema=self.inject_domain_schema,
                check_adequacy=self.check_adequacy,
            )

        (
            final_problem_file_path,
            final_plan_file_path,
            planning_successful,
            grounding_successful,
            _task_possible,
            _possibility_explanation,
            refinements_per_iteration,
            _domain_gen_time,
            problem_gen_time,
            refinement_time,
            total_llm_time,
            failure_stage,
            _failure_reason,
        ) = results

        # Validate generated plan
        val_valid = False
        up_sim_valid = False
        up_sim_failure = None
        plan_length = 0
        gif_path = None

        if planning_successful and grounding_successful and final_plan_file_path:
            if self.generate_domain:
                # Validate with generated domain + generated problem (GT predicates differ)
                generated_domain = results_problem_dir / "iteration_0" / "generated_domain.pddl"
                can_validate = (
                    generated_domain.exists()
                    and final_problem_file_path
                    and os.path.exists(final_problem_file_path)
                    and os.path.exists(final_plan_file_path)
                )
                if can_validate:
                    val_valid, _val_out = VAL_validate(
                        str(generated_domain), str(final_problem_file_path), final_plan_file_path
                    )
            else:
                can_validate = gt_problem_file.exists() and os.path.exists(final_plan_file_path)

                if can_validate:
                    val_valid, _val_out = VAL_validate(
                        str(domain_file), str(gt_problem_file), final_plan_file_path
                    )

                # --- UP SequentialSimulator + GIF only when using GT domain ---
                if can_validate:
                    up_sim_valid, up_sim_failure, action_strs = simulate_plan(
                        str(domain_file), str(gt_problem_file), final_plan_file_path
                    )
                    if up_sim_failure:
                        logger.debug(f"UP sim failure: {up_sim_failure}")

                    gif_out = str(results_problem_dir / "plan.gif")
                    rendered = render_plan_gif(
                        self.domain_name, str(domain_file), str(gt_problem_file),
                        final_plan_file_path, gif_out,
                    )
                    if rendered:
                        gif_path = gif_out
                        logger.info(f"GIF saved: {gif_out}")

            # Count plan steps
            try:
                plan_text = Path(final_plan_file_path).read_text().strip()
                plan_length = len([l for l in plan_text.splitlines() if l.strip() and not l.startswith(";")]) if plan_text else 0
            except Exception:
                pass

        total_refinements = sum(refinements_per_iteration)

        # Write per-problem manifold.json
        manifold = {
            "problem_id": problem_id,
            "domain": self.domain_name,
            "batch_id": self.batch_id,
            "planning_successful": planning_successful,
            "val_valid": val_valid,
            "up_sim_valid": up_sim_valid,
            "up_sim_failure": up_sim_failure,
            "gif": gif_path,
            "plan_length": plan_length,
            "pddl_refinements": total_refinements,
            "timing": {
                "problem_gen_s": round(problem_gen_time, 2),
                "refinement_s": round(refinement_time, 2),
                "total_llm_s": round(total_llm_time, 2),
            },
            "failure_stage": failure_stage,
        }
        with open(results_problem_dir / "manifold.json", "w") as f:
            json.dump(manifold, f, indent=2)

        status = "SUCCESS" if val_valid else ("PLAN_ONLY" if planning_successful else "FAILED")
        logger.info(
            f"Problem {problem_id}: {status} | "
            f"plan_len={plan_length} refinements={total_refinements} "
            f"llm_time={total_llm_time:.1f}s"
        )

        # Append to CSV
        csv_filepath = Path(results_dir) / "results.csv"
        with open(csv_filepath, "a", newline="") as f:
            csv.writer(f, delimiter="|").writerow([
                problem_id, planning_successful, val_valid, up_sim_valid, plan_length,
                total_refinements, round(problem_gen_time, 2),
                round(refinement_time, 2), round(total_llm_time, 2), failure_stage,
            ])
