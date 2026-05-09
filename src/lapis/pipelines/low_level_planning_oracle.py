import os
import sys
from pathlib import Path
import logging

from src.lapis.pipelines.base import BasePipeline
from src.lapis.logger_cfg import logger
from src.lapis.planner.low.planner_oracled import LowLevelPlanner

# For GT simulator setup
from unified_planning.shortcuts import SequentialSimulator
from unified_planning.io import PDDLReader


class LowLevelPlanningOraclePipeline(BasePipeline):

    def __init__(self,
                 gt_source_dir: str = "third-party/llm-pddl/domains",
                 **base_init_kwargs):
        super().__init__(**base_init_kwargs)
        self.gt_source_dir = Path(gt_source_dir)
        # Note: LowLevelPlanner created per task with simulator
        self.low_level_planner = None

    def _initialize_csv(self, csv_filepath):
        import csv
        header = ["Problem", "Domain", "Oracle Success", "Plan Found", "Error"]
        with open(csv_filepath, mode="w", newline='') as f:
            writer = csv.writer(f, delimiter='|')
            writer.writerow(header)

    def _process_task(self, task_data):
        domain_name = task_data["domain_name"]
        problem_id = task_data["problem_id"]
        task_description = task_data.get("task_description", "")

        logger.info(f"--- Problem {problem_id} ({domain_name}/) [ORACLE MODE] ---")

        # Locate GT files
        gt_domain_path = task_data.get("gt_domain_path") or \
                        self.gt_source_dir / domain_name / "domain.pddl"
        gt_problem_path = task_data.get("gt_problem_path") or \
                         self.gt_source_dir / domain_name / f"{problem_id}.pddl"

        if not gt_domain_path.exists() or not gt_problem_path.exists():
            logger.error(f"GT files not found: {gt_domain_path}, {gt_problem_path}")
            return {"success": False, "error": "GT files missing"}

        logger.info(f"GT domain: {gt_domain_path}")
        logger.info(f"GT problem: {gt_problem_path}")

        # Setup results directory
        results_dir = Path(self.results_dir) / self.experiment_name / self.timestamp / problem_id
        results_dir.mkdir(parents=True, exist_ok=True)

        # Setup GT simulator
        simulator = self._setup_gt_simulator(str(gt_domain_path), str(gt_problem_path), domain_name)
        if simulator is None:
            logger.warning("Failed to setup GT simulator - will run WITHOUT simulator validation")
            logger.warning("This means the oracle state grounding will be skipped")
            logger.warning("Falling back to standard LowLevelPlanning pipeline")
            # Continue without simulator - pipeline will work but without oracle grounding

        # Extract GT state and ground it
        logger.info("=== Oracle State Grounding ===")
        raw_report = self._get_simulator_raw_state_report(simulator)
        logger.info(f"Extracted GT state:\n{raw_report}")

        # Prepare nl_sections (minimal structure for oracle mode)
        nl_sections = {
            'description': f"Domain: {domain_name}",
            'actions': "",
            'preconditions': "",
            'effects': "",
            'objects': self._extract_objects_from_problem(str(gt_problem_path)),
            'initial_state': raw_report  # Will be replaced by assignment mapping
        }

        # Fair Assignment Mapping (oracle grounding)
        assignment = self._map_simulator_state_to_assignment(raw_report, nl_sections['objects'])

        if assignment:
            logger.info(f"Assignment Mapping successful. Grounded predicates:\n{assignment}")
            nl_sections['initial_state'] = assignment  # REPLACE with PDDL predicates
        else:
            logger.warning("Assignment Mapping failed, falling back to NL-based generation")

        # Create LowLevelPlanner with GT simulator for integrated validation
        logger.info("Creating low-level planner with GT simulator for validation")
        low_level_planner = LowLevelPlanner(
            agent=self.agent,
            use_vector_db=True,
            gt_simulator=simulator  # Pass simulator for validation during refinement
        )

        # Call LowLevelPlanner with grounded state
        success, plan_path, history = low_level_planner.plan(
            nl_sections=nl_sections,
            current_goal_text=task_description,
            results_dir=results_dir,
            workflow_iteration=0,
            planner_name="up_fd"
        )

        # Simulator validation is now integrated in the refinement loop
        # (happens inside low_level_planner.plan())

        result = {
            "success": success,
            "plan_path": plan_path,
            "refinement_history": history,
            "grounded_state": assignment
        }

        logger.info(f"Oracle planning completed. Success: {success}")
        return result

    def _setup_gt_simulator(self, gt_domain_path: str, gt_problem_path: str, domain_name: str):
        logger.info(f"Setting up UP SequentialSimulator for {domain_name}")

        # First, apply UP-specific preprocessing (handles tyreworld, storage, floortile)
        from src.lapis.utils.pddl_preprocessor import preprocess_pddl_for_up
        try:
            gt_domain_path, gt_problem_path = preprocess_pddl_for_up(
                gt_domain_path, gt_problem_path, domain_name
            )
            logger.info(f"Applied UP preprocessing for {domain_name}")
        except Exception as preproc_err:
            logger.debug(f"UP preprocessing not needed or failed: {preproc_err}")

        try:
            reader = PDDLReader()
            problem = reader.parse_problem(gt_domain_path, gt_problem_path)
            simulator = SequentialSimulator(problem)

            # Initialize state
            simulator.problem = problem
            simulator.current_state = simulator.get_initial_state()

            logger.info("GT simulator setup successful")
            return simulator

        except Exception as e:
            logger.warning(f"UP PDDLReader failed: {e}")

            # Strategy 1: Try UP-fixed version if available
            gt_domain_path_obj = Path(gt_domain_path)
            fixed_domain_path = gt_domain_path_obj.parent / "domain_up_fixed.pddl"

            if fixed_domain_path.exists():
                logger.info(f"Trying UP-fixed domain: {fixed_domain_path}")
                try:
                    reader = PDDLReader()
                    problem = reader.parse_problem(str(fixed_domain_path), gt_problem_path)
                    simulator = SequentialSimulator(problem)

                    # Initialize state
                    simulator.problem = problem
                    simulator.current_state = simulator.get_initial_state()

                    logger.info("✓ GT simulator setup successful with UP-fixed domain")
                    return simulator

                except Exception as e_fixed:
                    logger.warning(f"UP-fixed domain also failed: {e_fixed}")

            # Strategy 2: Try CPDDL preprocessing
            from src.lapis.utils.cpddl_preprocessor import is_adl_parsing_error, preprocess_pddl_with_cpddl

            if is_adl_parsing_error(e):
                logger.info("Attempting CPDDL preprocessing...")

                preprocessed = preprocess_pddl_with_cpddl(gt_domain_path, gt_problem_path)

                if preprocessed:
                    preprocessed_domain, preprocessed_problem = preprocessed
                    logger.info("Retrying simulator setup with CPDDL-preprocessed PDDL...")

                    try:
                        reader = PDDLReader()
                        problem = reader.parse_problem(preprocessed_domain, preprocessed_problem)
                        simulator = SequentialSimulator(problem)

                        # Initialize state
                        simulator.problem = problem
                        simulator.current_state = simulator.get_initial_state()

                        logger.info("✓ GT simulator setup successful with CPDDL preprocessing")
                        return simulator

                    except Exception as e2:
                        logger.error(f"Simulator setup failed even after CPDDL preprocessing: {e2}")
                        import traceback
                        traceback.print_exc()
                        return None
                else:
                    logger.error("CPDDL preprocessing failed")

            # All strategies failed
            logger.error("All simulator setup strategies failed")
            logger.error(f"Original error: {e}")
            import traceback
            traceback.print_exc()
            return None

    def _extract_objects_from_problem(self, problem_path: str) -> str:
        import re
        with open(problem_path) as f:
            content = f.read()

        match = re.search(r'\(:objects(.*?)\)', content, re.DOTALL | re.IGNORECASE)
        if match:
            return match.group(1).strip()
        return ""

    def _get_simulator_raw_state_report(self, simulator) -> str:
        # UP-based simulator (any PDDL domain)
        current_state = getattr(simulator, 'current_state', None)
        if not simulator or not current_state:
            return "No simulator state available."

        report = ["Current State Fluents (True only):"]
        state = current_state

        # Get full state values
        all_values = {}
        current = state
        while current:
            for fluent, value in current._values.items():
                if fluent not in all_values:
                    all_values[fluent] = value
            current = current._father

        for fluent, value in all_values.items():
            if value.is_bool_constant() and value.constant_value():
                fluent_name = fluent.fluent().name
                args = [str(arg).replace("'", "") for arg in fluent.args]
                if args:
                    report.append(f"- {fluent_name}({', '.join(args)}) is True")
                else:
                    report.append(f"- {fluent_name} is True")

        return "\n".join(report)

    def _map_simulator_state_to_assignment(self, raw_report: str, objects_list: str) -> str:
        system_prompt = (
            "You are a PDDL state translator. Your task is to translate a raw state report "
            "into valid PDDL predicates for the :init block."
        )

        user_prompt = f"""
        PDDL OBJECTS:
        {objects_list}

        RAW SIMULATOR STATE REPORT:
        {raw_report}

        INSTRUCTIONS:
        1. Convert each fact in the state report to a valid PDDL predicate
        2. Use ONLY the object names provided in the PDDL OBJECTS list
        3. Each predicate should be on its own line in the format: (predicate-name arg1 arg2 ...)
        4. Extract the predicate name and arguments from each fact
        5. DO NOT include the (:init ...) wrapper, just the predicates
        6. DO NOT add any explanatory text, ONLY output the predicates

        OUTPUT FORMAT:
        (predicate1 arg1 arg2)
        (predicate2 arg1)
        (predicate3 arg1 arg2 arg3)
        ...

        Example Input:
        - on(block_a, block_b) is True
        - clear(block_c) is True

        Example Output:
        (on block_a block_b)
        (clear block_c)
        """

        try:
            logger.info(f"Assignment Mapping - Raw Report:\n{raw_report}")
            assignment = self.agent.llm_call(system_prompt, user_prompt)
            logger.info(f"Assignment Mapping - LLM Response:\n{assignment}")

            # Basic cleanup
            assignment = assignment.replace(":init", "").replace("```pddl", "").replace("```", "").strip()

            # Remove any wrapping parentheses if present
            if assignment.startswith("(:init") and assignment.endswith(")"):
                # Extract content between (:init and final )
                import re
                match = re.search(r'\(:init\s+(.*)\s*\)', assignment, re.DOTALL)
                if match:
                    assignment = match.group(1).strip()

            return assignment
        except Exception as e:
            logger.error(f"Assignment mapping failed: {e}")
            import traceback
            traceback.print_exc()
            return ""
