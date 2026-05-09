import os
import re
import shutil
import logging
from pathlib import Path
from src.lapis.planner.low.planner_utils import plan_with_output
from src.lapis.planner.low.pddl_generation import (
    generate_domain, generate_problem, refine_problem, refine_domain,
    refine_domain_and_problem_unified, check_domain_adequacy, check_problem_adequacy,
    IssueStats
)
from src.lapis.planner.low.pddl_verification import translate_plan, VAL_validate, VAL_ground
from src.lapis.planner.low.semantic_verification import run_semantic_checks
from src.lapis.planner.low.utils.log import save_statistics

logger = logging.getLogger("my_logger")

class ObjectRegistry:
    def __init__(self, vocabulary: set[str]):
        # vocabulary should be set of object names
        self.vocab = {v.lower() for v in vocabulary}

    def scan_pddl(self, pddl_text: str) -> list[str]:
        """Return all object identifiers in the PDDL (:objects ...) block that are NOT in the vocabulary."""
        # Focus on the :objects section
        match = re.search(r'\(:objects(.*?)\)', pddl_text, re.DOTALL | re.IGNORECASE)
        if not match:
            return []
            
        objects_block = match.group(1).lower()
        # Find all identifiers (words starting with letter/dash/underscore)
        found_ids = set(re.findall(r'\b([a-z][\w-]*)\b', objects_block))
        
        # Keywords and common types to exclude
        keywords = {'and', 'or', 'not', 'goal', 'init', 'objects', 'domain', 'problem', 'define', 'item', 'entity', 'location', 'direction', 'color', 'room', 'agent'}
        
        return [fid for fid in found_ids if fid not in self.vocab and fid not in keywords]

def _parse_init_state(raw: str | None) -> str | None:
    """Normalize pddl_init_state to a clean PDDL fact list string."""
    if not raw:
        return None
    # Already looks like PDDL facts: (predicate arg1 arg2)
    if re.search(r'\(\s*[a-zA-Z][\w-]*', raw):
        return raw.strip()
    return None

class LowLevelPlanner:
    def __init__(self, agent, use_vector_db=True, gt_simulator=None):
        self.agent = agent
        self.use_vector_db = use_vector_db
        self.gt_simulator = gt_simulator  # Optional: for oracle mode with simulator validation

    def _parse_init_state(self, raw: str | None) -> str | None:
        """Normalize pddl_init_state to a clean PDDL fact list string."""
        if not raw:
            return None
        # Already looks like PDDL facts: (predicate arg1 arg2)
        if re.search(r'\(\s*[\w-][\w-]*', raw):
            return raw.strip()
        return None  # Not parseable — do not silently corrupt sync

    def plan(self, workflow_iteration="0", subgoal_id=None, **kwargs):
        """
        Execute the low-level planning process with iterative refinement.
        Implements L-1 to L-11 fixes for robustness.
        """
        goal_file_path = kwargs.get("goal_file_path")
        environment_str_arg = kwargs.get("environment_str")
        current_goal_text = kwargs.get("current_goal_text")
        results_dir = kwargs.get("results_dir")
        output_callback = kwargs.get("output_callback")
        
        # Harmonized inputs
        hl_domain = kwargs.get("hl_domain")
        hl_problem = kwargs.get("hl_problem")
        hl_actions = kwargs.get("hl_actions")
        hl_goal = kwargs.get("hl_goal")
        hl_constraints = kwargs.get("hl_constraints")
        nl_sections = kwargs.get("nl_sections")
        
        # L-1: Validate and Normalize pddl_init_state
        pddl_init_state = self._parse_init_state(kwargs.get("pddl_init_state"))
        
        # L-2: Object Registry Setup
        object_vocab = {"agent", "room"}
        if nl_sections and nl_sections.get('objects'):
            # Extract all alphanumeric words from the objects section
            object_vocab.update(re.findall(r'\b([a-zA-Z][\w-]*)\b', nl_sections['objects']))
        registry = ObjectRegistry(object_vocab)

        if not current_goal_text and hl_goal:
            current_goal_text = f"Goal: {hl_goal}"

        logs_dir = results_dir / "logs"
        logs_dir.mkdir(parents=True, exist_ok=True)
        
        environment_str = environment_str_arg or ""
        domain_description = kwargs.get("domain_description")
        problem_nl = ""

        if nl_sections:
            domain_description = f"{nl_sections['description']}\n\n{nl_sections['actions']}\n\n{nl_sections['preconditions']}\n\n{nl_sections['effects']}"
            environment_str = nl_sections['description']
            problem_nl = f"{nl_sections['objects']}\n\n{nl_sections['initial_state']}"
        elif not domain_description and hl_domain:
             domain_description = f"Domain: {hl_domain}"
             if hl_actions: domain_description += f"\nActions: {hl_actions}"

        # ------------------------------------------------------------------
        # 1. GENERATE DOMAIN
        # ------------------------------------------------------------------
        domain_file_path = results_dir / "domain.pddl"
        temp_goal_file = results_dir / "goal.txt"
        with open(temp_goal_file, "w") as f: f.write(current_goal_text or "")
        
        generate_domain(
            domain_file_path=str(domain_file_path),
            goal_file_path=str(temp_goal_file),
            domain_description=domain_description,
            environment=environment_str,
            agent=self.agent,
            logs_dir=str(logs_dir),
            temperature=0.1
        )
        
        # L-3: Domain Isolation Check
        val_success, val_log = self._check_val(str(domain_file_path))
        if not val_success:
            logger.warning("Domain failed initial VAL. Refining domain in isolation...")
            new_domain_str, _ = refine_domain(
                domain_file_path=str(domain_file_path),
                problem_file_path=None,
                environment=domain_description,
                task=current_goal_text,
                logs_dir=str(logs_dir),
                workflow_iteration=workflow_iteration,
                refinement_iteration=-1,
                agent=self.agent,
                nl_sections=nl_sections,
                VAL_validation_log=val_log,
                temperature=0.1
            )
            with open(domain_file_path, "w") as f: f.write(new_domain_str)
            val_success, val_log = self._check_val(str(domain_file_path))

        # ------------------------------------------------------------------
        # 1.5 DOMAIN ADEQUACY CHECK (Pre-Planning CoT)
        # ------------------------------------------------------------------
        if nl_sections and nl_sections.get('initial_state'):
            domain_pddl = open(domain_file_path).read()
            amended_domain = check_domain_adequacy(
                domain_pddl=domain_pddl,
                raw_observation=nl_sections['initial_state'],
                objects_list=nl_sections.get('objects', ''),
                agent=self.agent,
                logs_dir=str(logs_dir),
                temperature=0.1
            )
            if amended_domain != domain_pddl:
                with open(domain_file_path, "w") as f: f.write(amended_domain)
                val_success, val_log = self._check_val(str(domain_file_path))
                if not val_success:
                    logger.warning("Domain adequacy amendment broke VAL; reverting.")
                    with open(domain_file_path, "w") as f: f.write(domain_pddl)

        # ------------------------------------------------------------------
        # 2. GENERATE PROBLEM
        # ------------------------------------------------------------------
        problem_file_path = results_dir / "problem.pddl"
        generate_problem(
            domain_file_path=str(domain_file_path),
            task=current_goal_text,
            environment=environment_str,
            problem_file_path=str(problem_file_path),
            logs_dir=str(logs_dir),
            workflow_iteration=workflow_iteration,
            agent=self.agent,
            problem_nl=problem_nl,
            pddl_init_state=None,
            temperature=0.1
        )

        # ------------------------------------------------------------------
        # 2.5 PROBLEM ADEQUACY CHECK (Post-Generation CoT)
        # ------------------------------------------------------------------
        if nl_sections and nl_sections.get('initial_state'):
            problem_pddl = open(problem_file_path).read()
            domain_pddl = open(domain_file_path).read()
            amended_problem = check_problem_adequacy(
                problem_pddl=problem_pddl,
                domain_pddl=domain_pddl,
                raw_observation=nl_sections['initial_state'],
                objects_list=nl_sections.get('objects', ''),
                agent=self.agent,
                logs_dir=str(logs_dir),
                temperature=0.1
            )
            if amended_problem != problem_pddl:
                with open(problem_file_path, "w") as f: f.write(amended_problem)

        # ------------------------------------------------------------------
        # 2.6 SEMANTIC VERIFICATION (LAPIS Extension)
        # ------------------------------------------------------------------
        domain_pddl = open(domain_file_path).read()
        problem_pddl = open(problem_file_path).read()
        semantic_result = run_semantic_checks(domain_pddl, problem_pddl, strict=False, extractor_type="auto")
        if not semantic_result["passed"]:
            logger.warning(f"Semantic checks failed:\n{semantic_result['combined_diagnosis']}")
            # Save semantic check results for debugging
            semantic_log_path = logs_dir / "semantic_checks_initial.txt"
            with open(semantic_log_path, "w") as f:
                f.write(semantic_result["combined_diagnosis"])

        # ------------------------------------------------------------------
        # Initialize Issue Stats tracker
        # ------------------------------------------------------------------
        issue_stats = IssueStats()

        # ------------------------------------------------------------------
        # 3. REFINEMENT LOOP (L-11 Temperature Ladder & L-6 Trajectory)
        # ------------------------------------------------------------------
        max_refinements = 4
        full_history = []
        repair_trajectory = [] # L-6: Detailed refinement history
        
        for r in range(max_refinements):
            temp = 0.1 + (r * 0.2) # Escalating temperature
            refinement_dir = results_dir / f"refinement_{r}"
            refinement_dir.mkdir(parents=True, exist_ok=True)
            problems_dir = refinement_dir / "problems"
            problems_dir.mkdir(parents=True, exist_ok=True)
            
            current_domain_path = refinement_dir / "domain.pddl"
            current_prob_path = problems_dir / "problem.pddl"
            shutil.copy(domain_file_path, current_domain_path)
            shutil.copy(problem_file_path, current_prob_path)
            
            # L-2: Object Hallucination Check before planning
            with open(current_prob_path, 'r') as f: prob_pddl = f.read()
            bad_objs = registry.scan_pddl(prob_pddl)
            if bad_objs:
                logger.warning(f"L-2: Detected hallucinated objects: {bad_objs}. Forcing refinement.")
                hallucination_error = f"HALLUCINATION_ERROR: The following objects do NOT exist in the world: {bad_objs}. Do NOT use them."
                # We skip planning and go straight to refinement with this error
                plan, pddlenv_error, planner_error = None, hallucination_error, None
            else:
                # Try planning
                plan_file_out = refinement_dir / f"plan_{r}.out"
                plan, pddlenv_error, planner_error, stats = plan_with_output(
                    domain_file_path=str(current_domain_path),
                    problem_dir=str(problems_dir),
                    plan_file_path=str(plan_file_out),
                    planner_name=kwargs.get("planner_name", "fd")
                )

                # L-8: Detect and repair empty plan hallucinations
                if plan == "":
                     logger.warning("L-8: Empty plan detected. Verifying if goal is truly satisfied...")
                     hallucination_error = "HALLUCINATION_DETECTED: Planner returned empty plan on a non-trivial goal. You likely hallucinated the goal into the initial state or wrote an unreachable goal."
                     plan, pddlenv_error = None, hallucination_error

            if plan is not None:
                # Validate Plan with VAL
                VAL_successful, VAL_validation_log, VAL_ground_successful, VAL_grounding_log = \
                    self._check_val_planning(str(current_domain_path), str(current_prob_path), str(plan_file_out), f"{workflow_iteration}_{r}", refinement_dir)

                if VAL_successful and VAL_ground_successful:
                    # VAL passed - now check simulator if available
                    if self.gt_simulator is not None:
                        logger.info(f"VAL passed - validating plan on simulator...")
                        translated_plan_path = str(refinement_dir / f"translated_plan_{workflow_iteration}_{r}.txt")
                        sim_valid, sim_error = self._validate_plan_on_simulator(translated_plan_path)

                        if sim_valid:
                            logger.info(f"✓ Plan validated on simulator at refinement {r}")
                            full_history.append({"final_domain_file": str(current_domain_path), "final_problem_file": str(current_prob_path)})
                            return True, translated_plan_path, full_history
                        else:
                            logger.warning(f"✗ Plan failed simulator validation at refinement {r}")
                            logger.warning(f"Simulator feedback will be used for refinement")
                            # Use simulator error for refinement
                            planner_error = f"SIMULATOR_ERROR:\n{sim_error}"
                    else:
                        # No simulator - just use VAL validation
                        logger.info(f"Plan found and validated at refinement {r}")
                        full_history.append({"final_domain_file": str(current_domain_path), "final_problem_file": str(current_prob_path)})
                        return True, str(refinement_dir / f"translated_plan_{workflow_iteration}_{r}.txt"), full_history
                else:
                    # Plan failed VAL -> Extract error for refinement
                    planner_error = f"VAL_ERROR: {VAL_validation_log}"

            # 4. PERFORM REFINEMENT
            combined_error = f"{pddlenv_error or ''}\n{planner_error or ''}".strip()
            logger.info(f"Refinement {r}: Errors detected. Invoking repair (temp={temp})...")
            
            new_domain_str, new_problem_str, history = refine_domain_and_problem_unified(
                domain_file_path=str(current_domain_path),
                problem_file_path=str(current_prob_path),
                environment=environment_str,
                task=current_goal_text,
                logs_dir=str(logs_dir),
                workflow_iteration=workflow_iteration,
                refinement_iteration=r,
                agent=self.agent,
                nl_sections=nl_sections,
                pddlenv_error_log=combined_error,
                pddl_init_state=pddl_init_state,
                issue_stats=issue_stats,
                temperature=temp
            )
            
            with open(domain_file_path, "w") as f: f.write(new_domain_str)
            with open(problem_file_path, "w") as f: f.write(new_problem_str)
            full_history.extend(history)
            repair_trajectory.append({"r": r, "error": combined_error, "temp": temp})

            # Semantic verification after refinement (LAPIS Extension)
            semantic_result = run_semantic_checks(new_domain_str, new_problem_str, strict=False, extractor_type="auto")
            if not semantic_result["passed"]:
                logger.warning(f"Semantic checks failed after refinement {r}:\n{semantic_result['combined_diagnosis']}")
                semantic_log_path = logs_dir / f"semantic_checks_refinement_{r}.txt"
                with open(semantic_log_path, "w") as f:
                    f.write(semantic_result["combined_diagnosis"])

        # L-9: Final Regeneration Fallback
        logger.warning("L-9: All refinements failed. Attempting final regeneration...")
        generate_problem(
            domain_file_path=str(domain_file_path),
            task=current_goal_text,
            environment=environment_str,
            problem_file_path=str(problem_file_path),
            logs_dir=str(logs_dir),
            workflow_iteration=workflow_iteration,
            agent=self.agent,
            problem_nl=problem_nl,
            temperature=0.7
        )
        # Try one last plan
        plan_file_out = results_dir / "final_fallback_plan.out"
        plan, _, _, _ = plan_with_output(str(domain_file_path), str(results_dir), str(plan_file_out), kwargs.get("planner_name", "fd"))
        if plan:
             VAL_successful, _, VAL_ground_successful, _ = self._check_val_planning(str(domain_file_path), str(problem_file_path), str(plan_file_out), "final", results_dir)
             if VAL_successful and VAL_ground_successful:
                 return True, str(results_dir / "translated_plan_final.txt"), full_history

        # Log Issue Statistics
        logger.info(issue_stats.summary())
        return False, None, full_history

    def _validate_plan_on_simulator(self, plan_path: str) -> tuple[bool, str]:
        """
        Validate plan by executing it on the GT simulator.
        Returns (success, error_message).

        This provides rich feedback about why a plan fails in simulation,
        which can be used for refinement.
        """
        if self.gt_simulator is None:
            return True, ""  # No simulator available, skip validation

        try:
            from dataclasses import dataclass
            from unified_planning.io import PDDLReader

            # Read the plan file
            with open(plan_path) as f:
                plan_content = f.read()

            # Parse plan to actions
            actions = self._parse_plan_to_actions(plan_content)
            logger.info(f"Parsed {len(actions)} actions from plan for simulator validation")

            # Use the GT simulator's problem
            problem = self.gt_simulator.problem
            current_state = self.gt_simulator.get_initial_state()

            # Execute each action
            for i, action_obj in enumerate(actions):
                up_action_name = action_obj.action.name
                up_params = action_obj.actual_parameters

                # Find the action in the UP problem
                found_action = None
                for ua in problem.actions:
                    if ua.name.lower() == up_action_name.lower():
                        found_action = ua
                        break

                if not found_action:
                    error_msg = f"SIMULATOR ERROR: The simulator does not recognize action '{up_action_name}'.\n"
                    error_msg += f"This means the generated domain uses an action name that doesn't exist in the environment.\n"
                    error_msg += f"The action name may be misspelled, or a different action name should be used."
                    logger.error(f"Action {up_action_name} not found in simulator problem")
                    return False, error_msg

                # Map parameters (strings) to UP Objects
                up_args = []
                try:
                    for p_name in up_params:
                        if not problem.has_object(p_name):
                            # Try case-insensitive match
                            found_obj = None
                            for o in problem.all_objects:
                                if o.name.lower() == p_name.lower():
                                    found_obj = o
                                    break
                            if found_obj:
                                obj = found_obj
                            else:
                                error_msg = f"SIMULATOR ERROR: The simulator does not recognize object '{p_name}'.\n"
                                error_msg += f"This means the plan references an object that doesn't exist in the environment."
                                logger.error(f"Object {p_name} not found in simulator problem")
                                return False, error_msg
                        else:
                            obj = problem.object(p_name)
                        up_args.append(obj)

                    # Create the action instance
                    up_action = found_action(*up_args)

                except Exception as e:
                    error_msg = f"SIMULATOR ERROR: Failed to map parameters for action '{up_action_name}': {str(e)}"
                    logger.error(f"Failed to map params for action {up_action_name}: {e}")
                    return False, error_msg

                # Check if action is applicable
                if not self.gt_simulator.is_applicable(current_state, up_action):
                    # Generate detailed error feedback
                    error_msg = self._generate_simulator_error_feedback(
                        failed_step=i,
                        failed_action=up_action,
                        action_obj=action_obj,
                        current_state=current_state,
                        problem=problem,
                        executed_actions=actions[:i]
                    )
                    logger.error(f"Step {i}: Action {up_action} is NOT applicable in simulator state")
                    return False, error_msg

                # Apply the action
                try:
                    current_state = self.gt_simulator.apply(current_state, up_action)
                    logger.debug(f"Step {i}: Applied {up_action} in simulator")
                except Exception as e:
                    error_msg = f"SIMULATOR ERROR: Failed to apply action at step {i}:\n"
                    error_msg += f"Action: {up_action}\n"
                    error_msg += f"Error: {str(e)}"
                    logger.error(f"Step {i}: Failed to apply action {up_action}: {e}")
                    return False, error_msg

            # Check if goal is satisfied in final state
            # Use the simulator's is_goal method which properly evaluates all goals
            is_goal_satisfied = self.gt_simulator.is_goal(current_state)

            if is_goal_satisfied:
                logger.info("✓ All goals satisfied in simulator")
                return True, ""
            else:
                # Generate detailed goal failure feedback
                logger.error("✗ Plan executed but did NOT achieve the goal")
                error_msg = "SIMULATOR ERROR: Plan executed but did NOT achieve the goal.\n"
                error_msg += f"Total actions executed: {len(actions)}\n"
                error_msg += "\nGoal predicates NOT satisfied:\n"
                goals = problem.goals if hasattr(problem, 'goals') else [problem.goal]
                for goal in goals:
                    error_msg += f"  - {goal}\n"
                return False, error_msg

        except Exception as e:
            error_msg = f"SIMULATOR ERROR: Exception during validation: {str(e)}"
            logger.error(f"Simulator validation failed with exception: {e}")
            import traceback
            traceback.print_exc()
            return False, error_msg

    def _generate_simulator_error_feedback(self, failed_step: int, failed_action, action_obj,
                                            current_state, problem, executed_actions) -> str:
        """
        Generate detailed, actionable error feedback using LLM analysis.

        The LLM analyzes:
        - What action failed and why
        - Execution trace leading to failure
        - Observable environment state
        - Whether to refine domain, problem, or replan

        IMPORTANT: Only provides simulation observations, NOT GT structure.
        """
        from dataclasses import dataclass

        # Build execution trace
        trace_str = "Execution Trace (successful steps before failure):\n"
        for i, act in enumerate(executed_actions):
            trace_str += f"  Step {i}: {act.action.name}({', '.join(act.actual_parameters)})\n"

        # Get observable state
        state_values = []
        for fluent, value in list(current_state._values.items())[:50]:
            if value.is_bool_constant() and value.constant_value():
                fluent_name = fluent.fluent().name
                args = [str(arg).replace("'", "") for arg in fluent.args]
                if args:
                    state_values.append(f"({fluent_name} {' '.join(args)})")
                else:
                    state_values.append(f"({fluent_name})")

        state_str = "\n".join(state_values[:30])

        # Use LLM to analyze the failure
        system_prompt = """You are a PDDL planning expert analyzing simulation failures.

Your task: Analyze why an action failed in simulation and provide actionable feedback for refinement.

CRITICAL RULES:
1. You only see SIMULATION OBSERVATIONS (like robot sensors) - NOT ground truth structure
2. Do NOT reveal or compare to ground truth domain/problem
3. Focus on what the simulation showed, not what "should" be
4. Classify the error as: DOMAIN_EFFECTS, PROBLEM_INIT, or PLANNER_ERROR
5. Provide specific, actionable refinement suggestions"""

        user_prompt = f"""SIMULATION FAILURE ANALYSIS

Failed at Step: {failed_step}
Failed Action: {action_obj.action.name}({', '.join(action_obj.actual_parameters)})

{trace_str}

Observable Environment State (from simulation):
{state_str}

The simulator REJECTED the action - it was not applicable in this state.

ANALYZE:
1. Why did this action fail? (Look at the state and execution history)
2. Is this a DOMAIN_EFFECTS error (previous actions didn't produce expected state)?
3. Is this a PROBLEM_INIT error (initial state was wrong)?
4. Is this a PLANNER_ERROR (wrong parameters, but domain/problem are correct)?

Provide your analysis in this format:

ERROR CLASSIFICATION: [DOMAIN_EFFECTS | PROBLEM_INIT | PLANNER_ERROR]

DIAGNOSIS:
[2-3 sentences explaining what went wrong based on simulation observations]

ROOT CAUSE:
[1-2 sentences on why this happened]

RECOMMENDED FIX:
[Specific actionable steps - which component to refine and how]

HINT:
[Specific detail to look for when refining]"""

        # Get LLM analysis
        logger.info("Generating simulator error feedback via LLM analysis...")
        diagnosis = self.agent.llm_call(system_prompt, user_prompt, temperature=0.1)

        # Format the complete error message
        error_msg = f"SIMULATOR ERROR: Action NOT applicable at step {failed_step}\n\n"
        error_msg += f"Failed Action: {failed_action}\n"
        error_msg += f"  Name: {action_obj.action.name}\n"
        error_msg += f"  Parameters: {action_obj.actual_parameters}\n\n"
        error_msg += trace_str + "\n"
        error_msg += f"Observable Environment State:\n{state_str}\n\n"
        error_msg += "=" * 80 + "\n"
        error_msg += "LLM ANALYSIS & RECOMMENDED FIX:\n"
        error_msg += "=" * 80 + "\n\n"
        error_msg += diagnosis + "\n"

        return error_msg

    def _parse_plan_to_actions(self, plan_content: str):
        """Parse plan content into action objects."""
        from dataclasses import dataclass

        @dataclass
        class PDDLActionWrapper:
            name: str

        @dataclass
        class PDDLActionInstance:
            action: PDDLActionWrapper
            actual_parameters: list

        actions = []
        for line in plan_content.splitlines():
            line = line.strip()
            if not line or line.startswith(';'):
                continue

            # Replace parentheses and commas with spaces
            content = line.replace('(', ' ').replace(')', ' ').replace(',', ' ')
            parts = content.split()
            if not parts:
                continue

            # Handle optional step numbers
            if parts[0].endswith(':') or parts[0].isdigit():
                if len(parts) < 2:
                    continue
                action_name = parts[1]
                params = parts[2:]
            else:
                action_name = parts[0]
                params = parts[1:]

            action_obj = PDDLActionInstance(
                action=PDDLActionWrapper(name=action_name),
                actual_parameters=params
            )

            actions.append(action_obj)

        return actions

    def _check_val(self, domain_file_path, problem_file_path=None):
        """
        Helper method to perform VAL validation.
        If problem_file_path is provided, validates both domain and problem.
        Otherwise, validates only the domain.
        """
        if problem_file_path:
            logger.info(f"Checking problem and domain with VAL\n  Domain: {domain_file_path}\n  Problem: {problem_file_path}")
            VAL_successful, VAL_validation_log = VAL_validate(domain_file_path, problem_file_path)
        else:
            logger.info(f"Checking domain with VAL\n  Domain: {domain_file_path}")
            VAL_successful, VAL_validation_log = VAL_validate(domain_file_path)
        
        if not VAL_successful:
            logger.warning("VAL validation check failed")
            logger.debug(f"\n[VAL VALIDATION LOG START]\n{VAL_validation_log}\n[VAL VALIDATION LOG END]")
        else:
            logger.info("VAL validation check passed")
        return VAL_successful, VAL_validation_log

    def _check_val_planning(self, domain_file_path, problem_file_path, plan_file_path, translation_suffix, problem_dir):
        """
        Helper method to perform VAL validation and grounding for planning.
        Translates the plan using the given translation_suffix and then checks validation and grounding.
        Returns a tuple of (VAL_successful, VAL_validation_log, VAL_ground_successful, VAL_grounding_log).
        """
        # Translate the plan into a format parsable by VAL
        translated_plan_path = os.path.join(problem_dir, f"translated_plan_{translation_suffix}.txt")
        translate_plan(plan_file_path, translated_plan_path)
        
        logger.info(f"Checking planning with VAL\n  Domain: {domain_file_path}\n  Problem: {problem_file_path}\n  Plan: {translated_plan_path}")
        
        VAL_successful, VAL_validation_log = VAL_validate(domain_file_path, problem_file_path, translated_plan_path)
        if VAL_successful:
            logger.info("VAL validation check passed")
            logger.info("Attempting to ground the plan")
            VAL_ground_successful, VAL_grounding_log = VAL_ground(domain_file_path, problem_file_path)
            if VAL_ground_successful:
                logger.info("Plan is grounded")
            else:
                logger.warning("Plan is not grounded")
                logger.debug(f"\n[VAL GROUNDING LOG START]\n{VAL_grounding_log}\n[VAL GROUNDING LOG END]")
        else:
            logger.warning("VAL validation check failed")
            logger.debug(f"\n[VAL VALIDATION LOG START]\n{VAL_validation_log}\n[VAL VALIDATION LOG END]")
            VAL_ground_successful = False
            VAL_grounding_log = None
        return VAL_successful, VAL_validation_log, VAL_ground_successful, VAL_grounding_log

