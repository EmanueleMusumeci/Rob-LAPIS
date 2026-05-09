import os
import shutil
import logging
from pathlib import Path
from src.lapis.planner.low.planner_utils import plan_with_output
from src.lapis.planner.low.pddl_generation import generate_domain, generate_problem, refine_problem, refine_domain, refine_domain_and_problem_unified
from src.lapis.planner.low.pddl_verification import translate_plan, VAL_validate, VAL_ground
from src.lapis.utils.log import save_statistics

logger = logging.getLogger("my_logger")

class LowLevelPlanner:
    def __init__(self, agent, use_vector_db=False):
        self.agent = agent
        self.use_vector_db = use_vector_db

    def plan(self, workflow_iteration="0", subgoal_id=None, **kwargs):
        """
        Execute the low-level planning process.
        """
        goal_file_path = kwargs.get("goal_file_path")
        environment_str_arg = kwargs.get("environment_str")
        current_goal_text = kwargs.get("current_goal_text")
        results_dir = kwargs.get("results_dir")
        output_callback = kwargs.get("output_callback")  # Callback to write to pipeline output
        
        # Harmonized inputs from High-Level Planner
        hl_domain = kwargs.get("hl_domain")
        hl_problem = kwargs.get("hl_problem")
        hl_actions = kwargs.get("hl_actions")
        hl_goal = kwargs.get("hl_goal")
        hl_constraints = kwargs.get("hl_constraints")
        nl_sections = kwargs.get("nl_sections")
        
        # If explicit goal text is not provided, try to construct it from HL components
        if not current_goal_text and hl_goal:
            current_goal_text = f"Goal: {hl_goal}"
            # STRICT: hl_constraints (LTL) is NO LONGER appended to the goal or prompt here
            # as requested by the user to diminish LLM responsibility and avoid broken formulas.

        # Setup logging directories
        logs_dir = results_dir / "logs"
        logs_dir.mkdir(parents=True, exist_ok=True)
        
        # Determine scene description and domain description
        environment_str = environment_str_arg or ""
        domain_description = kwargs.get("domain_description")
        problem_nl = ""

        if nl_sections:
            # Domain gen gets only description + actions + pre + eff
            domain_description = f"{nl_sections['description']}\n\n{nl_sections['actions']}\n\n{nl_sections['preconditions']}\n\n{nl_sections['effects']}"
            # Environment for problem gen gets top description
            environment_str = nl_sections['description']
            # Objects and Init State for problem gen
            problem_nl = f"{nl_sections['objects']}\n\n{nl_sections['initial_state']}"
        elif not domain_description and hl_domain:
             domain_description = f"Domain: {hl_domain}"
             if hl_actions:
                 domain_description += f"\nActions: {hl_actions}"

        # ------------------------------------------------------------------
        # 1. GENERATE DOMAIN
        # ------------------------------------------------------------------
        print(f"\n[LLM] Generating domain PDDL...")
        logger.info("Generating domain PDDL")
        if output_callback:
            output_callback("Generated Domain (PDDL):")
        
        # generate_domain expects file paths to write to
        domain_file_path = results_dir / "domain.pddl"
        # We need a dummy goal file path if not provided, or just pass the text?
        # generate_domain signature: (domain_file_path, goal_file_path, domain_description, agent, logs_dir)
        # It reads goal_file_path to get task description for the prompt.
        # We can create a temporary goal file.
        temp_goal_file = results_dir / "goal.txt"
        
        with open(temp_goal_file, "w") as f:
            f.write(current_goal_text or "")
        

        generate_domain(
            domain_file_path=str(domain_file_path),
            goal_file_path=str(temp_goal_file),
            domain_description=domain_description,
            environment=environment_str,
            agent=self.agent,
            logs_dir=str(logs_dir)
        )
        
        print(f"  ✓ Generated domain: {domain_file_path}")
        
        # Write domain to output callback
        if output_callback and domain_file_path.exists():
            with open(domain_file_path, 'r') as f:
                domain_content = f.read()
            output_callback("-" * 80)
            output_callback(domain_content)
            output_callback("-" * 80)
            output_callback("")
        
        # Check domain with VAL
        val_success, val_log = self._check_val(str(domain_file_path))
        self._last_domain_val_log = val_log
        
        # ------------------------------------------------------------------
        # 2. GENERATE PROBLEM (Initial)
        # ------------------------------------------------------------------
        print(f"[LLM] Generating problem PDDL...")
        logger.info("Generating initial problem PDDL")
        if output_callback:
            output_callback("Generated Problem (PDDL):")
        
        problem_file_path = results_dir / "problem.pddl"
        
        generate_problem(
            domain_file_path=str(domain_file_path),
            task=current_goal_text,
            environment=environment_str,
            problem_file_path=str(problem_file_path),
            logs_dir=str(logs_dir),
            workflow_iteration="0",
            agent=self.agent,
            problem_nl=problem_nl
        )

        print(f"  ✓ Generated problem: {problem_file_path}")
        
        # Write problem to output callback
        if output_callback and problem_file_path.exists():
            with open(problem_file_path, 'r') as f:
                problem_content = f.read()
            output_callback("-" * 80)
            output_callback(problem_content)
            output_callback("-" * 80)
            output_callback("")
        
        # Check problem and domain with VAL
        val_success, val_log = self._check_val(str(domain_file_path), str(problem_file_path))
        
        # Save INITIAL_PLANNING
        if problem_file_path.exists():
            save_statistics(
                phase="INITIAL_PLANNING",
                dir=str(results_dir),
                workflow_iteration=workflow_iteration,
                subgoal_id=subgoal_id,
                VAL_validation_log=val_log
            )

        # ------------------------------------------------------------------
        # 3. ITERATIVE REFINEMENT LOOP
        # ------------------------------------------------------------------
        # Refinement options
        use_unified_refinement = kwargs.get("use_unified_refinement", True)
        use_two_step_refinement = kwargs.get("use_two_step_refinement", True)
        logger.info(f"Refinement mode: {'Unified' if use_unified_refinement else 'Separate'}, Two-step: {use_two_step_refinement}")

        # Track domain generation stats
        if domain_file_path.exists():
            save_statistics(
                phase="DOMAIN_GENERATION",
                dir=str(results_dir),
                workflow_iteration=workflow_iteration,
                subgoal_id=subgoal_id,
                VAL_validation_log=getattr(self, '_last_domain_val_log', None)
            )

        full_history = []
        max_iterations = 3
        max_refinements = 3
        
        for i in range(max_iterations):
            iteration_dir = results_dir / f"iteration_{i}"
            iteration_dir.mkdir(parents=True, exist_ok=True)
            
            # We'll keep refining the problem PDDL up to max_refinements
            # We work with files, so we copy the initial problem to the iteration dir
            current_problem_path = iteration_dir / "problem.pddl"
            shutil.copy(problem_file_path, current_problem_path)
            
            for r in range(max_refinements):
                logger.info(f"\n================================================================================")
                logger.info(f"------------ Refinement iteration {r+1}/{max_refinements} ------------")
                logger.info(f"================================================================================\n")
                
                refinement_dir = iteration_dir / f"refinement_{r}"
                refinement_dir.mkdir(parents=True, exist_ok=True)
                
                # Create problems directory to satisfy PDDLEnv requirements
                # PDDLEnv loads all .pddl files in the directory as problems, so we must separate domain
                problems_dir = refinement_dir / "problems"
                problems_dir.mkdir(parents=True, exist_ok=True)
                
                # Copy current PDDLs to refinement dir
                current_domain_path = refinement_dir / "domain.pddl"
                current_prob_path = problems_dir / "problem.pddl"
                shutil.copy(domain_file_path, current_domain_path)
                shutil.copy(current_problem_path, current_prob_path)
                
                # Check problem and domain with VAL BEFORE planning
                VAL_successful, VAL_validation_log = self._check_val(str(current_domain_path), str(current_prob_path))

                if not VAL_successful:
                    logger.warning("VAL check failed for problem. Refining immediately...")
                    
                    if use_unified_refinement:
                        # UNIFIED REFINEMENT: Refine domain and problem together
                        logger.info("Starting UNIFIED refinement (Domain + Problem)...")
                        new_domain_str, new_problem_str, history = refine_domain_and_problem_unified(
                            domain_file_path=str(current_domain_path),
                            problem_file_path=str(current_prob_path),
                            environment=environment_str,
                            task=current_goal_text,
                            logs_dir=str(logs_dir),
                            workflow_iteration=i,
                            refinement_iteration=r,
                            agent=self.agent,
                            nl_sections=nl_sections,
                            VAL_validation_log=VAL_validation_log,
                            use_two_step_refinement=use_two_step_refinement,
                            use_vector_db=kwargs.get("use_vector_db", False),
                            vector_db_path=kwargs.get("vector_db_path", None)
                        )
                        full_history.extend(history)
                        
                        # Update both files
                        logger.info(f"Updating domain file: {current_domain_path}")
                        with open(current_domain_path, "w") as f:
                            f.write(new_domain_str)
                        logger.info(f"Updating problem file: {current_problem_path}")
                        with open(current_problem_path, "w") as f:
                            f.write(new_problem_str)
                            
                        # CRITICAL FIX: Update the tracking paths so the next iteration uses the refined files
                        domain_file_path = current_domain_path
                            
                    else:
                        # SEPARATE REFINEMENT: Refine domain and problem separately
                        # Refine DOMAIN using VAL logs
                        logger.info("Starting SEPARATE refinement - Step 1: DOMAIN...")
                        new_domain_str, history = refine_domain(
                            domain_file_path=str(current_domain_path),
                            problem_file_path=str(current_prob_path),
                            environment=environment_str,
                            task=current_goal_text,
                            logs_dir=str(logs_dir),
                            workflow_iteration=i,
                            refinement_iteration=r,
                            agent=self.agent,
                            nl_sections=nl_sections,
                            VAL_validation_log=VAL_validation_log,
                            use_two_step_refinement=use_two_step_refinement,
                            use_vector_db=kwargs.get("use_vector_db", False),
                            vector_db_path=kwargs.get("vector_db_path", None),
                            use_taxonomy=kwargs.get("use_taxonomy", True),
                        )
                        full_history.extend(history)

                        # Update current domain file immediately so problem refinement uses the new domain
                        logger.info(f"Updating domain file: {current_domain_path}")
                        with open(current_domain_path, "w") as f:
                            f.write(new_domain_str)

                        # Refine problem using VAL logs (and potentially new domain)
                        logger.info("Starting SEPARATE refinement - Step 2: PROBLEM...")
                        new_problem_str, history = refine_problem(
                            domain_file_path=str(current_domain_path),
                            problem_file_path=str(current_prob_path),
                            environment=environment_str,
                            task=current_goal_text,
                            logs_dir=str(logs_dir),
                            workflow_iteration=i,
                            refinement_iteration=r,
                            agent=self.agent,
                            nl_sections=nl_sections,
                            VAL_validation_log=VAL_validation_log,
                            use_two_step_refinement=use_two_step_refinement,
                            use_vector_db=kwargs.get("use_vector_db", False),
                            vector_db_path=kwargs.get("vector_db_path", None),
                            use_taxonomy=kwargs.get("use_taxonomy", True),
                        )
                        full_history.extend(history)

                        # Update current problem file for next refinement
                        logger.info(f"Updating problem file: {current_prob_path}")
                        with open(current_prob_path, "w") as f:
                            f.write(new_problem_str)
                        
                        # Persist back to base files so next main iterations start from here
                        with open(domain_file_path, "w") as f:
                            f.write(new_domain_str)
                        with open(problem_file_path, "w") as f:
                            f.write(new_problem_str)
                        
                        save_statistics(
                            phase="PDDL_REFINEMENT",
                            dir=str(results_dir),
                            workflow_iteration=workflow_iteration,
                            subgoal_id=subgoal_id,
                            pddl_refinement_iteration=r,
                            VAL_validation_log=VAL_validation_log,
                            db_entries=history # history contains issue/solution/is_valid info
                        )
                        
                    continue # Skip planning and go to next refinement iteration

                # Try planning
                logger.info("Attempting to plan...")
                
                # Initialize VAL logs to None in case planning fails or VAL check is skipped
                VAL_grounding_log = None
                # VAL_validation_log is already set by _check_val above, but let's ensure it's available for refine_problem
                
                # plan_with_output signature: (domain_file_path, problem_dir, plan_file_path, ...)
                # It expects problem.pddl to be in problem_dir
                plan_file_out = refinement_dir / f"plan_{r}.out"
                
                plan, pddlenv_error, planner_error, stats = plan_with_output(
                    domain_file_path=str(current_domain_path),
                    problem_dir=str(problems_dir), # Use the problems subdirectory
                    plan_file_path=str(plan_file_out),
                    planner_name=kwargs.get("planner_name", "fd")
                )
                
                if plan is None:
                    # User request: If plan is not found, log output and make file empty
                    if planner_error:
                        logger.info(f"Planner Output: {planner_error}")
                    if pddlenv_error:
                        logger.error(f"PDDLEnv Error: {pddlenv_error}")
                        
                    # Clear the file (plan_with_output writes error to it by default)
                    with open(plan_file_out, 'w') as f:
                        pass
                
                if plan is not None:
                    if not plan:
                        logger.info("Goal already satisfied (empty plan).")
                                
                    if plan is not None:
                        logger.info(f"Plan found! Saved to {refinement_dir}")
                    
                        # Check planning with VAL
                        VAL_successful, VAL_validation_log, VAL_ground_successful, VAL_grounding_log = \
                            self._check_val_planning(
                                str(current_domain_path), 
                                str(current_prob_path), 
                                str(plan_file_out), 
                                translation_suffix=f"{i}_{r}", 
                                problem_dir=str(refinement_dir)
                            )
                        
                        if VAL_successful and VAL_ground_successful:
                            logger.info("Plan passed VAL checks.")
                            
                            final_translated_plan = refinement_dir / f"translated_plan_{i}_{r}.txt"
                            
                            full_history.append({
                                "final_domain_file": str(current_domain_path),
                                "final_problem_file": str(current_prob_path)
                            })
                            
                            return True, str(final_translated_plan), full_history
                        else:
                            logger.warning("Plan failed VAL checks. Treating as failure.")
                            
                logger.info("Plan not found or invalid. Refining...")
                
                new_domain_str, history = refine_domain(
                    domain_file_path=str(current_domain_path),
                    problem_file_path=str(current_prob_path),
                    environment=environment_str,
                    task=current_goal_text,
                    logs_dir=str(logs_dir),
                    workflow_iteration=i,
                    refinement_iteration=r,
                    agent=self.agent,
                    nl_sections=nl_sections,
                    pddlenv_error_log=pddlenv_error,
                    planner_error_log=planner_error,
                    VAL_validation_log=VAL_validation_log,
                    VAL_grounding_log=VAL_grounding_log,
                    use_vector_db=kwargs.get("use_vector_db", self.use_vector_db),
                    vector_db_path=kwargs.get("vector_db_path", None),
                    use_taxonomy=kwargs.get("use_taxonomy", True),
                )
                full_history.extend(history)

                # Update current domain file immediately
                with open(current_domain_path, "w") as f:
                    f.write(new_domain_str)
                # Persist to base domain
                with open(domain_file_path, "w") as f:
                    f.write(new_domain_str)

                new_problem_str, history = refine_problem(
                    domain_file_path=str(current_domain_path),
                    problem_file_path=str(current_prob_path),
                    environment=environment_str,
                    task=current_goal_text,
                    logs_dir=str(logs_dir),
                    workflow_iteration=i,
                    refinement_iteration=r,
                    agent=self.agent,
                    nl_sections=nl_sections,
                    pddlenv_error_log=pddlenv_error,
                    planner_error_log=planner_error,
                    VAL_validation_log=VAL_validation_log,
                    VAL_grounding_log=VAL_grounding_log,
                    use_vector_db=kwargs.get("use_vector_db", self.use_vector_db),
                    vector_db_path=kwargs.get("vector_db_path", None),
                    use_taxonomy=kwargs.get("use_taxonomy", True),
                )
                full_history.extend(history)

                # Update current problem file for next refinement
                current_problem_path = iteration_dir / "problem.pddl" # Overwrite for next step? Or keep separate?
                # Let's overwrite the one in iteration_dir to be used as base for next refinement
                with open(current_problem_path, "w") as f:
                    f.write(new_problem_str)
                # Persist to base problem
                with open(problem_file_path, "w") as f:
                    f.write(new_problem_str)

                save_statistics(
                    phase="PDDL_REFINEMENT",
                    dir=str(results_dir),
                    workflow_iteration=workflow_iteration,
                    subgoal_id=subgoal_id,
                    pddl_refinement_iteration=r,
                    plan_successful=plan is not None,
                    pddlenv_error_log=pddlenv_error,
                    planner_error_log=planner_error,
                    planner_statistics=stats,
                    VAL_validation_log=VAL_validation_log,
                    VAL_grounding_log=VAL_grounding_log,
                    db_entries=history
                )
            
            logger.info("Out of PDDL refinements")
        
        return False, None, full_history

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

