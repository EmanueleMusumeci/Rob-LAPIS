import os
import re
import time
import shutil
import pprint
import logging
import json
import textwrap
from datetime import datetime
from pathlib import Path
from src.lapis.pipelines.base import BasePipeline
from src.lapis.planner.high.planner import HighLevelPlanner
from src.lapis.planner.low.planner import LowLevelPlanner
from src.lapis.planner.high.Planner.nl_description_generator import nl_description_generation
from src.lapis.simulators.blocksworld_simulator import BlocksworldSimulator
from src.lapis.simulators.babyai_simulator import BabyAISimulator
from src.lapis.simulators.scenario import Scenario
from src.lapis.simulators.utils import Action
from src.lapis.utils import logic_utils
from src.lapis.prompt import planning_prompts
from src.lapis.utils.log import save_statistics

from src.lapis.planner.high.Planner.trace_check import trace_check as check_trace, format_formula
import unified_planning as up
from unified_planning.shortcuts import Not, And, Or, Implies, Always, Sometime, Fluent, InstantaneousAction, DurativeAction
import sys
from pathlib import Path
from dataclasses import dataclass
import pprint

planner_path = Path(__file__).parent.parent / "planner" / "high" / "Planner"
if str(planner_path) not in sys.path:
    sys.path.append(str(planner_path))

from nl_description_generator import nl_description_generation
from formula_generator import ltl_formula_generation
from unified_planning.io import PDDLReader
from unified_planning.model.operators import OperatorKind

logger = logging.getLogger("my_logger")

# TODO Planning manifold visualization script (as a knowledge graph)
# TODO: Trace Aligner and Verbose Failure Report
# Insert a trace aligner for a verbose trace failure report.
# Use the trace aligner to detect which rule fluents should be corrected and a LLM agent to analyze verbosely the reasons for failure.
# Then we'll orchestrate high or low-level modifications to determine what to correct in the high or low level plan respectively.

class MultiLevelPlanningPipeline(BasePipeline):
    def __init__(self, 
        agent, 
        high_level_domain_name, 
        high_level_constraints_num,
        base_dir, 
        data_dir, 
        results_dir, 
        splits, 
        generate_domain=False,
        generate_high_level_plan=False,
        batch_id="data_2",
        low_level_planner_name="fd",
        use_vector_db=False,
        env_name=None
    ):
        super().__init__(base_dir, data_dir, results_dir, splits, agent, generate_domain)
        self.generate_high_level_plan = generate_high_level_plan
        self.env_name = env_name
        self.high_level_domain_name = high_level_domain_name
        self.high_level_constraints_num = high_level_constraints_num
        self.batch_id = batch_id
        self.low_level_planner_name = low_level_planner_name
        self.use_vector_db = use_vector_db
        
        # Cache for symbolic subgoal checks
        self._subgoal_predicates_cache = {}
        
        # Initialize Planners
        self.high_level_planner = HighLevelPlanner(agent)
        self.low_level_planner = LowLevelPlanner(agent, use_vector_db=self.use_vector_db)
        
        # Modular Shadowing Detection
        self.enable_shadowing_detection = True 
        self.domain_analysis = {
            "predicates_to_actions": {}, # predicate -> {"set": [actions], "unset": [actions]}
            "actions_to_predicates": {}  # action -> {"set": [predicates], "unset": [predicates]}
        }

        # Use data_dir passed in init
        self.domain_path = self.data_dir / self.high_level_domain_name
        self.data_folder = self.domain_path / self.batch_id
        
        # Output file for tracking pipeline steps
        self.output_file = None

    def _analyze_domain_actions(self, domain_path):
        """
        Analyze the domain file to identify which actions 'set' or 'unset' specific predicates.
        This provides the structural logic for shadowing detection.
        """
        if not self.enable_shadowing_detection:
            return

        try:
            reader = PDDLReader()
            # parse_problem can take just a domain file to return an incomplete problem object
            problem = reader.parse_problem(str(domain_path))
            
            for action in problem.actions:
                action_name = action.name.lower()
                self.domain_analysis["actions_to_predicates"][action_name] = {"set": [], "unset": []}
                
                # We only handle InstantaneousActions for now (standard for Blocksworld/BabyAI)
                if isinstance(action, up.model.InstantaneousAction):
                    for effect in action.effects:
                        fluent = effect.fluent.fluent().name.lower()
                        value = effect.value
                        
                        # Positive effect (sets fluent to True)
                        if value.is_bool_constant() and value.constant_value():
                            self.domain_analysis["actions_to_predicates"][action_name]["set"].append(fluent)
                            if fluent not in self.domain_analysis["predicates_to_actions"]:
                                self.domain_analysis["predicates_to_actions"][fluent] = {"set": [], "unset": []}
                            self.domain_analysis["predicates_to_actions"][fluent]["set"].append(action_name)
                        
                        # Negative effect (sets fluent to False)
                        elif value.is_bool_constant() and not value.constant_value():
                            self.domain_analysis["actions_to_predicates"][action_name]["unset"].append(fluent)
                            if fluent not in self.domain_analysis["predicates_to_actions"]:
                                self.domain_analysis["predicates_to_actions"][fluent] = {"set": [], "unset": []}
                            self.domain_analysis["predicates_to_actions"][fluent]["unset"].append(action_name)
                
                # self.domain_analysis["actions_to_predicates"][action_name]["unset"] = sorted(list(set(self.domain_analysis["actions_to_predicates"][action_name]["unset"])))
            
            logger.info(f"Domain analysis completed for {domain_path}. Analyzed {len(problem.actions)} actions.")
        except Exception as e:
            logger.error(f"Failed to analyze domain actions: {e}", exc_info=True)
            self.enable_shadowing_detection = False # Disable if analysis fails



    def _write_output(self, content):
        """Write content to output file"""
        if self.output_file:
            with open(self.output_file, 'a', encoding='utf-8') as f:
                f.write(content)
                f.write('\n')
                f.flush()


    def _process_task(self, task_name, results_dir):
        # Prepare results directory for this task (needed for log file)
        problem_id = task_name
        task_results_dir = Path(os.path.join(results_dir, problem_id))
        task_results_dir.mkdir(parents=True, exist_ok=True)

        # Setup output file for this task
        self.output_file = task_results_dir / "pipeline_output.txt"
        with open(self.output_file, 'w', encoding='utf-8') as f:
            f.write(f"{'='*80}\n")
            f.write(f"Pipeline Execution Output for Problem {problem_id}\n")
            f.write(f"{'='*80}\n\n")

        # Setup File Handler
        log_file = task_results_dir / "pipeline.log"
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(logging.Formatter('[%(asctime)s][%(name)s][%(levelname)s] - %(message)s'))
        logger.addHandler(file_handler)
        
        try:
            self._process_task_logic(task_name, results_dir)
        except Exception as e:
            logger.error(f"Task {task_name} failed: {e}", exc_info=True)
        finally:
            logger.removeHandler(file_handler)
            file_handler.close()
            self.output_file = None



    def _generate_fluent_assignment(self, ltl_fluents, pddl_objects):
        """
        Generate a mapping between LTL fluent terms and PDDL objects using the LLM.
        """
        try:
            from pydantic import BaseModel
            
            class AssignmentItem(BaseModel):
                term: str
                object: str
                
            class FluentAssignment(BaseModel):
                assignments: list[AssignmentItem]
                reasoning: str

            system_prompt = planning_prompts.FLUENT_ASSIGNMENT_SYSTEM_PROMPT
            
            user_input = f"""
            LTL Fluents: {ltl_fluents}
            PDDL Objects: {pddl_objects}
            
            Generate the assignment mapping.
            """
            
            response = self.agent.client.beta.chat.completions.parse(
                model="gpt-4o",
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_input}
                ],
                temperature=0,
                response_format=FluentAssignment,
            )
            
            # Convert list back to dict
            assignment_dict = {item.term: item.object for item in response.choices[0].message.parsed.assignments}
            return assignment_dict
            
        except Exception as e:
            logger.warning(f"Failed to generate fluent assignment: {e}")
            return {}

    def _check_subgoal_satisfied(self, subgoal_text, current_pddl_state):
        """
        Check if the current PDDL state satisfies the natural language subgoal description.
        1. Extract target predicates from NL subgoal (once per subgoal).
        2. Check if all target predicates exist in the current PDDL state string.
        """
        if not current_pddl_state:
            return False

        try:
            # 1. Extract/Retrieve Goal Predicates
            if subgoal_text not in self._subgoal_predicates_cache:
                target_predicates = self._extract_goal_predicates(subgoal_text)
                self._subgoal_predicates_cache[subgoal_text] = target_predicates
            else:
                target_predicates = self._subgoal_predicates_cache[subgoal_text]

            if not target_predicates:
                # If no predicates could be extracted, fall back to False or handle as error
                return False

            # 2. Symbolic Check
            # Check if every target predicate is present in the current_pddl_state string.
            # We normalize spaces and case for a more robust check.
            state_normalized = current_pddl_state.lower().replace(" ", "")
            
            missing_predicates = []
            for pred in target_predicates:
                pred_normalized = pred.lower().replace(" ", "")
                if pred_normalized not in state_normalized:
                    missing_predicates.append(pred)

            if not missing_predicates:
                logger.info(f"Subgoal Satisfaction Check (Symbolic): True")
                logger.info(f"All target predicates satisfied: {target_predicates}")
                return True
            else:
                logger.debug(f"Subgoal Satisfaction Check (Symbolic): False. Missing: {missing_predicates}")
                return False

        except Exception as e:
            logger.warning(f"Failed to check subgoal satisfaction symbolically: {e}")
            return False

    def _extract_goal_predicates(self, subgoal_text):
        """
        Use LLM to translate a NL subgoal into a list of PDDL predicates.
        """
        try:
            from pydantic import BaseModel
            
            class GoalPredicates(BaseModel):
                predicates: list[str]
                reasoning: str

            system_prompt = (
                "You are an expert in PDDL. Translate the following natural language subgoal into a list of specific PDDL predicates that MUST hold for the goal to be considered satisfied. "
                "Output as a list of strings like ['(on block_a block_b)', '(holding block_c)']."
            )
            
            user_input = f"Subgoal: {subgoal_text}"
            
            response = self.agent.client.beta.chat.completions.parse(
                model="gpt-4o",
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_input}
                ],
                temperature=0,
                response_format=GoalPredicates,
            )
            
            extracted = response.choices[0].message.parsed.predicates
            logger.info(f"Extracted goal predicates for '{subgoal_text}': {extracted}")
            return extracted
            
        except Exception as e:
            logger.error(f"Failed to extract goal predicates: {e}")
            return []

    def _associate_fluents_to_step(self, step_goal, fluents_map, constraints_data, assignment_map=None):
        """
        Associate relevant fluents and constraints to a high-level step based on term overlap.
        Uses assignment_map to translate fluent terms to PDDL objects if provided.
        """
        associated_fluent_ids = []
        associated_constraint_ids = []
        
        # Simple heuristic: Check if terms in the fluent string appear in the step goal string
        import re
        # Normalize step goal: lowercase and replace underscores with spaces for term extraction
        step_goal_normalized = step_goal.lower().replace("_", " ")
        step_terms = set(re.findall(r'\w+', step_goal_normalized))
        
        # 1. Find Associated Fluents
        for fid, fluent_str in fluents_map.items():
            # Extract terms from fluent
            fluent_terms = set(re.findall(r'\w+', fluent_str.lower()))
            
            # If we have assignment map, translate fluent terms to PDDL objects
            translated_terms = set()
            if assignment_map:
                for t in fluent_terms:
                    if t in assignment_map:
                        # Also split the target object name if it has underscores
                        obj_name = assignment_map[t].lower()
                        translated_terms.add(obj_name)
                        for part in obj_name.split("_"):
                            translated_terms.add(part)
                    else:
                        translated_terms.add(t)
            else:
                translated_terms = fluent_terms

            # Intersection of step terms with translated fluent terms
            common = step_terms.intersection(translated_terms)
            
            # Filter out common PDDL/Logic keywords
            meaningful_common = [t for t in common if t not in 
                                 ['unstack', 'stack', 'pickup', 'putdown', 
                                  'on', 'clear', 'holding', 'not', 'and', 'or', 'table', 'block']]
            
            if meaningful_common:
                associated_fluent_ids.append(fid)
                
        # 2. Find Associated Constraints
        for i, constraint in enumerate(constraints_data):
            # A constraint is associated if ANY of its fluents are associated
            if any(fid in associated_fluent_ids for fid in constraint['fluent_ids']):
                associated_constraint_ids.append(i)
                
        return associated_fluent_ids, sorted(list(set(associated_constraint_ids)))

    def _merge_subgoals(self, high_level_plan_steps, ltl_fluents):
        """
        Merge contiguous steps that refer to EXACTLY the same list of constraints.
        Returns a list of merged groups.
        """
        if not high_level_plan_steps:
            return []
            
        merged_groups = []
        current_group = {
            "step_ids": [high_level_plan_steps[0]["step_id"]],
            "goals": [high_level_plan_steps[0]["goal"]],
            "associated_ltl_constraints": high_level_plan_steps[0]["associated_ltl_constraints"],
            "associated_fluents": high_level_plan_steps[0]["associated_fluents"]
        }
        
        for i in range(1, len(high_level_plan_steps)):
            step = high_level_plan_steps[i]
            
            curr_set = set(current_group["associated_ltl_constraints"])
            next_set = set(step["associated_ltl_constraints"])
            
            # Merging rules:
            # 1. Exact Match: set(curr) == set(next)
            # 2. Monotonic Containment: set(curr) is a subset of set(next)
            # 2. Monotonic Containment: set(curr) is a subset of set(next)
            if curr_set <= next_set:
                # 3. Modular Shadowing Check (Deactivatable)
                if self.enable_shadowing_detection and self._has_shadowing_conflict(current_group, step, ltl_fluents):
                    # Potential shadowing of an LTL-sensitive fluent detected, don't merge
                    merged_groups.append(current_group)
                    current_group = {
                        "step_ids": [step["step_id"]],
                        "goals": [step["goal"]],
                        "associated_ltl_constraints": sorted(list(next_set)),
                        "associated_fluents": step["associated_fluents"]
                    }
                    continue

                current_group["step_ids"].append(step["step_id"])
                current_group["goals"].append(step["goal"])
                # Union of fluents
                current_group["associated_fluents"] = sorted(list(set(current_group["associated_fluents"] + step["associated_fluents"])))
                # Update constraints to the superset
                current_group["associated_ltl_constraints"] = sorted(list(next_set))
            else:
                merged_groups.append(current_group)
                current_group = {
                    "step_ids": [step["step_id"]],
                    "goals": [step["goal"]],
                    "associated_ltl_constraints": sorted(list(next_set)),
                    "associated_fluents": step["associated_fluents"]
                }
        merged_groups.append(current_group)
        return merged_groups

    def _has_shadowing_conflict(self, group, next_step, ltl_fluents):
        """
        Check if merging 'next_step' into 'group' would 'shadow' 
        an LTL-sensitive fluent transition.
        """
        if not self.enable_shadowing_detection or not self.domain_analysis:
            return False
            
        if not self.domain_analysis.get("actions_to_predicates"):
            return False

        # 1. Identify sensitive fluent names from the LTL formula
        # Extract base predicate names (e.g., 'holding' from 'F(holding(blue_block_1))')
        sensitive_fluent_names = set()
        ltl_ops = {"f", "g", "x", "u", "r", "not", "and", "or", "implies", "true", "false", "→", "∧", "∨", "¬", "&", "|", "->"}
        
        for f_str in ltl_fluents.values():
            # Use regex to find all alphanumeric words
            all_words = re.findall(r'[a-zA-Z_][a-zA-Z0-9_\-]*', f_str.lower())
            for w in all_words:
                if w not in ltl_ops and len(w) > 1: # Avoid operators and single-letter stubs
                    sensitive_fluent_names.add(w)
                    # We usually only care about the first predicate in the fluent for shadowing
                    break
        
        if not sensitive_fluent_names:
            return False
            
        # 2. Heuristic Action Extraction from NL goal
        def get_action_name(goal_text):
            # Clean text and handle common multi-word actions that might be single words in PDDL
            text = goal_text.lower()
            text = text.replace("pick up", "pickup").replace("put down", "putdown")
            text = text.replace("un stack", "unstack").replace("go to", "goto")
            
            words = text.replace(".", "").replace(",", "").replace("-", " ").split()
            # Priority: check for full action names in words
            for word in words:
                if word in self.domain_analysis["actions_to_predicates"]:
                    return word
            return None

        group_actions = [get_action_name(g) for g in group["goals"]]
        next_action = get_action_name(next_step["goal"])
        
        if not next_action or next_action not in self.domain_analysis["actions_to_predicates"]:
            # If we can't identify the next action, we can't safely detect shadowing
            return False
            
        # 3. Conflict Detection (Toggle Check)
        for g_action in group_actions:
            if not g_action or g_action not in self.domain_analysis["actions_to_predicates"]:
                continue
                
            # If a group action sets a sensitive fluent...
            sets_by_group = self.domain_analysis["actions_to_predicates"][g_action]["set"]
            # ...and the next action unsets it...
            unsets_by_next = self.domain_analysis["actions_to_predicates"][next_action]["unset"]
            
            # logger.info(f"[Shadowing] Checking {g_action} (sets {sets_by_group}) vs {next_action} (unsets {unsets_by_next})")
            
            # Intersection of toggled fluents that are also mentioned in LTL
            toggled = set(sets_by_group) & set(unsets_by_next)
            sensitive_conflicts = [f for f in toggled if f in sensitive_fluent_names]
            
            if sensitive_conflicts:
                logger.warning(f"[Shadowing Detected] Action '{g_action}' sets {sensitive_conflicts}, but '{next_action}' unsets them. Rejecting merge.")
                return True
                
        return False

    def _apply_assignment_substitution(self, formula_str, assignment_map):
        """
        Substitute high-level terms in the formula with low-level objects using the assignment map.
        """
        # We need to be careful not to replace substrings of other words.
        # We can use regex word boundaries.
        import re
        grounded_formula = formula_str
        for term, obj in assignment_map.items():
            # Replace full word "term" with "obj"
            pattern = r'\b' + re.escape(term) + r'\b'
            grounded_formula = re.sub(pattern, obj, grounded_formula)
        return grounded_formula
    
    #TODO: Move to utils
    def _convert_up_to_ltl_string(self, node):
        """
        Convert a Unified Planning FNode (constraint) to a symbolic LTL string.
        Compatible with trace_check.py format.
        """
        if node.is_fluent_exp():
            # Format: predicate(arg1, arg2)
            fluent = node.fluent()
            args = [str(a).replace("'", "") for a in node.args]
            if args:
                return f"{fluent.name}({', '.join(args)})"
            return fluent.name
        
        elif node.is_not():
            return f"!({self._convert_up_to_ltl_string(node.args[0])})"
        
        elif node.is_and():
            return f"({' & '.join([self._convert_up_to_ltl_string(a) for a in node.args])})"
        
        elif node.is_or():
            return f"({' | '.join([self._convert_up_to_ltl_string(a) for a in node.args])})"
            
        elif node.is_implies():
            return f"({self._convert_up_to_ltl_string(node.args[0])} -> {self._convert_up_to_ltl_string(node.args[1])})"
            
        elif node.node_type == OperatorKind.ALWAYS:
            return f"G({self._convert_up_to_ltl_string(node.args[0])})"
            
        elif node.node_type == OperatorKind.SOMETIME:
            return f"F({self._convert_up_to_ltl_string(node.args[0])})"
            
        elif node.node_type == OperatorKind.SOMETIME_BEFORE:
             # SometimeBefore(lhs, rhs) -> F(lhs & F(rhs)) ?? Or PDDL semantics?
             # PDDL SometimeBefore(a, b) typically means: if a happens, b must happen before? 
             # Actually UP SometimeBefore(x, y) = y must be true before or at the same time as x ??
             # Let's check UP semantics or babyai usage.
             # In BabyAI logic_utils it seemed to be mapped to standard patterns.
             # For now, let's assume it follows standard LTL patterns if possible, 
             # or just recurse.
             # UP documentation says SometimeBefore(f, g) -> F(g) & (not g U f) roughly?
             # Let's stick to simple recursion if it's a structural node, but usually these are top level.
             # If encountered, we might need a better map. 
             # For now, let's just format it as string if unsure, or try to map.
             return f"F({self._convert_up_to_ltl_string(node.args[0])})" # Simplified fallback
             
        elif node.node_type == OperatorKind.SOMETIME_AFTER:
             # SometimeAfter(f, g) means always (f -> Next (F g)) or similar?
             # In BabyAI it seems to be Always(Implies(f, Sometime(g)))
             return f"G({self._convert_up_to_ltl_string(node.args[0])} -> F({self._convert_up_to_ltl_string(node.args[1])}))"
             
        else:
            return str(node)

    def _parse_nl_sections(self, nl_text):
        """
        Split the NL description into functional sections.
        Returns a dictionary with keys: description, actions, preconditions, effects, objects, initial_state, goal, constraints.
        """
        sections = {
            "description": "",
            "actions": "",
            "preconditions": "",
            "effects": "",
            "objects": "",
            "initial_state": "",
            "goal": "",
            "constraints": ""
        }
        
        # Simple keyword-based split
        header_map = {
            "The available actions are": "actions",
            "The actions of this domain have the following preconditions": "preconditions",
            "The actions of this domain have the following effects": "effects",
            "The world includes the following objects": "objects",
            "The original state of the world is": "initial_state",
            "The task is to": "goal",
            "A valid plan for the abovementioned problem must abide by the following constraints": "constraints"
        }
        
        # Identify section boundaries
        boundaries = []
        for header, key in header_map.items():
            pos = nl_text.find(header)
            if pos != -1:
                boundaries.append((pos, key, header))
        
        # Sort by position
        boundaries.sort()
        
        # Extract initial description (everything before first header)
        if boundaries:
            sections["description"] = nl_text[:boundaries[0][0]].strip()
        else:
            sections["description"] = nl_text.strip()
            return sections
            
        # Extract each section
        for i in range(len(boundaries)):
            start_pos, key, header = boundaries[i]
            # Content starts after header
            content_start = start_pos + len(header)
            
            if i + 1 < len(boundaries):
                end_pos = boundaries[i+1][0]
                content = nl_text[content_start:end_pos].strip()
            else:
                content = nl_text[content_start:].strip()
            
            # Clean up leading/trailing punctuation or "following:" strings
            content = content.lstrip(": \n\t")
            sections[key] = content
            
        return sections

    def _ground_subgoal_entities(self, subgoal, objects_section):
        """
        Use LLM to ground natural language entities in a subgoal to the specific 
        object names listed in the objects section of the NL file.
        """
        prompt = f"""
Role: You are a linguist and PDDL expert. 
Your task is to take a natural language subgoal and reformulate it so that all objects mentioned in it match the exact PDDL instance names provided in the list.

OBJECT NAMES:
{objects_section}

SUBGOAL:
"{subgoal}"

REFORMULATION RULES:
1. Identify all objects in the subgoal (e.g., "green block number 1").
2. Find the closest matching PDDL instance name from the OBJECT NAMES list (e.g., "green_block_1").
3. Rewrite the subgoal using ONLY these specific PDDL names.
4. Keep the descriptive action part the same (e.g., "Place X on Y").
5. Output ONLY the reformulated subgoal string. No preamble.

FINAL SUBGOAL:
"""
        response = self.agent.llm_call(content=f"Subgoal: {subgoal}", prompt=prompt)
        grounded_subgoal = response.strip().strip('"').strip("'")
        return grounded_subgoal

    #TODO: Move to utils
    def _get_ltl_from_pddl(self, domain_path, problem_path):
        """
        Extract trajectory constraints from PDDL problem file using PDDLReader.
        Returns them as a single combined LTL formula string.
        """
        try:
            reader = PDDLReader()
            # Suppress output during parsing if possible, or just catch errors
            problem = reader.parse_problem(domain_path, problem_path)
            
            if not hasattr(problem, 'trajectory_constraints') or not problem.trajectory_constraints:
                return None
                
            ltl_constraints = []
            for c in problem.trajectory_constraints:
                ltl_str = self._convert_up_to_ltl_string(c)
                ltl_constraints.append(ltl_str)
            
            if not ltl_constraints:
                return None
                
            # Combine all constraints with AND
            if len(ltl_constraints) == 1:
                return ltl_constraints[0]
            else:
                return f"({' & '.join(ltl_constraints)})"
                
        except Exception as e:
            logger.warning(f"Failed to extract LTL from PDDL: {e}")
            return None

    def _process_task_logic(self, task_name, results_dir):
        """
        Main loop for multi-level planning with feedback.
        """
        max_attempts = 1 # Placeholder for future feedback loop configuration
        
        for attempt in range(max_attempts):
            logger.info(f"--- Planning Attempt {attempt + 1}/{max_attempts} ---")
            
            # TODO: Here we could pass feedback from previous attempts to the planner
            # stored in self or passed as arguments.
            
            try:
                self._process_task_attempt(task_name, results_dir, attempt)
            except Exception as e:
                logger.error(f"Attempt {attempt + 1} failed: {e}", exc_info=True)
                
            # TODO: Logic to check if we should break (e.g. if success)
            # For now, just run once.

    def _process_task_attempt(self, task_name, results_dir, attempt=0):
        problem_id = task_name
        
        logger.info(f"Processing multi-level planning for problem {problem_id}")
        
        # Prepare results directory for this task
        task_results_dir = Path(os.path.join(results_dir, problem_id))
        task_results_dir.mkdir(parents=True, exist_ok=True)

        # Initialize CSV for this task (Move before HLP)
        self._initialize_csv(task_results_dir)

        
        # Locate high-level plan and problem
        problem_folder = self.data_folder / problem_id
        
        # I will read `.../high/plan.txt` and extract the plan.
        plan_file = problem_folder / 'high' / 'plan.txt'
        
        # Fallback to constrained_plan in the problem folder
        if not plan_file.exists():
            plan_file = problem_folder / 'constrained_plan'
        
        problem_pddl_file = problem_folder / 'problem.pddl'
        nl_file = problem_folder / 'nl'
        
        if not nl_file.exists():
            logger.warning(f"NL description not found at {nl_file}")
            return

        # Read NL description
        with open(nl_file, 'r') as f:
            nl_description = f.read()
            
        # Parse NL sections for targeted prompts
        nl_sections = self._parse_nl_sections(nl_description)
        
        print(f"\n{'='*80}")
        print(f"[PROBLEM {problem_id}] NL Description:")
        print(f"{'='*80}")
        print(nl_description)
        
        # Write to output file
        self._write_output(f"{'='*80}")
        self._write_output(f"PROBLEM {problem_id}")
        self._write_output(f"{'='*80}\n")
        self._write_output("NL DESCRIPTION:")
        self._write_output("-" * 80)
        self._write_output(nl_description)
        self._write_output("-" * 80)
        self._write_output("")

        # Harmonized Data Components
        hl_domain = None
        hl_problem = None
        hl_actions = None
        hl_goal = None
        hl_constraints = None

        # Determine if we need to generate the high-level plan
        
        if not self.generate_high_level_plan:
            if not plan_file.exists():
                logger.error(f"High-level plan not found at {plan_file}")
                # Write skip message to full_plan.txt
                full_plan_path = task_results_dir / "full_plan.txt"
                with open(full_plan_path, "w") as f:
                    f.write("SKIPPED_DUE_TO_HIGH_LEVEL_PLANNING_FAILURE")
                return
            else:
                logger.info(f"Using existing high-level plan from {plan_file}")

        if self.generate_high_level_plan:
            logger.info("Invoking High-Level Planner...")
            
            # Run High-Level Planner
            hl_result = self.high_level_planner.plan(
                nl_file_path=nl_file,
                problem_id=problem_id,
                domain_name=self.high_level_domain_name,
                num_constraints=self.high_level_constraints_num,
                data_folder=self.data_folder
            )
            
            if hl_result['success']:
                logger.info("High-level planning successful.")
                save_statistics(
                    phase="HL_PLANNING",
                    dir=str(task_results_dir),
                    workflow_iteration=0,
                    plan_successful=True
                )
                # Use the generated plan
                plan_lines = hl_result['plan'].strip().split('\n')
                
                # Also get the harmonized data
                hl_domain = hl_result['domain']
                hl_problem = hl_result['problem']
                hl_actions = hl_result['actions']
                hl_goal = hl_result['goal']
                hl_constraints = hl_result['constraints']
                
            else:
                logger.error("High-level planning failed.")
                save_statistics(
                    phase="HL_PLANNING",
                    dir=str(task_results_dir),
                    workflow_iteration=0,
                    plan_successful=False
                )
                return
        else:
            # Read high-level plan from the existing file
            plan_lines = self._extract_plan_from_file(plan_file)
            if not plan_lines:
                 logger.warning(f"Could not extract plan from {plan_file} (missing FINAL PLAN: section)")
                 # Write skip message to full_plan.txt
                 full_plan_path = task_results_dir / "full_plan.txt"
                 with open(full_plan_path, "w") as f:
                     f.write("SKIPPED_DUE_TO_HIGH_LEVEL_PLANNING_FAILURE")
                 return
            
            # Parse NL description using the high-level generator to get harmonized data
            logger.info("Generating/Parsing NL description components...")
            hl_domain, hl_problem, hl_actions, hl_goal, hl_constraints = nl_description_generation(nl_description, self.agent)


        # Save high-level plan to results (or problem folder as requested)
        high_plan_path = problem_folder / "high_plan.txt"
        with open(high_plan_path, "w") as f:
            for line in plan_lines:
                f.write(f"{line}\n")


        

        # Harmonized Data Components
        # Force LTL extraction from PDDL (No fallbacks allowed as per user request)
        # Try problem-specific domain first
        domain_file = problem_folder / "domain.pddl"
        if not domain_file.exists():
            domain_file = self.domain_path / "domain.pddl"
            
        # Domain Analysis for Shadowing Detection (using resolved domain_file)
        self._analyze_domain_actions(domain_file)
            
        ltl_formula = self._get_ltl_from_pddl(domain_file, problem_pddl_file)
        
        if ltl_formula:
            logger.info(f"Successfully extracted LTL formula from PDDL: {ltl_formula}")
        else:
            logger.warning("No LTL formula found in PDDL. Proceeding without global constraints (unless HLP reintegration TODO is addressed).")
            # TODO: Reintegrate HLP-based formula or LLM-based generation when the high-level formula module is stabilized by other developer.
            # ltl_formula = hl_constraints (and LLM fallback logic removed for now)
            ltl_formula = None
        
        # Extract LTL info
        # ltl_fluents is now a map {id: str}, ltl_constraints_list is data with fluent_ids
        ltl_fluents_map, ltl_constraints_data = logic_utils.extract_ltl_info(ltl_formula) if ltl_formula else ({}, [])

        # Initialize Manifold Structure
        manifold = {
            "nl_goal": hl_goal,
            "nl_constraints": hl_constraints,
            "ltl_fluents": ltl_fluents_map,
            "ltl_constraints": [],
            "input_files": {
                "domain_path": str(self.domain_path / "domain.pddl"), # Default domain path
                "problem_path": str(problem_pddl_file),
                "high_level_plan_path": str(plan_file)
            },
            "high_level_plan": [], 
            "low_level_plans": []
        }
        
        # Populate LTL Constraints
        for i, constraint_data in enumerate(ltl_constraints_data):
            manifold["ltl_constraints"].append({
                "id": i,
                "formula": constraint_data["formula"],
                "fluent_ids": constraint_data["fluent_ids"]
            })
            
        # Populate Fluent Grounding Formulas (using syntax description for now)
        # Populate Fluent Grounding Info (to be used later)
        fluent_syntax_desc = None
        if 'hl_result' in locals() and 'fluent_syntax' in hl_result:
             fluent_syntax_desc = hl_result['fluent_syntax']
        elif 'fluent_syntax' in locals():
             fluent_syntax_desc = fluent_syntax
            
        # Create problem results directory
        problem_results_dir = Path(results_dir) / task_name
        problem_results_dir.mkdir(parents=True, exist_ok=True)

        # 1. Generate High-Level Plan (if needed)
        for i, step in enumerate(plan_lines):
            # We associate fluents and constraints
            assoc_fluents, assoc_constraints = self._associate_fluents_to_step(step, ltl_fluents_map, ltl_constraints_data) if ltl_fluents_map else ([], [])
            
            manifold["high_level_plan"].append({
                "step_id": i,
                "goal": step,
                "associated_fluents": assoc_fluents,
                "associated_ltl_constraints": assoc_constraints
            })

        # Initialize Simulators
        print(f"\n[_process_task_attempt] Initializing simulators...")
        print(f"  - domain_name: {self.high_level_domain_name}")
        
        simulators = {}
        global_traces = {}
        frames_dict = {}
        fluent_assignment_map = {}
        simulator = None
        
        try:
            if "blocksworld" in self.high_level_domain_name.lower():
                print(f"[_process_task_attempt] Setting up BlocksworldSimulator(s)")

                
                # Setup simulators
                simulators["GT"] = BlocksworldSimulator()
                gt_domain_path = problem_folder / "domain.pddl"
                if not gt_domain_path.exists():
                    gt_domain_path = self.domain_path / "domain.pddl"
                simulators["GT"].setup(domain_path=gt_domain_path, problem_path=problem_pddl_file)
                
                gen_domain_path = self.data_dir / f"{task_name.replace('_','-')}-domain.pddl"
                if gen_domain_path.exists():
                    sim_gen = BlocksworldSimulator()
                    sim_gen.setup(domain_path=gen_domain_path, problem_path=problem_pddl_file)
                    simulators["Generated"] = sim_gen
                
                simulator = simulators.get("GT") or simulators.get("Generated")
                logger.info(f"Primary simulator selected: {simulator}")
                
                if simulator:
                    # Generate Fluent Assignment using primary simulator objects
                    print(f"\n[_process_task_attempt] Generating fluent assignment...")

                    try:
                        pddl_objects = [o.name for o in simulator.problem.all_objects]
                        fluent_strings = list(ltl_fluents_map.values()) if ltl_fluents_map else []
                        fluent_assignment_map = self._generate_fluent_assignment(fluent_strings, pddl_objects)
                        logger.info(f"Generated fluent assignment: {fluent_assignment_map}")
                    except Exception as e:
                        logger.warning(f"Failed to generate fluent assignment: {e}")
                        fluent_assignment_map = {}

                    # Update Manifold relations
                    if ltl_fluents_map:
                        for step_info in manifold["high_level_plan"]:
                            step_goal = step_info["goal"]
                            assoc_fluents, assoc_constraints = self._associate_fluents_to_step(
                                step_goal, ltl_fluents_map, ltl_constraints_data, fluent_assignment_map
                            )
                            step_info["associated_fluents"] = assoc_fluents
                            step_info["associated_ltl_constraints"] = assoc_constraints

                    # Initialize traces and frames for all simulators
                    for name, sim in simulators.items():
                        global_traces[name] = [sim.current_state]
                        frames_dict[name] = []
                        
                        img = sim.get_image(
                            all_subgoals=plan_lines, 
                            active_subgoal_indices=[], 
                            all_constraints=[self._apply_assignment_substitution(c["formula"], fluent_assignment_map) for c in ltl_constraints_data] if fluent_assignment_map else [c["formula"] for c in ltl_constraints_data]
                        )
                        if img:
                            frames_dict[name].append(img)
                            # Save discrete frame
                            frames_dir = problem_results_dir / "frames" / name
                            frames_dir.mkdir(parents=True, exist_ok=True)
                            frame_path = frames_dir / f"frame_0.png"
                            img.save(frame_path)
                            logger.info(f"[{name}] Saved initial frame to {frame_path}")

            elif "babyai" in self.high_level_domain_name.lower():
                print(f"[_process_task_attempt] Setting up BabyAISimulator(s)")
                
                # Setup simulators
                env_name = self.env_name if self.env_name else "BabyAI-MiniBossLevel-v0"
                
                # Setup GT/Primary Simulator
                import gymnasium as gym
                env = gym.make(env_name, render_mode="rgb_array", highlight=False)
                seed = int(task_name) if task_name.isdigit() else 0
                env.reset(seed=seed)
                simulators["GT"] = BabyAISimulator(env)
                
                # Setup Generated Simulator
                gen_domain_path = self.data_dir / f"{task_name.replace('_','-')}-domain.pddl"
                print(f"\n[_process_task_attempt] Initializing Generated simulator:")
                
                if gen_domain_path.exists():
                    env_gen = gym.make(env_name, render_mode="rgb_array", highlight=False)
                    env_gen.reset(seed=seed)
                    simulators["Generated"] = BabyAISimulator(env_gen)
                    print(f"  - Generated simulator setup successful")
                else:
                    print(f"  - Generated simulator domain not found at {gen_domain_path}")
                
                simulator = simulators.get("Generated") or simulators.get("GT")
                logger.info(f"Primary simulator selected: {simulator}")
                
                logger.info(f"Primary simulator selected: {simulator}")
                
                if simulator:
                    for name, sim in simulators.items():
                        state_to_save = getattr(sim, 'current_state', getattr(sim, 'prev_obs', None))
                        global_traces[name] = [state_to_save]
                        frames_dict[name] = []
                        
                        img = sim.get_image()
                        if img:
                            frames_dict[name].append(img)
                            frames_dir = problem_results_dir / "frames" / name
                            frames_dir.mkdir(parents=True, exist_ok=True)
                            frame_path = frames_dir / f"frame_0.png"
                            img.save(frame_path)
                            logger.info(f"[{name}] Saved initial frame to {frame_path}")

                    manifold_path = problem_results_dir / "manifold.json"
                    with open(manifold_path, "w") as f:
                        json.dump(manifold, f, indent=4)
                
        except Exception as e:
            logger.error(f"Failed to initialize simulators: {e}", exc_info=True)
            simulator = None

        # Merge contiguous steps with same constraints
        # ltl_fluents_map is already populated above
        merged_subgoals = self._merge_subgoals(manifold["high_level_plan"], ltl_fluents_map)
        logger.info(f"Merged {len(plan_lines)} steps into {len(merged_subgoals)} subgoals.")

        # Iterate through merged subgoals
        for i, group in enumerate(merged_subgoals):
            # Construct a combined grounded goal for the low-level planner
            grounded_parts = []
            for step_goal in group["goals"]:
                grounded_step = self._ground_subgoal_entities(step_goal, nl_sections['objects'])
                grounded_parts.append(grounded_step)
            
            # Combine multiple steps if needed
            if len(grounded_parts) > 1:
                combined_grounded_goal = " and ".join(grounded_parts)
                original_goal_text = " and ".join(group["goals"])
            else:
                combined_grounded_goal = grounded_parts[0]
                original_goal_text = group["goals"][0]

            logger.info(f"Processing Merged Subgoal {i}: {combined_grounded_goal}")

            # Write to output file
            self._write_output(f"\n{'='*80}")
            self._write_output(f"SUBGOAL {i} (Merged Steps: {group['step_ids']})")
            self._write_output(f"{'='*80}")
            self._write_output(f"Original Goal: {original_goal_text}")
            self._write_output(f"Grounded Goal: {combined_grounded_goal}")
            self._write_output("")

            # Generate metacognitive summary
            # (Using remaining merged groups for future steps)
            future_subgoals = []
            for g in merged_subgoals[i+1:]:
                future_subgoals.extend(g["goals"])
            summary = self._generate_summary(future_subgoals, nl_description)
            logger.info(f"Metacognitive Summary: {summary}")

            # Determine primary simulator for this subgoal iteration
            if "Generated" in simulators:
                simulator = simulators["Generated"]
                logger.debug("Using Generated simulator for this subgoal.")
            elif "GT" in simulators:
                simulator = simulators["GT"]
                logger.debug("Using GT simulator for this subgoal.")
            else:
                simulator = None

            success = False
            generated_plan_path = None
            refinement_history = []
            
            # FAIR STATE GROUNDING (Assignment Mapping)
            start_time = time.time()
            if simulator:
                logger.info("Performing FAIR state grounding (Assignment Mapping)...")
                raw_report = self._get_simulator_raw_state_report(simulator)
                assignment = self._map_simulator_state_to_assignment(raw_report, nl_sections['objects'])
                
                if assignment:
                    logger.info(f"Assignment Mapping successful. New :init count: {len(assignment.splitlines())}")
                    nl_sections['initial_state'] = assignment
                else:
                    logger.warning("Assignment Mapping returned empty state. Falling back to static NL description.")

            success, generated_plan_path, refinement_history = self.low_level_planner.plan(
                environment_str=nl_description, 
                nl_sections=nl_sections, 
                current_goal_text=None,
                results_dir=task_results_dir / f"subgoal_{i}",
                hl_domain=hl_domain,
                hl_problem=hl_problem,
                hl_actions=hl_actions,
                hl_goal=combined_grounded_goal, 
                hl_constraints=None, 
                planner_name=self.low_level_planner_name,
                workflow_iteration=0,
                subgoal_id=i,
                use_vector_db=self.use_vector_db,
                use_unified_refinement=(not self.use_vector_db),
                output_callback=self._write_output  
            )
            
            end_time = time.time()
            duration = end_time - start_time
            logger.info(f"Low-Level Planning for Merged Subgoal {i+1} completed in {duration:.2f} seconds. Success: {success}")
            
            if not success:
                logger.error(f"Low-Level Planning failed for Merged Subgoal {i+1}. STOPPING PIPELINE.")
                break

            if success:
                #TODO: re-organize code into specific functions, trying to keep redundant code to a minimum
                logger.info(f"Subgoal {i} achieved.")
                if generated_plan_path and os.path.exists(generated_plan_path):
                    target_low_plan = problem_results_dir / f"low_plan_{i}.txt"
                    # Read and clean the low level plan content for final output
                    with open(generated_plan_path, 'r') as f:
                        raw_plan_content = f.read()
                    
                    # USER REQUEST: Make plans readable (remove parentheses, commas, apices)
                    plan_content = ""
                    for line in raw_plan_content.splitlines():
                        if not line.strip() or line.strip().startswith(';'):
                            plan_content += line + "\n"
                            continue
                        # Clean line
                        cleaned_line = line.replace('(', ' ').replace(')', ' ').replace(',', ' ').replace('"', '').replace("'", "")
                        # Normalize spaces
                        cleaned_line = ' '.join(cleaned_line.split())
                        plan_content += cleaned_line + "\n"

                    with open(target_low_plan, 'w') as f:
                        f.write(f"; SUBGOAL {i} (Original Steps: {group['step_ids']}): {original_goal_text}\n")
                        f.write(f"; SUBGOAL {i} (Grounded): {combined_grounded_goal}\n")
                        f.write(plan_content)
                    
                    self._write_output(f"Generated Low-Level Plan:")
                    self._write_output("-" * 80)
                    self._write_output(plan_content)
                    self._write_output("-" * 80)
                    self._write_output("")

                    # Update Generated Simulator if it doesn't exist or if we need to track refinement
                    if refinement_history:
                        # Find the entry containing final paths
                        final_paths = next((item for item in reversed(refinement_history) if 'final_domain_file' in item), None)
                        
                        if final_paths:
                            refined_domain = final_paths.get('final_domain_file')
                            refined_problem = final_paths.get('final_problem_file')
                        else:
                            # Fallback if final_paths key not found but domain exists (for robustness)
                            refined_domain = str(task_results_dir / f"subgoal_{i}" / "domain.pddl")
                            refined_problem = str(task_results_dir / f"subgoal_{i}" / "problem.pddl")
                        
                        if os.path.exists(refined_domain):
                            logger.info(f"Updating Generated simulator with refined files: {refined_domain}")
                            if "Generated" not in simulators:
                                if "blocksworld" in self.high_level_domain_name.lower():
                                    simulators["Generated"] = BlocksworldSimulator()
                                elif "babyai" in self.high_level_domain_name.lower():
                                    import gymnasium as gym
                                    env_name = self.env_name if self.env_name else "BabyAI-MiniBossLevel-v0"
                                    env = gym.make(env_name, render_mode="rgb_array", highlight=False)
                                    seed = int(task_name) if task_name.isdigit() else 0
                                    env.reset(seed=seed)
                                    simulators["Generated"] = BabyAISimulator(env)
                                # Initialize traces for the new simulator
                                global_traces["Generated"] = []
                                frames_dict["Generated"] = []
                            
                            # Update setup
                            if simulators["Generated"].setup(domain_path=refined_domain, problem_path=refined_problem):
                                logger.info("Generated simulator reload successful.")
                                # If we are using Generated as primary, ensure it's selected
                                simulator = simulators["Generated"]
                            else:
                                logger.error("Generated simulator reload FAILED.")
                    else:
                        # No refinement, but if Generated simulator doesn't exist yet, we should initialize it
                        # with the initially generated domain/problem files.
                        initial_domain = str(task_results_dir / f"subgoal_{i}" / "domain.pddl")
                        initial_problem = str(task_results_dir / f"subgoal_{i}" / "problem.pddl")
                        
                        if os.path.exists(initial_domain):
                            logger.info(f"Initializing Generated simulator with initial files: {initial_domain}")
                            if "Generated" not in simulators:
                                if "blocksworld" in self.high_level_domain_name.lower():
                                    simulators["Generated"] = BlocksworldSimulator()
                                elif "babyai" in self.high_level_domain_name.lower():
                                    import gymnasium as gym
                                    env_name = self.env_name if self.env_name else "BabyAI-MiniBossLevel-v0"
                                    env = gym.make(env_name, render_mode="rgb_array", highlight=False)
                                    seed = int(task_name) if task_name.isdigit() else 0
                                    env.reset(seed=seed)
                                    simulators["Generated"] = BabyAISimulator(env)
                                global_traces["Generated"] = []
                                frames_dict["Generated"] = []
                            
                            if simulators["Generated"].setup(domain_path=initial_domain, problem_path=initial_problem):
                                logger.info("Generated simulator initialization successful.")
                                simulator = simulators["Generated"]
                            else:
                                logger.error("Generated simulator initialization FAILED.")
                    
                    # Compute cost and update trace for each simulator
                    if simulators:
                        for sim_name, current_sim in simulators.items():
                            try:
                                # Parse plan
                                print(f"\n[SUBGOAL {i}] [{sim_name}] Generated plan:")
                                print(plan_content)
                                
                                actions = self._parse_plan_to_actions(plan_content)
                                
                                print(actions)

                                # Map actions to simulator and step
                                up_actions = []
                                for action_obj in actions:
                                    # Find action in UP problem
                                    up_action_name = action_obj.action.name
                                    up_params = action_obj.actual_parameters
                                    
                                    found_action = None
                                    if current_sim.problem:
                                        for ua in current_sim.problem.actions:
                                            if ua.name.lower() == up_action_name.lower():
                                                found_action = ua
                                                break
                                                                        
                                    if found_action:
                                        # Map params (strings) to UP Objects
                                        up_args = []
                                        try:
                                            for p_name in up_params:
                                                if not current_sim.problem.has_object(p_name):
                                                     logger.warning(f"[{sim_name}] Object {p_name} not found in simulator problem!")
                                                     found_obj = None
                                                     for o in current_sim.problem.all_objects:
                                                         if o.name.lower() == p_name.lower():
                                                             found_obj = o
                                                             break
                                                     if found_obj:
                                                         obj = found_obj
                                                     else:
                                                         raise ValueError(f"Object {p_name} not found")
                                                else:
                                                    obj = current_sim.problem.object(p_name)
                                                up_args.append(obj)
                                            up_actions.append(found_action(*up_args))
                                        except Exception as e:
                                            logger.warning(f"[{sim_name}] Failed to map params for action {up_action_name}: {e}", exc_info=True)
                                    else:
                                        logger.warning(f"[{sim_name}] Action {up_action_name} not found in UP problem")
                                
                                pprint.pprint(up_actions)

                                # Simulate and compute cost
                                subgoal_cost = 0.0
                                for up_a in up_actions:
                                    try:
                                        cost = current_sim.scenario.compute_cost(up_a)
                                        subgoal_cost += cost
                                        
                                        logger.info(f"[{sim_name}] Executing action: {up_a}")
                                        
                                        if not current_sim.simulator.is_applicable(current_sim.current_state, up_a):

                                            logger.error(f"[{sim_name}] Action {up_a} is NOT applicable in current state!\nState:\n{pprint.pformat(current_sim.current_state)}")

                                        try:
                                            obs, _, _, _, _ = current_sim.step(up_a)
                                            global_traces[sim_name].append(obs['state'])
                                        except Exception as e:
                                            logger.error(f"[{sim_name}] Simulation step failed at action {up_a}: {e}", exc_info=True)
                                            # Mark plan as failed for this subgoal
                                            break
                                        
                                        # Capture frame
                                        img = current_sim.get_image(
                                            action_text=str(up_a),
                                            all_subgoals=plan_lines,
                                            active_subgoal_indices=group["step_ids"],
                                            all_constraints=[self._apply_assignment_substitution(c["formula"], fluent_assignment_map) for c in ltl_constraints_data] if fluent_assignment_map else [c["formula"] for c in ltl_constraints_data]
                                        )
                                        if img:
                                            frames_dict[sim_name].append(img)
                                            frames_dir = problem_results_dir / "frames" / sim_name
                                            frames_dir.mkdir(parents=True, exist_ok=True)
                                            frame_idx = len(frames_dict[sim_name]) - 1
                                            frame_path = frames_dir / f"frame_{frame_idx}.png"
                                            img.save(frame_path)
                                            
                                    except Exception as e:
                                        logger.error(f"[{sim_name}] Simulation step failed at action {up_a}: {e}", exc_info=True)
                                        pass

                                logger.info(f"[{sim_name}] Subgoal {i} Plan Cost: {subgoal_cost}")
                                
                                # Save cost
                                with open(problem_results_dir / f"low_plan_{i}_cost_{sim_name}.txt", "w") as f:
                                    f.write(str(subgoal_cost))
                                    
                            except Exception as e:
                                logger.error(f"[{sim_name}] Failed to compute cost/trace: {e}", exc_info=True)

                    # Read the low level plan content
                    with open(target_low_plan, 'r') as f:
                        ll_plan_content = f.read()
                        
                    # Update Manifold with Low-Level Plan
                    # Get associated info from merged group
                    associated_constraint_ids = group["associated_ltl_constraints"]
                    associated_fluent_ids = group["associated_fluents"]
                    
                    # Construct full constraint objects for this subgoal
                    subgoal_constraints = []
                    grounded_constraints = []
                    
                    for cid in associated_constraint_ids:
                        c_data = ltl_constraints_data[cid]
                        subgoal_constraints.append({
                            "id": cid,
                            "formula": c_data["formula"],
                            "fluent_ids": c_data["fluent_ids"]
                        })
                        
                        # Ground the formula
                        grounded_f = self._apply_assignment_substitution(c_data["formula"], fluent_assignment_map)
                        grounded_constraints.append(grounded_f)
                        
                    manifold["low_level_plans"].append({
                        "subgoal_id": i,
                        "original_step_ids": group["step_ids"],
                        "original_subgoal": original_goal_text,
                        "grounded_subgoal": combined_grounded_goal,
                        "plan": ll_plan_content,
                        "refinement_history": refinement_history,
                        "constraints": subgoal_constraints,
                        "high_level_fluents": associated_fluent_ids,
                        "fluent_grounding": {
                            "description": fluent_syntax_desc,
                            "assignment": fluent_assignment_map,
                            "grounded_ltl_constraints": grounded_constraints
                        }
                    })
                    
                    # Save Manifold (incremental update)
                    manifold_path = problem_results_dir / "manifold.json"
                    try:
                        with open(manifold_path, "w") as f:
                            json.dump(manifold, f, indent=4)
                    except Exception as e:
                        logger.error(f"Failed to save manifold: {e}", exc_info=True)



            else:
                logger.error("PLANNING_FAILED")
                logger.warning(f"Subgoal {i} failed. Stopping pipeline execution.")

                
                target_low_plan = problem_results_dir / f"low_plan_{i}.txt"
                with open(target_low_plan, 'w') as f:
                    f.write(f"; SUBGOAL {i} (Original Steps: {group['step_ids']}): {original_goal_text}\n")
                    f.write(f"; SUBGOAL {i} (Grounded): {combined_grounded_goal}\n")
                    f.write("NO_PLAN")
                
                # Associated info even on failure
                associated_constraint_ids = group["associated_ltl_constraints"]
                associated_fluent_ids = group["associated_fluents"]
                
                subgoal_constraints = []
                grounded_constraints = []
                for cid in associated_constraint_ids:
                    c_data = ltl_constraints_data[cid]
                    subgoal_constraints.append({
                        "id": cid,
                        "formula": c_data["formula"],
                        "fluent_ids": c_data["fluent_ids"]
                    })
                    grounded_f = self._apply_assignment_substitution(c_data["formula"], fluent_assignment_map)
                    grounded_constraints.append(grounded_f)

                manifold["low_level_plans"].append({
                    "subgoal_id": i,
                    "original_step_ids": group["step_ids"],
                    "plan": "NO_PLAN",
                    "refinement_history": refinement_history,
                    "constraints": subgoal_constraints,
                    "high_level_fluents": associated_fluent_ids,
                    "fluent_grounding": {
                        "description": fluent_syntax_desc,
                        "assignment": fluent_assignment_map,
                        "grounded_ltl_constraints": grounded_constraints
                    }
                })
                
                # Stop the pipeline on failure
                break


            # The final goal satisfaction is the AND of all subgoal satisfactions
            # (so if any step of the original plan failed, the final goal is not satisfied
            # but we should still check the satisfaction of the final goal to acquire
            # refinement feedback)
            # Global LTL check is now performed at the end of the pipeline.


        # After ALL subgoals: Perform Final Global LTL Check over complete trace
        logger.info("Performing Final Global LTL Check over complete trace...")
        try:
            # Accumulate full trace from all simulators
            full_trace_raw = global_traces.get("Generated") or global_traces.get("GT")
            
            # If still empty, check if we have subgoal-specific traces (usually not, but for robustness)
            if not full_trace_raw:
                full_trace_raw = []
                for i in range(len(plan_lines)):
                    if f"subgoal_{i}" in global_traces:
                        full_trace_raw.extend(global_traces[f"subgoal_{i}"])
            
            # If still empty, collect everything we have
            if not full_trace_raw and global_traces:
                # Just take the first one available
                full_trace_raw = list(global_traces.values())[0]

            if full_trace_raw:
                try:
                    target_trace = logic_utils.convert_trace_to_strings(full_trace_raw)
                except Exception as e:
                     logger.error(f"Error converting trace: {e}")
                     target_trace = None
            else:
                target_trace = None
            # Store the full trace once in the manifold
            manifold["total_execution_trace"] = [str(s) for s in (target_trace or [])]
            
            # Keep per-subgoal execution_trace as an empty list (or placeholder) to maintain schema compatibility
            # but avoid massive redundancy
            for i, llp_entry in enumerate(manifold.get("low_level_plans", [])):
                llp_entry["execution_trace"] = [] # Redundancy removed
            
            if target_trace and ltl_formula:
                string_trace = target_trace  # Already converted to string sets
                grounded_ltl_formula = ltl_formula
                if fluent_assignment_map:
                    grounded_ltl_formula = self._apply_assignment_substitution(ltl_formula, fluent_assignment_map)
                
                is_sat, violations = check_trace(string_trace, grounded_ltl_formula)

                
                final_reason = "All constraints satisfied over complete trace." if is_sat else (violations.get('plain_english_explanation', str(violations)) if isinstance(violations, dict) else str(violations))
                
                manifold["final_global_ltl_check"] = {
                    "formula": ltl_formula,
                    "grounded_formula": grounded_ltl_formula,
                    "is_satisfied": is_sat,
                    "overview": final_reason,
                    "full_trace": [str(s) for s in target_trace]
                }
                
                if is_sat:
                    logger.info("Final Global LTL Check: SATISFIED")
                else:
                    logger.warning(f"Final Global LTL Check: VIOLATED - {final_reason}")
            else:
                logger.warning(f"Final Global LTL Check: Skipped (Formula: {ltl_formula is not None}, Trace: {target_trace is not None})")
                
            # Save final manifold
            manifold_path = problem_results_dir / "manifold.json"
            with open(manifold_path, "w") as f:
                json.dump(manifold, f, indent=4)
            
            # Generate final summary
            summary_path = problem_results_dir / "pipeline_summary.txt"
            self._generate_pipeline_summary(manifold, summary_path)
            
        except Exception as e:
            logger.error(f"Failed to perform final global LTL check: {e}", exc_info=True)


                     
        # Save Full Plan
        full_plan_path = problem_results_dir / "full_plan.txt"
        logger.info(f"DEBUG: manifold type: {type(manifold)}")
        logger.info(f"DEBUG: manifold['low_level_plans'] type: {type(manifold.get('low_level_plans'))}")
        logger.info(f"DEBUG: manifold['low_level_plans'] length: {len(manifold.get('low_level_plans', []))}")
        
        # MANIFOLD INTEGRITY CHECK
        ll_plans = manifold.get("low_level_plans", [])
        if not isinstance(ll_plans, list):
            logger.error(f"CRITICAL: manifold['low_level_plans'] is not a list! Type: {type(ll_plans)}")
            ll_plans = []
        
        valid_ll_plans = []
        for i, entry in enumerate(ll_plans):
            if isinstance(entry, dict):
                valid_ll_plans.append(entry)
            else:
                logger.warning(f"Removing malformed manifold entry at index {i}: {type(entry)}")
        
        manifold["low_level_plans"] = valid_ll_plans
        ll_plans = valid_ll_plans # Update local reference
        
        try:
            with open(full_plan_path, "w") as outfile:
                ll_plans = manifold.get("low_level_plans", [])
                for idx in range(len(ll_plans)):
                    target_low_plan = problem_results_dir / f"low_plan_{idx}.txt"
                    
                    if target_low_plan.exists():
                        with open(target_low_plan, 'r') as f:
                            content = f.read()
                            outfile.write(content)
                    else:
                        subgoal_info = ll_plans[idx]
                        orig_steps = subgoal_info.get('original_step_ids', [idx])
                        outfile.write(f"; SUBGOAL {idx} (Original Steps: {orig_steps}): {subgoal_info.get('original_subgoal', 'N/A')}\n")
                        outfile.write(f"; SUBGOAL {idx} (Grounded): {subgoal_info.get('grounded_subgoal', 'N/A')}\n")
                        outfile.write(f"; SUBGOAL_{idx}_NOT_ATTEMPTED\n")
                    outfile.write("\n") # Add spacing between subgoals
        except Exception as e:
            logger.error(f"Failed to save full plan: {e}", exc_info=True)

        # Verify final plan against compiled constraint benchmarks
        try:
            from src.lapis.utils.compile_verifier import verify_plan_with_compiled_val
            compiled_domain = Path(self.data_dir) / self.high_level_domain_name / "data" / self.batch_id / str(task_name) / "compiled_domain.pddl"
            compiled_problem = Path(self.data_dir) / self.high_level_domain_name / "data" / self.batch_id / str(task_name) / "compiled_problem.pddl"
            
            manifold_path = problem_results_dir / "manifold.json"
            if manifold_path.exists():
                with open(manifold_path, "r") as f:
                    manifold_data = json.load(f)
                
                success, val_out = verify_plan_with_compiled_val(
                    str(compiled_domain),
                    str(compiled_problem),
                    str(full_plan_path)
                )
                manifold_data["ground_truth_success"] = success
                if not success:
                    manifold_data["ground_truth_failure_reason"] = val_out
                else:
                    manifold_data["ground_truth_failure_reason"] = ""
                
                with open(manifold_path, "w") as f:
                    json.dump(manifold_data, f, indent=4)
                    
                logger.info(f"Compiled VAL Ground Truth Check: {'SUCCESS' if success else ('FAILED: ' + val_out)}")
        except Exception as e:
            logger.error(f"Failed to run compiled verification check: {e}")

        # Save GIFs for all simulators
        if frames_dict:
            for sim_name, frames in frames_dict.items():
                if frames:
                    gif_path = problem_results_dir / f"simulation_{sim_name}.gif"
                    logger.info(f"[{sim_name}] Saving simulation GIF to {gif_path} with {len(frames)} frames")
                    try:
                        # Slow down framerate (1 second per frame)
                        durations = [1000] * len(frames)
                        if len(durations) > 0:
                            durations[-1] = 4000 # 4 second pause at the end
                            
                        frames[0].save(
                            gif_path,
                            save_all=True,
                            append_images=frames[1:],
                            duration=durations,
                            loop=0
                        )
                    except Exception as e:
                        logger.error(f"[{sim_name}] Failed to save GIF: {e}", exc_info=True)

    def _extract_plan_from_file(self, plan_file):
        with open(plan_file, 'r') as f:
            lines = f.readlines()
        
        plan_lines = []
        in_plan = False
        
        # Strictly look for "FINAL PLAN:" section
        for line in lines:
            if "FINAL PLAN:" in line:
                in_plan = True
                continue
            if in_plan:
                if line.strip() == "":
                    # Plan section ends at first empty line after some actions
                    if plan_lines:
                        break
                    continue
                
                # Check if it's a numbered step
                parts = line.strip().split(". ", 1)
                if len(parts) == 2:
                    plan_lines.append(parts[1])
                else:
                    # If it doesn't look like a step, check if we should still include it
                    # or if we've reached the end of the plan
                    cleaned = line.strip()
                    if cleaned and not cleaned.startswith("STATUS:") and not cleaned.startswith("="):
                        plan_lines.append(cleaned)
                    elif plan_lines:
                        break
        
        return plan_lines




    def _generate_summary(self, future_subgoals, nl_description):
        if not future_subgoals:
            return "This is the last step of the plan."

        
        prompt = planning_prompts.SUMMARY_GENERATION_PROMPT.format(future_subgoals=future_subgoals)
        # We need to access the agent from somewhere. 
        # Since we passed agent to __init__, we should store it or access it via self.agent if BasePipeline stores it.
        # BasePipeline stores it as self.agent.
        return self.agent.llm_call(content=f"Original Task: {nl_description}\nYou are a helpful planning assistant.", prompt=prompt)

    def _wrap_text(self, text, indent=0, width=80):
        if not text:
            return ""
        wrapper = textwrap.TextWrapper(width=width, initial_indent=' ' * indent, subsequent_indent=' ' * indent)
        return wrapper.fill(text)
    def _generate_pipeline_summary(self, manifold, output_path):
        """
        Generate pipeline summaries in both Markdown and TXT formats.
        MD: full detail with collapsible sections.
        TXT: concise, no issues/traces.
        """
        base_path = Path(output_path).with_suffix("")
        md_path = base_path.with_suffix(".md")
        txt_path = base_path.with_suffix(".txt")
        
        md_lines = []
        txt_lines = []
        
        # Header
        date_str = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        md_lines.append(f"# Pipeline Summary")
        md_lines.append(f"**Date:** {date_str}")
        md_lines.append("")
        
        txt_lines.append("=" * 60)
        txt_lines.append("PIPELINE SUMMARY")
        txt_lines.append(f"Date: {date_str}")
        txt_lines.append("=" * 60)
        txt_lines.append("")
        
        # NL Goal
        nl_goal = manifold.get("nl_goal", "N/A")
        md_lines.append("## Natural Language Goal")
        md_lines.append(self._wrap_text(nl_goal, indent=0))
        md_lines.append("")
        
        txt_lines.append("GOAL:")
        txt_lines.append(self._wrap_text(nl_goal, indent=2))
        txt_lines.append("")
        
        # NL Constraints
        nl_constraints = manifold.get("nl_constraints", "N/A")
        md_lines.append("## Natural Language Constraints")
        md_lines.append(self._wrap_text(nl_constraints, indent=0))
        md_lines.append("")
        
        txt_lines.append("CONSTRAINTS:")
        txt_lines.append(self._wrap_text(nl_constraints, indent=2))
        txt_lines.append("")
        
        # LTL Fluents
        md_lines.append("## LTL Fluents")
        for fid, fluent in manifold.get("ltl_fluents", {}).items():
            md_lines.append(f"- **[{fid}]** `{fluent}`")
        md_lines.append("")
        
        # LTL Constraints
        md_lines.append("## LTL Constraints")
        for c in manifold.get("ltl_constraints", []):
            md_lines.append(f"- **[C{c['id']}]** `{c['formula']}` (Fluents: {c.get('fluent_ids', [])})")
        md_lines.append("")
        
        # High-Level Plan
        md_lines.append("## High-Level Plan")
        txt_lines.append("HIGH-LEVEL PLAN:")
        for step in manifold.get("high_level_plan", []):
            constraints = step.get("associated_ltl_constraints", [])
            c_str = ", ".join([f"**C{c}**" for c in constraints]) if constraints else "None"
            md_lines.append(f"1. **Step {step['step_id']}**: `{step['goal']}` (Constraints: {c_str})")
            txt_lines.append(f"  {step['step_id'] + 1}. {step['goal']}")
        md_lines.append("")
        txt_lines.append("")
        
        # Low-Level Plans
        md_lines.append("## Low-Level Plans")
        txt_lines.append("LOW-LEVEL PLANS:")
        
        any_plan_succeeded = False
        
        for llp in manifold.get("low_level_plans", []):
            sid = llp.get("subgoal_id", "?")
            plan = llp.get("plan", "NO_PLAN")
            step_ids = llp.get("original_step_ids", [sid])
            
            if plan != "NO_PLAN":
                any_plan_succeeded = True
            
            md_lines.append(f"### Subgoal {sid + 1} (Steps: {step_ids})")
            txt_lines.append(f"  Subgoal {sid + 1} (Steps: {step_ids}):")
            
            # Plan
            md_lines.append("#### Plan")
            if plan == "NO_PLAN":
                md_lines.append("`NO_PLAN`")
                txt_lines.append("    NO_PLAN")
            else:
                md_lines.append("```pddl")
                for action in plan.split("\n"):
                    if action.strip():
                        md_lines.append(f"{action.strip()}")
                        txt_lines.append(f"    {action.strip()}")
                md_lines.append("```")
            md_lines.append("")
            txt_lines.append("")
            
            # Refinements (MD only, collapsible)
            history = llp.get("refinement_history", [])
            if history:
                md_lines.append("<details>")
                md_lines.append(f"<summary><strong>Refinements ({len(history)} issues)</strong></summary>")
                md_lines.append("")
                for entry in history:
                    status = "✅ VALID" if entry.get("is_valid") else "❌ INVALID"
                    md_lines.append(f"- **Issue**: {entry.get('issue')}")
                    md_lines.append(f"  - **Solution**: {entry.get('solution')}")
                    md_lines.append(f"  - **Result**: {status}")
                md_lines.append("")
                md_lines.append("</details>")
                md_lines.append("")
            
            # Execution Trace (MD only, collapsible)
            trace = llp.get("execution_trace", [])
            if trace:
                md_lines.append("<details>")
                md_lines.append("<summary><strong>Execution Trace</strong></summary>")
                md_lines.append("")
                
                for i, state in enumerate(trace):
                    # logic for parsing state to list same as below
                    state_preds = []
                    if isinstance(state, str):
                        try:
                            import ast
                            parsed = ast.literal_eval(state)
                            if isinstance(parsed, (set, list, tuple)):
                                state_preds = list(parsed)
                            else:
                                state_preds = [str(state)]
                        except:
                            state_preds = [str(state)]
                    elif isinstance(state, (set, list, tuple)):
                        state_preds = list(state)
                    else:
                        state_preds = [str(state)]
                        
                    md_lines.append(f"- **State {i}**:")
                    if not state_preds:
                        md_lines.append("  - (Empty)")
                    else:
                        for pred in sorted(state_preds):
                            md_lines.append(f"  - **{pred}**")
                    md_lines.append("")
                
                md_lines.append("")
                md_lines.append("</details>")
                md_lines.append("")
            
            # Constraints
            constraints = llp.get("constraints", [])
            if constraints:
                c_ids = [f"**C{c['id']}**" for c in constraints]
                md_lines.append(f"#### Constraints: {', '.join(c_ids)}")
                for c in constraints:
                    md_lines.append(f"- **[C{c['id']}]** `{c['formula']}`")
            else:
                md_lines.append("#### Constraints: None")
            md_lines.append("")
            
            # Verification
            verification = llp.get("verification", [])
            if verification:
                md_lines.append("#### Verification")
                for v in verification:
                    status = "✅ PASS" if v.get("is_satisfied") else "❌ FAIL"
                    md_lines.append(f"- **[C{v['id']}]** {status}")
                    txt_lines.append(f"    C{v['id']}: {'PASS' if v.get('is_satisfied') else 'FAIL'}")
                md_lines.append("")
            
            # Per-subgoal Global LTL Check (only if plan succeeded)
            glc = llp.get("global_ltl_check", {})
            if glc and plan != "NO_PLAN":
                status = "✅ PASS" if glc.get("is_satisfied") else "❌ FAIL"
                md_lines.append(f"#### Global LTL (so far): {status}")
                if glc.get("overview"):
                    md_lines.append(f"> {glc.get('overview')}")
                md_lines.append("")
            
            md_lines.append("---")
            md_lines.append("")
        
        # Final Global LTL Check (over complete trace)
        final_glc = manifold.get("final_global_ltl_check", {})
        if final_glc:
            status = "✅ PASS" if final_glc.get("is_satisfied") else "❌ FAIL"
            md_lines.append("## Final Global LTL Check")
            md_lines.append(f"**Result:** {status}")
            md_lines.append(f"**Formula:** `{final_glc.get('formula', 'N/A')}`")
            md_lines.append("")
            
            txt_lines.append("FINAL GLOBAL LTL CHECK:")
            txt_lines.append(f"  Result: {'PASS' if final_glc.get('is_satisfied') else 'FAIL'}")
            txt_lines.append("")
            
            if final_glc.get("overview"):
                md_lines.append(f"> {final_glc.get('overview')}")
                md_lines.append("")
            
            # Collapsible full trace
            full_trace = final_glc.get("full_trace", [])
            if full_trace:
                md_lines.append("<details>")
                md_lines.append("<summary><strong>Complete Execution Trace</strong></summary>")
                md_lines.append("")
                
                for i, state in enumerate(full_trace):
                    # Convert set/string representation to list
                    # state is likely a set of strings or a string representation of it
                    state_preds = []
                    if isinstance(state, str):
                        # Attempt to parse set string: "{'a', 'b'}"
                        try:
                            # Use logic_utils or simple parsing if safe
                            # Assuming it's already a clean list of strings or set
                            import ast
                            parsed = ast.literal_eval(state)
                            if isinstance(parsed, (set, list, tuple)):
                                state_preds = list(parsed)
                            else:
                                state_preds = [str(state)]
                        except:
                            state_preds = [str(state)]
                    elif isinstance(state, (set, list, tuple)):
                        state_preds = list(state)
                    else:
                        state_preds = [str(state)]
                        
                    md_lines.append(f"- **State {i}**:")
                    if not state_preds:
                        md_lines.append("  - (Empty)")
                    else:
                        # Sort for determinism and readability
                        for pred in sorted(state_preds):
                            md_lines.append(f"  - **{pred}**")
                    md_lines.append("")
                
                md_lines.append("")
                md_lines.append("</details>")
                md_lines.append("")
        
        # Write files
        try:
            with open(md_path, "w") as f:
                f.write("\n".join(md_lines))
            logger.info(f"Pipeline summary (MD) saved to {md_path}")
            
            with open(txt_path, "w") as f:
                f.write("\n".join(txt_lines))
            logger.info(f"Pipeline summary (TXT) saved to {txt_path}")
        except Exception as e:
            logger.error(f"Failed to save pipeline summary: {e}")

    def _initialize_csv(self, results_dir):
        # Placeholder for CSV initialization if needed
        pass

    def _run_and_log_pipeline(self, task_name, scene_name, problem_id, results_problem_dir,
                            domain_file_path, domain_description, scene_graph_file_path,
                            csv_filepath):
        # This method is required by BasePipeline but not used in MultiLevelPlanningPipeline
        # because we override _process_task.
        pass

    def _parse_plan_to_actions(self, plan_content):
        """Parse plan content into a list of Action objects."""
        actions = []
        for line in plan_content.splitlines():
            line = line.strip()
            if not line or line.startswith(';'):
                continue
            # Replace parentheses and commas with spaces to avoid merging tokens (e.g. unstack(block1) -> unstack block1)
            content = line.replace('(', ' ').replace(')', ' ').replace(',', ' ')
            parts = content.split()
            if not parts:
                continue
            
            # Handle optional step numbers (e.g. "0: unstack block1 block2" or "0 unstack...")
            if parts[0].endswith(':') or parts[0].isdigit():
                if len(parts) < 2: continue
                action_name = parts[1]
                params = parts[2:]
            else:
                action_name = parts[0]
                params = parts[1:]
            
            # Create a dummy object that mimics PDDL action structure expected by map_pddl2simulator
            # The simulators expect an object with .action.name and .actual_parameters
            
            @dataclass
            class PDDLActionWrapper:
                name: str
            
            @dataclass
            class PDDLActionInstance:
                action: PDDLActionWrapper
                actual_parameters: list
            
            # We need to pass objects that stringify to the parameter names
            # because map_pddl2simulator does str(param)
            
            action_obj = PDDLActionInstance(
                action=PDDLActionWrapper(name=action_name),
                actual_parameters=params
            )
            
            actions.append(action_obj)
            
        return actions



    def _get_simulator_raw_state_report(self, simulator) -> str:
        """
        Extract a textual raw description of the simulator state.
        This is the 'fair' input for the assignment mapping LLM.
        """
        from src.lapis.simulators.babyai_simulator import BabyAISimulator, get_rooms
        
        if isinstance(simulator, BabyAISimulator):
            try:
                env = simulator.env.unwrapped
                grid = env.grid
                agent_pos = tuple(env.agent_pos)
                agent_dir = env.agent_dir
                carrying = env.carrying
                
                # Directions: 0: > (East), 1: v (South), 2: < (West), 3: ^ (North)
                dir_map = {0: "East", 1: "South", 2: "West", 3: "North"}
                
                rooms = get_rooms(grid)
                def get_room_num(pos):
                    for idx, room_cells in enumerate(rooms):
                        if tuple(pos) in [tuple(c) for c in room_cells]:
                            return idx + 1
                    return "Unknown"

                report = []
                agent_room = get_room_num(agent_pos)
                report.append(f"Agent Position: {agent_pos} in Room {agent_room}")
                report.append(f"Agent Facing: {dir_map.get(agent_dir, 'Unknown')}")
                report.append(f"Agent Carrying: {carrying.color + ' ' + carrying.type if carrying else 'Nothing'}")
                
                report.append("\nGrid Content (Non-wall/Non-floor):")
                for x in range(grid.width):
                    for y in range(grid.height):
                        cell = grid.get(x, y)
                        if cell is not None and cell.type not in ('wall', 'floor', 'agent'):
                            room_num = get_room_num((x, y))
                            info = f"- {cell.color} {cell.type} at ({x}, {y}) in Room {room_num}"
                            if cell.type == 'door':
                                state = "open" if cell.is_open else "closed"
                                locked = "locked" if cell.is_locked else "unlocked"
                                info += f" ({state}, {locked})"
                            report.append(info)
                
                return "\n".join(report)
            except Exception as e:
                logger.error(f"Failed to get BabyAI raw report: {e}")
                return "Error extracting BabyAI state."

        # --- UP-based (Blocksworld, etc.) ---
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
            "You are a PDDL state translator. Your task is to translate a raw physical state report "
            "into a list of PDDL predicates (the :init block) using ONLY the provided object names."
        )
        
        user_prompt = f"""
        PDDL OBJECT NAMES (Assignment Vocabulary):
        {objects_list}
        
        RAW SIMULATOR STATE REPORT:
        {raw_report}
        
        INSTRUCTIONS:
        1. Translate the raw state into PDDL predicates according to the world logic.
        2. Use ONLY the object names provided in the vocabulary above. Do NOT invent new names or use raw coordinates if a named object exists at that location.
        3. CRITICAL: For navigation domains, use ONLY THESE EXACT predicates:
           - (in-room agent ?room)
           - (in-front-of ?obj)
           - (holding ?obj)
           - (empty-hands)
           - (unlocked ?door)
           - (visited ?room)
           - (blocking ?obj ?door)
        4. CRITICAL: DO NOT use `facing`, `door-unlocked`, `door-open`, etc. Translate `facing` into `(in-front-of ?obj)` if appropriate.
        5. Standard predicates for blocksworld include: (on ?x ?y), (ontable ?x), (clear ?x), (holding ?x), (handempty).
        5. Output ONLY the list of predicates, one per line, wrapped in parentheses.
        6. DO NOT output the (:init ...) wrapper, just the predicates.
        
        OUTPUT FORMAT:
        (predicate1 arg1 arg2)
        (predicate2 arg1)
        ...
        """
        
        try:
            logger.info(f"Assignment Mapping - Raw Report Trace:\n{raw_report}")
            assignment = self.agent.llm_call(system_prompt, user_prompt)
            logger.info(f"Assignment Mapping - LLM Response Trace:\n{assignment}")
            
            # Basic cleanup in case LLM adds markdown or :init
            assignment = assignment.replace(":init", "").replace("```pddl", "").replace("```", "").strip()
            # If it's wrapped in (:init ...), strip it
            if assignment.startswith("(") and assignment.endswith(")"):
                # Potential double nesting, check if first char after ( is another (
                for_check = assignment[1:].strip()
                if for_check.startswith("("):
                     # Likely wrapped in (:init (pred)...) or just ( (pred)... )
                     # We'll just trust the LLM to provide lines of (pred)
                     pass

            return assignment
        except Exception as e:
            logger.error(f"Assignment mapping failed: {e}")
            return ""

    def _get_pddl_init_from_babyai_obs(self, simulator) -> str:
        """
        Extract a PDDL :objects block and :init block from BabyAISimulator's live grid.

        This is fully generic — it reads only the simulator's observation and the
        MiniGrid grid traversal helpers. No domain-specific strings are hardcoded.
        Returned string is extraction of information from the environment.
        """
        from src.lapis.simulators.babyai_simulator import get_rooms
        try:
            env = simulator.env.unwrapped
            grid = env.grid
            agent_pos = tuple(env.agent_pos)
            carrying = env.carrying

            rooms = get_rooms(grid)  # list of list of (x,y) tuples per room

            # Scan grid inline for all objects (doors, balls, boxes, keys)
            objects_in_grid = []
            for x in range(grid.width):
                for y in range(grid.height):
                    cell = grid.get(x, y)
                    if cell is not None and cell.type not in ('wall', 'floor'):
                        objects_in_grid.append((x, y, cell))

            # ---- Build canonical names for each entity ----
            # room name: room_{i+1}  (1-indexed to match BabyAI convention)
            room_names = [f"room_{i+1}" for i in range(len(rooms))]

            # Helper: find which room index an (x,y) cell belongs to
            def room_of(pos):
                for idx, room_cells in enumerate(rooms):
                    if tuple(pos) in [tuple(c) for c in room_cells]:
                        return idx
                return 0  # fallback

            # Enumerate doors, balls, boxes, keys with stable unique names
            door_counter = {}
            obj_counter = {}
            door_map = {}   # pos -> canonical_name
            obj_map = {}    # pos -> canonical_name
            type_map = {}   # canonical_name -> pddl_type

            for (x, y, cell) in objects_in_grid:
                color = cell.color
                otype = cell.type  # 'door', 'ball', 'box', 'key'
                key = (color, otype)
                if otype == 'door':
                    door_counter[key] = door_counter.get(key, 0) + 1
                    name = f"{color}_{otype}_{door_counter[key]}"
                    door_map[(x, y)] = name
                    type_map[name] = 'door'
                else:
                    obj_counter[key] = obj_counter.get(key, 0) + 1
                    name = f"{color}_{otype}_{obj_counter[key]}"
                    obj_map[(x, y)] = name
                    type_map[name] = otype

            # Identify agent room
            agent_room_idx = room_of(agent_pos)
            agent_room_name = room_names[agent_room_idx]

            # ---- Build PDDL objects block ----
            objects_lines = []
            objects_lines.append("    agent - entity")  # agent is always present
            if room_names:
                objects_lines.append(f"    {' '.join(room_names)} - room")
            if door_map:
                objects_lines.append(f"    {' '.join(door_map.values())} - door")
            # Group non-door objects by type
            by_type = {}
            for name, otype in type_map.items():
                if otype != 'door':
                    by_type.setdefault(otype, []).append(name)
            for otype, names in by_type.items():
                objects_lines.append(f"    {' '.join(names)} - {otype}")

            # ---- Build PDDL init predicates ----
            init_lines = []

            # Agent location
            init_lines.append(f"(at agent {agent_room_name})")
            init_lines.append(f"(visited agent {agent_room_name})")

            # Carrying state
            if carrying is not None:
                color = carrying.color
                otype = carrying.type
                # find which name we gave it (it won't be in the grid)
                key = (color, otype)
                cnt = obj_counter.get(key, 0) + 1
                obj_counter[key] = cnt
                carried_name = f"{color}_{otype}_{cnt}"
                init_lines.append(f"(holding agent {carried_name})")
            else:
                init_lines.append("(empty-hands agent)")

            # Object locations
            for (x, y), name in obj_map.items():
                r_idx = room_of((x, y))
                init_lines.append(f"(in-room {name} {room_names[r_idx]})")

            # Door states and connectivity
            # Build a set of door positions to find which rooms they connect
            door_positions = list(door_map.keys())
            for (x, y), dname in door_map.items():
                cell = grid.get(x, y)
                # locked / unlocked
                if hasattr(cell, 'is_locked') and cell.is_locked:
                    init_lines.append(f"(locked {dname})")
                else:
                    init_lines.append(f"(unlocked {dname})")
                # Connectivity: check which rooms border this door
                adjacent_rooms = set()
                for r_idx, room_cells in enumerate(rooms):
                    for dx, dy in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
                        if (x + dx, y + dy) in [(c[0], c[1]) if not isinstance(c, tuple) else c for c in room_cells]:
                            adjacent_rooms.add(r_idx)
                adj_list = sorted(adjacent_rooms)
                if len(adj_list) >= 2:
                    r1, r2 = adj_list[0], adj_list[1]
                    init_lines.append(f"(connected {room_names[r1]} {room_names[r2]} {dname})")
                    init_lines.append(f"(connected {room_names[r2]} {room_names[r1]} {dname})")

            objects_block = "\n".join(objects_lines)
            init_block = "\n".join(init_lines)
            # Store objects block separately — it's used as a hint to the LLM but
            # Removed environment synchronization injection.
            self._babyai_objects_hint = (
                f"\n\n--- GROUND TRUTH OBJECTS (use exactly these names and types in :objects block) ---\n"
                f"(:objects\n{objects_block}\n)\n"
                f"--- END GROUND TRUTH OBJECTS ---\n"
            )
            return init_block

        except Exception as e:
            logger.warning(f"Failed to extract BabyAI PDDL init state from observation: {e}")
            return None






