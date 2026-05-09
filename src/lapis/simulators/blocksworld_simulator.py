"""Blocksworld simulator implementation using unified_planning."""

from typing import List, Tuple, Dict, Any, Optional
import unified_planning
from unified_planning.shortcuts import SequentialSimulator
from unified_planning.model import Problem
from unified_planning.io import PDDLReader
import logging
from pathlib import Path

from .base_simulator import BaseSimulator
# Adjust import path based on project structure
# Assuming domains is importable from root or we need to add it to path
import sys
import os

# Add project root to path to ensure domains can be imported
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../.."))
if project_root not in sys.path:
    sys.path.append(project_root)

try:
    from third_party.lexicon_neurips.domains.blocksworld.up_domain import get_blocksworld_problem
except ImportError:
    # Try alternative path if third_party is not a package
    try:
        sys.path.append(os.path.join(project_root, "third-party/lexicon_neurips"))
        from domains.blocksworld.up_domain import get_blocksworld_problem
    except ImportError:
         print("Warning: Could not import get_blocksworld_problem. BlocksworldSimulator will not work.")
         get_blocksworld_problem = None


from .scenario import Scenario

class BlocksworldScenario(Scenario):
    """Scenario for Blocksworld domain."""
    
    def compute_cost(self, action: Any, state: Any = None) -> float:
        """Compute cost for Blocksworld actions.
        
        All actions in Blocksworld have unitary cost.
        """
        return 1.0


class BlocksworldSimulator(BaseSimulator):
    """Blocksworld simulator using unified_planning SequentialSimulator.
    
    This simulator generates Blocksworld problems and simulates them using
    the PDDL-based definition.
    """

    def __init__(self, env_name: str = "blocksworld", seed: Optional[int] = None, scenario: Optional[Scenario] = None):
        """Initialize Blocksworld simulator.
        
        Args:
            env_name: Name of the environment
            seed: Random seed
            scenario: The scenario associated with this simulator
        """
        self.seed = seed
        self.problem = None
        self.simulator = None
        self.current_state = None
        
        if scenario is None:
            scenario = BlocksworldScenario()
            
        # BaseSimulator expects an env, but we don't have a gym env here.
        # We pass None and handle it.
        super().__init__(None, scenario=scenario)

    def setup(self, domain_path: Path, problem_path: Path, **kwargs) -> bool:
        """Setup the simulator from PDDL files.
        
        Args:
            domain_path: Path to the domain PDDL file
            problem_path: Path to the problem PDDL file
            
        Returns:
            bool: True if setup successful
        """
        logger = logging.getLogger(__name__)
        try:
            domain_path = Path(domain_path)
            problem_path = Path(problem_path)
            
            if not domain_path.exists():
                logger.error(f"Domain path {domain_path} does not exist.")
                return False
            if not problem_path.exists():
                logger.error(f"Problem path {problem_path} does not exist.")
                return False
            
            reader = PDDLReader()
            try:
                # Parse problem
                problem = reader.parse_problem(str(domain_path), str(problem_path))
                
                # Create a clean problem for simulation (remove constraints/metrics)
                clean_problem = Problem(problem.name)
                for f in problem.fluents:
                    clean_problem.add_fluent(f)
                clean_problem.add_objects(problem.all_objects)
                clean_problem.add_actions(problem.actions)
                for f, v in problem.initial_values.items():
                    clean_problem.set_initial_value(f, v)
                for goal in problem.goals:
                    clean_problem.add_goal(goal)
                
                self.problem = clean_problem
                self.simulator = SequentialSimulator(clean_problem)
                self.current_state = self.simulator.get_initial_state()
                logger.info(f"Loaded and cleaned PDDL problem for simulation from {problem_path} (Domain: {domain_path})")
                return True
            except Exception as e:
                logger.error(f"Failed to parse PDDL for simulation: {e}", exc_info=True)
                return False
        except Exception as e:
            logger.error(f"Failed to setup blocksworld simulator: {e}", exc_info=True)
            return False

    def reset(self, seed: Optional[int] = None, **kwargs) -> Tuple[Dict, Dict]:
        """Reset the simulator to an initial state.
        
        Args:
            seed: Random seed
            **kwargs: Additional parameters (e.g. num_blocks)
            
        Returns:
            observation: Initial observation (current state fluents)
            info: Additional info
        """
        if seed is not None:
            self.seed = seed
            
        num_blocks = kwargs.get("num_blocks", 5)
        
        if get_blocksworld_problem is None:
            raise ImportError("Blocksworld domain logic not found.")
            
        self.problem = get_blocksworld_problem(num_blocks=num_blocks, seed=self.seed)
        self.simulator = SequentialSimulator(self.problem)
        self.current_state = self.simulator.get_initial_state()
        
        return self.observation(self.current_state), {"problem": self.problem}

    def map_pddl2simulator(self, action: Any) -> Any:
        """Map PDDL action to simulator action.
        
        Since we use UP simulator, the PDDL action (UP ActionInstance) 
        is already in the correct format or needs to be converted to one.
        
        Args:
            action: PDDL action object
            
        Returns:
            Simulator-specific action
        """
        # If action is already a UP ActionInstance, return it.
        # If it's a plan item wrapper, extract the action.
        # SequentialSimulator.apply expects a UP ActionInstance (plan.actions elements)
        return action

    def map_plan2simulator(self, plan: List[Any]) -> List[Any]:
        """Map PDDL plan to simulator actions.
        
        Args:
            plan: List of PDDL actions
            
        Returns:
            List of simulator actions
        """
        if hasattr(plan, "actions"):
            return plan.actions
        return plan

    def step(self, action: Any) -> Tuple[Dict, float, bool, bool, Dict]:
        """Execute one step in the simulator.
        
        Args:
            action: Action to execute
            
        Returns:
            observation: New observation
            reward: Reward (1 if goal reached, else 0)
            terminated: Whether goal is reached
            truncated: False
            info: Additional info
        """
        if self.simulator is None:
            raise RuntimeError("Simulator not initialized. Call reset() first.")

        #print(self.current_state)

        if not self.simulator.is_applicable(self.current_state, action):
            # Action not applicable
            # We can raise error or return same state with penalty
            # For now, let's raise error to match strict simulation
            raise ValueError(f"Action {action} is not applicable in current state.")

        self.current_state = self.simulator.apply(self.current_state, action)
        
        # Check goal
        is_goal = self.simulator.is_goal(self.current_state)
        reward = 1.0 if is_goal else 0.0
        
        return self.observation(self.current_state), reward, is_goal, False, {}

    def observation(self, state: Any) -> Dict:
        """Process state into observation.
        
        Args:
            state: UP State
            
        Returns:
            Dictionary of fluents
        """
        # Convert UP state to dictionary representation
        # state.values is a dictionary of {Fluent: Value}
        # We'll convert it to a string representation or keep as is
        return {"state": state}

    def render(self):
        """Render the current state in ASCII."""
        if self.current_state is None:
            print("No state to render.")
            return

        # Extract state info
        # We assume standard blocksworld predicates: on(x, y), ontable(x), clear(x), holding(x), handempty()
        
        state = self.current_state
        # We need to access the problem to get objects
        objects = self.problem.all_objects
        blocks = [o for o in objects if o.type.name == 'block'] # Assuming type name is 'block' or similar
        
        # If types are not explicitly 'block', we might need to check all objects
        if not blocks:
            blocks = objects
            
        # Build stacks
        # on(x, y) -> x is on y
        # ontable(x) -> x is on table
        # holding(x) -> holding x
        
        # Helper to get fluent value
        def get_fluent(name, *args):
            # This depends on how UP state exposes values. 
            # state.get_value(fluent, args)
            try:
                fluent = self.problem.fluent(name)
                val = state.get_value(fluent(*args))
                return val.constant_value() if val.is_bool_constant() else val
            except Exception:
                return False

        holding = None
        stacks = []
        
        # Find what is holding
        for b in blocks:
            if get_fluent("holding", b):
                holding = b
                break
                
        # Find stacks (bottom up)
        # First find all blocks on table
        on_table_blocks = []
        has_ontable = False
        try:
            self.problem.fluent("ontable")
            has_ontable = True
        except Exception:
            pass

        for b in blocks:
            if has_ontable:
                if get_fluent("ontable", b):
                    on_table_blocks.append(b)
            else:
                is_on = any(get_fluent("on", b, other) for other in blocks)
                is_held = get_fluent("holding", b)
                if not is_on and not is_held:
                    on_table_blocks.append(b)
                
        for base in on_table_blocks:
            stack = [base]
            current = base
            # Find what is on current
            while True:
                found_next = False
                for b in blocks:
                    if get_fluent("on", b, current):
                        stack.append(b)
                        current = b
                        found_next = True
                        break
                if not found_next:
                    break
            stacks.append(stack)
            
        print("\n--- Blocksworld State ---")
        if holding:
            print(f"Holding: [{holding.name}]")
        else:
            print("Hand Empty")
            
        print("Table:")
        if not stacks:
            print("  (Empty)")
        else:
            # Find max height
            max_height = max(len(s) for s in stacks) if stacks else 0
            for h in range(max_height - 1, -1, -1):
                row = ""
                for stack in stacks:
                    if h < len(stack):
                        row += f"[{stack[h].name}] "
                    else:
                        row += "    "
                print(f"  {row}")
        print("-------------------------\n")

    def get_image(self, action_text=None, all_subgoals=None, active_subgoal_indices=None, all_constraints=None):
        """Render the current state to a PIL Image."""
        try:
            from PIL import Image, ImageDraw, ImageFont
        except ImportError:
            print("PIL not installed, cannot render image.")
            return None

        # Dimensions: 600x800
        width = 600
        render_height = 400
        text_area_height = 400
        height = render_height + text_area_height
        
        img = Image.new('RGB', (width, height), color = (255, 255, 255))
        d = ImageDraw.Draw(img)
        
        if self.current_state is None:
            d.text((10, 10), "No state", fill=(0,0,0))
            return img

        state = self.current_state
        objects = self.problem.all_objects
        blocks = [o for o in objects if o.type.name == 'block']
        if not blocks:
            # Fallback if type name is different
            blocks = objects

        # Helper to get fluent value
        def get_fluent(name, *args):
            try:
                fluent = self.problem.fluent(name)
                val = state.get_value(fluent(*args))
                return val.constant_value() if val.is_bool_constant() else val
            except Exception:
                return False

        # Build stacks
        blocks = sorted(list(set(blocks)), key=lambda x: x.name)
        num_blocks = len(blocks)

        on_table_blocks = []
        has_ontable = False
        try:
            self.problem.fluent("ontable")
            has_ontable = True
        except Exception:
            pass

        for b in blocks:
            if has_ontable:
                if get_fluent("ontable", b):
                    on_table_blocks.append(b)
            else:
                is_on = any(get_fluent("on", b, other) for other in blocks)
                is_held = get_fluent("holding", b)
                if not is_on and not is_held:
                    on_table_blocks.append(b)
                
        stacks = []
        for base in on_table_blocks:
            stack = [base]
            current = base
            while True:
                found_next = False
                for b in blocks:
                    if get_fluent("on", b, current):
                        stack.append(b)
                        current = b
                        found_next = True
                        break
                if not found_next:
                    break
            stacks.append(stack)
            
        # Dynamic scaling based on number of blocks
        margin = 30
        available_width = width - 2 * margin
        
        if num_blocks > 0:
            block_width = available_width / (1.2 * num_blocks - 0.2) if num_blocks > 1 else 80
            block_width = min(80, block_width) # Cap at 80
            spacing = 0.2 * block_width
        else:
            block_width = 80
            spacing = 40
            
        block_height = min(60, 300 / (num_blocks + 1)) if num_blocks > 0 else 60
        block_height = max(30, block_height)
        
        start_x = margin
        base_y = render_height - 30
        
        # Color map based on name (simple hash)
        def get_color(name):
            if "red" in name.lower(): return (255, 100, 100)
            if "blue" in name.lower(): return (100, 100, 255)
            if "green" in name.lower(): return (100, 255, 100)
            if "orange" in name.lower(): return (255, 165, 0)
            if "yellow" in name.lower(): return (255, 255, 100)
            if "brown" in name.lower(): return (165, 42, 42)
            if "black" in name.lower(): return (80, 80, 80)
            if "white" in name.lower(): return (240, 240, 240)
            # Default hash color
            h = hash(name)
            return ((h & 0xFF), ((h >> 8) & 0xFF), ((h >> 16) & 0xFF))

        # Try to load fonts
        try:
            # Common paths for bold fonts on Linux
            font_bold_path = "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"
            if not os.path.exists(font_bold_path):
                font_bold_path = "/usr/share/fonts/TTF/DejaVuSans-Bold.ttf"
            
            font_bold = ImageFont.truetype(font_bold_path, 14) if os.path.exists(font_bold_path) else ImageFont.load_default()
            font_reg = ImageFont.load_default()
        except:
            font_bold = ImageFont.load_default()
            font_reg = ImageFont.load_default()

        for i, stack in enumerate(stacks):
            x = start_x + i * (block_width + spacing)
            for j, block in enumerate(stack):
                y = base_y - (j + 1) * block_height
                color = get_color(block.name)
                # Adjust text color for contrast
                text_color = (255, 255, 255) if sum(color) < 380 else (0, 0, 0)
                
                d.rectangle([x, y, x + block_width, y + block_height], outline=(0,0,0), fill=color)
                
                # Draw block name
                name_to_draw = block.name
                if len(name_to_draw) > block_width // 8:
                    name_to_draw = name_to_draw[:max(1, int(block_width // 8))]
                
                d.text((x + 5, y + 5), name_to_draw, fill=text_color)

        # Draw holding
        holding = None
        for b in blocks:
            if get_fluent("holding", b):
                holding = b
                break
                
        # FIXED POSITION for Hand state to avoid overlap
        hand_state_x = width - 150
        hand_state_y = 20
        
        if holding:
            box_w, box_h = block_width, block_height
            d.text((hand_state_x, hand_state_y), "Holding:", fill=(0,0,0), font=font_bold)
            
            # Draw held block below label
            x_h = hand_state_x + 10
            y_h = hand_state_y + 25
            color = get_color(holding.name)
            text_color = (255, 255, 255) if sum(color) < 380 else (0, 0, 0)
            d.rectangle([x_h, y_h, x_h + box_w, y_h + box_h], outline=(0,0,0), fill=color)
            
            name_to_draw = holding.name
            if len(name_to_draw) > box_w // 8:
                name_to_draw = name_to_draw[:max(1, int(box_w // 8))]
            d.text((x_h + 5, y_h + 5), name_to_draw, fill=text_color)
        else:
            d.text((hand_state_x, hand_state_y), "Hand Empty", fill=(0,0,0), font=font_bold)
            
        # Draw separator line
        d.line([(0, render_height), (width, render_height)], fill=(0, 0, 0), width=3)

        # Draw goal and constraint text at the bottom
        import textwrap
        y_text = render_height + 15
        
        # Action text
        if action_text:
            d.text((20, y_text), "Current Action:", fill=(0,0,0), font=font_bold)
            d.text((150, y_text), action_text, fill=(0,0,0))
            y_text += 30

        # Goals Section
        d.text((20, y_text), "GOALS", fill=(0,0,0), font=font_bold)
        y_text += 25
        
        if all_subgoals:
            for idx, sg in enumerate(all_subgoals):
                is_active = active_subgoal_indices and idx in active_subgoal_indices
                color = (255, 0, 0) if is_active else (100, 100, 100)
                font = font_bold if is_active else font_reg
                
                prefix = "> " if is_active else "  "
                wrapped_sg = textwrap.fill(f"{prefix}{sg}", width=80)
                for line in wrapped_sg.split('\n'):
                    d.text((30, y_text), line, fill=color, font=font)
                    y_text += 18
            y_text += 10
        elif action_text == None and not all_subgoals: # Fallback for initial state if not passed correctly
             d.text((30, y_text), "No subgoals provided", fill=(100,100,100))
             y_text += 20

        # Constraints Section
        y_text = max(y_text, render_height + 250) # Ensure some space
        d.text((20, y_text), "CONSTRAINTS", fill=(0,0,0), font=font_bold)
        y_text += 25
        
        if all_constraints:
            if isinstance(all_constraints, str):
                all_constraints = [all_constraints]
            
            for c in all_constraints:
                wrapped_c = textwrap.fill(f"• {c}", width=80)
                for line in wrapped_c.split('\n'):
                    d.text((30, y_text), line, fill=(0, 0, 255))
                    y_text += 18
        else:
             d.text((30, y_text), "None", fill=(100,100,100))
            
        return img
