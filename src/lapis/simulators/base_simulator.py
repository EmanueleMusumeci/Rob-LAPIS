"""Base simulator interface for all domain simulators."""

from abc import ABC, abstractmethod
from typing import List, Tuple, Any, Optional, Dict, TYPE_CHECKING

if TYPE_CHECKING:
    from .scenario import Scenario


class BaseSimulator(ABC):
    """Abstract base class for domain simulators.
    
    All simulators should inherit from this class and implement the required methods.
    This provides a consistent interface for integrating different simulators with the
    planning framework.
    """

    def __init__(self, env, scenario: Optional['Scenario'] = None):
        """Initialize the simulator with an environment.
        
        Args:
            env: The underlying environment (e.g., gym environment)
            scenario: The scenario associated with this simulator
        """
        self.env = env
        self.scenario = scenario
        if self.scenario:
            self.scenario.set_simulator(self)
        self.prev_obs = None

    @abstractmethod
    def reset(self, seed: Optional[int] = None, **kwargs) -> Tuple[Dict, Dict]:
        """Reset the simulator to an initial state.
        
        Args:
            seed: Random seed for reproducibility
            **kwargs: Additional environment-specific parameters
            
        Returns:
            observation: Initial observation
            info: Additional information dictionary
        """
        pass

    @abstractmethod
    def setup(self, **kwargs) -> bool:
        """Setup the simulator with necessary data (e.g. loading domains/problems).

        Args:
            **kwargs: Setup parameters (e.g. domain_path, problem_path)

        Returns:
            bool: True if setup successful, False otherwise
        """
        pass

    @abstractmethod
    def map_pddl2simulator(self, action: Any) -> Any:
        """Map a PDDL action to simulator-specific action format.
        
        Args:
            action: PDDL action object
            
        Returns:
            Simulator-specific action representation
        """
        pass

    @abstractmethod
    def map_plan2simulator(self, plan: List[Any]) -> List[Any]:
        """Map a PDDL plan (list of actions) to simulator format.
        
        Args:
            plan: List of PDDL actions
            
        Returns:
            List of simulator-specific actions
        """
        pass

    @abstractmethod
    def step(self, action: Any) -> Tuple[Dict, float, bool, bool, Dict]:
        """Execute one step in the simulator.
        
        Args:
            action: Action to execute (in simulator format)
            
        Returns:
            observation: New observation after action
            reward: Reward received
            terminated: Whether episode ended
            truncated: Whether episode was truncated
            info: Additional information
        """
        pass

    @abstractmethod
    def observation(self, obs: Any) -> Dict:
        """Process and format observation from environment.
        
        Args:
            obs: Raw observation from environment
            
        Returns:
            Formatted observation dictionary
        """
        pass

    def is_feasible_plan(self, plan: List[Any]) -> bool:
        """Check if a plan is feasible by simulating it.
        
        Args:
            plan: List of PDDL actions
            
        Returns:
            True if plan executes successfully, False otherwise
        """
        simulator_plan = self.map_plan2simulator(plan)
        try:
            for action in simulator_plan:
                self.step(action)
            return True
        except Exception as e:
            print(f"Plan infeasible: {e}")
            return False

    def is_feasible_action(self, action: Any) -> bool:
        """Check if a single action is feasible.
        
        Args:
            action: PDDL action
            
        Returns:
            True if action executes successfully, False otherwise
        """
        simulator_action = self.map_pddl2simulator(action)
        try:
            self.step(simulator_action)
            return True
        except Exception as e:
            print(f"Action infeasible: {e}")
            return False

    def verify_plan(self, plan: Any, problem: Any, optimal_plan: Optional[Any] = None) -> Dict[str, Any]:
        """Verify a PDDL plan by simulating it step-by-step.
        
        Args:
            plan: The PDDL plan to verify
            problem: The PDDL problem instance
            optimal_plan: Optional optimal plan for optimality comparison
            
        Returns:
            Dictionary containing verification results:
            - status: "OPTIMAL", "SUBOPTIMAL", or "INVALID"
            - failure_reason: Description of failure if invalid
            - unsatisfied_goals: List of unsatisfied goals if any
            - unsatisfied_conditions: List of unsatisfied preconditions if any
            - cost: Plan length/cost
            - optimal_cost: Optimal plan cost (if provided)
        """
        from unified_planning.shortcuts import SequentialSimulator
        
        result = {
            "status": "INVALID",
            "failure_reason": None,
            "unsatisfied_goals": [],
            "unsatisfied_conditions": [],
            "cost": len(plan.actions) if plan else 0,
            "optimal_cost": len(optimal_plan.actions) if optimal_plan else None
        }

        if not plan:
            result["failure_reason"] = "Empty plan"
            return result

        simulator = SequentialSimulator(problem)
        state = simulator.get_initial_state()
        
        # Step 1: Simulate plan actions
        for action in plan.actions:
            state, unsatisfied_conditions = self.step_and_verify(
                action, state, problem, simulator
            )
            
            if unsatisfied_conditions:
                result["status"] = "INVALID"
                result["unsatisfied_conditions"] = unsatisfied_conditions
                result["failure_reason"] = f"Action {action.action.name} failed: {unsatisfied_conditions}"
                return result

        # Step 2: Check goal satisfaction
        if state:
            unsatisfied_goals = simulator.get_unsatisfied_goals(state)
            if len(unsatisfied_goals) > 0:
                result["status"] = "INVALID"
                result["unsatisfied_goals"] = unsatisfied_goals
                result["failure_reason"] = f"Goals not reached: {unsatisfied_goals}"
                return result
            else:
                result["status"] = "VALID"

        # Step 3: Check optimality if valid and optimal plan provided
        if result["status"] == "VALID" and optimal_plan:
            if self.is_optimal_plan(plan, len(optimal_plan.actions)):
                result["status"] = "OPTIMAL"
            else:
                result["status"] = "SUBOPTIMAL"
                
        return result

    def step_and_verify(self, action: Any, state: Any, problem: Any, simulator: Any) -> Tuple[Any, Optional[List]]:
        """Execute one step in the simulator with verification.
        
        Args:
            action: Action to execute
            state: Current state
            problem: PDDL problem
            simulator: UP simulator instance
            
        Returns:
            Tuple of (new_state, unsatisfied_conditions)
            new_state is None if action fails
            unsatisfied_conditions is None if action succeeds
        """
        if simulator.is_applicable(state, action):
            new_state = simulator.apply(state, action)
            return new_state, None
        else:
            try:
                unsatisfied_conditions = simulator.get_unsatisfied_conditions(
                    state, action
                )
            except Exception as e:
                return None, [f"exception_during_simulation: {str(e)}"]
            return None, unsatisfied_conditions

    def is_optimal_plan(self, plan: Any, optimal_cost: int) -> bool:
        """Check if a plan is optimal based on cost.
        
        Args:
            plan: The plan to check
            optimal_cost: The known optimal cost
            
        Returns:
            True if plan cost <= optimal_cost
        """
        if not plan:
            return False
        return len(plan.actions) <= optimal_cost
