from abc import ABC, abstractmethod
from typing import Any, Optional

class Scenario(ABC):
    """Base class for scenarios.
    
    A Scenario defines the context and rules for a simulation, including
    cost computation for actions.
    """
    
    def __init__(self):
        self.simulator = None

    def set_simulator(self, simulator):
        """Link a simulator to this scenario."""
        self.simulator = simulator

    @abstractmethod
    def compute_cost(self, action: Any, state: Any = None) -> float:
        """Compute the cost of an action in a given state.
        
        Args:
            action: The action to compute cost for (PDDL action).
            state: The current state (optional, depending on domain).
            
        Returns:
            float: The cost of the action.
        """
        pass
