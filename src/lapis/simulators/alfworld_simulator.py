"""AlfWorld simulator implementation.

This simulator integrates the AlfWorld environment for household task planning.
AlfWorld is based on the ALFRED benchmark and provides embodied AI tasks in
simulated household environments.

Required packages:
    - alfworld: Install with `pip install alfworld`
    
Usage:
    To use this simulator, you need to have AlfWorld installed and configured.
    See: https://github.com/alfworld/alfworld
"""

import os
import re
from typing import List, Tuple, Dict, Any, Optional
import gymnasium as gym

from .base_simulator import BaseSimulator
from .utils import InfeasiblePlan


class AlfWorldSimulator(BaseSimulator):
    """AlfWorld simulator with PDDL action mapping.
    
    This simulator wraps an AlfWorld environment and provides PDDL action
    translation and execution for household tasks.
    
    Actions in AlfWorld include:
        - goto <location>: Navigate to a location
        - open/close <receptacle>: Manipulate containers
        - take/put <object>: Pick up or place objects
        - clean/heat/cool <object>: Manipulate object states
        - slice <object>: Cut objects
        - toggle <object>: Turn lights on/off
        - inventory: Check inventory
        - look: Observe surroundings
        - examine <object/receptacle>: Get detailed info
    """

    def __init__(self, env_name: str = "AlfWorld-v2", **env_kwargs):
        """Initialize AlfWorld simulator.
        
        Args:
            env_name: Name of the AlfWorld environment
            **env_kwargs: Additional arguments for environment creation
        """
        try:
            # Import AlfredTWEnv from the submodule
            import yaml
            from alfworld.agents.environment.alfred_tw_env import AlfredTWEnv
        except ImportError as e:
            raise ImportError(f"Could not import alfworld. Make sure the submodule is initialized and dependencies are installed. Error: {e}")

        # Load default config
        project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../.."))
        config_path = os.path.join(project_root, "third-party/alfworld/configs/base_config.yaml")
        
        if not os.path.exists(config_path):
             raise FileNotFoundError(f"AlfWorld config not found at {config_path}")

        with open(config_path) as f:
            config = yaml.safe_load(f)

        alfred_env = AlfredTWEnv(config, **env_kwargs)
        self.env = alfred_env.init_env(batch_size=1)
        super().__init__(self.env)
        
        self.current_obs = None
        self.inventory = []
        self.location = None

    def setup(self, **kwargs) -> bool:
        """Setup the simulator.
        
        AlfWorld simulator is set up during initialization.
        """
        return True

    def reset(self, seed: Optional[int] = None, **kwargs) -> Tuple[Dict, Dict]:
        """Reset the AlfWorld environment.
        
        Args:
            seed: Random seed for reproducibility
            **kwargs: Additional reset parameters
            
        Returns:
            observation: Initial observation with 'feedback' text
            info: Additional information dictionary
        """
        if seed is not None:
            self.env.seed(seed)
        
        obs, info = self.env.reset()
        self.current_obs = obs
        self.inventory = []
        self.location = self._parse_location(obs)
        
        return {"observation": obs, "location": self.location, "inventory": self.inventory}, info

    def map_pddl2simulator(self, action: Any) -> str:
        """Map PDDL action to AlfWorld text command.
        
        Args:
            action: PDDL action object with action.name and action.actual_parameters
            
        Returns:
            Text command string for AlfWorld
        """
        action_mapper = {
            "gotolocation": self._map_goto_location,
            "look": lambda params: "look",
            "inventory": lambda params: "inventory",
            "examinereceptacle": self._map_examine_receptacle,
            "examineobject": self._map_examine_object,
            "openobject": self._map_open_object,
            "closeobject": self._map_close_object,
            "pickupobject": self._map_pickup_object,
            "putobject": self._map_put_object,
            "cleanobject": self._map_clean_object,
            "heatobject": self._map_heat_object,
            "coolobject": self._map_cool_object,
            "toggleobject": self._map_toggle_object,
            "sliceobject": self._map_slice_object,
        }
        
        action_name = action.action.name.lower()
        params = [str(p) for p in action.actual_parameters]
        
        if action_name in action_mapper:
            return action_mapper[action_name](params)
        else:
            raise ValueError(f"Unknown action: {action_name}")

    def map_plan2simulator(self, plan: List[Any]) -> List[str]:
        """Map PDDL plan to AlfWorld text commands.
        
        Args:
            plan: List of PDDL actions
            
        Returns:
            List of text command strings
        """
        return [self.map_pddl2simulator(action) for action in plan]

    def step(self, action: str) -> Tuple[Dict, float, bool, bool, Dict]:
        """Execute a text command in AlfWorld.
        
        Args:
            action: Text command string
            
        Returns:
            observation: New observation with feedback
            reward: Reward (0 or 1 for task completion)
            terminated: Whether task is complete
            truncated: Whether episode was truncated
            info: Additional information
        """
        try:
            obs, reward, done, info = self.env.step([action])
            
            self.current_obs = obs
            self.location = self._parse_location(obs)
            self.inventory = self._parse_inventory(obs)
            
            observation = {
                "observation": obs,
                "location": self.location,
                "inventory": self.inventory,
                "feedback": obs if isinstance(obs, str) else obs.get("feedback", "")
            }
            
            return observation, reward, done, False, info
            
        except Exception as e:
            raise InfeasiblePlan(f"Action '{action}' failed: {str(e)}")

    def observation(self, obs: Any) -> Dict:
        """Process raw observation from AlfWorld.
        
        Args:
            obs: Raw observation (usually text)
            
        Returns:
            Formatted observation dictionary
        """
        return {
            "observation": obs,
            "location": self._parse_location(obs),
            "inventory": self._parse_inventory(obs),
        }

    # Helper methods for action mapping
    
    def _map_goto_location(self, params: List[str]) -> str:
        """Map gotolocation action: agent, location_start, location_end"""
        # params: [agent, location_start, location_end]
        if len(params) >= 3:
            return f"go to {params[2]}"
        return f"go to {params[-1]}"

    def _map_examine_receptacle(self, params: List[str]) -> str:
        """Map examinereceptacle action: agent, receptacle"""
        return f"examine {params[1]}"

    def _map_examine_object(self, params: List[str]) -> str:
        """Map examineobject action: agent, object"""
        return f"examine {params[1]}"

    def _map_open_object(self, params: List[str]) -> str:
        """Map openobject action: agent, location, receptacle"""
        return f"open {params[2]}"

    def _map_close_object(self, params: List[str]) -> str:
        """Map closeobject action: agent, location, receptacle"""
        return f"close {params[2]}"

    def _map_pickup_object(self, params: List[str]) -> str:
        """Map pickupobject action: agent, location, object, receptacle"""
        return f"take {params[2]} from {params[3]}"

    def _map_put_object(self, params: List[str]) -> str:
        """Map putobject action: agent, location, object, receptacle, object_type, receptacle_type"""
        return f"put {params[2]} in/on {params[3]}"

    def _map_clean_object(self, params: List[str]) -> str:
        """Map cleanobject action: agent, location, receptacle, object"""
        return f"clean {params[3]} with {params[2]}"

    def _map_heat_object(self, params: List[str]) -> str:
        """Map heatobject action: agent, location, receptacle, object"""
        return f"heat {params[3]} with {params[2]}"

    def _map_cool_object(self, params: List[str]) -> str:
        """Map coolobject action: agent, location, receptacle, object"""
        return f"cool {params[3]} with {params[2]}"

    def _map_toggle_object(self, params: List[str]) -> str:
        """Map toggleobject action: agent, location, object, receptacle"""
        return f"use {params[2]}"

    def _map_slice_object(self, params: List[str]) -> str:
        """Map sliceobject action: agent, location, object_to_slice, knife"""
        return f"slice {params[2]} with {params[3]}"

    # Helper methods for parsing observations
    
    def _parse_location(self, obs: Any) -> Optional[str]:
        """Extract current location from observation text."""
        if isinstance(obs, str):
            # Look for "You are in <location>"
            match = re.search(r"You are (?:in|at) (?:the )?(\w+)", obs)
            if match:
                return match.group(1)
        return self.location  # Keep previous location if not found

    def _parse_inventory(self, obs: Any) -> List[str]:
        """Extract inventory items from observation text."""
        if isinstance(obs, str):
            # Look for "You are carrying: <items>"
            match = re.search(r"You are carrying:(.+?)(?:\n|$)", obs)
            if match:
                items_text = match.group(1).strip()
                if items_text.lower() == "nothing":
                    return []
                return [item.strip() for item in items_text.split(",")]
        return self.inventory  # Keep previous inventory if not found


# Wrapper for integration with existing lexicon code
class AlfWorldPDDLWrapper(AlfWorldSimulator):
    """Compatibility wrapper for AlfWorld to match BabyAI interface."""
    
    def is_feasible_low_level(self, plan) -> bool:
        """Check if a plan is feasible (compatibility method)."""
        return self.is_feasible_plan(plan)
    
    def is_feasible_low_level_action(self, action) -> bool:
        """Check if an action is feasible (compatibility method)."""
        return self.is_feasible_action(action)
