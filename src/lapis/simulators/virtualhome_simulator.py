"""VirtualHome simulator implementation.

This simulator integrates the VirtualHome environment for household activity simulation.
VirtualHome provides detailed simulation of household activities with realistic physics
and multi-agent support.

Required packages:
    - virtualhome: Install from https://github.com/xavierpuigf/virtualhome
    
Usage:
    To use this simulator, you need to have VirtualHome installed.
    See: https://github.com/xavierpuigf/virtualhome
"""

import json
from typing import List, Tuple, Dict, Any, Optional
import numpy as np

from .base_simulator import BaseSimulator
from .utils import InfeasiblePlan


class VirtualHomeSimulator(BaseSimulator):
    """VirtualHome simulator with PDDL action mapping.
    
    This simulator wraps a VirtualHome environment and provides PDDL action
    translation and execution for complex household activities.
    
    VirtualHome actions follow the format:
        [ActionName] <object_name> (character_id)
    
    Common actions:
        - [Walk] <object>
        - [Find] <object>
        - [Grab] <object>
        - [Put] <object> <container>
        - [Open] <object>
        - [Close] <object>
        - [SwitchOn] <object>
        - [SwitchOff] <object>
        - [Sit] <object>
        - [StandUp]
        - [Drink] <object>
        - [TurnTo] <object>
        - [LookAt] <object>
        - [Touch] <object>
        - [PlugIn] <object>
        - [PlugOut] <object>
        - [Cut] <object>
        - [Eat] <object>
        - [Sleep]
        - [WakeUp]
        - [Release] <object>
    """

    def __init__(
        self,
        scene_id: int = 0,
        num_agents: int = 1,
        rendering: bool = False,
        port: str = "8080",
        **kwargs
    ):
        """Initialize VirtualHome simulator.
        
        Args:
            scene_id: Scene/apartment ID (0-6)
            num_agents: Number of agents in the environment
            rendering: Whether to enable Unity rendering
            port: Port for Unity communication
            **kwargs: Additional simulator parameters
        """
        try:
            from virtualhome.simulation.unity_simulator import UnityCommunication
        except ImportError:
            raise ImportError(
                "VirtualHome not installed. Install from:\n"
                "git clone https://github.com/xavierpuigf/virtualhome.git\n"
                "cd virtualhome && pip install -e ."
            )
        
        # Initialize Unity communication
        self.comm = UnityCommunication(port=port)
        self.scene_id = scene_id
        self.num_agents = num_agents
        self.rendering = rendering
        
        # Initialize environment
        self.comm.reset(scene_id)
        
        # Initialize base simulator without passing env (we manage it differently)
        self.env = None  # VirtualHome uses communication interface
        self.prev_obs = None
        self.current_graph = None
        self.agent_states = [{"location": None, "holding": []} for _ in range(num_agents)]

    def setup(self, **kwargs) -> bool:
        """Setup the simulator.
        
        VirtualHome simulator is set up during initialization.
        """
        return True

    def reset(self, seed: Optional[int] = None, scene_id: Optional[int] = None) -> Tuple[Dict, Dict]:
        """Reset the VirtualHome environment.
        
        Args:
            seed: Random seed (VirtualHome doesn't use seeds directly)
            scene_id: Scene to load (if different from initialization)
            
        Returns:
            observation: Initial scene graph and agent states
            info: Additional information dictionary
        """
        if scene_id is not None:
            self.scene_id = scene_id
        
        # Reset scene
        success = self.comm.reset(self.scene_id)
        
        if not success:
            raise RuntimeError(f"Failed to reset VirtualHome scene {self.scene_id}")
        
        # Get initial scene graph
        scene_graph = self.comm.environment_graph()
        self.current_graph = scene_graph
        
        # Reset agent states
        self.agent_states = [{"location": None, "holding": []} for _ in range(self.num_agents)]
        
        observation = {
            "scene_graph": scene_graph,
            "agents": self.agent_states,
        }
        
        info = {
            "scene_id": self.scene_id,
            "num_agents": self.num_agents,
        }
        
        return observation, info

    def map_pddl2simulator(self, action: Any) -> Tuple[str, int]:
        """Map PDDL action to VirtualHome script command.
        
        Args:
            action: PDDL action object with action.name and action.actual_parameters
            
        Returns:
            Tuple of (action_script, agent_id)
        """
        action_mapper = {
            "walk": self._map_walk,
            "find": self._map_find,
            "grab": self._map_grab,
            "put": self._map_put,
            "putin": self._map_putin,
            "open": self._map_open,
            "close": self._map_close,
            "switchon": self._map_switchon,
            "switchoff": self._map_switchoff,
            "sit": self._map_sit,
            "standup": self._map_standup,
            "drink": self._map_drink,
            "turnto": self._map_turnto,
            "lookat": self._map_lookat,
            "touch": self._map_touch,
            "plugin": self._map_plugin,
            "plugout": self._map_plugout,
            "cut": self._map_cut,
            "eat": self._map_eat,
            "sleep": self._map_sleep,
            "wakeup": self._map_wakeup,
            "release": self._map_release,
        }
        
        action_name = action.action.name.lower()
        params = [str(p) for p in action.actual_parameters]
        
        if action_name in action_mapper:
            return action_mapper[action_name](params)
        else:
            raise ValueError(f"Unknown action: {action_name}")

    def map_plan2simulator(self, plan: List[Any]) -> List[Tuple[str, int]]:
        """Map PDDL plan to VirtualHome script commands.
        
        Args:
            plan: List of PDDL actions
            
        Returns:
            List of (action_script, agent_id) tuples
        """
        return [self.map_pddl2simulator(action) for action in plan]

    def step(self, action: Tuple[str, int]) -> Tuple[Dict, float, bool, bool, Dict]:
        """Execute an action script in VirtualHome.
        
        Args:
            action: Tuple of (action_script, agent_id)
            
        Returns:
            observation: New scene graph and agent states
            reward: Reward (needs to be defined based on goal)
            terminated: Whether goal is achieved
            truncated: Whether episode was truncated
            info: Additional information including success status
        """
        script, agent_id = action
        
        try:
            # Execute script
            success, message = self.comm.render_script(
                [script],
                recording=self.rendering,
                skip_animation=not self.rendering,
                camera_mode="FIRST_PERSON"
            )
            
            if not success:
                raise InfeasiblePlan(f"Action '{script}' failed: {message}")
            
            # Get updated scene graph
            self.current_graph = self.comm.environment_graph()
            
            # Update agent state
            self._update_agent_state(agent_id)
            
            observation = {
                "scene_graph": self.current_graph,
                "agents": self.agent_states,
            }
            
            info = {
                "success": success,
                "message": message,
                "agent_id": agent_id,
            }
            
            # Reward and termination need to be defined based on task
            reward = 0.0
            terminated = False
            truncated = False
            
            return observation, reward, terminated, truncated, info
            
        except Exception as e:
            raise InfeasiblePlan(f"Action '{script}' failed: {str(e)}")

    def observation(self, obs: Any) -> Dict:
        """Process raw observation from VirtualHome.
        
        Args:
            obs: Raw observation (scene graph)
            
        Returns:
            Formatted observation dictionary
        """
        return {
            "scene_graph": obs,
            "agents": self.agent_states,
        }

    # Helper methods for action mapping
    
    def _extract_agent_id(self, params: List[str]) -> int:
        """Extract agent ID from parameters (usually first or last param)."""
        # Check if last param is agent/character ID
        for param in params:
            if param.startswith("agent") or param.startswith("char"):
                try:
                    # Extract numeric ID
                    agent_id = int(''.join(filter(str.isdigit, param)))
                    return agent_id
                except ValueError:
                    pass
        return 0  # Default to agent 0

    def _map_walk(self, params: List[str]) -> Tuple[str, int]:
        """Map walk action: agent, target_location"""
        agent_id = self._extract_agent_id(params)
        target = params[1] if len(params) > 1 else params[0]
        return f"[Walk] <{target}> ({agent_id})", agent_id

    def _map_find(self, params: List[str]) -> Tuple[str, int]:
        """Map find action: agent, object"""
        agent_id = self._extract_agent_id(params)
        obj = params[1] if len(params) > 1 else params[0]
        return f"[Find] <{obj}> ({agent_id})", agent_id

    def _map_grab(self, params: List[str]) -> Tuple[str, int]:
        """Map grab action: agent, object"""
        agent_id = self._extract_agent_id(params)
        obj = params[1] if len(params) > 1 else params[0]
        return f"[Grab] <{obj}> ({agent_id})", agent_id

    def _map_put(self, params: List[str]) -> Tuple[str, int]:
        """Map put action: agent, object, container"""
        agent_id = self._extract_agent_id(params)
        obj = params[1] if len(params) > 2 else params[0]
        container = params[2] if len(params) > 2 else params[1]
        return f"[Put] <{obj}> <{container}> ({agent_id})", agent_id

    def _map_putin(self, params: List[str]) -> Tuple[str, int]:
        """Map putin action: agent, object, container"""
        return self._map_put(params)  # Same as put

    def _map_open(self, params: List[str]) -> Tuple[str, int]:
        """Map open action: agent, object"""
        agent_id = self._extract_agent_id(params)
        obj = params[1] if len(params) > 1 else params[0]
        return f"[Open] <{obj}> ({agent_id})", agent_id

    def _map_close(self, params: List[str]) -> Tuple[str, int]:
        """Map close action: agent, object"""
        agent_id = self._extract_agent_id(params)
        obj = params[1] if len(params) > 1 else params[0]
        return f"[Close] <{obj}> ({agent_id})", agent_id

    def _map_switchon(self, params: List[str]) -> Tuple[str, int]:
        """Map switchon action: agent, object"""
        agent_id = self._extract_agent_id(params)
        obj = params[1] if len(params) > 1 else params[0]
        return f"[SwitchOn] <{obj}> ({agent_id})", agent_id

    def _map_switchoff(self, params: List[str]) -> Tuple[str, int]:
        """Map switchoff action: agent, object"""
        agent_id = self._extract_agent_id(params)
        obj = params[1] if len(params) > 1 else params[0]
        return f"[SwitchOff] <{obj}> ({agent_id})", agent_id

    def _map_sit(self, params: List[str]) -> Tuple[str, int]:
        """Map sit action: agent, object"""
        agent_id = self._extract_agent_id(params)
        obj = params[1] if len(params) > 1 else params[0]
        return f"[Sit] <{obj}> ({agent_id})", agent_id

    def _map_standup(self, params: List[str]) -> Tuple[str, int]:
        """Map standup action: agent"""
        agent_id = self._extract_agent_id(params)
        return f"[StandUp] ({agent_id})", agent_id

    def _map_drink(self, params: List[str]) -> Tuple[str, int]:
        """Map drink action: agent, object"""
        agent_id = self._extract_agent_id(params)
        obj = params[1] if len(params) > 1 else params[0]
        return f"[Drink] <{obj}> ({agent_id})", agent_id

    def _map_turnto(self, params: List[str]) -> Tuple[str, int]:
        """Map turnto action: agent, object"""
        agent_id = self._extract_agent_id(params)
        obj = params[1] if len(params) > 1 else params[0]
        return f"[TurnTo] <{obj}> ({agent_id})", agent_id

    def _map_lookat(self, params: List[str]) -> Tuple[str, int]:
        """Map lookat action: agent, object"""
        agent_id = self._extract_agent_id(params)
        obj = params[1] if len(params) > 1 else params[0]
        return f"[LookAt] <{obj}> ({agent_id})", agent_id

    def _map_touch(self, params: List[str]) -> Tuple[str, int]:
        """Map touch action: agent, object"""
        agent_id = self._extract_agent_id(params)
        obj = params[1] if len(params) > 1 else params[0]
        return f"[Touch] <{obj}> ({agent_id})", agent_id

    def _map_plugin(self, params: List[str]) -> Tuple[str, int]:
        """Map plugin action: agent, object"""
        agent_id = self._extract_agent_id(params)
        obj = params[1] if len(params) > 1 else params[0]
        return f"[PlugIn] <{obj}> ({agent_id})", agent_id

    def _map_plugout(self, params: List[str]) -> Tuple[str, int]:
        """Map plugout action: agent, object"""
        agent_id = self._extract_agent_id(params)
        obj = params[1] if len(params) > 1 else params[0]
        return f"[PlugOut] <{obj}> ({agent_id})", agent_id

    def _map_cut(self, params: List[str]) -> Tuple[str, int]:
        """Map cut action: agent, object"""
        agent_id = self._extract_agent_id(params)
        obj = params[1] if len(params) > 1 else params[0]
        return f"[Cut] <{obj}> ({agent_id})", agent_id

    def _map_eat(self, params: List[str]) -> Tuple[str, int]:
        """Map eat action: agent, object"""
        agent_id = self._extract_agent_id(params)
        obj = params[1] if len(params) > 1 else params[0]
        return f"[Eat] <{obj}> ({agent_id})", agent_id

    def _map_sleep(self, params: List[str]) -> Tuple[str, int]:
        """Map sleep action: agent"""
        agent_id = self._extract_agent_id(params)
        return f"[Sleep] ({agent_id})", agent_id

    def _map_wakeup(self, params: List[str]) -> Tuple[str, int]:
        """Map wakeup action: agent"""
        agent_id = self._extract_agent_id(params)
        return f"[WakeUp] ({agent_id})", agent_id

    def _map_release(self, params: List[str]) -> Tuple[str, int]:
        """Map release action: agent, object"""
        agent_id = self._extract_agent_id(params)
        obj = params[1] if len(params) > 1 else params[0]
        return f"[Release] <{obj}> ({agent_id})", agent_id

    def _update_agent_state(self, agent_id: int):
        """Update agent state from current scene graph."""
        if self.current_graph is None:
            return
        
        # Parse scene graph to update agent state
        # This is a simplified version - full implementation would parse the graph
        # to determine agent location and what they're holding
        if "nodes" in self.current_graph:
            for node in self.current_graph["nodes"]:
                if node.get("class_name") == "character" and node.get("id") == agent_id:
                    self.agent_states[agent_id]["location"] = node.get("in_room")
                    # Check for held objects
                    held_objects = [
                        edge["to_id"]
                        for edge in self.current_graph.get("edges", [])
                        if edge["from_id"] == agent_id and edge["relation_type"] == "HOLDS_RH"
                    ]
                    self.agent_states[agent_id]["holding"] = held_objects

    def close(self):
        """Close the VirtualHome simulator."""
        if hasattr(self, 'comm'):
            self.comm.close()


# Wrapper for integration with existing lexicon code
class VirtualHomePDDLWrapper(VirtualHomeSimulator):
    """Compatibility wrapper for VirtualHome to match BabyAI interface."""
    
    def is_feasible_low_level(self, plan) -> bool:
        """Check if a plan is feasible (compatibility method)."""
        return self.is_feasible_plan(plan)
    
    def is_feasible_low_level_action(self, action) -> bool:
        """Check if an action is feasible (compatibility method)."""
        return self.is_feasible_action(action)
