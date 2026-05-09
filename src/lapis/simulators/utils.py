"""Utility classes and exceptions for simulators."""

from dataclasses import dataclass
from typing import Tuple, Any


class InfeasiblePlan(Exception):
    """Exception raised when a plan cannot be executed."""
    
    def __init__(self, message: str):
        self.message = message
        super().__init__(self.message)


@dataclass
class Object:
    """Represents an object in the environment."""
    color: str
    type: str
    room: int


@dataclass
class EmptyObject:
    """Represents an empty cell in the environment."""
    cur_pos: Tuple[int, int]
    type: str = "empty"


@dataclass
class Door:
    """Represents a door between rooms."""
    color: str
    adj_rooms: Tuple[int, int]


@dataclass
class Room:
    """Represents a room in the environment."""
    room_id: int


@dataclass
class Action:
    """Represents an action with parameters."""
    action: Any
    actual_parameters: list
