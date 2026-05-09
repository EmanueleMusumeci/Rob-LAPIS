"""grippers_simulator.py — Visualizer for the Grippers IPC domain.

Renders rooms, robots, and ball positions using PIL.
"""

from __future__ import annotations

from typing import Optional
from .pddl_simulator import PDDLSimulator

_COLORS = {
    "room_bg": (30, 30, 60),
    "room_border": (100, 120, 200),
    "ball": (255, 180, 50),
    "robot": (80, 200, 120),
    "text": (220, 220, 220),
    "holding": (255, 100, 100),
    "bg": (15, 15, 35),
    "header": (50, 50, 90),
}

W, H = 800, 500
ROOM_H = 300
ROOM_TOP = 120


class GrippersSimulator(PDDLSimulator):
    """Visualizer for the Grippers planning domain."""

    def __init__(self, seed=None):
        super().__init__("grippers", seed=seed)

    def get_image(self, action_text: Optional[str] = None, **kwargs):
        try:
            from PIL import Image, ImageDraw, ImageFont
        except ImportError:
            return None

        if self.problem is None or self.current_state is None:
            return None

        img = Image.new("RGB", (W, H), _COLORS["bg"])
        draw = ImageDraw.Draw(img)

        # Header
        draw.rectangle([0, 0, W, 60], fill=_COLORS["header"])
        draw.text((20, 15), "Grippers Domain", fill=_COLORS["text"])
        if action_text:
            draw.text((20, 38), f"Action: {action_text}", fill=(180, 180, 180))

        # Gather rooms, robots, balls from fluents
        all_objects = list(self.problem.all_objects)
        rooms  = [o.name for o in all_objects if "room"  in o.type.name.lower()]
        robots = [o.name for o in all_objects if "robot" in o.type.name.lower()]
        balls  = [o.name for o in all_objects if "ball"  in o.type.name.lower()]

        if not rooms:
            # Fall back: guess from object names
            rooms  = [o.name for o in all_objects if o.name.lower().startswith("room")]
            robots = [o.name for o in all_objects if o.name.lower().startswith("robo")]
            balls  = [o.name for o in all_objects if o.name.lower().startswith("ball")]

        n_rooms = max(len(rooms), 1)
        room_w = (W - 40) // n_rooms

        room_positions: dict[str, tuple[int, int, int, int]] = {}
        for i, room in enumerate(rooms):
            x0 = 20 + i * room_w
            x1 = x0 + room_w - 10
            y0, y1 = ROOM_TOP, ROOM_TOP + ROOM_H
            room_positions[room] = (x0, y0, x1, y1)
            draw.rectangle([x0, y0, x1, y1], outline=_COLORS["room_border"], width=3)
            draw.text((x0 + 8, y0 + 6), room, fill=_COLORS["room_border"])

        def _is_at(obj, room_name, fluent_name="at"):
            for fn in [fluent_name, "at-robby", "at_robby"]:
                try:
                    fluent = self.problem.fluent(fn)
                    room_obj = next((o for o in all_objects if o.name == room_name), None)
                    obj_o = next((o for o in all_objects if o.name == obj), None)
                    if room_obj and obj_o:
                        from unified_planning.shortcuts import get_environment
                        env = get_environment()
                        val = self.current_state.get_value(fluent(
                            env.expression_manager.ObjectExp(obj_o),
                            env.expression_manager.ObjectExp(room_obj),
                        ))
                        if val.bool_constant_value():
                            return True
                except Exception:
                    pass
            return False

        # Draw balls in their rooms
        for ball in balls:
            for room, (x0, y0, x1, y1) in room_positions.items():
                if _is_at(ball, room, "at"):
                    cx = (x0 + x1) // 2
                    cy = y0 + 80 + balls.index(ball) * 30
                    r = 12
                    draw.ellipse([cx - r, cy - r, cx + r, cy + r], fill=_COLORS["ball"])
                    draw.text((cx - 10, cy - 8), ball[:4], fill=(30, 30, 30))

        # Draw robots in their rooms
        for robot in robots:
            for room, (x0, y0, x1, y1) in room_positions.items():
                if _is_at(robot, room, "at-robby") or _is_at(robot, room, "at_robby"):
                    cx = (x0 + x1) // 2
                    cy = y1 - 60
                    draw.rectangle([cx - 15, cy - 25, cx + 15, cy + 25],
                                   fill=_COLORS["robot"])
                    draw.text((cx - 12, cy - 12), robot[:4], fill=(30, 30, 30))

        return img
