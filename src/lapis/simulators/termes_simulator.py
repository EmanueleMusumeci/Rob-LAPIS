"""termes_simulator.py — Visualizer for the Termes IPC domain.

Renders a height grid with color gradient (blue=0 → red=max),
robot position, and depot location.
"""

from __future__ import annotations

from typing import Optional
from .pddl_simulator import PDDLSimulator

W, H = 640, 540
MARGIN = 60
HEADER_H = 70


class TermesSimulator(PDDLSimulator):

    def __init__(self, seed=None):
        super().__init__("termes", seed=seed)

    def get_image(self, action_text: Optional[str] = None, **kwargs):
        try:
            from PIL import Image, ImageDraw
        except ImportError:
            return None

        if self.problem is None or self.current_state is None:
            return None

        all_objects = list(self.problem.all_objects)

        # Find npos (grid positions) and robots/depots
        npos = [o for o in all_objects if "npos" in o.type.name.lower() or
                o.name.lower().startswith("n")]
        robots = [o for o in all_objects if "robot" in o.type.name.lower() or
                  o.name.lower().startswith("r") and "robo" in o.name.lower()]
        depots = [o for o in all_objects if "depot" in o.name.lower() or
                  o.type.name.lower() == "depot"]

        # Parse grid coordinates from npos names (e.g., n0_1 → x=0, y=1)
        coords: dict[str, tuple[int, int]] = {}
        import re
        for p in npos:
            m = re.search(r"(\d+)[_x,\s](\d+)", p.name)
            if m:
                coords[p.name] = (int(m.group(1)), int(m.group(2)))
            else:
                m2 = re.search(r"(\d+)$", p.name)
                if m2:
                    idx = int(m2.group(1))
                    coords[p.name] = (idx % 5, idx // 5)

        if not coords:
            img = Image.new("RGB", (W, H), (20, 20, 40))
            draw = ImageDraw.Draw(img)
            draw.text((20, 20), "Termes: no grid positions found", fill=(200, 200, 200))
            return img

        xs = [c[0] for c in coords.values()]
        ys = [c[1] for c in coords.values()]
        gw, gh = max(xs) + 1, max(ys) + 1

        cell_size = min(
            (W - 2 * MARGIN) // max(gw, 1),
            (H - MARGIN - HEADER_H) // max(gh, 1)
        )

        img = Image.new("RGB", (W, H), (15, 15, 35))
        draw = ImageDraw.Draw(img)

        # Header
        draw.rectangle([0, 0, W, HEADER_H], fill=(40, 40, 80))
        draw.text((20, 14), "Termes Domain", fill=(220, 220, 220))
        if action_text:
            draw.text((20, 40), f"Action: {action_text}", fill=(160, 160, 160))

        # Gather heights
        heights: dict[str, int] = {}
        for pos_name, (gx, gy) in coords.items():
            try:
                h_val = self._get_height(pos_name, all_objects) or 0
                heights[pos_name] = h_val
            except Exception:
                heights[pos_name] = 0

        max_h = max(heights.values(), default=1) or 1

        def _height_color(h: int) -> tuple:
            ratio = h / max_h
            r = int(50 + 200 * ratio)
            b = int(200 - 180 * ratio)
            return (r, 60, b)

        # Draw grid cells
        robot_at: dict[str, str] = {}
        depot_at: set[str] = set()

        # Find robot positions
        for robot in robots:
            for pos_name in coords:
                if self._check_at(robot.name, pos_name, all_objects):
                    robot_at[robot.name] = pos_name

        # Find depot positions
        for depot in depots:
            for pos_name in coords:
                if self._check_at(depot.name, pos_name, all_objects) or \
                   self._check_located(depot.name, pos_name, all_objects):
                    depot_at.add(pos_name)

        for pos_name, (gx, gy) in coords.items():
            x0 = MARGIN + gx * cell_size
            y0 = HEADER_H + MARGIN // 2 + gy * cell_size
            x1, y1 = x0 + cell_size - 2, y0 + cell_size - 2

            h = heights.get(pos_name, 0)
            fill = _height_color(h)
            draw.rectangle([x0, y0, x1, y1], fill=fill, outline=(80, 80, 100))

            # Height label
            draw.text((x0 + 4, y0 + 4), str(h), fill=(220, 220, 220))

            # Depot marker
            if pos_name in depot_at:
                draw.text((x0 + cell_size // 2 - 6, y0 + cell_size // 2 - 8),
                          "D", fill=(255, 215, 0))

            # Robot marker
            for robot_name, at_pos in robot_at.items():
                if at_pos == pos_name:
                    cx, cy = (x0 + x1) // 2, (y0 + y1) // 2
                    r = cell_size // 4
                    draw.ellipse([cx - r, cy - r, cx + r, cy + r], fill=(0, 255, 100))

        return img

    def _get_height(self, pos_name: str, all_objects: list) -> int:
        """Try to read height fluent for a position."""
        try:
            fluent = self.problem.fluent("height")
            pos_obj = next((o for o in all_objects if o.name == pos_name), None)
            if pos_obj is None:
                return 0
            from unified_planning.shortcuts import get_environment
            env = get_environment()
            val = self.current_state.get_value(
                fluent(env.expression_manager.ObjectExp(pos_obj))
            )
            return int(val.constant_value()) if not val.is_bool_constant() else 0
        except Exception:
            return 0

    def _check_at(self, obj_name: str, pos_name: str, all_objects: list) -> bool:
        for fname in ["at", "robot-at", "robot_at"]:
            try:
                fluent = self.problem.fluent(fname)
                obj = next((o for o in all_objects if o.name == obj_name), None)
                pos = next((o for o in all_objects if o.name == pos_name), None)
                if obj and pos:
                    from unified_planning.shortcuts import get_environment
                    env = get_environment()
                    val = self.current_state.get_value(
                        fluent(env.expression_manager.ObjectExp(obj),
                               env.expression_manager.ObjectExp(pos))
                    )
                    if val.bool_constant_value():
                        return True
            except Exception:
                pass
        return False

    def _check_located(self, obj_name: str, pos_name: str, all_objects: list) -> bool:
        try:
            fluent = self.problem.fluent("depot-at")
            obj = next((o for o in all_objects if o.name == obj_name), None)
            pos = next((o for o in all_objects if o.name == pos_name), None)
            if obj and pos:
                from unified_planning.shortcuts import get_environment
                env = get_environment()
                val = self.current_state.get_value(
                    fluent(env.expression_manager.ObjectExp(obj),
                           env.expression_manager.ObjectExp(pos))
                )
                return val.bool_constant_value()
        except Exception:
            pass
        return False
