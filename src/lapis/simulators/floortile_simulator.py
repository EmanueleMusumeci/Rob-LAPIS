"""floortile_simulator.py — Visualizer for the Floortile IPC domain.

Renders a grid showing painted cells (blank/black/white), robot positions,
and color availability.
"""

from __future__ import annotations

from typing import Optional
from .pddl_simulator import PDDLSimulator

W, H = 700, 560
_BG = (18, 18, 28)
_HDR = (38, 38, 72)
_TEXT = (210, 210, 220)
_GRID_BORDER = (70, 70, 110)
_BLANK = (50, 50, 70)
_BLACK_PAINT = (20, 20, 20)
_WHITE_PAINT = (230, 230, 240)
_ROBOT_COLOR = (80, 200, 120)
MARGIN = 40
HEADER_H = 70


class FloortileSimulator(PDDLSimulator):

    def __init__(self, seed=None):
        super().__init__("floortile", seed=seed)

    def get_image(self, action_text: Optional[str] = None, **kwargs):
        try:
            from PIL import Image, ImageDraw
        except ImportError:
            return None

        if self.problem is None or self.current_state is None:
            return None

        img = Image.new("RGB", (W, H), _BG)
        draw = ImageDraw.Draw(img)

        draw.rectangle([0, 0, W, HEADER_H], fill=_HDR)
        draw.text((20, 14), "Floortile Domain", fill=_TEXT)
        if action_text:
            draw.text((20, 40), f"Action: {action_text}", fill=(150, 150, 170))

        all_objects = list(self.problem.all_objects)

        tiles  = [o for o in all_objects if "tile" in o.type.name.lower() or
                  "tile" in o.name.lower()]
        robots = [o for o in all_objects if "robot" in o.type.name.lower() or
                  "robot" in o.name.lower()]

        if not tiles:
            draw.text((20, HEADER_H + 10), "No tiles found", fill=_TEXT)
            return img

        # Infer grid dimensions from tile names (e.g. tile_1_2 → row=1, col=2)
        import re
        coords: dict[str, tuple[int, int]] = {}
        for tile in tiles:
            m = re.search(r"(\d+)[_\-\s](\d+)", tile.name)
            if m:
                coords[tile.name] = (int(m.group(1)), int(m.group(2)))
            else:
                idx = tiles.index(tile)
                coords[tile.name] = (idx // 5, idx % 5)

        rows = max(c[0] for c in coords.values()) + 1
        cols = max(c[1] for c in coords.values()) + 1
        cell_size = min(
            (W - 2 * MARGIN) // max(cols, 1),
            (H - HEADER_H - MARGIN) // max(rows, 1),
        )

        def _check(fname, *onames):
            try:
                fluent = self.problem.fluent(fname)
                from unified_planning.shortcuts import get_environment
                env = get_environment()
                objs = [env.expression_manager.ObjectExp(
                    next(o for o in all_objects if o.name == n))
                    for n in onames]
                val = self.current_state.get_value(fluent(*objs))
                return val.bool_constant_value()
            except Exception:
                return False

        # Draw tiles
        robot_positions: dict[str, str] = {}
        for robot in robots:
            for tile in tiles:
                if _check("robot-at", robot.name, tile.name) or \
                   _check("at", robot.name, tile.name):
                    robot_positions[robot.name] = tile.name

        for tile in tiles:
            row, col = coords[tile.name]
            x0 = MARGIN + col * cell_size
            y0 = HEADER_H + MARGIN // 2 + row * cell_size
            x1, y1 = x0 + cell_size - 2, y0 + cell_size - 2

            if _check("painted-black", tile.name):
                fill = _BLACK_PAINT
                text_color = (200, 200, 200)
                label = "■"
            elif _check("painted-white", tile.name):
                fill = _WHITE_PAINT
                text_color = (30, 30, 30)
                label = "□"
            else:
                fill = _BLANK
                text_color = (120, 120, 150)
                label = "·"

            draw.rectangle([x0, y0, x1, y1], fill=fill, outline=_GRID_BORDER)
            draw.text((x0 + cell_size // 2 - 5, y0 + cell_size // 2 - 8), label,
                      fill=text_color)

            # Robot on this tile
            for robot_name, at_tile in robot_positions.items():
                if at_tile == tile.name:
                    cx, cy = (x0 + x1) // 2, (y0 + y1) // 2
                    r = cell_size // 5
                    draw.ellipse([cx - r, cy - r, cx + r, cy + r], fill=_ROBOT_COLOR)

        return img
