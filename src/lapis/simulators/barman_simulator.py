"""barman_simulator.py — Visualizer for the Barman IPC domain.

Renders the bar counter with dispensers, shakers, glasses,
the barman's two hands, and liquid contents.
"""

from __future__ import annotations

from typing import Optional
from .pddl_simulator import PDDLSimulator

W, H = 900, 540
_BG = (12, 10, 22)
_HDR = (35, 30, 65)
_TEXT = (210, 210, 230)
_COUNTER = (50, 38, 28)
_COUNTER_TOP = (80, 60, 40)
_SHAKER = (120, 130, 150)
_GLASS = (100, 180, 220)
_DISPENSER = (160, 60, 70)
_HAND = (200, 160, 120)
_LIQUID_COLORS = [
    (220, 80, 80), (80, 180, 200), (100, 200, 100),
    (220, 200, 80), (180, 80, 220), (80, 120, 220),
]


class BarmanSimulator(PDDLSimulator):

    def __init__(self, seed=None):
        super().__init__("barman", seed=seed)

    def get_image(self, action_text: Optional[str] = None, **kwargs):
        try:
            from PIL import Image, ImageDraw
        except ImportError:
            return None

        if self.problem is None or self.current_state is None:
            return None

        img = Image.new("RGB", (W, H), _BG)
        draw = ImageDraw.Draw(img)

        draw.rectangle([0, 0, W, 60], fill=_HDR)
        draw.text((20, 15), "Barman Domain", fill=_TEXT)
        if action_text:
            draw.text((20, 38), f"Action: {action_text}", fill=(150, 140, 190))

        # Bar counter
        draw.rectangle([20, 280, W - 20, 420], fill=_COUNTER)
        draw.rectangle([20, 265, W - 20, 285], fill=_COUNTER_TOP)

        all_objects = list(self.problem.all_objects)

        def names(keyword):
            return [o for o in all_objects
                    if keyword.lower() in o.type.name.lower() or keyword.lower() in o.name.lower()]

        dispensers = names("ingredient") or names("dispenser")
        shakers    = names("shaker")
        shots      = names("shot")
        bartenders = names("bartender")
        hands_     = names("hand")
        cocktails  = names("cocktail")

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

        # Draw dispensers along the back wall
        for i, disp in enumerate(dispensers[:8]):
            dx = 40 + i * 100
            draw.rectangle([dx, 100, dx + 60, 264], fill=_DISPENSER, outline=(200, 80, 90))
            draw.text((dx + 4, 104), disp.name[:7], fill=_TEXT)
            # Dispense button
            draw.ellipse([dx + 20, 220, dx + 40, 240], fill=(220, 50, 50))

        # Draw shakers on counter
        for i, shaker in enumerate(shakers):
            sx = 50 + i * 120
            sy = 200
            draw.rectangle([sx, sy, sx + 40, 265], fill=_SHAKER, outline=(180, 190, 200))
            draw.text((sx + 2, sy + 2), shaker.name[:6], fill=(30, 30, 30))

            # Is something in shaker?
            contents = []
            for cocktail in cocktails:
                if _check("contains", shaker.name, cocktail.name):
                    contents.append(cocktail.name[:4])
            if contents:
                liq_color = _LIQUID_COLORS[i % len(_LIQUID_COLORS)]
                draw.rectangle([sx + 4, 240, sx + 36, 263], fill=liq_color)
                draw.text((sx + 2, 243), contents[0][:4], fill=(30, 30, 30))

        # Draw shot glasses on counter
        for i, shot in enumerate(shots[:6]):
            gx = 300 + i * 90
            gy = 225
            draw.rectangle([gx, gy, gx + 35, 265], fill=_GLASS, outline=(60, 150, 200))
            draw.text((gx + 2, gy + 2), shot.name[:5], fill=(20, 20, 40))

            # Contents in shot glass
            for cocktail in cocktails:
                if _check("contains", shot.name, cocktail.name):
                    liq_color = _LIQUID_COLORS[cocktails.index(cocktail) % len(_LIQUID_COLORS)]
                    draw.rectangle([gx + 4, 245, gx + 31, 263], fill=liq_color)

        # Draw barman hands
        hand_names = [h.name for h in hands_] or ["left", "right"]
        for j, hand_name in enumerate(hand_names[:2]):
            hx = 200 + j * 400
            hy = 430
            draw.ellipse([hx, hy, hx + 60, hy + 40], fill=_HAND, outline=(180, 140, 100))
            draw.text((hx + 4, hy + 12), hand_name[:5], fill=(80, 60, 40))

            # Is hand holding something?
            for container in shakers + shots:
                if _check("holding", hand_name, container.name) or \
                   _check("grasping", hand_name, container.name):
                    draw.text((hx + 4, hy - 20), f"← {container.name[:6]}", fill=_TEXT)

        return img
