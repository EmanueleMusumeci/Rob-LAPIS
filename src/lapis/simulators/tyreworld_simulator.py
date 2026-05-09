"""tyreworld_simulator.py — Visualizer for the Tyreworld IPC domain.

Renders wheels/hubs, boot contents, tool statuses (jack, wrench, pump),
and nut tightness states.
"""

from __future__ import annotations

from typing import Optional
from .pddl_simulator import PDDLSimulator

W, H = 800, 520
_BG = (18, 18, 32)
_HDR = (40, 40, 80)
_HUB_COLOR = (80, 90, 110)
_FLAT_COLOR = (180, 60, 60)
_INTACT_COLOR = (60, 180, 100)
_BOOT_COLOR = (50, 50, 80)
_TEXT = (210, 210, 220)
_TIGHT_COLOR = (255, 200, 50)
_LOOSE_COLOR = (200, 100, 50)


class TyreworldSimulator(PDDLSimulator):

    def __init__(self, seed=None):
        super().__init__("tyreworld", seed=seed)

    def get_image(self, action_text: Optional[str] = None, **kwargs):
        try:
            from PIL import Image, ImageDraw
        except ImportError:
            return None

        if self.problem is None or self.current_state is None:
            return None

        img = Image.new("RGB", (W, H), _BG)
        draw = ImageDraw.Draw(img)

        # Header
        draw.rectangle([0, 0, W, 60], fill=_HDR)
        draw.text((20, 14), "Tyreworld Domain", fill=_TEXT)
        if action_text:
            draw.text((20, 38), f"Action: {action_text}", fill=(160, 160, 160))

        all_objects = list(self.problem.all_objects)

        hubs  = [o for o in all_objects if "hub"   in o.type.name.lower() or
                 o.name.lower().startswith("hub") or o.name.lower().startswith("wheel")]
        nuts  = [o for o in all_objects if "nut"   in o.type.name.lower() or
                 "nut" in o.name.lower()]
        tyres = [o for o in all_objects if "tyre"  in o.type.name.lower() or
                 "tyre" in o.name.lower()]
        tools = [o for o in all_objects if any(t in o.name.lower()
                 for t in ["jack", "wrench", "pump", "spanner", "boot"])]
        boots = [o for o in all_objects if "boot" in o.name.lower()]

        n_hubs = max(len(hubs), 1)
        hub_w = min(180, (W - 40) // n_hubs)

        def _check_fluent(name, *obj_names):
            try:
                fluent = self.problem.fluent(name)
                from unified_planning.shortcuts import get_environment
                env = get_environment()
                objs = []
                for oname in obj_names:
                    obj = next((o for o in all_objects if o.name == oname), None)
                    if obj is None:
                        return False
                    objs.append(env.expression_manager.ObjectExp(obj))
                val = self.current_state.get_value(fluent(*objs))
                return val.bool_constant_value()
            except Exception:
                return False

        # Draw hubs with tyre/nut info
        for i, hub in enumerate(hubs):
            x0 = 20 + i * hub_w
            y0, y1 = 80, 280
            draw.rectangle([x0, y0, x0 + hub_w - 10, y1], fill=_HUB_COLOR,
                           outline=(100, 110, 140), width=2)
            draw.text((x0 + 6, y0 + 4), hub.name[:8], fill=_TEXT)

            # Is tyre on hub?
            for tyre in tyres:
                if _check_fluent("on", tyre.name, hub.name) or \
                   _check_fluent("on-hub", tyre.name, hub.name):
                    color = _FLAT_COLOR if _check_fluent("flat", tyre.name) else _INTACT_COLOR
                    draw.ellipse([x0 + 10, y0 + 30, x0 + hub_w - 20, y0 + 100],
                                 fill=color, outline=(200, 200, 200))
                    label = "flat" if _check_fluent("flat", tyre.name) else "ok"
                    draw.text((x0 + 20, y0 + 56), label, fill=(30, 30, 30))

            # Nut status
            for nut in nuts:
                is_tight = _check_fluent("tight", nut.name, hub.name)
                is_loose = _check_fluent("loose", nut.name, hub.name)
                if is_tight:
                    draw.ellipse([x0 + hub_w // 2 - 8, y0 + 110,
                                  x0 + hub_w // 2 + 8, y0 + 126],
                                 fill=_TIGHT_COLOR)
                    draw.text((x0 + 6, y0 + 130), "tight", fill=_TIGHT_COLOR)
                elif is_loose:
                    draw.ellipse([x0 + hub_w // 2 - 8, y0 + 110,
                                  x0 + hub_w // 2 + 8, y0 + 126],
                                 fill=_LOOSE_COLOR)
                    draw.text((x0 + 6, y0 + 130), "loose", fill=_LOOSE_COLOR)

        # Draw boot contents
        boot_x0, boot_y0 = 20, 300
        boot_w, boot_h = W - 40, H - 320
        draw.rectangle([boot_x0, boot_y0, boot_x0 + boot_w, boot_y0 + boot_h],
                       outline=(80, 90, 130), width=2)
        draw.text((boot_x0 + 6, boot_y0 + 4), "Boot", fill=(130, 130, 180))

        # Items in boot
        items_in_boot = []
        for obj in all_objects:
            if boots and _check_fluent("in", obj.name, boots[0].name):
                items_in_boot.append(obj.name)

        for j, item_name in enumerate(items_in_boot):
            draw.text((boot_x0 + 10 + j * 100, boot_y0 + 28), item_name, fill=_TEXT)

        return img
