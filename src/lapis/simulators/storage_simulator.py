"""storage_simulator.py — Visualizer for the Storage IPC domain.

Renders depot/container/store zones, crates on pallets, and hoists.
"""

from __future__ import annotations

from typing import Optional
from .pddl_simulator import PDDLSimulator

W, H = 800, 520
_BG = (15, 20, 35)
_TEXT = (210, 210, 220)
_HDR = (35, 45, 80)
_DEPOT_BG = (30, 50, 80)
_CONTAINER_BG = (50, 35, 70)
_STORE_BG = (25, 60, 45)
_CRATE_COLOR = (200, 150, 60)
_HOIST_COLOR = (160, 80, 200)


class StorageSimulator(PDDLSimulator):

    def __init__(self, seed=None):
        super().__init__("storage", seed=seed)

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
        draw.text((20, 15), "Storage Domain", fill=_TEXT)
        if action_text:
            draw.text((20, 38), f"Action: {action_text}", fill=(150, 150, 170))

        all_objects = list(self.problem.all_objects)

        def names_of_type(*keywords):
            return [o for o in all_objects
                    if any(k.lower() in o.type.name.lower() or k.lower() in o.name.lower()
                           for k in keywords)]

        depots     = names_of_type("depot")
        containers = names_of_type("container")
        stores     = names_of_type("store")
        hoists     = names_of_type("hoist")
        crates     = names_of_type("crate")

        def _check(fname, *obj_names):
            try:
                fluent = self.problem.fluent(fname)
                from unified_planning.shortcuts import get_environment
                env = get_environment()
                objs = [env.expression_manager.ObjectExp(
                    next(o for o in all_objects if o.name == n))
                    for n in obj_names]
                val = self.current_state.get_value(fluent(*objs))
                return val.bool_constant_value()
            except Exception:
                return False

        # Draw three zone columns
        zones = [
            ("Depots",      depots,     _DEPOT_BG,     80),
            ("Containers",  containers, _CONTAINER_BG, 290),
            ("Storeareas",  stores,     _STORE_BG,     500),
        ]

        for label, zone_objs, bg, x0 in zones:
            x1 = x0 + 190
            draw.rectangle([x0, 70, x1, H - 20], fill=bg, outline=(80, 90, 120), width=2)
            draw.text((x0 + 6, 74), label, fill=_TEXT)

            for j, zone_obj in enumerate(zone_objs):
                y0_ = 95 + j * 60
                draw.text((x0 + 10, y0_), zone_obj.name[:14], fill=(180, 200, 220))

                # Crates on this surface
                for crate in crates:
                    if _check("in", crate.name, zone_obj.name) or \
                       _check("at", crate.name, zone_obj.name):
                        rx = x0 + 110 + (crates.index(crate) % 3) * 20
                        ry = y0_ - 5
                        draw.rectangle([rx, ry, rx + 16, ry + 16], fill=_CRATE_COLOR,
                                       outline=(100, 80, 40))
                        draw.text((rx + 2, ry + 2), crate.name[:2], fill=(30, 30, 30))

        # Draw hoists
        for i, hoist in enumerate(hoists):
            hx = 20 + i * 70
            draw.rectangle([hx, H - 115, hx + 50, H - 25], fill=_HOIST_COLOR,
                           outline=(120, 60, 180), width=2)
            draw.text((hx + 4, H - 112), hoist.name[:6], fill=(220, 220, 240))

            # Hoist holding crate?
            for crate in crates:
                if _check("lifting", hoist.name, crate.name):
                    draw.rectangle([hx + 5, H - 140, hx + 45, H - 120],
                                   fill=_CRATE_COLOR, outline=(100, 80, 40))
                    draw.text((hx + 8, H - 137), crate.name[:4], fill=(30, 30, 30))

        return img
