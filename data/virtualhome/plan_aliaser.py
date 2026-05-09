"""
Plan aliaser for VirtualHome: maps LLM-generated action names to GT canonical names.

Used before GT-valid VAL check so that semantically correct plans with
non-canonical action names can still be validated against the GT domain.
"""

import re
from dataclasses import dataclass
from typing import Optional, List


@dataclass
class AliasRule:
    """Map one or more induced names to a GT canonical name."""
    canonical: str
    aliases: List[str]


# GT canonical action names (from virtualhome.pddl)
_GT_ACTIONS = {
    "walk_into", "walk_towards", "turn_to", "find", "grab",
    "put_on", "put_inside", "put_inside_no_open",
    "open_obj", "close_obj",
    "switch_on", "switch_off", "plug_in",
    "sit", "standup", "lie",
    "drink", "drink_from_recipient", "read", "eat",
    "wash", "pour", "drop", "look_at",
}

_RULES: List[AliasRule] = [
    AliasRule("walk_into",           ["walk_in", "enter", "go_into", "move_into"]),
    AliasRule("walk_towards",        ["walk_to", "walk", "move_to", "go_to", "approach", "navigate_to", "move_towards"]),
    AliasRule("turn_to",             ["turn", "face", "turn_towards", "face_towards"]),
    AliasRule("grab",                ["pick_up", "pickup", "take", "grasp", "pick", "get"]),
    AliasRule("put_on",              ["place_on", "put", "place", "put_onto", "set_on", "set_onto"]),
    AliasRule("put_inside",          ["put_in", "place_inside", "insert", "place_in", "put_into"]),
    AliasRule("put_inside_no_open",  ["put_in_closed", "insert_no_open", "place_no_open"]),
    AliasRule("open_obj",            ["open"]),
    AliasRule("close_obj",           ["close"]),
    AliasRule("switch_on",           ["turn_on", "switch on", "power_on", "activate"]),
    AliasRule("switch_off",          ["turn_off", "switch off", "power_off", "deactivate"]),
    AliasRule("plug_in",             ["connect", "plug", "plugin"]),
    AliasRule("standup",             ["stand_up", "stand", "get_up", "rise"]),
    AliasRule("lie",                 ["lie_down", "sleep", "lay_down", "lay"]),
    AliasRule("drink_from_recipient",["drink_from", "drink_recipient"]),
    AliasRule("drop",                ["release", "put_down", "set_down"]),
    AliasRule("look_at",             ["look", "watch", "observe", "stare_at"]),
]

# Build lookup: alias → canonical (case-insensitive)
_ALIAS_MAP: dict = {}
for rule in _RULES:
    for alias in rule.aliases:
        _ALIAS_MAP[alias.lower()] = rule.canonical


def _alias_action_name(name: str) -> str:
    if name.lower() in _GT_ACTIONS:
        return name  # already canonical
    return _ALIAS_MAP.get(name.lower(), name)  # pass-through if unknown


def alias_vh_plan(plan_text: str, enabled: bool = True) -> str:
    """Alias action names in a VAL-format plan to GT canonical names."""
    if not enabled or not plan_text:
        return plan_text

    out_lines = []
    for line in plan_text.splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith(";"):
            out_lines.append(line)
            continue

        # VAL plan format: (action_name arg1 arg2 ...) or with cost annotation
        m = re.match(r'^(\s*)\((\S+)(.*?)\)\s*(;.*)?$', line)
        if m:
            indent, action, rest, comment = m.group(1), m.group(2), m.group(3), m.group(4) or ""
            canonical = _alias_action_name(action)
            out_lines.append(f"{indent}({canonical}{rest}){' ' + comment if comment else ''}")
        else:
            out_lines.append(line)

    return "\n".join(out_lines)
