"""
AlfWorld name resolution utilities for LAPIS benchmark.

symbol_table: dict mapping clean semantic names → GT coordinate-encoded IDs
  e.g. "soap_bar_1" → "SoapBar_bar__minus_03_dot_57_bar__plus_00_dot_22_bar__plus_00_dot_94"

These clean names are injected into the problem generation prompt so the LLM uses them
consistently, then resolved back to GT IDs before GT-valid VAL validation.
"""

import re
from typing import Optional


def objects_block(symbol_table: dict) -> Optional[str]:
    """
    Return a multi-line string listing all clean object names for LAPIS to use in :objects.

    Passed as `objects_list` to grounded_planning() so the LLM uses these identifiers exactly.
    Always includes `agent1` to match the GT problem's agent declaration.
    Returns None if the table is empty (no injection).
    """
    if not symbol_table:
        return None
    names = sorted(symbol_table.keys())
    # Prepend the GT-canonical agent name so the LLM uses it verbatim
    return "agent1\n" + "\n".join(names)


def resolve_plan(raw_plan: str, symbol_table: dict) -> str:
    """
    Replace every clean semantic name in a plan with its GT coordinate-encoded ID.

    Uses whole-word matching sorted by descending length to prevent partial replacements
    (e.g. "arm_chair_1" must replace before "arm_chair_").
    """
    if not symbol_table or not raw_plan:
        return raw_plan

    resolved = raw_plan
    # Sort longest-first so e.g. "arm_chair_10" replaces before "arm_chair_1"
    for clean_name in sorted(symbol_table.keys(), key=len, reverse=True):
        gt_id = symbol_table[clean_name]
        resolved = re.sub(
            r'(?<![A-Za-z0-9_])' + re.escape(clean_name) + r'(?![A-Za-z0-9_])',
            gt_id,
            resolved,
        )

    return resolved
