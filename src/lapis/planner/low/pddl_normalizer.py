"""
PDDL Domain Normalizer — corrective actions for each error class in pddl_taxonomy.py.

Corrective methods (in order applied by normalize_domain()):
  1. fix_empty_effects()          — P0: (:effect ()) → (:effect (and))
  2. flatten_type_hierarchy()     — P1.1: TYPE - PARENT → TYPE (UP detects, regex flattens)
  3. fix_reserved_type_rename()   — P1.1: object → physical-object everywhere (types + predicates + actions)

CPDDL connection (requires ./bin/pddl binary, not currently installed):
  - P0 detection:  ./bin/pddl --pddl-stop --pedantic domain.pddl problem.pddl
  - P0 normalize:  ./bin/pddl --pddl-domain-out d.pddl --pddl-stop domain.pddl problem.pddl
  - P0 cond-eff:   ./bin/pddl --pddl-ce --pddl-domain-out d.pddl --pddl-stop ...
  - P1.1 types:    CPDDL preserves hierarchy (research tool) → must use UP + regex instead

Reference: /DATA/PlanGen/REPAIR_ONTOLOGY.md, pddl_taxonomy.py
"""

import re
import logging

logger = logging.getLogger(__name__)


# ── Helpers ──────────────────────────────────────────────────────────────────

def _find_paren_block(text: str, start: int) -> tuple[int, int]:
    """Return [start, end) of the balanced paren block beginning at start ('(')."""
    depth = 0
    i = start
    while i < len(text):
        if text[i] == '(':
            depth += 1
        elif text[i] == ')':
            depth -= 1
            if depth == 0:
                return start, i + 1
        i += 1
    return start, len(text)


# ── P0: Empty effect bodies ───────────────────────────────────────────────────

def fix_empty_effects(domain_text: str) -> str:
    """
    P0 / empty_effect_body.

    FastDownward's translator rejects ':effect ()' — it requires the effect
    body to be non-empty. Replace with ':effect (and)' which FD accepts as
    an empty conjunction (no-op).

    Corrective method: preprocess_regex
    CPDDL: unclear whether it handles this; empirically use regex.
    """
    fixed = re.sub(r':effect\s*\(\s*\)', ':effect (and)', domain_text)
    if fixed != domain_text:
        count = len(re.findall(r':effect\s*\(\s*\)', domain_text))
        logger.info("[NORMALIZER] P0/empty_effect_body: fixed %d empty :effect clauses", count)
    return fixed


# ── P1.1: Hierarchical type hierarchy ────────────────────────────────────────

def detect_hierarchical_types_up(domain_path: str) -> bool:
    """
    P1.1 / hierarchical_type_hierarchy — detection via UnifiedPlanning.

    UP's PDDLReader supports hierarchical types. After parsing, inspect
    user_types for any type that has a non-None .father attribute.
    Returns True if hierarchy is present.

    CPDDL alternative: ./bin/pddl --pddl-stop --pedantic would NOT catch this
    as an error (CPDDL also supports hierarchical types). UP is the right tool
    here because FD (via UP) raises UnsupportedProblemTypeError: HIERARCHICAL_TYPING.
    """
    try:
        from unified_planning.io import PDDLReader
        reader = PDDLReader()
        problem = reader.parse_problem(domain_path)
        for t in problem.environment.type_manager.user_types:
            if t.father is not None:
                logger.info(
                    "[NORMALIZER] P1.1/hierarchical_type: UP detected '%s - %s'",
                    t.name, t.father.name
                )
                return True
        return False
    except Exception as e:
        logger.debug("[NORMALIZER] UP hierarchy detection failed (non-critical): %s", e)
        return False


def detect_hierarchical_types_regex(domain_text: str) -> bool:
    """Fast regex fallback — looks for ' - WORD' inside the (:types ...) block."""
    m = re.search(r'\(:types\b', domain_text)
    if not m:
        return False
    _, end = _find_paren_block(domain_text, m.start())
    types_block = domain_text[m.start():end]
    return bool(re.search(r'\b\w[\w-]*\s+-\s+\w[\w-]*', types_block))


def flatten_type_hierarchy(domain_text: str) -> str:
    """
    P1.1 / hierarchical_type_hierarchy — corrective action.

    Strips '- PARENTTYPE' from every declaration line inside (:types ...).
    Preserves all type names; only removes the inheritance annotation.

    Example:
        (:types character - entity  obj - entity)
        → (:types character obj)

    UP is used for detection; this regex is the actual fix.
    CPDDL would preserve the hierarchy (it supports hierarchical types).
    """
    match = re.search(r'\(:types\b', domain_text)
    if not match:
        return domain_text

    start = match.start()
    _, end = _find_paren_block(domain_text, start)
    types_block = domain_text[start:end]

    # Remove '[ \t]- WORD' patterns (type-parent annotation)
    flat_block = re.sub(r'[ \t]+-[ \t]+\w[\w-]*', '', types_block)

    if flat_block != types_block:
        logger.info("[NORMALIZER] P1.1/hierarchical_type: flattened type hierarchy in (:types ...)")

    return domain_text[:start] + flat_block + domain_text[end:]


# ── P0: Conditional effects (CPDDL --pddl-ce equivalent) ─────────────────────

def has_conditional_effects(domain_text: str) -> bool:
    """Detect LLM-generated 'when' conditional effects (requires CPDDL --pddl-ce to compile away)."""
    return bool(re.search(r'\(\s*when\b', domain_text, re.IGNORECASE))


# ── P1.1: Reserved type name 'object' ────────────────────────────────────────

def fix_reserved_type_rename(domain_text: str) -> str:
    """
    P1.1 / reserved_type_name_object — rename 'object' → 'physical-object' consistently.

    'object' is PDDL's implicit root type; redeclaring it as a user type confuses
    parsers. LLMs commonly do this. The rename must be applied in three contexts:

      (a) "object -" patterns  (object used as child type before a parent)
      (b) "- object" patterns  (object used as type annotation in parameters)
      (c) standalone in :types block (flat list entry without hierarchy context)

    FD symptom when rename is partial: KeyError: 'physical-object' — types block
    still says 'object' while predicates/actions say 'physical-object'.

    Corrective method: preprocess_regex (P1.1/reserved_type_name_object in taxonomy).
    """
    if 'object' not in domain_text:
        return domain_text

    # Only rename if this file declares a :types block — problem files use
    # '- object' as PDDL's built-in root type (valid; must not be renamed).
    if not re.search(r'\(:types\b', domain_text, re.IGNORECASE):
        return domain_text

    # (a) object followed by " -" (acts as a child type in a hierarchy)
    domain_text = re.sub(r'\bobject\b(?=\s*-)', 'physical-object', domain_text)
    # (b) object preceded by "- " (type annotation in predicate/action parameters)
    domain_text = re.sub(r'(?<=- )\bobject\b', 'physical-object', domain_text)
    # (c) standalone in :types block (flat entries not caught by (a) or (b))
    match = re.search(r'\(:types\b', domain_text, re.IGNORECASE)
    if match:
        _, end = _find_paren_block(domain_text, match.start())
        types_block = domain_text[match.start():end]
        renamed = re.sub(r'(?<!-)\bobject\b', 'physical-object', types_block)
        if renamed != types_block:
            logger.info("[NORMALIZER] P1.1/reserved_type_name_object: renamed 'object' in :types block")
            domain_text = domain_text[:match.start()] + renamed + domain_text[end:]

    return domain_text


# ── Master normalizer ─────────────────────────────────────────────────────────

def normalize_domain(domain_text: str, domain_path: str | None = None) -> str:
    """
    Apply all deterministic normalizations to a generated PDDL domain text.

    Order matters: fix structural issues before type issues.

    Args:
        domain_text:  Raw PDDL domain string (just generated by LLM).
        domain_path:  Optional path to domain file on disk — enables UP-based
                      hierarchical type detection (more precise than regex).

    Returns:
        Normalized PDDL domain string ready for FD / UP planner.
    """
    # P0.1 — Fix empty effect bodies (FD translator rejects '(:effect ())')
    domain_text = fix_empty_effects(domain_text)

    # P1.1 — Rename reserved type 'object' → 'physical-object' consistently
    domain_text = fix_reserved_type_rename(domain_text)

    # P1.1 — Flatten type hierarchy (FD/UP raises HIERARCHICAL_TYPING)
    has_hierarchy = False
    if domain_path:
        has_hierarchy = detect_hierarchical_types_up(domain_path)
    if not has_hierarchy:
        has_hierarchy = detect_hierarchical_types_regex(domain_text)
    if has_hierarchy:
        domain_text = flatten_type_hierarchy(domain_text)

    # Warn about conditional effects (need CPDDL --pddl-ce if present)
    if has_conditional_effects(domain_text):
        logger.warning(
            "[NORMALIZER] P0/conditional_effects: domain contains '(when ...)' — "
            "CPDDL '--pddl-ce' required to compile away, or update domain.nl prompt."
        )

    return domain_text
