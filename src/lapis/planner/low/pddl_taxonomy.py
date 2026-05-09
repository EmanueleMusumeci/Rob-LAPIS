"""
PDDL Domain Error Taxonomy — adapted from CoSTL/PlanGen REPAIR_ONTOLOGY.md
for the LAPIS VirtualHome pipeline.

Each ErrorClass maps to a CorrectionMethod. The taxonomy is used by
pddl_normalizer.py (algorithmic fixes) and the LLM refinement loop
(prompt-based fixes).

Reference: /DATA/PlanGen/REPAIR_ONTOLOGY.md
"""

from dataclasses import dataclass
from enum import StrEnum


class ErrorClass(StrEnum):
    P0_SYNTAX          = "P0"       # Structural — parse failure, empty effect body
    P1_1_TYPES         = "P1.1"     # Declarative — type errors, hierarchical typing
    P1_2_ARITY         = "P1.2"     # Declarative — predicate arity mismatch
    P2_1_HALLUCINATION = "P2.1"     # Grounding — invented objects / predicate names
    P2_2_DRIFT         = "P2.2"     # Metacognitive — goal drift / simplification
    P3_1_PRECONDITION  = "P3.1"     # Semantic — precondition gap (unsolvable)
    P3_2_EFFECT        = "P3.2"     # Semantic — effect divergence (VAL ok, sim fails)
    P4_TEMPORAL        = "P4"       # Temporal — LTL / liveness violations


class CorrectionMethod(StrEnum):
    PREPROCESS_REGEX   = "preprocess_regex"   # Deterministic regex in _preprocess_pddl
    UP_PARSE_FLATTEN   = "up_parse_flatten"   # UP PDDLReader detects → regex flattens
    CPDDL_NORMALIZE    = "cpddl_normalize"    # ./bin/pddl --pddl-stop / normalization pass
    PROMPT_INJECTION   = "prompt_injection"   # Add constraint to domain generation prompt
    LLM_REFINEMENT     = "llm_refinement"     # Feed VAL error back to LLM for repair
    OUT_OF_SCOPE       = "out_of_scope"       # Not handled in current pipeline


@dataclass(frozen=True)
class TaxonomyEntry:
    error_class: ErrorClass
    subtype: str           # fine-grained label
    diagnostic_markers: tuple[str, ...]
    description: str
    correction_method: CorrectionMethod
    correction_note: str


TAXONOMY: list[TaxonomyEntry] = [
    # ── P0: Structural / Syntax ──────────────────────────────────────────────
    TaxonomyEntry(
        error_class=ErrorClass.P0_SYNTAX,
        subtype="empty_effect_body",
        diagnostic_markers=("(:effect ())", "Missing fields", "translate exit code: 31"),
        description=(
            "FD's translator requires :effect to have non-empty content. "
            "LLMs encode no-op actions as (:effect ()) which FD rejects."
        ),
        correction_method=CorrectionMethod.PREPROCESS_REGEX,
        correction_note="Replace ':effect ()' with ':effect (and)' in _preprocess_pddl.",
    ),
    TaxonomyEntry(
        error_class=ErrorClass.P0_SYNTAX,
        subtype="bracket_imbalance",
        diagnostic_markers=(
            "unexpected token", "mismatched parenthesis", "parse error",
            "Could not parse domain file", "Could not parse problem file",
            "Missing ')'", "FD translate failed (rc=31)",
        ),
        description="The PDDL file has unbalanced parentheses — not valid Lisp-like syntax.",
        correction_method=CorrectionMethod.LLM_REFINEMENT,
        correction_note=(
            "VAL reports the offending location; feed error into the refinement prompt."
        ),
    ),
    TaxonomyEntry(
        error_class=ErrorClass.P0_SYNTAX,
        subtype="objects_in_domain",
        diagnostic_markers=("(:objects", "in domain file"),
        description="LLM hallucinated an (:objects ...) block inside the domain file.",
        correction_method=CorrectionMethod.PREPROCESS_REGEX,
        correction_note=(
            "Strip (:objects ...) from domain files — already handled in _preprocess_pddl."
        ),
    ),

    # ── P1.1: Declarative — Types ────────────────────────────────────────────
    TaxonomyEntry(
        error_class=ErrorClass.P1_1_TYPES,
        subtype="hierarchical_type_hierarchy",
        diagnostic_markers=(
            "HIERARCHICAL_TYPING", "character - entity", "obj - entity",
            "UnsupportedProblemTypeError",
        ),
        description=(
            "LLM generates hierarchical types (TYPE - PARENT) which FastDownward does not "
            "support. UP's PDDLReader can parse the domain to confirm hierarchy presence; "
            "the types block must then be flattened to bare type names.\n"
            "CPDDL note: CPDDL also supports hierarchical types (it is a research tool), "
            "so '--pddl-domain-out' preserves the hierarchy rather than flattening it. "
            "CPDDL is NOT the right tool here."
        ),
        correction_method=CorrectionMethod.UP_PARSE_FLATTEN,
        correction_note=(
            "IMPLEMENTED in pddl_normalizer.py: "
            "(1) detect_hierarchical_types_up() uses unified_planning.io.PDDLReader; "
            "inspect UserType.father for any non-None parent. "
            "(2) flatten_type_hierarchy() strips '- PARENT' from (:types ...) block "
            "via balanced-paren walk + regex. "
            "Called automatically from _preprocess_pddl() in pddl_generation.py."
        ),
    ),
    TaxonomyEntry(
        error_class=ErrorClass.P1_1_TYPES,
        subtype="reserved_type_name_object",
        diagnostic_markers=(
            "'object' is a reserved", "redefinition of built-in",
            "KeyError: 'physical-object'", "keyerror",
        ),
        description=(
            "LLM uses 'object' as a user-defined type, conflicting with PDDL's built-in "
            "root type. The rename 'object' → 'physical-object' must be applied in all "
            "three contexts: (a) type annotations ('- object'), (b) child-type patterns "
            "('object -'), and (c) standalone entries in the :types block. A partial rename "
            "(e.g. only predicate signatures, not :types) causes FD's translator to crash "
            "with KeyError: 'physical-object' because the type is referenced but not declared."
        ),
        correction_method=CorrectionMethod.PREPROCESS_REGEX,
        correction_note=(
            "IMPLEMENTED in pddl_normalizer.fix_reserved_type_rename(): renames 'object' "
            "consistently in (a) '- object' parameter annotations, (b) 'object -' child "
            "patterns, and (c) standalone entries in the :types block via balanced-paren "
            "extraction. Called from normalize_domain() before flatten_type_hierarchy()."
        ),
    ),
    TaxonomyEntry(
        error_class=ErrorClass.P1_1_TYPES,
        subtype="undefined_type",
        diagnostic_markers=("undefined type", "type not declared", "unknown type"),
        description="A predicate or action parameter references a type not in (:types ...).",
        correction_method=CorrectionMethod.LLM_REFINEMENT,
        correction_note="Feed VAL type error into the domain refinement prompt.",
    ),

    # ── P1.2: Declarative — Arity ────────────────────────────────────────────
    TaxonomyEntry(
        error_class=ErrorClass.P1_2_ARITY,
        subtype="predicate_arity_mismatch",
        diagnostic_markers=("wrong number of arguments", "undeclared predicate", "arity"),
        description=(
            "A predicate is called with more or fewer arguments than its declaration."
        ),
        correction_method=CorrectionMethod.LLM_REFINEMENT,
        correction_note=(
            "VAL reports the offending predicate; feed into refinement prompt with the "
            "correct arity from the generated (:predicates ...) block."
        ),
    ),

    # ── P2.1: Grounding — Hallucination ──────────────────────────────────────
    TaxonomyEntry(
        error_class=ErrorClass.P2_1_HALLUCINATION,
        subtype="invented_predicate",
        diagnostic_markers=("object not found", "non-canonical name", "predicate not in domain",
                            "Not able to handle", "undeclared predicate"),
        description=(
            "The generated problem uses a predicate name not declared in the generated domain "
            "(e.g. char_inside when the domain declares inside). Distinct from constant_name_drift: "
            "here the LLM invents a new predicate symbol; there, it uses valid predicates but "
            "with wrong GT object identifiers."
        ),
        correction_method=CorrectionMethod.LLM_REFINEMENT,
        correction_note=(
            "Feed the UP/VAL error ('Not able to handle: (pred ...)') + the domain (:predicates) "
            "block back to the problem refinement prompt. inject_domain_schema=True should prevent "
            "this but does not reliably do so across domain+problem generation boundaries."
        ),
    ),
    TaxonomyEntry(
        error_class=ErrorClass.P2_1_HALLUCINATION,
        subtype="domain_problem_predicate_drift",
        diagnostic_markers=("is not well-formed", "UPTypeError", "wrong argument type",
                            "type mismatch in init"),
        description=(
            "The generated problem uses a predicate with the correct name but wrong argument "
            "types, because domain and problem generation are independent LLM calls. "
            "Example (VirtualHome run 2): domain declares inside(?c-character, ?r-obj); "
            "problem generates (inside couch home_office) — both args are obj, not character+obj. "
            "The correct predicate for obj-in-room is inside_obj(?o-obj, ?c-obj). "
            "inject_domain_schema=True is insufficient to prevent this drift."
        ),
        correction_method=CorrectionMethod.LLM_REFINEMENT,
        correction_note=(
            "UP reports 'expression is not well-formed' with line/col. Feed UP error + "
            "the full domain (:predicates) block with typed signatures into problem refinement. "
            "If drift persists across 3 refinements, consider adding explicit predicate-usage "
            "examples to the problem generation prompt (PROMPT_INJECTION)."
        ),
    ),
    TaxonomyEntry(
        error_class=ErrorClass.P2_1_HALLUCINATION,
        subtype="constant_name_drift",
        diagnostic_markers=("GT validation failed", "symbol not in GT problem",
                            "AlarmClock_2", "lifted symbol"),
        description=(
            "The LLM uses internally consistent names that diverge from GT coordinate-encoded "
            "identifiers because the NL-barrier withholds GT IDs. VAL self-validation passes; "
            "GT validation fails. LLM refinement cannot fix this — the LLM has no access to "
            "GT IDs. Requires lifted-symbol grounding + bijective post-hoc resolution. "
            "AlfWorld-specific as implemented; VH would need a parallel vh_name_remapper.py."
        ),
        correction_method=CorrectionMethod.PROMPT_INJECTION,
        correction_note=(
            "IMPLEMENTED (AlfWorld) in data/alfworld/name_remapper.py: "
            "inject canonical symbol list via objects_list kwarg in generate_problem(); "
            "resolve plan post-hoc via bijective symbol table before GT VAL check. "
            "VH: VH plan aliaser (plan_aliaser.py) handles action name drift; "
            "object identifier drift is less severe since VH uses human-readable IDs."
        ),
    ),

    # ── P2.2: Metacognitive — Goal Drift ─────────────────────────────────────
    TaxonomyEntry(
        error_class=ErrorClass.P2_2_DRIFT,
        subtype="goal_simplification",
        diagnostic_markers=("Goal Mismatch", "Integrity Auditor: FAIL"),
        description="The LLM simplified the goal to bypass planning difficulty.",
        correction_method=CorrectionMethod.LLM_REFINEMENT,
        correction_note=(
            "Re-anchor to the original task NL description; re-generate problem from scratch."
        ),
    ),

    # ── P3.1: Semantic — Precondition Gap ────────────────────────────────────
    TaxonomyEntry(
        error_class=ErrorClass.P3_1_PRECONDITION,
        subtype="missing_init_fluent",
        diagnostic_markers=(
            "action inapplicable", "unsolvable", "precondition not met",
            "no solution!", "initial state is a dead end", "dead end",
            "completely explored state space",
        ),
        description=(
            "An action's precondition is not satisfied in the initial state — "
            "likely a missing fluent in (:init ...)."
        ),
        correction_method=CorrectionMethod.LLM_REFINEMENT,
        correction_note=(
            "Feed planner output (unsolvable) + the precondition gap to the problem "
            "refinement prompt; ask to add the missing init fluent."
        ),
    ),

    # ── P3.2: Semantic — Effect Divergence ───────────────────────────────────
    TaxonomyEntry(
        error_class=ErrorClass.P3_2_EFFECT,
        subtype="effect_divergence",
        diagnostic_markers=("VAL: valid", "Sim: failure", "State Divergence"),
        description=(
            "The PDDL plan is VAL-valid but fails in the simulator because "
            "the action effects do not match ground-truth simulator transitions."
        ),
        correction_method=CorrectionMethod.LLM_REFINEMENT,
        correction_note=(
            "Use simulator feedback (GT-valid check / execution trace) to refine "
            "the domain action effects. Out of scope without Unity runtime."
        ),
    ),

    # ── P4: Temporal / Liveness ──────────────────────────────────────────────
    TaxonomyEntry(
        error_class=ErrorClass.P4_TEMPORAL,
        subtype="ltl_violation",
        diagnostic_markers=("LTL violation", "Trace end reached without goal"),
        description="Plan is locally valid but violates global temporal/safety constraints.",
        correction_method=CorrectionMethod.OUT_OF_SCOPE,
        correction_note="Out of scope for VirtualHome — no LTL constraints in this benchmark.",
    ),
]

# Fast lookup: diagnostic marker → TaxonomyEntry
_MARKER_INDEX: dict[str, TaxonomyEntry] = {}
for _entry in TAXONOMY:
    for _marker in _entry.diagnostic_markers:
        _MARKER_INDEX[_marker.lower()] = _entry


def classify(text: str) -> TaxonomyEntry | None:
    """Return the best-matching TaxonomyEntry for a diagnostic message, or None."""
    text_lower = text.lower()
    for marker, entry in _MARKER_INDEX.items():
        if marker in text_lower:
            return entry
    return None


def detect_all(text: str) -> list[TaxonomyEntry]:
    """Return all TaxonomyEntries whose diagnostic_markers appear in text (deduplicated)."""
    text_lower = text.lower()
    seen: set[str] = set()
    found: list[TaxonomyEntry] = []
    for marker, entry in _MARKER_INDEX.items():
        key = f"{entry.error_class}/{entry.subtype}"
        if key not in seen and marker in text_lower:
            seen.add(key)
            found.append(entry)
    return found


# Targeted refinement guidance for each error class / correction method.
# These are generic enough to apply across domains; domain-specific details
# come from the VAL/FD error text already present in the refinement prompt.
_REFINEMENT_HINTS: dict[str, str] = {
    "P1.1/hierarchical_type_hierarchy": (
        "TAXONOMY P1.1: FastDownward requires a flat type hierarchy. "
        "Remove all '- PARENT' annotations from (:types ...) — list bare type names only."
    ),
    "P1.1/reserved_type_name_object": (
        "TAXONOMY P1.1: 'object' is PDDL's built-in root type. "
        "Rename every occurrence to a domain-specific name (e.g. 'physical-object') "
        "consistently in (:types), (:predicates), and all action :parameters."
    ),
    "P1.1/undefined_type": (
        "TAXONOMY P1.1: Every type used in predicate or action parameter signatures "
        "must appear in (:types). Add the missing type or replace it with a declared one."
    ),
    "P1.2/predicate_arity_mismatch": (
        "TAXONOMY P1.2: The predicate is called with the wrong number of arguments. "
        "Check the (:predicates) declaration and use exactly that many arguments in every call."
    ),
    "P2.1/invented_predicate": (
        "TAXONOMY P2.1: The problem uses a predicate name not declared in the domain. "
        "Use ONLY predicate names from the domain's (:predicates) block — do not invent new ones."
    ),
    "P2.1/domain_problem_predicate_drift": (
        "TAXONOMY P2.1: The predicate exists in the domain but is called with wrong argument types. "
        "Re-read the domain (:predicates) signatures and match argument types exactly."
    ),
    "P2.1/constant_name_drift": (
        "TAXONOMY P2.1: Use only the object names provided in the OBJECT NAMES list. "
        "Do not invent alternative identifiers or use abbreviated names."
    ),
    "P2.2/goal_simplification": (
        "TAXONOMY P2.2: The goal must match the original task description exactly. "
        "Do not simplify or drop conditions from the goal to make planning easier."
    ),
    "P3.1/missing_init_fluent": (
        "TAXONOMY P3.1: The planner found no solution. Examine each action needed to achieve the "
        "goal and identify which specific precondition fluent is not satisfied. For each such action, "
        "determine the exact PDDL literal (predicate + args) that must be true to make it applicable. "
        "Report these as 'missing_init_facts' in the JSON (e.g. [\"(plugged_in microwave_1)\"]). "
        "Do NOT propose weakening, rewriting, or simplifying action preconditions — action signatures "
        "must remain unchanged. Only report a fact as missing if it is genuinely absent from (:init), "
        "not if the fix belongs in the domain action logic."
    ),
}


def refinement_hint(entries: list[TaxonomyEntry]) -> str:
    """Return a block of targeted guidance to prepend to the LLM refinement prompt.

    Only entries with LLM_REFINEMENT or PROMPT_INJECTION correction methods are included
    (PREPROCESS_REGEX errors are fixed algorithmically before this is called).
    Returns an empty string if no actionable hints exist.
    """
    lines: list[str] = []
    for e in entries:
        if e.correction_method in (CorrectionMethod.LLM_REFINEMENT, CorrectionMethod.PROMPT_INJECTION):
            key = f"{e.error_class}/{e.subtype}"
            hint = _REFINEMENT_HINTS.get(key)
            if hint:
                lines.append(hint)
    if not lines:
        return ""
    return "\n\nKNOWN ERROR PATTERNS DETECTED — apply these fixes:\n" + "\n".join(f"• {l}" for l in lines)
