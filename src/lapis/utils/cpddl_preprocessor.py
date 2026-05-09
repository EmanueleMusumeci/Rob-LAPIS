"""
CPDDL diagnostic and normalization tools for LAPIS taxonomy-driven error correction.

SIF location: third-party/cpddl/cpddl_latest.sif
Requires: apptainer

Taxonomy mapping
----------------
Tool / flags                                    Taxonomy entry
----------------------------------------------  -----------------------------------------------
--pddl-stop --pedantic                          P0/bracket_imbalance
                                                P1.1/undefined_type
                                                P1.2/predicate_arity_mismatch
--pddl-ce --pddl-domain-out --pddl-stop         P0/conditional_effects  (compile away 'when')
--pddl-domain-out --pddl-stop                   P0/type_union_either, P0/quantified_precondition
--strips-as-py --strips-stop                    P3.1/missing_init_fluent  (forward reachability)
--strips-h2-dump --strips-stop                  P3.1/missing_init_fluent  (mutex-based)
--lmg --lmg-stop                                P2.1/domain_problem_predicate_drift (lifted mutexes)
"""
from __future__ import annotations

import ast
import os
import re
import subprocess
import tempfile
import logging
from dataclasses import dataclass, field
from pathlib import Path

logger = logging.getLogger(__name__)

CPDDL_SIF = Path(__file__).parents[3] / "third-party" / "cpddl" / "cpddl_latest.sif"
_TIMEOUT = 30  # seconds per CPDDL call

# AlfWorld and some planners write (FALSE (predicate args)) in :init to explicitly
# mark fluents as false. CPDDL's STRIPS maker rejects this (it only accepts
# positive ground atoms). Strip these before passing to CPDDL.
# Pattern handles one level of nesting: (FALSE) / (FALSE pred args) / (FALSE (pred args))
_FALSE_ATOM_RE = re.compile(r'\(\s*FALSE\b(?:[^()]*|\([^()]*\))*\)', re.IGNORECASE)


def _strip_false_atoms(pddl_text: str) -> str:
    """Remove (FALSE ...) atoms from :init — CPDDL only accepts positive facts."""
    return _FALSE_ATOM_RE.sub('', pddl_text)


def _write_cleaned_problem(problem_path: str, tmp_dir: str) -> str:
    """Write a CPDDL-compatible copy of the problem with (FALSE ...) atoms removed."""
    text = Path(problem_path).read_text()
    cleaned = _strip_false_atoms(text)
    out = os.path.join(tmp_dir, "problem_clean.pddl")
    Path(out).write_text(cleaned)
    return out


def _available() -> bool:
    return CPDDL_SIF.exists()


def _run(args: list[str], cwd: str | None = None) -> tuple[int, str, str]:
    """Run CPDDL via apptainer. Returns (returncode, stdout, stderr)."""
    if not _available():
        raise FileNotFoundError(f"CPDDL SIF not found: {CPDDL_SIF}")
    cmd = ["apptainer", "run", str(CPDDL_SIF)] + args
    try:
        r = subprocess.run(
            cmd, capture_output=True, text=True,
            timeout=_TIMEOUT, cwd=cwd,
        )
        return r.returncode, r.stdout, r.stderr
    except subprocess.TimeoutExpired:
        return -1, "", f"CPDDL timed out after {_TIMEOUT}s"
    except Exception as e:
        return -1, "", str(e)


# ── P0 / P1 — Parse-time diagnostics ─────────────────────────────────────────

def cpddl_parse_errors(domain_path: str, problem_path: str) -> list[str]:
    """
    P0/bracket_imbalance, P1.1/undefined_type, P1.2/predicate_arity_mismatch.

    Run CPDDL with --pddl-stop --pedantic to get structured parse errors with
    line/column pointers.  Returns a list of error message strings (empty if clean).
    """
    if not _available():
        return []
    with tempfile.TemporaryDirectory() as tmp:
        clean_problem = _write_cleaned_problem(problem_path, tmp)
        _rc, _out, stderr = _run(["--pddl-stop", "--pedantic", domain_path, clean_problem])
    errors = []
    for line in stderr.splitlines():
        if "error" in line.lower() or "fatal" in line.lower():
            errors.append(line.strip())
    if errors:
        logger.info("[CPDDL] parse errors: %s", errors)
    return errors


# ── P0 — Compile ADL features ─────────────────────────────────────────────────

def cpddl_compile_conditional_effects(domain_path: str, problem_path: str) -> str | None:
    """
    P0/conditional_effects.

    Compile away 'when' conditional effects using --pddl-ce.
    Returns the normalised domain PDDL text, or None on failure.
    """
    if not _available():
        return None
    with tempfile.TemporaryDirectory() as tmp:
        clean_problem = _write_cleaned_problem(problem_path, tmp)
        out_domain = os.path.join(tmp, "domain_ce.pddl")
        _rc, _out, stderr = _run([
            "--pddl-ce",
            "--pddl-domain-out", out_domain,
            "--pddl-stop",
            domain_path, clean_problem,
        ])
        if os.path.exists(out_domain):
            return Path(out_domain).read_text()
        logger.warning("[CPDDL] conditional effects compile failed: %s", stderr[:300])
        return None


def cpddl_compile_adl(domain_path: str, problem_path: str) -> tuple[str, str] | None:
    """
    P0/type_union_either, P0/quantified_precondition.

    Compile away ADL features (quantifiers, disjunctions, type unions) to STRIPS.
    Returns (domain_text, problem_text) or None on failure.
    """
    if not _available():
        return None
    with tempfile.TemporaryDirectory() as tmp:
        clean_problem = _write_cleaned_problem(problem_path, tmp)
        out_d = os.path.join(tmp, "domain_strips.pddl")
        out_p = os.path.join(tmp, "problem_strips.pddl")
        _rc, _out, stderr = _run([
            "--pddl-domain-out", out_d,
            "--pddl-problem-out", out_p,
            "--pddl-stop",
            domain_path, clean_problem,
        ])
        if os.path.exists(out_d) and os.path.exists(out_p):
            return Path(out_d).read_text(), Path(out_p).read_text()
        logger.warning("[CPDDL] ADL compile failed: %s", stderr[:300])
        return None


# ── P3.1 — Forward-reachability diagnosis ────────────────────────────────────

@dataclass
class StripsDiagnosis:
    goal_is_unreachable: bool = False
    unreachable_goal_facts: list[str] = field(default_factory=list)
    unreachable_operator_names: list[str] = field(default_factory=list)
    all_facts: list[str] = field(default_factory=list)
    reachable_facts: list[str] = field(default_factory=list)


def cpddl_strips_diagnosis(domain_path: str, problem_path: str) -> StripsDiagnosis | None:
    """
    P3.1/missing_init_fluent.

    Dump grounded STRIPS as Python dict via --strips-as-py, then compute
    forward reachability from init to identify which goal facts are unreachable.

    The Python dict has:
        'fact'  : list[str]  — all ground fact names
        'op'    : list[dict] — each with 'name', 'pre', 'add', 'del' (index sets)
        'init'  : list[int]  — fact indices true in the initial state
        'goal'  : list[int]  — fact indices required by the goal
        'goal_is_unreachable': bool

    We compute the forward reachability closure and diff against goal.
    """
    if not _available():
        return None
    with tempfile.TemporaryDirectory() as tmp:
        clean_problem = _write_cleaned_problem(problem_path, tmp)
        strips_py = os.path.join(tmp, "strips.py")
        _rc, _out, stderr = _run([
            "--strips-as-py", strips_py,
            "--strips-stop",
            domain_path, clean_problem,
        ])
        if not os.path.exists(strips_py):
            # Common causes: ADL quantifiers (exists/forall) produce internal (FALSE)
            # sentinel during grounding; (FALSE ...) atoms in init not yet stripped.
            # Fall back to generic P3.1 hint — not a fatal error.
            logger.debug("[CPDDL] strips-as-py failed (rc=%d): %s", _rc, stderr[:300])
            return None

        try:
            src = Path(strips_py).read_text().strip()
            # CPDDL writes either "var = {...}" or bare "{...}"
            body = src.split("=", 1)[1].strip() if "=" in src else src
            data = ast.literal_eval(body)
        except Exception as e:
            logger.warning("[CPDDL] failed to parse strips.py: %s", e)
            return None

    facts: list[str] = data.get("fact", [])
    ops: list[dict] = data.get("op", [])
    init_ids: set[int] = set(data.get("init", []))
    goal_ids: list[int] = data.get("goal", [])
    goal_unreachable_flag: bool = bool(data.get("goal_is_unreachable", False))

    # Forward reachability (relaxed — delete effects ignored)
    reachable: set[int] = set(init_ids)
    changed = True
    while changed:
        changed = False
        for op in ops:
            pre: set = op.get("pre", set())
            add: set = op.get("add", set())
            if pre.issubset(reachable) and not add.issubset(reachable):
                reachable |= add
                changed = True

    unreachable_goal = [facts[i] for i in goal_ids if i not in reachable]
    unreachable_ops = [
        op["name"] for op in ops
        if not op.get("pre", set()).issubset(reachable)
    ]

    diag = StripsDiagnosis(
        goal_is_unreachable=goal_unreachable_flag or bool(unreachable_goal),
        unreachable_goal_facts=unreachable_goal,
        unreachable_operator_names=unreachable_ops[:20],  # cap for prompt length
        all_facts=facts,
        reachable_facts=[facts[i] for i in sorted(reachable)],
    )
    logger.info(
        "[CPDDL] P3.1 diagnosis: goal_unreachable=%s, missing=%s",
        diag.goal_is_unreachable, diag.unreachable_goal_facts,
    )
    return diag


# ── P3.1 / P2.1 — h² mutex diagnosis ─────────────────────────────────────────

def cpddl_h2_mutexes(domain_path: str, problem_path: str) -> list[tuple[str, str]]:
    """
    P3.1/missing_init_fluent (mutex variant), P2.1/domain_problem_predicate_drift.

    Run h² and return mutex pairs as (fact_a, fact_b) string tuples.
    A goal fact that is mutex with an init fact → structurally unsolvable.
    """
    if not _available():
        return []
    with tempfile.TemporaryDirectory() as tmp:
        clean_problem = _write_cleaned_problem(problem_path, tmp)
        h2_out = os.path.join(tmp, "h2.txt")
        _rc, _out, stderr = _run([
            "--strips-h2-dump", h2_out,
            "--strips-stop",
            domain_path, clean_problem,
        ])
        if not os.path.exists(h2_out):
            logger.debug("[CPDDL] h2-dump failed: %s", stderr[:200])
            return []
        pairs = []
        for line in Path(h2_out).read_text().splitlines():
            parts = re.findall(r'\(([^)]+)\)', line)
            if len(parts) == 2:
                pairs.append((parts[0], parts[1]))
        return pairs


# ── P2.1 — Lifted mutex groups (schema-level type incompatibilities) ──────────

def cpddl_lifted_mutex_groups(domain_path: str, problem_path: str) -> list[str]:
    """
    P2.1/domain_problem_predicate_drift.

    Lifted mutex groups (LMGs) detect schema-level incompatibilities — e.g. a
    predicate used with the wrong object type.  Returns a list of LMG descriptions
    from stderr (CPDDL logs them at INFO level).
    """
    if not _available():
        return []
    with tempfile.TemporaryDirectory() as tmp:
        clean_problem = _write_cleaned_problem(problem_path, tmp)
        _rc, _out, stderr = _run(["--lmg", "--lmg-stop", domain_path, clean_problem])
    groups = [l.strip() for l in stderr.splitlines() if "lmg" in l.lower() or "mutex" in l.lower()]
    return groups


# ── Convenience: format P3.1 hint from diagnosis ─────────────────────────────

def format_p31_hint(diag: StripsDiagnosis) -> str:
    """Format a targeted P3.1 refinement hint from a StripsDiagnosis."""
    if not diag or not diag.goal_is_unreachable:
        return ""
    lines = ["CPDDL REACHABILITY: The following goal facts are unreachable from the initial state:"]
    for f in diag.unreachable_goal_facts[:10]:
        lines.append(f"  • {f}")
    if diag.unreachable_operator_names:
        lines.append(
            f"Actions with unsatisfied preconditions ({len(diag.unreachable_operator_names)} total): "
            + ", ".join(diag.unreachable_operator_names[:5])
        )
    if diag.unreachable_goal_facts:
        lines.append(
            "For each unreachable fact, identify the domain action that should establish it and examine "
            "its preconditions. If a precondition literal is genuinely absent from (:init) (not a logic "
            "error in the action), report it in 'missing_init_facts'. Do NOT add facts to (:init) whose "
            "absence reflects a domain-logic error rather than a truly missing initial condition."
        )
    return "\n".join(lines)
