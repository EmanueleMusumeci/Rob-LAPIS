import json
import os
import re
from pathlib import Path
from src.lapis.agents.agent import Agent
from src.lapis.logger_cfg import logger

# ─── Issue Classification ───────────────────────────────────────────────────

PRIORITY_ORDER = ['P0_SYNTAX', 'P1_TYPE', 'P1_PREDICATE', 'P2_ACTION', 'P3_OTHER']

def classify_issue(issue_text: str) -> str:
    """
    Classify a PDDL issue into a priority category using keyword matching.
    Issues are fixed in priority order: syntax first, then types, etc.
    """
    text = issue_text.lower()

    # Phase 1: Unambiguous syntax errors
    if any(k in text for k in ['parenthes', 'bracket', 'parse error', 'syntax error']):
        return 'P0_SYNTAX'

    # Phase 2: Domain-specific categories
    if any(k in text for k in ['predicate', 'arity', 'fluent', 'wrong number of arguments']):
        return 'P1_PREDICATE'
    if any(k in text for k in ['type', 'undefined type', 'undeclared', 'not declared', 'object']):
        return 'P1_TYPE'
    if any(k in text for k in ['precondition', 'effect', 'action', 'operator']):
        return 'P2_ACTION'

    # Phase 3: Ambiguous syntax-ish terms
    if any(k in text for k in ['unexpected', 'expected', 'invalid', 'malformed']):
        return 'P0_SYNTAX'

    return 'P3_OTHER'


# Map taxonomy error_class → priority bucket used for bundled correction.
_TAX_CLASS_TO_BUCKET = {
    'P0':   'P0_SYNTAX',
    'P1.1': 'P1_TYPE',
    'P1.2': 'P1_PREDICATE',
    'P2.1': 'P2_ACTION',
    'P2.2': 'P2_ACTION',
    'P3.1': 'P3_OTHER',
    'P3.2': 'P3_OTHER',
}


def _bucket_for_issue(item: dict, taxonomy_entries) -> str:
    """Priority bucket for an issue. Prefer taxonomy class; fall back to keyword classifier."""
    if taxonomy_entries:
        cls = getattr(taxonomy_entries[0], "error_class", None)
        if cls in _TAX_CLASS_TO_BUCKET:
            return _TAX_CLASS_TO_BUCKET[cls]
    return classify_issue(item.get("issue", "") if isinstance(item, dict) else str(item))


class IssueStats:
    """
    Tracks issue classification counts and resolution rates across a planning call.
    Surfaced in pipeline logs and summary.
    """
    def __init__(self):
        self.counts: dict[str, int] = {}
        self.resolved: dict[str, int] = {}

    def record(self, category: str, was_resolved: bool):
        self.counts[category] = self.counts.get(category, 0) + 1
        if was_resolved:
            self.resolved[category] = self.resolved.get(category, 0) + 1

    def summary(self) -> str:
        """Human-readable summary for logs."""
        if not self.counts:
            return "Issue Resolution Statistics: No issues recorded."
        lines = ["Issue Resolution Statistics:"]
        for cat in PRIORITY_ORDER:
            if cat in self.counts:
                total = self.counts[cat]
                res = self.resolved.get(cat, 0)
                rate = (res / total * 100) if total else 0
                lines.append(f"  {cat}: {res}/{total} resolved ({rate:.0f}%)")
        # Any categories not in PRIORITY_ORDER
        for cat in sorted(self.counts):
            if cat not in PRIORITY_ORDER:
                total = self.counts[cat]
                res = self.resolved.get(cat, 0)
                rate = (res / total * 100) if total else 0
                lines.append(f"  {cat}: {res}/{total} resolved ({rate:.0f}%)")
        return '\n'.join(lines)

def _strip_domain_constants_from_problem(problem_pddl: str, domain_pddl: str) -> str:
    """
    Remove domain :constants from the problem's :objects block.

    Domain constants are pre-declared objects; re-declaring them in :objects
    causes FD translate errors (P0/constant_redeclaration).  Also deduplicates
    any object name that appears more than once in :objects.
    """
    m = re.search(r'\(:constants\s+(.*?)\)', domain_pddl, re.DOTALL | re.IGNORECASE)
    if not m:
        return problem_pddl

    const_block = m.group(1)
    constant_names: set[str] = set()
    for segment in re.split(r'\s+-\s+\w[\w-]*', const_block):
        for tok in segment.split():
            if tok and not tok.startswith(';'):
                constant_names.add(tok)

    if not constant_names:
        return problem_pddl

    # Locate the :objects block via balanced-paren scan
    obj_match = re.search(r'\(:objects', problem_pddl, re.IGNORECASE)
    if not obj_match:
        return problem_pddl

    start = obj_match.start()
    depth = 0
    end = start
    for i in range(start, len(problem_pddl)):
        if problem_pddl[i] == '(':
            depth += 1
        elif problem_pddl[i] == ')':
            depth -= 1
            if depth == 0:
                end = i + 1
                break

    inner = problem_pddl[start + len('(:objects'):end - 1]

    seen: set[str] = set()
    filtered_lines: list[str] = []
    for line in inner.split('\n'):
        stripped = line.strip()
        if not stripped or stripped.startswith(';'):
            filtered_lines.append(line)
            continue
        # Match "name1 name2 ... - type" group
        type_match = re.match(r'^([\w][\w\-\s]*?)\s*-\s*([\w][\w-]*)$', stripped)
        if type_match:
            names = type_match.group(1).split()
            type_name = type_match.group(2)
            remaining = []
            for n in names:
                if n in constant_names or n in seen:
                    continue
                seen.add(n)
                remaining.append(n)
            if remaining:
                indent = len(line) - len(line.lstrip())
                filtered_lines.append(' ' * indent + ' '.join(remaining) + ' - ' + type_name)
        else:
            filtered_lines.append(line)

    new_block = '(:objects' + '\n'.join(filtered_lines) + ')'
    return problem_pddl[:start] + new_block + problem_pddl[end:]


# ── Per-issue taxonomy fix map ────────────────────────────────────────────────
# Maps PREPROCESS_REGEX entry subtypes to the specific normalizer function.
# Only the matched function is applied — not the full normalize_domain() stack.
def _get_entry_fix_map():
    from src.lapis.planner.low.pddl_normalizer import (
        fix_empty_effects, fix_reserved_type_rename, flatten_type_hierarchy,
    )
    return {
        "empty_effect_body":        fix_empty_effects,
        "reserved_type_name_object": fix_reserved_type_rename,
        "hierarchical_type_hierarchy": flatten_type_hierarchy,
        # objects_in_domain is handled inside _preprocess_pddl already
    }


def _classify_issue_with_llm(issue_text: str, solution_text: str, agent) -> list:
    """LLM judge: map a natural-language diagnosed issue to taxonomy entries.

    Used in the per-issue correction loop where the trigger is an LLM-generated
    sentence, not a structured planner error.  String matching is inappropriate
    there; an LLM judge can reason semantically over the issue description.

    Returns a list of TaxonomyEntry objects (may be empty if no match).
    """
    from src.lapis.planner.low.pddl_taxonomy import TAXONOMY, CorrectionMethod

    actionable = [e for e in TAXONOMY if e.correction_method != CorrectionMethod.OUT_OF_SCOPE]
    taxonomy_summary = "\n".join(
        f'- "{e.subtype}" ({e.error_class}): {e.description.splitlines()[0][:120]}'
        for e in actionable
    )
    system_prompt = (
        "You are a PDDL error classifier. Given a diagnosed PDDL issue and its proposed solution, "
        "identify which entries from a known error taxonomy clearly apply. "
        "Be conservative — only select entries with direct evidence in the text."
    )
    user_prompt = (
        f"Issue: {issue_text}\n"
        f"Proposed solution: {solution_text}\n\n"
        f"Taxonomy:\n{taxonomy_summary}\n\n"
        "Return a JSON list of matched subtype strings, e.g. [\"empty_effect_body\"]. "
        "Return [] if none clearly match. Output JSON only."
    )
    try:
        response = agent.llm_call(system_prompt, user_prompt)
        clean = response.strip().replace("```json", "").replace("```", "").strip()
        matched_subtypes = json.loads(clean)
        subtype_map = {e.subtype: e for e in TAXONOMY}
        return [subtype_map[s] for s in matched_subtypes if s in subtype_map]
    except Exception:
        return []


def _preprocess_pddl(pddl_str):
    """
    Last-resort cleanup of non-standard PDDL constructs that break UP/FD parsers.

    Universal fixes only — no domain-specific assumptions.
    Error taxonomy: see src/lapis/planner/low/pddl_taxonomy.py
    Normalizer:     see src/lapis/planner/low/pddl_normalizer.py
    """
    # Replace -(either type1 type2 ...) with -<first_type>
    def replace_either(m):
        return f'- {m.group(1).strip().split()[0]}'
    pddl_str = re.sub(r'-\s*\(\s*either\s+([^)]+)\)', replace_either, pddl_str)

    # Strip (:objects ...) hallucinated inside domain files
    if '(define (domain' in pddl_str:
        pddl_str = re.sub(r'\(\s*:objects\b[^)]*\)', '', pddl_str, flags=re.IGNORECASE)

        # Rename `up` to `up_dir` to avoid collision with `unified_planning` (up)
        # Note: 'object' → 'physical-object' rename is handled by normalize_domain()
        #       (P1.1/reserved_type_name_object in pddl_normalizer.py)
        pddl_str = re.sub(r'\bup\b', 'up_dir', pddl_str)

    # P0/empty_effect_body + P1.1/hierarchical_type_hierarchy (taxonomy-driven)
    from src.lapis.planner.low.pddl_normalizer import normalize_domain
    pddl_str = normalize_domain(pddl_str)

    # If this is a problem file truncated before (:goal ...), add a minimal placeholder
    # so FD's parser doesn't crash with StopIteration. Refinement will fix the real goal.
    if '(define (problem' in pddl_str and '(:goal' not in pddl_str:
        logger.warning("[PAREN_BALANCE] problem truncated before (:goal) — inserting placeholder goal")
        pddl_str = pddl_str.rstrip().rstrip(')').rstrip() + '\n  (:goal (and))\n)\n'

    # P0/bracket_imbalance: if the LLM truncated its output (hits token limit), the
    # file ends mid-expression with unclosed parens. Two-step fix:
    #   1. Strip any trailing (:action ...) block that is missing :effect (incomplete).
    #   2. Append the exact number of missing close parens.
    # This is deterministic and does not alter any declared PDDL semantics.
    if '(define (domain' in pddl_str:
        # Step 1: detect and drop the last action if it lacks :effect
        action_starts = [m.start() for m in re.finditer(r'\(:action\b', pddl_str)]
        if action_starts:
            last_action_text = pddl_str[action_starts[-1]:]
            if ':effect' not in last_action_text:
                logger.warning(
                    "[PAREN_BALANCE] last action is truncated (no :effect) — removing it"
                )
                pddl_str = pddl_str[:action_starts[-1]].rstrip()

    # Step 2: balance remaining parens
    _open = pddl_str.count('(')
    _close = pddl_str.count(')')
    if _open > _close:
        missing = _open - _close
        logger.warning(
            "[PAREN_BALANCE] domain truncated — appending %d missing ')' (open=%d close=%d)",
            missing, _open, _close,
        )
        pddl_str = pddl_str.rstrip() + '\n' + ')' * missing + '\n'

    return pddl_str

from src.lapis.utils.vector_db import VectorDB
from src.lapis.planner.low.pddl_verification import VALVerifier
from src.lapis.planner.low.heuristics import check_pddl_heuristics

def prioritize_issues(val_issues, heuristic_warnings):
    """
    Filter and prioritize issues based on dependency tree.
    Order: 
    1. Syntax/Parse Errors (Blocker)
    2. Domain declaration issues (Types/Predicates)
    3. Problem consistency (Objects/Init)
    4. Goal/Logic issues
    """
    all_issues = []
    
    # Convert heuristic warnings to issues
    for w in heuristic_warnings:
        all_issues.append(f"HEURISTIC: {w}")
        
    all_issues.extend(val_issues)
    
    # Simple Keyword Classification
    syntax_keywords = ["syntax", "parse", "expected", "parenthesis"]
    domain_keywords = ["undeclared predicate", "undeclared type", "unknown type"]
    problem_keywords = ["object", "init", "unknown literal"]
    goal_keywords = ["goal", "unbound", "quantifier"]
    
    syntax_issues = [i for i in all_issues if any(k in i.lower() for k in syntax_keywords)]
    if syntax_issues:
        return syntax_issues[:2] # Return only top syntax issues to avoid noise
        
    domain_issues = [i for i in all_issues if any(k in i.lower() for k in domain_keywords)]
    if domain_issues:
        return domain_issues
        
    # If no fundamental domain issues, return problem/goal issues
    # But prioritize "Heuristic" warnings as they are high-confidence generic fixes
    heuristic_issues = [i for i in all_issues if i.startswith("HEURISTIC:")]
    
    # Return Heuristics + whatever VAL specifically complains about regarding problem/goal
    remaining = [i for i in all_issues if i not in syntax_issues and i not in domain_issues]
    
    # Sort: Heuristics first
    remaining.sort(key=lambda x: 0 if x.startswith("HEURISTIC:") else 1)
    
    return remaining##########
# DOMAIN #
##########

def extract_schema(domain_pddl: str) -> dict:
    """
    Dynamically extract types, predicates, and constants from a PDDL domain string.
    Used to build schema-aware prompts for problem generation (Approach A).
    Returns a dict with keys: types (list of str), predicates (list of str signatures),
    constants (list of 'name - type' strings).
    """
    schema = {"types": [], "predicates": [], "constants": []}

    # Types: parse segments separated by " - parent"
    m = re.search(r'\(:types\s+(.*?)\)', domain_pddl, re.DOTALL | re.IGNORECASE)
    if m:
        for segment in re.split(r'\s+-\s+\S+', m.group(1)):
            for t in segment.split():
                if t and not t.startswith(';'):
                    schema["types"].append(t)
        # Also include parent types
        for parent in re.findall(r'-\s+(\S+)', m.group(1)):
            if parent not in schema["types"]:
                schema["types"].append(parent)

    # Constants
    m = re.search(r'\(:constants\s+(.*?)\)', domain_pddl, re.DOTALL | re.IGNORECASE)
    if m:
        schema["constants"] = [
            f"{name} - {ctype}"
            for name, ctype in re.findall(r'(\w[\w-]*)\s*-\s*(\w[\w-]*)', m.group(1))
        ]

    # Predicates: extract from (:predicates ...) block
    pred_block = re.search(r'\(:predicates(.*?)(?=\s*\(:action|\s*\)$)', domain_pddl, re.DOTALL | re.IGNORECASE)
    if not pred_block:
        pred_block = re.search(r'\(:predicates(.*?)\)', domain_pddl, re.DOTALL | re.IGNORECASE)
    if pred_block:
        for pm in re.finditer(r'\((\w[\w-]*)([^)]*)\)', pred_block.group(1)):
            name = pm.group(1)
            params = re.findall(r'\?(\w[\w-]*)\s*-\s*(\w[\w-]*)', pm.group(2))
            if params:
                sig = f"({name} {' '.join(f'?{v} - {t}' for v, t in params)})"
            else:
                sig = f"({name})"
            schema["predicates"].append(sig)

    return schema


def _format_schema_block(schema: dict) -> str:
    """Format extracted schema as a prompt constraint block."""
    lines = []
    if schema["types"]:
        lines.append(f"VALID TYPES: {', '.join(sorted(set(schema['types'])))}")
        lines.append("  Only use these in (:objects). Do NOT invent new types.")
    if schema["constants"]:
        lines.append(f"CONSTANTS (already in domain — do NOT re-declare in :objects):")
        for c in schema["constants"]:
            lines.append(f"  {c}")
    if schema["predicates"]:
        lines.append("PREDICATE SIGNATURES — match arities exactly:")
        for p in schema["predicates"]:
            lines.append(f"  {p}")
    return "\n".join(lines)


def generate_domain(
    domain_file_path,
    domain_description,
    agent: Agent,
    logs_dir=None,
    environment=None,       # kept for signature compatibility
    goal_file_path=None,    # kept for signature compatibility
    clean_domain_prompt=True,
    temperature=0.1,
):
    """
    Generate a PDDL domain file from a natural-language description.

    Ablation axis — clean_domain_prompt:
      True  (Approach A): minimal prompt, no hardcoded domain-specific hints.
      False (baseline):   legacy prompt with hardcoded navigation/agent hints.
    """
    if clean_domain_prompt:
        prompt = f"""Role: You are an expert PDDL domain generator.
Task: Generate a PDDL domain file from the action descriptions below.

INSTRUCTIONS:
- Model ONLY what is described. Do NOT add types, predicates, or actions not mentioned.
- DO NOT use `object` as a type name — it is PDDL's reserved root type.
- EVERY variable (?x) in a precondition/effect MUST be an :parameters variable.
- Use PURE STRIPS effects ONLY — no `when`, `forall`, or `exists` in :effect blocks.
  Every effect must be a simple add or delete of a single grounded predicate.
- The domain must be generic and reusable across problem instances.

DOMAIN DESCRIPTION:
{domain_description}"""
    else:
        # Legacy baseline prompt (hardcoded navigation hints — kept for ablation)
        prompt = f"""
    Role: You are an excellent PDDL domain generator.
    Task: Given a description of the planning domain (actions, preconditions, and effects), generate a PDDL domain file.

    STRICT INSTRUCTIONS:
    - ONLY use the logic provided in the DOMAIN DESCRIPTION below.
    - DO NOT include any goal-specific or problem-specific logic.
    - The domain should be generic and reusable.
    - CRITICAL: NO Type Redundancy: DO NOT declare `agent` as a type. It is already provided as a constant.
    - CRITICAL: DO NOT use the type name `object`, as it is a reserved root type in PDDL. Use `item` instead for physical objects like balls, boxes, or keys.
    - CRITICAL: For navigation domains, use exactly these predicate signatures:
        (in-room ?e - entity ?r - room)
        (in-front-of ?e - entity)
        (holding ?x - item)
        (empty-hands)
        (unlocked ?d - door)
        (visited ?r - room)
        (blocking ?x - item ?d - door)
    - CRITICAL: EVERY variable (starting with `?`) used in an effect or precondition MUST be either an action parameter or bound by a quantifier (e.g., `forall` or `exists`).
    - CRITICAL: DO NOT use `exists` in action effects. Standard PDDL only supports `forall` quantifier in effects. If you need to clear a state for all entities, use `(forall (?any - entity) (not (in-front-of ?any)))`.
    - CRITICAL: If an action involves a variable that is not a parameter (e.g. checking if *any* object is blocking a door), you MUST use a quantifier. Example: `(not (exists (?x - item) (blocking ?x ?d)))`.
    - CRITICAL: You MUST declare `agent - entity` in the `(:constants ...)` block of the domain.

    DOMAIN DESCRIPTION:
    {domain_description}
    """

    user_prompt = "Generate the PDDL PDDL domain within <DOMAIN></DOMAIN> tags."
    
    # Fairness Guard: Ensure we aren't passing a PDDL file path as domain description
    if domain_description and domain_description.endswith(".pddl"):
         raise ValueError(f"FAIRNESS VIOLATION: domain_description appears to be a PDDL file path ('{domain_description}'). In NL-to-PDDL mode, it must be a string or NL content.")

    print(f"DEBUG: Calling agent.llm_call for domain generation (NL-based)...")
    answer = agent.llm_call(prompt, user_prompt, temperature=temperature)

    _save_prompt_response(
        prompt=f"{prompt}\n\n{user_prompt}",
        response=answer,
        prefix="strict_pddl_domain_generation",
        suffix="",
        output_dir=logs_dir
    )

    match = re.search(r'<DOMAIN>(.*?)</DOMAIN>', answer, re.DOTALL)
    if match:
        domain_pddl = match.group(1).strip()
    else:
        greedy = re.search(r'<DOMAIN>(.*)', answer, re.DOTALL)
        if greedy:
            domain_pddl = greedy.group(1).strip()
        else:
            domain_pddl = answer.replace("`", "").replace("pddl", "").replace("lisp", "").strip()
        domain_pddl = re.sub(r'</?DOMAIN>', '', domain_pddl).strip()

    # Strip markdown code fences that some models wrap PDDL in
    domain_pddl = re.sub(r'^```[a-zA-Z]*\s*', '', domain_pddl.strip())
    domain_pddl = re.sub(r'\s*```\s*$', '', domain_pddl.strip())

    domain_pddl = _preprocess_pddl(domain_pddl)

    # Domain completion pass: if the NL description names actions that are absent
    # from the generated domain (stochastic truncation by the LLM), make a single
    # targeted follow-up call asking only for the missing actions.
    if domain_description and agent is not None:
        # Extract action names from NL description; deduplicate preserving order
        # (action names appear multiple times: in the actions list, preconditions, and effects sections)
        _all_names = re.findall(r'"(\w+)(?:\s[^""]*)?"(?:\s*\([^)]*\))?\s*:', domain_description)
        expected = list(dict.fromkeys(_all_names))
        generated_actions = set(re.findall(r'\(:action\s+(\S+)', domain_pddl))
        missing = [a for a in expected if a not in generated_actions]
        if missing:
            logger.warning(
                "[DOMAIN_COMPLETION] %d action(s) missing from generated domain: %s — requesting completion",
                len(missing), missing,
            )
            # Ask only for the missing action blocks (not the full domain) to stay
            # under the model's per-call output token cap (~1309 for gemini-3-flash-preview).
            # We splice the returned action blocks into the existing domain programmatically.
            _completion_system = (
                "You are an expert PDDL domain writer. "
                "Output ONLY the (:action ...) blocks for the listed missing actions — "
                "nothing else. No domain wrapper, no predicates, no types. "
                "Follow the style of the existing domain exactly."
            )
            _missing_descriptions = []
            for _act in missing:
                _m = re.search(
                    rf'"{re.escape(_act)}[^"]*"\s*:.*?(?=\n\s+"|\Z)',
                    domain_description, re.DOTALL,
                )
                if _m:
                    _missing_descriptions.append(_m.group(0).strip())
            _completion_user = (
                "Generate PDDL (:action ...) blocks for ONLY these missing actions:\n"
                + "\n".join(f"  - {a}" for a in missing)
                + (
                    "\n\nNL descriptions:\n" + "\n\n".join(_missing_descriptions)
                    if _missing_descriptions else ""
                )
                + f"\n\nExisting domain (for style reference — DO NOT repeat these actions):\n{domain_pddl}\n\n"
                "Output ONLY the new (:action ...) blocks, between <ACTIONS></ACTIONS> tags."
            )
            try:
                _completion_answer = agent.llm_call(_completion_system, _completion_user, temperature=max(temperature, 0.3))
                _am = re.search(r'<ACTIONS>(.*?)</ACTIONS>', _completion_answer, re.DOTALL)
                if not _am:
                    _am = re.search(r'<ACTIONS>(.*)', _completion_answer, re.DOTALL)
                if _am:
                    _new_action_blocks = _am.group(1).strip()
                    # Splice: insert the new action blocks before the final ')' of the domain
                    _splice_point = domain_pddl.rfind('\n)')
                    if _splice_point != -1:
                        _completed = (
                            domain_pddl[:_splice_point].rstrip()
                            + '\n\n'
                            + _new_action_blocks
                            + '\n)\n'
                        )
                    else:
                        _completed = domain_pddl.rstrip() + '\n\n' + _new_action_blocks + '\n'
                    _completed = _preprocess_pddl(_completed)
                    _new_actions = set(re.findall(r'\(:action\s+(\S+)', _completed))
                    if generated_actions.issubset(_new_actions):
                        domain_pddl = _completed
                        logger.info("[DOMAIN_COMPLETION] accepted — domain now has actions: %s", sorted(_new_actions))
                    else:
                        logger.warning("[DOMAIN_COMPLETION] completion dropped existing actions — discarding")
                else:
                    logger.warning("[DOMAIN_COMPLETION] no <ACTIONS> tags in completion response")
            except Exception as _ce:
                logger.warning("[DOMAIN_COMPLETION] completion call failed: %s", _ce)

    with open(domain_file_path, "w") as file:
        file.write(domain_pddl)

    return





###########
# PROBLEM #
###########

# PROBLEM GENERATION #

def generate_problem(
    domain_file_path,
    task,
    environment,
    problem_file_path,
    agent: Agent,
    logs_dir=None,
    workflow_iteration="0",
    problem_nl=None,
    ADD_PREDICATE_UNDERSCORE_EXAMPLE=False,
    inject_domain_schema=True,
    pddl_init_state=None,  # NEW: Oracle grounding - pre-grounded :init predicates
    temperature=0.1,
    objects_list: str | None = None,  # lifted symbols to enforce in :objects
):
    """
    Generate a PDDL problem file.

    Ablation axis — inject_domain_schema:
      True  (Approach A): extract types/predicates/constants from domain and inject
                          them as explicit constraints in the prompt.
      False (baseline):   provide domain PDDL as context only, no extracted schema.

    Oracle grounding (pddl_init_state):
      If provided, the grounded :init predicates are appended to the context_nl
      so the LLM generates a problem using these exact predicates.
    """
    with open(domain_file_path, "r") as file:
        domain = file.read()

    context_nl = problem_nl if problem_nl else environment

    # Oracle grounding: Append grounded :init predicates to context (like CoSTL)
    # The LLM must properly format these into the PDDL problem - this is "fair" grounding
    if pddl_init_state:
        logger.info(f"Oracle grounding: Adding {len(pddl_init_state.splitlines())} grounded predicates to context")
        context_nl = f"{context_nl}\n\nINITIAL STATE PREDICATES (grounded from simulator):\n{pddl_init_state}"

    objects_constraint = (
        f"\nOBJECT NAMES (use EXACTLY these identifiers in (:objects) — no substitutions):\n{objects_list}\n"
        if objects_list else ""
    )

    if inject_domain_schema:
        schema = extract_schema(domain)
        schema_block = _format_schema_block(schema)
        prompt = f"""Role: You are an expert PDDL problem generator.
Task: Generate a PDDL problem consistent with the given domain.

SCHEMA CONSTRAINTS (extracted from domain — must match exactly):
{schema_block}
{objects_constraint}
CONTENT RULES:
- (:init) must reflect the WORLD DESCRIPTION exactly. Do NOT hallucinate facts.
- (:goal) must reflect ONLY the SUBGOAL below. Do NOT add extra conditions.
- Do NOT use (not ...) in (:init) — PDDL uses the closed-world assumption.
- Do NOT include (:action ...) blocks in the problem file.
- Use ONLY the predicate names listed in PREDICATE SIGNATURES above. Do NOT invent alternative names (e.g. if the domain declares `inside`, do not use `char_inside`).

WORLD DESCRIPTION (Objects and Initial State):
{context_nl}

SUBGOAL (Target to reach):
{task}"""
    else:
        # Baseline prompt — domain as context, no extracted schema
        prompt = f"""
    Role: You are an excellent PDDL problem generator.
    Task: Given a PDDL domain, an initial world state, a list of objects, and a specific SUBGOAL, generate a PDDL problem file.

    STRICT INSTRUCTIONS:
    - The `(:init)` block must reflect the provided WORLD DESCRIPTION exactly.
    - The `(:goal)` block must reflect ONLY the provided SUBGOAL.
    - DO NOT add extra goal conditions from common sense or global tasks.
    - Ensure all object types match those defined in the domain.
    - REMOVE any (:action ...) blocks from the problem file.
    - Use character tokens <PROBLEM> and </PROBLEM> to wrap your output.
    {objects_constraint}
    DOMAIN PDDL:
    {domain}

    WORLD DESCRIPTION (Objects and Initial State):
    {context_nl}

    SUBGOAL (Target to reach):
    {task}
    """

    user_prompt = "Generate the PDDL problem within <PROBLEM></PROBLEM> tags."
    
    # Fairness Guard: Ensure the domain context is coming from the intended source (generated vs GT)
    try:
        dfp_str = str(domain_file_path) if domain_file_path else ""
        if "data/llmpp" in dfp_str and "/domain.pddl" in dfp_str:
             # This is the ground truth path. We should only use this if we are in condition A/B/C.
             # In condition D (full_adequacy), we must use the generated domain.
             logger.info(f"FAIRNESS CHECK: Using GROUND TRUTH domain for problem generation context: {dfp_str}")
        elif "generated_domain.pddl" in dfp_str:
             logger.info(f"FAIRNESS CHECK: Using GENERATED domain for problem generation context: {dfp_str}")
    except Exception as e:
        logger.warning(f"Fairness Guard encountered an error (non-critical): {e}")

    print(f"DEBUG: Calling agent.llm_call for problem generation (inject_schema={inject_domain_schema})...")

    answer = agent.llm_call(prompt, user_prompt)

    _save_prompt_response(
        prompt=f"{prompt}\n\n{user_prompt}",
        response=answer,
        prefix="strict_pddl_problem_generation",
        suffix=str(workflow_iteration),
        output_dir=logs_dir
    )

    match = re.search(r'<PROBLEM>(.*?)</PROBLEM>', answer, re.DOTALL)
    if match:
        problem_pddl = match.group(1).strip()
    else:
        # Greedy fallback: take everything after <PROBLEM> if closing tag is missing
        greedy = re.search(r'<PROBLEM>(.*)', answer, re.DOTALL)
        if greedy:
            problem_pddl = greedy.group(1).strip()
        else:
            problem_pddl = answer.replace("`", "").replace("pddl", "").replace("lisp", "").strip()
        # Strip any residual XML tags that snuck in
        problem_pddl = re.sub(r'</?PROBLEM>', '', problem_pddl).strip()

    problem_pddl = _preprocess_pddl(problem_pddl)
    problem_pddl = _strip_domain_constants_from_problem(problem_pddl, domain)

    with open(problem_file_path, "w") as file:
        file.write(problem_pddl)

    return

# PROBLEM REFINEMENT #

def refine_problem(
    domain_file_path,
    problem_file_path,
    environment=None,
    task=None,
    logs_dir=None,
    workflow_iteration=None,
    refinement_iteration=None,
    agent: Agent = None,
    nl_sections=None,
    pddlenv_error_log = None,
    planner_error_log = None,
    VAL_validation_log = None,
    VAL_grounding_log = None,
    scene_graph_grounding_log = None,
    use_two_step_refinement = True,
    use_vector_db = False,
    vector_db_path = None,
    scene_graph = None,
    temperature=0.1,
    use_taxonomy = True,
    bundle_by_error_class: bool = True,
):
    # Accept scene_graph as alias for environment
    if environment is None and scene_graph is not None:
        environment = scene_graph
    """
    Refine a PDDL problem file based on feedback.
    Strict Policy: Uses segmented NL sections for objects and initial state.
    """

    # 1) Read the relevant files
    with open(problem_file_path, "r") as file:
        problem_pddl = file.read()

    with open(domain_file_path, "r") as file:
        domain_pddl = file.read()

    # 2) Parse VAL Output
    verifier = VALVerifier()
    val_issues = []
    if VAL_validation_log:
        val_issues = verifier.parse_output(VAL_validation_log)

    # ----------------------------------------------------------
    # STEP A: Diagnosis (Structured)
    # ----------------------------------------------------------
    # 2) Setup Targeted NL Context
    problem_nl = f"{nl_sections['objects']}\n\n{nl_sections['initial_state']}" if nl_sections else environment

    reason_system_prompt = (
        "You are an expert PDDL diagnostician. Your job is to analyze PDDL problem "
        "failures and suggest precise fixes while strictly respecting the provided "
        "ground truth for objects and the initial state."
    )
    
    reason_user_prompt = f"""
    Below is the domain PDDL file (for context):
    ```
    {domain_pddl}
    ```
    Below is the problem PDDL file that failed:
    ```
    {problem_pddl}
    ```

    GROUND TRUTH WORLD DESCRIPTION (Objects and Initial State):
    {problem_nl}

    SUBGOAL (Target to reach):
    {task}
    """

    if pddlenv_error_log:
        reason_user_prompt += f"\nPDDLEnv Error:\n```{pddlenv_error_log}```\n"
    if planner_error_log:
        reason_user_prompt += f"\nPlanner Error:\n```{planner_error_log}```\n"

    # ── Taxonomy (bulk): string-match raw planner/VAL error text ─────────────
    # detect_all() is appropriate here — we are matching structured error strings.
    if use_taxonomy:
        from src.lapis.planner.low.pddl_taxonomy import detect_all, refinement_hint
        _error_text = " ".join(filter(None, [pddlenv_error_log, planner_error_log, " ".join(val_issues)]))
        _detected = detect_all(_error_text)
        _hint = refinement_hint(_detected)
        if _hint:
            reason_user_prompt += _hint + "\n"

        # ── CPDDL augmentation: targeted diagnostics for matched entries ──────
        _detected_subtypes = {e.subtype for e in _detected}
        try:
            from src.lapis.utils.cpddl_preprocessor import (
                cpddl_parse_errors, cpddl_strips_diagnosis, format_p31_hint,
            )
            if "missing_init_fluent" in _detected_subtypes:
                _diag = cpddl_strips_diagnosis(domain_file_path, problem_file_path)
                _cpddl_hint = format_p31_hint(_diag)
                if _cpddl_hint:
                    reason_user_prompt += "\n" + _cpddl_hint + "\n"
            if _detected_subtypes & {"bracket_imbalance", "undefined_type", "predicate_arity_mismatch"}:
                _parse_errs = cpddl_parse_errors(domain_file_path, problem_file_path)
                if _parse_errs:
                    reason_user_prompt += (
                        "\nCPDDL PARSE ERRORS (line/column):\n"
                        + "\n".join(f"  • {e}" for e in _parse_errs[:15])
                        + "\n"
                    )
        except Exception as _cpddl_exc:
            logger.debug("[CPDDL] augmentation skipped: %s", _cpddl_exc)
        # ─────────────────────────────────────────────────────────────────────
    # ──────────────────────────────────────────────────────────────────────────

    issues_list = []

    # --- HEURISTIC CHECK ---
    heuristic_warnings = check_pddl_heuristics(domain_pddl, problem_pddl)

    # --- PRIORITIZATION ---
    # Merge VAL issues and Heuristic warnings
    final_issues_to_report = prioritize_issues(val_issues, heuristic_warnings)

    if not final_issues_to_report:
         # No obvious issues? Maybe logic error.
         reason_user_prompt += "\nNo syntax/type errors found by VAL. Use your own reasoning to check for logical errors."
    else:
         reason_user_prompt += "\nDETECTED ISSUES (Prioritized):\n"
         for issue in final_issues_to_report:
             reason_user_prompt += f"- {issue}\n"

    reason_user_prompt += """
    Please analyze the errors and provide a list of specific issues and solutions in the following JSON format:
    {
        "issues": [
            {
                "issue": "Description of the specific issue (e.g. 'Object x is type A but defined as B')",
                "solution": "Detailed plan to fix this specific issue (e.g. 'Change type of x to B')",
                "correctness_check": "Optional check (e.g. 'Verify x is now type B')",
                "missing_init_facts": []
            }
        ]
    }

    For P3.1 (no solution found) issues ONLY: if and only if the root cause is a literal that is
    genuinely absent from (:init) — NOT a wrong action precondition or wrong action effect — list
    the exact PDDL literals (e.g. ["(plugged_in microwave_1)", "(open fridge_1)"]) in
    'missing_init_facts'. Leave 'missing_init_facts' as [] if the fix belongs in the domain logic.
    Do NOT propose weakening or rewriting action preconditions.

    CRITICAL: You MUST NOT suggest changes that modify the physical configuration of objects in the
    `(:init)` block EXCEPT for literals explicitly listed in 'missing_init_facts'. The (:init) block
    represents the Ground Truth of the simulator. Fix action preconditions/effects or domain logic
    rather than patching (:init) when the issue is a logic error.
    ONLY output the JSON.
    """

    # Call LLM — use low temperature for diagnosis (want reliable JSON), ramp on correction
    diagnosis_response = agent.llm_call(reason_system_prompt, reason_user_prompt, temperature=0.1)

    # Save log
    _save_prompt_response(
        prompt=reason_system_prompt + "\n\n" + reason_user_prompt,
        response=diagnosis_response,
        prefix="problem_diagnosis_json",
        suffix=str(workflow_iteration)+"_"+str(refinement_iteration),
        output_dir=logs_dir
    )

    # Parse JSON
    try:
        # cleanup markdown code blocks if present
        clean_response = diagnosis_response.replace("```json", "").replace("```", "").strip()
        diagnosis_data = json.loads(clean_response)
    except Exception as e:
        print(f"Error parsing diagnosis JSON: {e}")
        # Fallback: if parsing fails, we might just return the original problem or try a fallback refinement
        # For now, let's treat it as empty issues
        diagnosis_data = {"issues": []}

    # ----------------------------------------------------------
    # STEP B: Vector DB Init & Search
    # ----------------------------------------------------------
    vdb = None
    if use_vector_db:
        if not vector_db_path:
             # Default path
             base_dir = Path(logs_dir).parent.parent # Assuming logs_dir is deeper like results/run/logs
             vector_db_path = os.path.join(base_dir, "vector_db.json")
        
        # Check if agent has client, otherwise try to use os.getenv or fail gracefully
        client = getattr(agent, "client", None)
        if client:
            vdb = VectorDB(vector_db_path, client)
        else:
            print("Warning: Agent does not have 'client' attribute. Vector DB disabled.")

    # ----------------------------------------------------------
    # STEP C: Correction (bundled by error class, or per-issue sequential)
    # ----------------------------------------------------------
    current_problem_pddl = problem_pddl

    issues_list = diagnosis_data.get("issues", [])
    refinement_history = []

    # G6: Structural pre-check — deterministically detect invented predicates and
    # constant_name_drift via UP-style domain/problem AST walk, BEFORE relying on
    # the LLM diagnosis to flag them. Synthetic issues come pre-classified with
    # the right TaxonomyEntry, so bucketing handles them naturally. This closes
    # v5 detection misses on T231358 / T152158 (isCold / isHeated invented).
    structural_issues_count = 0
    if use_taxonomy:
        try:
            from src.lapis.planner.low.domain_parser import structural_problem_issues
            structural_issues = structural_problem_issues(domain_pddl, problem_pddl)
            if structural_issues:
                logger.info(
                    "[STRUCTURAL] pre-check found %d issue(s) before LLM diagnosis",
                    len(structural_issues),
                )
                # Prepend so they get processed first when bundles share priority.
                issues_list = structural_issues + issues_list
                structural_issues_count = len(structural_issues)
        except Exception as _struct_exc:
            logger.warning("[STRUCTURAL] pre-check skipped: %s", _struct_exc)

    # Pre-classify each issue's taxonomy entries so we can bucket by error_class.
    # G4b: prefer bulk-detected matches (already computed above); only fall back to
    # the LLM judge per-issue when bulk found nothing.
    # Structural-pre-check issues already carry _taxonomy_entries — reuse them.
    issues_meta = []
    for _item in issues_list:
        _pre_entries = _item.get("_taxonomy_entries") if isinstance(_item, dict) else None
        if _pre_entries:
            _entries = list(_pre_entries)
        elif use_taxonomy:
            if _detected:
                _entries = list(_detected)
            else:
                _entries = _classify_issue_with_llm(
                    _item.get("issue", ""), _item.get("solution", ""), agent
                )
        else:
            _entries = []
        issues_meta.append({"item": _item, "entries": _entries})

    # G1: apply PREPROCESS_REGEX domain fixes ONCE (no LLM call needed).
    # All current PREPROCESS_REGEX entries (empty_effect_body, reserved_type_name_object,
    # hierarchical_type_hierarchy) target the domain — applying them to the problem
    # would be a no-op or destructive.
    if use_taxonomy:
        from src.lapis.planner.low.pddl_taxonomy import CorrectionMethod
        _fix_map = _get_entry_fix_map()
        _seen_subtypes = set()
        for _meta in issues_meta:
            for _entry in _meta["entries"]:
                if (_entry.correction_method == CorrectionMethod.PREPROCESS_REGEX
                        and _entry.subtype not in _seen_subtypes):
                    _seen_subtypes.add(_entry.subtype)
                    _fix_fn = _fix_map.get(_entry.subtype)
                    if _fix_fn:
                        try:
                            with open(domain_file_path, "r") as _df:
                                _dom = _df.read()
                            _dom_fixed = _fix_fn(_dom)
                            with open(domain_file_path, "w") as _df:
                                _df.write(_dom_fixed)
                        except Exception as _fix_exc:
                            logger.warning("[G1] domain regex fix failed: %s", _fix_exc)

    # Build buckets. With bundle_by_error_class=True (default), issues sharing a
    # priority bucket (P0_SYNTAX, P1_TYPE, P1_PREDICATE, P2_ACTION, P3_OTHER) are
    # fixed together in a single LLM call, in priority order. With False, each
    # issue gets its own bucket (legacy per-issue mode).
    if bundle_by_error_class and issues_meta:
        buckets: dict = {}
        for _meta in issues_meta:
            _bk = _bucket_for_issue(_meta["item"], _meta["entries"])
            buckets.setdefault(_bk, []).append(_meta)
        bucket_order = [b for b in PRIORITY_ORDER if b in buckets]
        bucket_order += [b for b in buckets if b not in PRIORITY_ORDER]
    else:
        buckets = {}
        bucket_order = []
        for _i, _meta in enumerate(issues_meta):
            _key = f"_seq_{_i}"
            buckets[_key] = [_meta]
            bucket_order.append(_key)

    # Process each bucket: ONE LLM correction call per bucket, addressing all its issues.
    for step_idx, bk in enumerate(bucket_order):
        bundle = buckets[bk]
        if not bundle:
            continue

        # Union taxonomy entries across the bundle (for refinement_hint)
        _bundle_entries = []
        _seen_keys = set()
        for _m in bundle:
            for _e in _m["entries"]:
                _k = (_e.error_class, _e.subtype)
                if _k not in _seen_keys:
                    _bundle_entries.append(_e)
                    _seen_keys.add(_k)
        _bundle_hint = ""
        if use_taxonomy and _bundle_entries:
            from src.lapis.planner.low.pddl_taxonomy import refinement_hint
            _bundle_hint = refinement_hint(_bundle_entries)

        # Vector DB context — balanced retrieval (positive/negative exemplars).
        # Query with first issue's text (representative of the bucket). The DB is
        # per-task so positives/negatives only reflect this task's history.
        similar_solution = ""
        db_match = False
        _query_text = bundle[0]["item"].get("issue", "")
        if vdb and _query_text:
            try:
                from src.lapis.utils.vector_db import format_rag_context
                _positives, _negatives = vdb.search_balanced(_query_text, k_per_outcome=2)
                _rag = format_rag_context(_positives, _negatives)
                if _rag:
                    similar_solution = _rag
                    db_match = bool(_positives)
            except Exception as _vdb_exc:
                logger.debug("[VDB] balanced search failed, falling back to flat: %s", _vdb_exc)
                results = vdb.search(_query_text)
                if results:
                    top_match = results[0]
                    if top_match.get('similarity', 0) > 0.8:
                        similar_solution = f"Similar past solution found: {top_match.get('solution')}"
                        db_match = True

        # G3b: union of missing_init_facts across the bundle
        union_facts: list = []
        for _m in bundle:
            for _f in (_m["item"].get("missing_init_facts") or []):
                if _f not in union_facts:
                    union_facts.append(_f)
        if union_facts:
            _facts_block = "\n".join(f"  {_f}" for _f in union_facts)
            _init_lock_clause = (
                f"The diagnosis identified these specific init-state literals as required-but-absent:\n"
                f"{_facts_block}\n"
                "Add EXACTLY these literals to the (:init) block and nothing else. "
                "Do NOT add any other facts, do NOT remove or modify existing init facts, "
                "do NOT modify action preconditions or effects."
            )
        else:
            _init_lock_clause = (
                "CRITICAL: You MUST NOT modify the (:init) block in any way. "
                "It represents the unchangeable Ground Truth of the simulator. "
                "Fix the issue by correcting action preconditions, effects, or domain logic instead."
            )

        # Build issues section: single-issue or bundled
        if len(bundle) == 1:
            _it = bundle[0]["item"]
            _issue_text = _it.get("issue", "")
            _solution_text = _it.get("solution", "")
            _cc = _it.get("correctness_check", "")
            issues_section = (
                f"Specific Issue to Fix: {_issue_text}\n"
                f"Proposed Solution: {_solution_text}"
            )
            if _cc:
                issues_section += f"\nCorrectness Check: {_cc}"
        else:
            _bucket_label = bk if bk in PRIORITY_ORDER else "MULTI"
            issues_section = (
                f"Apply ALL of the following {len(bundle)} fixes (priority bucket: "
                f"{_bucket_label}) in a single corrected PDDL output:\n"
            )
            for _n, _m in enumerate(bundle, 1):
                _it = _m["item"]
                issues_section += (
                    f"\n[Fix {_n}]\n"
                    f"  Issue: {_it.get('issue', '')}\n"
                    f"  Solution: {_it.get('solution', '')}"
                )
                _cc = _it.get("correctness_check", "")
                if _cc:
                    issues_section += f"\n  Check: {_cc}"

        correction_system_prompt = (
             "You are a helpful planning assistant. You are fixing a PDDL problem file based on the listed issues. "
             "Apply ALL proposed solutions in a single corrected PDDL output."
        )

        correction_user_prompt = f"""
        Current PDDL Problem:
        ```
        {current_problem_pddl}
        ```

        {issues_section}

        {similar_solution}

        {_init_lock_clause}

        Please apply the fix(es) to the PDDL.
        Output the full corrected PDDL problem in the format:
        <PROBLEM>
        ... content ...
        </PROBLEM>
        """

        if _bundle_hint:
            correction_user_prompt += f"\n{_bundle_hint}"

        response = agent.llm_call(correction_system_prompt, correction_user_prompt, temperature=temperature)

        # Extract PDDL — retry once on missing tags rather than silently skipping.
        extraction_failed = False
        if "<PROBLEM>" in response and "</PROBLEM>" in response:
            new_content = response.split("<PROBLEM>")[1].split("</PROBLEM>")[0]
            new_content = new_content.replace("`", "").replace("pddl", "").replace("lisp", "").strip()
            refined_pddl_candidate = new_content
        else:
            retry_prompt = correction_user_prompt + (
                "\n\nIMPORTANT: Your previous response did not include the required <PROBLEM>...</PROBLEM> wrapper. "
                "Wrap the corrected PDDL in <PROBLEM>...</PROBLEM> tags and output the full PDDL now."
            )
            response = agent.llm_call(correction_system_prompt, retry_prompt, temperature=temperature)
            if "<PROBLEM>" in response and "</PROBLEM>" in response:
                new_content = response.split("<PROBLEM>")[1].split("</PROBLEM>")[0]
                new_content = new_content.replace("`", "").replace("pddl", "").replace("lisp", "").strip()
                refined_pddl_candidate = new_content
            else:
                logger.warning("Could not extract PDDL from correction step %d (after retry)", step_idx)
                refined_pddl_candidate = current_problem_pddl
                extraction_failed = True

        # Log this step
        _save_prompt_response(
             prompt=correction_system_prompt + "\n\n" + correction_user_prompt,
             response=response,
             prefix=f"problem_refinement_step_{step_idx}",
             suffix=str(workflow_iteration)+"_"+str(refinement_iteration),
             output_dir=logs_dir,
        )

        # Validate the candidate
        refined_pddl_candidate = _preprocess_pddl(refined_pddl_candidate)
        temp_prob_path = os.path.join(logs_dir, f"temp_problem_{workflow_iteration}_{refinement_iteration}_{step_idx}.pddl")
        with open(temp_prob_path, "w") as f:
             f.write(refined_pddl_candidate)

        is_valid, _ = verifier.verify(domain_file_path, temp_prob_path)

        # G5: solvability check — structural VAL alone is insufficient; a problem can
        # parse correctly while remaining unsolvable. Only add to VDB when the goal
        # is also reachable, to avoid polluting the DB with "fixes" that pass syntax
        # but don't actually resolve the planning failure.
        is_solvable = False
        if is_valid:
            try:
                from src.lapis.utils.cpddl_preprocessor import cpddl_strips_diagnosis
                _sol_diag = cpddl_strips_diagnosis(domain_file_path, temp_prob_path)
                is_solvable = not _sol_diag.goal_is_unreachable
            except Exception as _g5_exc:
                logger.debug("[G5] solvability check failed: %s", _g5_exc)
                is_solvable = True

        current_problem_pddl = refined_pddl_candidate
        if is_valid:
            if is_solvable:
                print(f"Step {step_idx} (bucket={bk}, n={len(bundle)}) Result: VALID+SOLVABLE.")
                if vdb:
                    for _m in bundle:
                        vdb.add(
                            _m["item"].get("issue", ""),
                            _m["item"].get("solution", ""),
                            outcome="positive",
                        )
            else:
                print(f"Step {step_idx} (bucket={bk}, n={len(bundle)}) Result: VALID (unreachable, not committed to DB).")
                # Goal-unreachable fixes are recorded as negatives so future
                # buckets can avoid the same dead-end pattern.
                if vdb:
                    for _m in bundle:
                        vdb.add(
                            _m["item"].get("issue", ""),
                            _m["item"].get("solution", ""),
                            outcome="negative",
                        )
        else:
            print(f"Step {step_idx} (bucket={bk}, n={len(bundle)}) Result: INVALID.")
            if vdb:
                for _m in bundle:
                    vdb.add(
                        _m["item"].get("issue", ""),
                        _m["item"].get("solution", ""),
                        outcome="negative",
                    )

        # One refinement_history entry per issue — preserves per-issue measurement
        # while sharing the bucket's outcome (is_valid/is_solvable).
        for _m in bundle:
            refinement_history.append({
                "issue": _m["item"].get("issue", ""),
                "solution": _m["item"].get("solution", ""),
                "missing_init_facts": _m["item"].get("missing_init_facts", []),
                "is_valid": is_valid,
                "is_solvable": is_solvable,
                "db_match": db_match,
                "extraction_failed": extraction_failed,
                "bucket": bk,
                "bundle_size": len(bundle),
                "source": _m["item"].get("_source", "llm_diagnosis"),
                "taxonomy_matches": [
                    {"error_class": e.error_class, "subtype": e.subtype}
                    for e in _m["entries"]
                ],
            })

        if os.path.exists(temp_prob_path):
             os.remove(temp_prob_path)

    current_problem_pddl = _preprocess_pddl(current_problem_pddl)
    with open(domain_file_path, "r") as _df:
        current_problem_pddl = _strip_domain_constants_from_problem(current_problem_pddl, _df.read())
    return current_problem_pddl, refinement_history




# DOMAIN REFINEMENT #

def refine_domain(
    domain_file_path,
    problem_file_path,
    environment,
    task,
    logs_dir,
    workflow_iteration,
    refinement_iteration,
    agent: Agent,
    nl_sections=None,
    pddlenv_error_log = None,
    planner_error_log = None,
    VAL_validation_log = None,
    VAL_grounding_log = None,
    scene_graph_grounding_log = None,
    use_two_step_refinement = True,
    use_vector_db = False,
    vector_db_path = None,
    temperature=0.1,
    use_taxonomy = True,
    bundle_by_error_class: bool = True,
):
    """
    Refine a PDDL domain file based on feedback.
    Strict Policy: Uses segmented NL sections for actions and effects.
    """

    # 1) Read the relevant files
    if problem_file_path is not None:
        with open(problem_file_path, "r") as file:
            problem_pddl = file.read()
    else:
        problem_pddl = "(define (problem placeholder) (:domain ?) (:objects) (:init) (:goal (and)))"

    with open(domain_file_path, "r") as file:
        domain_pddl = file.read()

    # 2) Parse VAL Output
    verifier = VALVerifier()
    val_issues = []
    if VAL_validation_log:
        val_issues = verifier.parse_output(VAL_validation_log)


    if VAL_validation_log:
        val_issues = verifier.parse_output(VAL_validation_log)

    # ----------------------------------------------------------
    # STEP A: Diagnosis (Structured)
    # ----------------------------------------------------------
    # 3) Setup Targeted NL Context
    domain_nl = f"{nl_sections['description']}\n\n{nl_sections['actions']}\n\n{nl_sections['preconditions']}\n\n{nl_sections['effects']}" if nl_sections else environment

    reason_system_prompt = (
        "You are an expert PDDL diagnostician. Your job is to analyze PDDL domain "
        "failures and suggest precise fixes while strictly respecting the provided "
        "ground truth for actions and world logic."
    )
    
    reason_user_prompt = f"""
    Below is the domain PDDL file:
    ```
    {domain_pddl}
    ```
    Below is the problem PDDL file:
    ```
    {problem_pddl}
    ```

    GROUND TRUTH DOMAIN DESCRIPTION (Actions and Effects):
    {domain_nl}

    SUBGOAL (Target to reach):
    {task}
    """

    if pddlenv_error_log:
        reason_user_prompt += f"\nPDDLEnv Error:\n```{pddlenv_error_log}```\n"
    if planner_error_log:
        reason_user_prompt += f"\nPlanner Error:\n```{planner_error_log}```\n"

    # ── Taxonomy (bulk): string-match raw planner/VAL error text ─────────────
    # detect_all() is appropriate here — we are matching structured error strings.
    if use_taxonomy:
        from src.lapis.planner.low.pddl_taxonomy import detect_all, refinement_hint
        _error_text = " ".join(filter(None, [pddlenv_error_log, planner_error_log, " ".join(val_issues)]))
        _detected = detect_all(_error_text)
        _hint = refinement_hint(_detected)
        if _hint:
            reason_user_prompt += _hint + "\n"

        # ── CPDDL augmentation: targeted diagnostics for matched entries ──────
        _detected_subtypes = {e.subtype for e in _detected}
        try:
            from src.lapis.utils.cpddl_preprocessor import (
                cpddl_parse_errors, cpddl_strips_diagnosis, format_p31_hint,
            )
            if "missing_init_fluent" in _detected_subtypes:
                _diag = cpddl_strips_diagnosis(domain_file_path, problem_file_path)
                _cpddl_hint = format_p31_hint(_diag)
                if _cpddl_hint:
                    reason_user_prompt += "\n" + _cpddl_hint + "\n"
            if _detected_subtypes & {"bracket_imbalance", "undefined_type", "predicate_arity_mismatch"}:
                _parse_errs = cpddl_parse_errors(domain_file_path, problem_file_path)
                if _parse_errs:
                    reason_user_prompt += (
                        "\nCPDDL PARSE ERRORS (line/column):\n"
                        + "\n".join(f"  • {e}" for e in _parse_errs[:15])
                        + "\n"
                    )
        except Exception as _cpddl_exc:
            logger.debug("[CPDDL] augmentation skipped: %s", _cpddl_exc)
        # ─────────────────────────────────────────────────────────────────────
    # ──────────────────────────────────────────────────────────────────────────

    issues_list = []

    # --- HEURISTIC CHECK ---
    heuristic_warnings = check_pddl_heuristics(domain_pddl, problem_pddl)

    # --- PRIORITIZATION ---
    # Merge VAL issues and Heuristic warnings
    final_issues_to_report = prioritize_issues(val_issues, heuristic_warnings)

    if not final_issues_to_report:
         # No obvious issues? Maybe logic error.
         reason_user_prompt += "\nNo syntax/type errors found by VAL. Use your own reasoning to check for logical errors."
    else:
         reason_user_prompt += "\nDETECTED ISSUES (Prioritized):\n"
         for issue in final_issues_to_report:
             reason_user_prompt += f"- {issue}\n"

    reason_user_prompt += """
    Please analyze the errors and provide a list of specific DOMAIN issues and solutions in the following JSON format:
    {
        "issues": [
            {
                "issue": "Description of the specific domain issue (e.g. 'Action X has incorrect preconditions')",
                "solution": "Detailed plan to fix this specific issue (e.g. 'Add predicate Y to preconditions of X')",
                "correctness_check": "Optional check"
            }
        ]
    }
    ONLY output the JSON.
    """

    # Call LLM — diagnosis always at low temperature for reliable JSON
    diagnosis_response = agent.llm_call(reason_system_prompt, reason_user_prompt, temperature=0.1)

    # Save log
    _save_prompt_response(
        prompt=reason_system_prompt + "\n\n" + reason_user_prompt,
        response=diagnosis_response,
        prefix="domain_diagnosis_json",
        suffix=str(workflow_iteration)+"_"+str(refinement_iteration),
        output_dir=logs_dir
    )

    # Parse JSON
    try:
        clean_response = diagnosis_response.replace("```json", "").replace("```", "").strip()
        diagnosis_data = json.loads(clean_response)
    except Exception as e:
        print(f"Error parsing diagnosis JSON: {e}")
        diagnosis_data = {"issues": []}

    # ----------------------------------------------------------
    # STEP B: Vector DB Init & Search
    # ----------------------------------------------------------
    vdb = None
    if use_vector_db:
        if not vector_db_path:
             base_dir = Path(logs_dir).parent.parent
             vector_db_path = os.path.join(base_dir, "vector_db.json")
        
        client = getattr(agent, "client", None)
        if client:
            vdb = VectorDB(vector_db_path, client)

    # ----------------------------------------------------------
    # STEP C: Correction (bundled by error class, or per-issue sequential)
    # ----------------------------------------------------------
    current_domain_pddl = domain_pddl

    issues_list = diagnosis_data.get("issues", [])
    refinement_history = []

    # Pre-classify each issue's taxonomy entries
    issues_meta = []
    for _item in issues_list:
        _entries = []
        if use_taxonomy:
            _entries = _classify_issue_with_llm(
                _item.get("issue", ""), _item.get("solution", ""), agent
            )
        issues_meta.append({"item": _item, "entries": _entries})

    # Apply PREPROCESS_REGEX domain fixes ONCE (no LLM call)
    if use_taxonomy:
        from src.lapis.planner.low.pddl_taxonomy import CorrectionMethod
        _fix_map = _get_entry_fix_map()
        _seen_subtypes = set()
        for _meta in issues_meta:
            for _entry in _meta["entries"]:
                if (_entry.correction_method == CorrectionMethod.PREPROCESS_REGEX
                        and _entry.subtype not in _seen_subtypes):
                    _seen_subtypes.add(_entry.subtype)
                    _fix_fn = _fix_map.get(_entry.subtype)
                    if _fix_fn:
                        try:
                            current_domain_pddl = _fix_fn(current_domain_pddl)
                        except Exception as _fix_exc:
                            logger.warning("[G1] domain regex fix failed: %s", _fix_exc)

    # Build buckets (priority-ordered when bundle_by_error_class=True)
    if bundle_by_error_class and issues_meta:
        buckets: dict = {}
        for _meta in issues_meta:
            _bk = _bucket_for_issue(_meta["item"], _meta["entries"])
            buckets.setdefault(_bk, []).append(_meta)
        bucket_order = [b for b in PRIORITY_ORDER if b in buckets]
        bucket_order += [b for b in buckets if b not in PRIORITY_ORDER]
    else:
        buckets = {}
        bucket_order = []
        for _i, _meta in enumerate(issues_meta):
            _key = f"_seq_{_i}"
            buckets[_key] = [_meta]
            bucket_order.append(_key)

    for step_idx, bk in enumerate(bucket_order):
        bundle = buckets[bk]
        if not bundle:
            continue

        # Union taxonomy entries across the bundle
        _bundle_entries = []
        _seen_keys = set()
        for _m in bundle:
            for _e in _m["entries"]:
                _k = (_e.error_class, _e.subtype)
                if _k not in _seen_keys:
                    _bundle_entries.append(_e)
                    _seen_keys.add(_k)
        _bundle_hint = ""
        if use_taxonomy and _bundle_entries:
            from src.lapis.planner.low.pddl_taxonomy import refinement_hint
            _bundle_hint = refinement_hint(_bundle_entries)

        # Vector DB context
        similar_solution = ""
        db_match = False
        _query_text = bundle[0]["item"].get("issue", "")
        if vdb and _query_text:
            results = vdb.search(_query_text)
            if results:
                top_match = results[0]
                if top_match.get('similarity', 0) > 0.8:
                    similar_solution = f"Similar past solution found: {top_match.get('solution')}"
                    db_match = True

        # Build issues section
        if len(bundle) == 1:
            _it = bundle[0]["item"]
            _issue_text = _it.get("issue", "")
            _solution_text = _it.get("solution", "")
            _cc = _it.get("correctness_check", "")
            issues_section = (
                f"Specific Issue to Fix: {_issue_text}\n"
                f"Proposed Solution: {_solution_text}"
            )
            if _cc:
                issues_section += f"\nCorrectness Check: {_cc}"
        else:
            _bucket_label = bk if bk in PRIORITY_ORDER else "MULTI"
            issues_section = (
                f"Apply ALL of the following {len(bundle)} fixes (priority bucket: "
                f"{_bucket_label}) in a single corrected PDDL output:\n"
            )
            for _n, _m in enumerate(bundle, 1):
                _it = _m["item"]
                issues_section += (
                    f"\n[Fix {_n}]\n"
                    f"  Issue: {_it.get('issue', '')}\n"
                    f"  Solution: {_it.get('solution', '')}"
                )
                _cc = _it.get("correctness_check", "")
                if _cc:
                    issues_section += f"\n  Check: {_cc}"

        correction_system_prompt = (
             "You are a helpful planning assistant. You are fixing a PDDL DOMAIN file based on the listed issues. "
             "Apply ALL proposed solutions in a single corrected PDDL output."
        )

        correction_user_prompt = f"""
        Current PDDL Domain:
        ```
        {current_domain_pddl}
        ```

        {issues_section}

        {similar_solution}

        Please apply the fix(es) to the DOMAIN PDDL.
        CRITICAL: PREDICATE NAMES MUST MATCH EXACTLY BETWEEN DOMAIN AND PROBLEM. If the original problem used `ontable`, the domain MUST use `ontable` (no hyphens). Do NOT invent new hyphens or remove existing ones.
        Output the full corrected PDDL domain in the format:
        <DOMAIN>
        ... content ...
        </DOMAIN>
        """

        if _bundle_hint:
            correction_user_prompt += f"\n{_bundle_hint}"

        response = agent.llm_call(correction_system_prompt, correction_user_prompt, temperature=temperature)

        # Extract PDDL — retry once on missing tags
        extraction_failed = False
        if "<DOMAIN>" in response and "</DOMAIN>" in response:
            new_content = response.split("<DOMAIN>")[1].split("</DOMAIN>")[0]
            new_content = new_content.replace("`", "").replace("pddl", "").replace("lisp", "").strip()
            refined_domain_candidate = new_content
        else:
            retry_prompt = correction_user_prompt + (
                "\n\nIMPORTANT: Your previous response did not include the required <DOMAIN>...</DOMAIN> wrapper. "
                "Wrap the corrected PDDL in <DOMAIN>...</DOMAIN> tags and output the full PDDL now."
            )
            response = agent.llm_call(correction_system_prompt, retry_prompt, temperature=temperature)
            if "<DOMAIN>" in response and "</DOMAIN>" in response:
                new_content = response.split("<DOMAIN>")[1].split("</DOMAIN>")[0]
                new_content = new_content.replace("`", "").replace("pddl", "").replace("lisp", "").strip()
                refined_domain_candidate = new_content
            else:
                logger.warning("Could not extract PDDL from domain correction step %d (after retry)", step_idx)
                refined_domain_candidate = current_domain_pddl
                extraction_failed = True

        # Log this step
        _save_prompt_response(
            prompt=correction_system_prompt + "\n\n" + correction_user_prompt,
            response=response,
            prefix=f"domain_refinement_step_{step_idx}",
            suffix=str(workflow_iteration)+"_"+str(refinement_iteration),
            output_dir=logs_dir,
        )

        refined_domain_candidate = _preprocess_pddl(refined_domain_candidate)
        temp_domain_path = os.path.join(logs_dir, f"temp_domain_{workflow_iteration}_{refinement_iteration}_{step_idx}.pddl")
        with open(temp_domain_path, "w") as f:
             f.write(refined_domain_candidate)

        is_valid, _ = verifier.verify(temp_domain_path, problem_file_path)

        current_domain_pddl = refined_domain_candidate
        if is_valid:
             print(f"Domain Step {step_idx} (bucket={bk}, n={len(bundle)}) Result: VALID.")
             if vdb:
                 for _m in bundle:
                     vdb.add(_m["item"].get("issue", ""), _m["item"].get("solution", ""))
        else:
             print(f"Domain Step {step_idx} (bucket={bk}, n={len(bundle)}) Result: INVALID.")

        for _m in bundle:
            refinement_history.append({
                "issue": _m["item"].get("issue", ""),
                "solution": _m["item"].get("solution", ""),
                "is_valid": is_valid,
                "db_match": db_match,
                "extraction_failed": extraction_failed,
                "bucket": bk,
                "bundle_size": len(bundle),
                "taxonomy_matches": [
                    {"error_class": e.error_class, "subtype": e.subtype}
                    for e in _m["entries"]
                ],
            })

        if os.path.exists(temp_domain_path):
             os.remove(temp_domain_path)

    current_domain_pddl = _preprocess_pddl(current_domain_pddl)
    return current_domain_pddl, refinement_history


#########
# OTHER #
#########


def _save_prompt_response(prompt: str, response: str, output_dir: str, prefix: str, suffix: str) -> None:
    """Save prompt and response to separate files."""
        
    base_name = f"{prefix}_{suffix}"
    
    # Save prompt
    prompt_file = os.path.join(output_dir, f"{base_name}_prompt.log")
    with open(prompt_file, "w") as f:
        f.write(prompt)
        
    # Save response
    response_file = os.path.join(output_dir, f"{base_name}_response.log")
    with open(response_file, "w") as f:
        f.write(json.dumps(response, indent=4))


from src.lapis.logger_cfg import logger as _logger

####################
# ADEQUACY CHECKS  #
####################

def check_domain_adequacy(domain_pddl, raw_observation, objects_list, agent, logs_dir, temperature=0.1):
    """
    Pre-planning CoT check: can the generated domain represent what we observe?

    Three Chain-of-Thought steps (each a separate LLM call):
      Step 1: Extract predicate signatures from the domain.
      Step 2: Match each observation fact to a predicate; flag critical gaps.
      Step 3: If critical gaps exist, amend the domain with new predicates.

    Returns: amended domain string (or unchanged if adequate).
    """
    step1_prompt = f"""Analyze this PDDL domain and extract ALL predicate definitions.

DOMAIN:
```
{domain_pddl}
```

For each predicate, output its name, parameters, and parameter types in this exact format:
PREDICATE: <name> | PARAMS: <param_list_with_types>

Example:
PREDICATE: in-room | PARAMS: ?e - entity, ?r - room
PREDICATE: holding | PARAMS: ?x - item

Output ONLY the predicate list, nothing else."""

    predicates_raw = agent.llm_call(
        "You are a PDDL syntax analyzer. Extract predicate signatures.",
        step1_prompt
    )

    if logs_dir:
        _save_prompt_response(
            prompt="[CoT Step 1: Extract predicates]",
            response=predicates_raw,
            prefix="domain_adequacy_step1",
            suffix="predicates",
            output_dir=logs_dir
        )

    step2_prompt = f"""You are checking whether a PDDL domain can represent all observed world facts.

DOMAIN PREDICATES:
{predicates_raw}

OBSERVED WORLD STATE:
{raw_observation}

KNOWN OBJECTS:
{objects_list}

For each observation fact:
1. Identify which predicate can represent it → mark as MAPPED
2. If no predicate can represent it → mark as UNMAPPABLE
3. For each UNMAPPABLE fact, explain whether it is:
   a) CRITICAL: needed for planning (e.g., spatial relations, object properties)
   b) IRRELEVANT: not needed (e.g., visual details, coordinates)

Output format:
MAPPED: "<observation>" → (<predicate_name> <args>)
UNMAPPABLE [CRITICAL]: "<observation>" — Reason: <why>
UNMAPPABLE [IRRELEVANT]: "<observation>" — Reason: <why>

Then summarize:
CRITICAL_GAPS: <number>
TOTAL_OBSERVATIONS: <number>"""

    gap_analysis = agent.llm_call(
        "You are a PDDL domain adequacy checker. Match observations to predicates.",
        step2_prompt
    )

    if logs_dir:
        _save_prompt_response(
            prompt="[CoT Step 2: Gap analysis]",
            response=gap_analysis,
            prefix="domain_adequacy_step2",
            suffix="gaps",
            output_dir=logs_dir
        )

    if "CRITICAL_GAPS: 0" in gap_analysis or "UNMAPPABLE [CRITICAL]" not in gap_analysis:
        _logger.info("Domain Adequacy: No critical gaps found. Domain is adequate.")
        return domain_pddl

    _logger.info("Domain Adequacy: Critical gaps detected. Amending domain...")

    step3_prompt = f"""Based on this gap analysis, amend the PDDL domain to add missing predicates.

CURRENT DOMAIN:
```
{domain_pddl}
```

GAP ANALYSIS:
{gap_analysis}

RULES:
1. ONLY add predicates for CRITICAL gaps (ignore IRRELEVANT ones).
2. New predicates must use types already defined in the domain.
3. Do NOT modify existing predicates or actions.
4. Do NOT add actions — only predicates that can represent the observed state.
5. Output the COMPLETE amended domain wrapped in <DOMAIN>...</DOMAIN> tags.
6. If on reflection no amendments are truly needed, output NO_AMENDMENT."""

    amendment_response = agent.llm_call(
        "You are a PDDL domain engineer. Add missing predicates for observed facts.",
        step3_prompt
    )

    if logs_dir:
        _save_prompt_response(
            prompt="[CoT Step 3: Domain amendment]",
            response=amendment_response,
            prefix="domain_adequacy_step3",
            suffix="amendment",
            output_dir=logs_dir
        )

    if "NO_AMENDMENT" in amendment_response:
        _logger.info("Domain Adequacy: LLM decided no amendments are needed after reflection.")
        return domain_pddl

    if "<DOMAIN>" in amendment_response and "</DOMAIN>" in amendment_response:
        amended = amendment_response.split("<DOMAIN>")[1].split("</DOMAIN>")[0]
        amended = amended.replace("`", "").replace("pddl", "").replace("lisp", "").strip()
        amended = _preprocess_pddl(amended)
        _logger.info("Domain Adequacy: Domain amended with new predicates.")
        return amended

    _logger.warning("Domain Adequacy: Could not parse amendment. Returning original domain.")
    return domain_pddl


def check_problem_adequacy(problem_pddl, domain_pddl, raw_observation, objects_list, agent, logs_dir, temperature=0.1):
    """
    Post-generation CoT check: does the problem correctly formalize the observation?

    Uses a single combined CoT prompt (simpler than domain check):
      - Verify each (:init) predicate exists in the domain and matches observation.
      - Flag hallucinated facts, missing facts, domain mismatches, hallucinated objects.
      - If issues found, output a corrected problem.

    Returns: amended problem string (or unchanged if adequate).
    """
    cot_prompt = f"""You are a PDDL problem adequacy checker. Verify the problem matches the observation.

DOMAIN:
```
{domain_pddl}
```

PROBLEM:
```
{problem_pddl}
```

OBSERVED WORLD STATE:
{raw_observation}

KNOWN OBJECTS:
{objects_list}

CHAIN OF THOUGHT — work through these steps:

STEP 1: List all objects in (:objects) and all predicates in (:init).

STEP 2: For each (:init) predicate:
  - Does it use a predicate defined in the domain? (If not → DOMAIN_MISMATCH)
  - Does it correspond to an observed fact? (If not → HALLUCINATED_FACT)

STEP 3: For each critical observation fact:
  - Is it represented in (:init)? (If not → MISSING_FACT)
  - Note: Not all observations need init predicates (some are action-derived)

STEP 4: For each object in (:objects):
  - Is it mentioned in the observation or objects list? (If not → HALLUCINATED_OBJECT)

STEP 5: Summarize:
  DOMAIN_MISMATCHES: <count>
  HALLUCINATED_FACTS: <count>
  MISSING_FACTS: <count>
  HALLUCINATED_OBJECTS: <count>

STEP 6: If any issues found, output the corrected problem wrapped in <PROBLEM>...</PROBLEM> tags.
  If no issues, output ADEQUATE.

IMPORTANT: Do NOT change the (:goal) block. Only fix (:objects) and (:init)."""

    response = agent.llm_call(
        "You are a PDDL problem adequacy checker. Verify and fix the problem.",
        cot_prompt
    )

    if logs_dir:
        _save_prompt_response(
            prompt="[CoT: Problem adequacy check]",
            response=response,
            prefix="problem_adequacy",
            suffix="check",
            output_dir=logs_dir
        )

    if "ADEQUATE" in response and "<PROBLEM>" not in response:
        _logger.info("Problem Adequacy: Problem is adequate.")
        return problem_pddl

    if "<PROBLEM>" in response and "</PROBLEM>" in response:
        amended = response.split("<PROBLEM>")[1].split("</PROBLEM>")[0]
        amended = amended.replace("`", "").replace("pddl", "").replace("lisp", "").strip()
        amended = _preprocess_pddl(amended)
        _logger.info("Problem Adequacy: Problem amended.")
        return amended

    _logger.warning("Problem Adequacy: Could not parse response. Returning original problem.")
    return problem_pddl


# UNIFIED DOMAIN AND PROBLEM REFINEMENT #

def refine_domain_and_problem_unified(
    domain_file_path,
    problem_file_path,
    environment,
    task,
    logs_dir,
    workflow_iteration,
    refinement_iteration,
    agent: Agent,
    nl_sections=None,
    pddlenv_error_log = None,
    planner_error_log = None,
    VAL_validation_log = None,
    VAL_grounding_log = None,
    scene_graph_grounding_log = None,
    use_two_step_refinement = True,
    use_vector_db = False,
    vector_db_path = None,
    temperature=0.2,
    pddl_init_state=None,
    issue_stats=None
):
    """
    Unified refinement that handles both domain and problem together. 
    Strict Policy: Uses targeted NL sections for grounding and integrates Vector DB.
    """
    
    # 1) Read both files
    with open(problem_file_path, "r") as file:
        problem_pddl = file.read()
    
    with open(domain_file_path, "r") as file:
        domain_pddl = file.read()
    
    # ----------------------------------------------------------
    # STEP A: Separate Diagnoses (Domain + Problem)
    # ----------------------------------------------------------
    
    # Domain Diagnosis
    domain_diagnosis_system = (
        "You are a helpful planning assistant. Analyze the DOMAIN PDDL and error logs. "
        "Focus on issues in the DOMAIN definition (types, predicates, actions)."
    )
    domain_diagnosis_user = f"""
Below is the domain PDDL file:
```
{domain_pddl}
```

Below is the problem PDDL file (for context):
```
{problem_pddl}
```

The original task is: {task}
The scene is: {environment}

"""

    if pddlenv_error_log:
        domain_diagnosis_user += f"PDDLEnv Error: ```{pddlenv_error_log}```\n"
    if planner_error_log:
        domain_diagnosis_user += f"Planner Error: ```{planner_error_log}```\n"
    if VAL_validation_log:
        domain_diagnosis_user += f"VAL Validation Error: ```{VAL_validation_log}```\n"
    if VAL_grounding_log:
        domain_diagnosis_user += f"VAL Grounding Error: ```{VAL_grounding_log}```\n"

    domain_diagnosis_user += """
Analyze the DOMAIN and identify issues:
1. Incorrect predicate definitions (arity, types)
2. Inconsistent action preconditions/effects
3. Type hierarchy problems

Provide detailed suggestions. Do NOT rewrite the PDDL yet.
"""

    domain_diagnosis = agent.llm_call(domain_diagnosis_system, domain_diagnosis_user, temperature=temperature)
    
    _save_prompt_response(
        prompt=domain_diagnosis_system + "\n\n" + domain_diagnosis_user,
        response=domain_diagnosis,
        prefix="unified_domain_diagnosis",
        suffix=str(workflow_iteration)+"_"+str(refinement_iteration),
        output_dir=logs_dir
    )
    
    # Problem Diagnosis
    problem_diagnosis_system = (
        "You are a helpful planning assistant. Analyze the PROBLEM PDDL and error logs. "
        "Focus on issues in the PROBLEM definition (objects, init state, goal)."
    )
    problem_diagnosis_user = f"""
Below is the domain PDDL file (for context):
```
{domain_pddl}
```

Below is the problem PDDL file:
```
{problem_pddl}
```

The original task is: {task}
The scene is: {environment}

"""

    if pddlenv_error_log:
        problem_diagnosis_user += f"PDDLEnv Error: ```{pddlenv_error_log}```\n"
    if planner_error_log:
        problem_diagnosis_user += f"Planner Error: ```{planner_error_log}```\n"
    if VAL_validation_log:
        problem_diagnosis_user += f"VAL Validation Error: ```{VAL_validation_log}```\n"
    if VAL_grounding_log:
        problem_diagnosis_user += f"VAL Grounding Error: ```{VAL_grounding_log}```\n"

    # --- HEURISTIC CHECK ---
    heuristic_warnings = check_pddl_heuristics(domain_pddl, problem_pddl)
    if heuristic_warnings:
        problem_diagnosis_user += "\nDETECTED ISSUES (High Confidence Heuristics):\n"
        for w in heuristic_warnings:
             problem_diagnosis_user += f"- {w}\n"

    problem_diagnosis_user += """
Analyze the PROBLEM and identify issues:
1. Objects with incorrect or missing types
2. Invalid predicates in init state
3. Goal predicates not matching domain
4. Missing objects or predicates

Provide detailed suggestions. Do NOT rewrite the PDDL yet.
"""

    problem_diagnosis = agent.llm_call(problem_diagnosis_system, problem_diagnosis_user, temperature=temperature)
    
    _save_prompt_response(
        prompt=problem_diagnosis_system + "\n\n" + problem_diagnosis_user,
        response=problem_diagnosis,
        prefix="unified_problem_diagnosis",
        suffix=str(workflow_iteration)+"_"+str(refinement_iteration),
        output_dir=logs_dir
    )
    
    # Combine diagnoses for refinement
    combined_diagnosis = f"""
DOMAIN DIAGNOSIS:
{domain_diagnosis}

PROBLEM DIAGNOSIS:
{problem_diagnosis}
"""

    _save_prompt_response(
        prompt="Combined diagnosis from separate domain and problem analyses",
        response=combined_diagnosis,
        prefix="unified_combined_diagnosis",
        suffix=str(workflow_iteration)+"_"+str(refinement_iteration),
        output_dir=logs_dir
    )

    # ----------------------------------------------------------
    # STEP B: Unified Refinement Logic
    # ----------------------------------------------------------
    
    if use_two_step_refinement:
        # =============== TWO-STEP UNIFIED REFINEMENT ===============
        # Step 1: Generate Combined Refinement Plan
        plan_system_prompt = (
            "You are a helpful planning assistant. Based on the diagnosis, "
            "your job is to create a detailed, step-by-step refinement plan for BOTH the domain and problem PDDL. "
            "Do NOT rewrite the PDDL yet; just create a clear plan covering both files."
        )
        plan_user_prompt = f"""
Below is the current domain PDDL:
```
{domain_pddl}
```

Below is the current problem PDDL:
```
{problem_pddl}
```

Diagnosis of the issue:
```
{combined_diagnosis}
```

Now please create a detailed refinement plan with TWO sections:

### DOMAIN REFINEMENT PLAN:
1. List each predicate that needs to be modified, added, or removed (with specific arity and parameter types)
2. For each action that needs changes, specify:
   - Which preconditions to add/modify/remove
   - Which effects to add/modify/remove
3. List any type definitions that need to be added or changed

### PROBLEM REFINEMENT PLAN:
1. List each object that needs to be modified, added, or removed (with specific types)
2. List initial state predicates that need to be added, modified, or removed
3. Identify any goal predicates that need adjustment

IMPORTANT: Ensure changes in domain and problem are CONSISTENT with each other.
For example, if you add a type to the domain, make sure objects of that type are properly defined in the problem.

CRITICAL PDDL SYNTAX RULES:
- The domain uses :requirements :strips :typing
- :strips does NOT support quantifiers (exists, forall)
- All types defined in domain MUST match object types in problem
- Use simple preconditions only: (and), (not), or atomic predicates
- If you see unsupported constructs in the current PDDL, plan to REMOVE or REPLACE them
- Example: Instead of (exists (?x - object) ...), add a predicate like (hands-free ?agent)
- DO NOT use ADL type unions like (either block surface) in predicate parameters
- Each parameter must have a SINGLE type, not a union of types
- If you need a predicate to accept multiple types, create separate predicates for each type
- Example: Instead of (PREDICATE-A ?x - typeA ?y - (either typeA typeB)), use (PREDICATE-B ?x - typeA ?y - typeA) and (PREDICATE-C ?x - typeA). Where PREDICATE-A is a valid PDDL predicate defined in your domain.
- DO NOT define problem-specific objects as constants in the domain
- DO NOT create duplicate action definitions
- CRITICAL: PREDICATE NAMES MUST MATCH EXACTLY BETWEEN DOMAIN AND PROBLEM. If the simulator/init state uses `ontable` and `handempty`, the domain MUST define `ontable` and `handempty` exactly (no hyphens).
- DO NOT invent new hyphens or remove existing ones if it breaks the link between the domain and problem states.

Output your plan in a structured format with clear sections.
"""

        
        # Call LLM for refinement plan
        print(f"\n[REFINEMENT PLAN PROMPT]\n{plan_system_prompt}\n\n{plan_user_prompt}\n")
        refinement_plan = agent.llm_call(plan_system_prompt, plan_user_prompt, temperature=temperature)
        print(f"\n[REFINEMENT PLAN RESPONSE]\n{refinement_plan}\n")
        
        # Save the plan
        _save_prompt_response(
            prompt=plan_system_prompt + "\n\n" + plan_user_prompt,
            response=refinement_plan,
            prefix="unified_refinement_plan",
            suffix=str(workflow_iteration)+"_"+str(refinement_iteration),
            output_dir=logs_dir
        )
        
        # Step 2: Apply Unified Refinement Plan
        application_system_prompt = (
            "You are a helpful planning assistant. You have a detailed refinement plan "
            "for fixing BOTH the domain and problem PDDL. Your job is to carefully apply each change "
            "to generate the corrected domain and problem."
        )
        application_user_prompt = f"""
Below is the current domain PDDL:
```
{domain_pddl}
```

Below is the current problem PDDL:
```
{problem_pddl}
```

Below is the refinement plan to apply:
```
{refinement_plan}
```

The original task is: {task}
The environment is: {environment}

Now please:
1. Apply DOMAIN changes specified in the plan systematically
2. Apply PROBLEM changes specified in the plan systematically
3. REMOVE any unsupported PDDL constructs from BOTH domain and problem:
   - NO quantifiers (exists, forall) - :strips requirement doesn't support them
   - Replace quantified expressions with predicates or simplified logic
   - NO ADL type unions like (either block surface) in predicate parameters
   - Each parameter must have a SINGLE type, not a union of types
4. Ensure predicate usage is consistent between domain and problem
5. Verify that all objects in problem match type definitions in domain
6. Ensure type consistency: if domain defines 'agent - locatable', problem must use 'agent - locatable' (not 'agent - agent')
7. Double-check that every predicate call has the correct number of parameters
8. DO NOT define problem-specific objects as constants in the domain
9. DO NOT create duplicate action definitions
10. CRITICAL: The PROBLEM file must NEVER contain (:action ...) blocks. Actions belong ONLY in the domain file. If the problem file has any (:action ...) blocks, remove them entirely.
11. CRITICAL: PREDICATE NAMES MUST MATCH EXACTLY BETWEEN DOMAIN AND PROBLEM. If the original problem used `ontable`, the domain MUST use `ontable` (no hyphens). Do NOT invent new hyphens or remove existing ones if it breaks the link between the domain and problem states.
11. CRITICAL: You MUST NOT change the physical configuration of objects in the `(:init)` block. It represents the unchangeable Ground Truth of the simulator. If the goal is unsolvable from this state, you must fix the action preconditions/effects or domain logic, NOT the initial state.
12. Output BOTH corrected files with the following format:

<DOMAIN>
The new corrected PDDL domain here.
</DOMAIN>

<PROBLEM>
The new corrected PDDL problem here.
</PROBLEM>

Use always the tokens otherwise the entire system fails.
"""

        
        # Call LLM to apply the plan
        print(f"\n[REFINEMENT APPLICATION PROMPT]\n{application_system_prompt}\n\n{application_user_prompt}\n")
        llm_response = agent.llm_call(application_system_prompt, application_user_prompt)
        print(f"\n[REFINEMENT APPLICATION RESPONSE]\n{llm_response}\n")
        
        # Save prompt & response
        _save_prompt_response(
            prompt=application_system_prompt + "\n\n" + application_user_prompt,
            response=llm_response,
            prefix="unified_refinement_application",
            suffix=str(workflow_iteration)+"_"+str(refinement_iteration),
            output_dir=logs_dir
        )
    else:
        # =============== SINGLE-STEP UNIFIED REFINEMENT ===============
        # Targeted context for refinement
        if nl_sections:
            domain_nl = f"{nl_sections['description']}\n\n{nl_sections['actions']}\n\n{nl_sections['preconditions']}\n\n{nl_sections['effects']}"
            problem_nl = f"{nl_sections['objects']}\n\n{nl_sections['initial_state']}"
        else:
            domain_nl = environment
            problem_nl = environment

        correction_system_prompt = (
            "You are a helpful planning assistant expert in PDDL. "
            "Your job is to fix PDDL files based on execution / validation feedback while strictly "
            "adhering to the provided Natural Language ground truth."
        )
        correction_user_prompt = f"""
We have a PDDL domain and problem that failed to solve.

DOMAIN DESCRIPTION (Ground Truth for actions):
{domain_nl}

PROBLEM DESCRIPTION (Ground Truth for objects/state):
{problem_nl}

SUBGOAL (Target to reach):
{task}

DIAGNOSIS OF FAILURE:
{combined_diagnosis}

CURRENT DOMAIN:
```pddl
{domain_pddl}
```

CURRENT PROBLEM:
```pddl
{problem_pddl}
```

NOW REPAIR THE FILES:
1. Fix domain syntax, types, predicates, and actions.
2. Fix problem objects, init state, and goal to match the SUBGOAL exactly.
3. CRITICAL: The (:init) block MUST match the PROBLEM DESCRIPTION. Do NOT change it to bypass errors.
4. Output BOTH corrected files within <DOMAIN></DOMAIN> and <PROBLEM></PROBLEM> tags.
"""

        # ----------------------------------------------------------
        # STEP B: Vector DB Init & Search
        # ----------------------------------------------------------
        vdb = None
        similar_solution = ""
        if use_vector_db:
            if not vector_db_path:
                 base_dir = Path(logs_dir).parent.parent
                 vector_db_path = os.path.join(base_dir, "vector_db.json")
            
            client = getattr(agent, "client", None)
            if client:
                vdb = VectorDB(vector_db_path, client)
                # Search using the combined diagnosis
                results = vdb.search(combined_diagnosis)
                if results:
                    top_match = results[0]
                    if top_match.get('similarity', 0) > 0.8:
                        similar_solution = f"\nSimilar past solution found:\n{top_match.get('solution')}\n"

        if similar_solution:
            correction_user_prompt += f"\nREFERENCE SOLUTION:\n{similar_solution}\n"

        # Call the LLM for correction
        llm_response = agent.llm_call(correction_system_prompt, correction_user_prompt)

        # Save prompt & response
        _save_prompt_response(
            prompt=correction_system_prompt + "\\n\\n" + correction_user_prompt,
            response=llm_response,
            prefix="unified_refinement",
            suffix=str(workflow_iteration)+"_"+str(refinement_iteration),
            output_dir=logs_dir
        )

    # ----------------------------------------------------------
    # Extract both domain and problem from LLM response
    # ----------------------------------------------------------
    try:
        new_domain = llm_response.split("<DOMAIN>")[1].split("</DOMAIN>")[0]
    except IndexError:
        # Fallback
        new_domain = domain_pddl
    
    try:
        new_problem = llm_response.split("<PROBLEM>")[1].split("</PROBLEM>")[0]
    except IndexError:
        # Fallback
        new_problem = problem_pddl

    # Cleanup strings
    new_domain = new_domain.replace("`", "").replace("pddl", "").replace("lisp", "")
    new_problem = new_problem.replace("`", "").replace("pddl", "").replace("lisp", "")

    # Log the iteration
    iteration_log = f"Iteration #\nReason of failure:\n{combined_diagnosis}\n\nLLM correction:\n{llm_response}"
    unified_log_file_path = os.path.join(logs_dir, f"unified_refinement_{workflow_iteration}_{refinement_iteration}.txt")
    with open(unified_log_file_path, "a") as log_file:
        log_file.write(iteration_log)

    refinement_history = [{
        "issue": "Unified Diagnosis (Domain & Problem)",
        "solution": "Unified Refinement",
        "diagnosis": combined_diagnosis,
        "is_valid": False # Will be updated if VDB learned
    }]

    # 4) Internal Verification for VDB Learning
    if use_vector_db and vdb:
        # We need to save the candidate to a temporary file for VAL
        temp_domain_path = os.path.join(logs_dir, f"temp_domain_unified_{workflow_iteration}_{refinement_iteration}.pddl")
        temp_prob_path = os.path.join(logs_dir, f"temp_problem_unified_{workflow_iteration}_{refinement_iteration}.pddl")
        
        with open(temp_domain_path, "w") as f:
             f.write(new_domain)
        with open(temp_prob_path, "w") as f:
             f.write(new_problem)
        
        # Use VALVerifier
        verifier = VALVerifier()
        is_valid, _ = verifier.verify(temp_domain_path, temp_prob_path)
        
        if is_valid:
             print(f"Unified Refinement Result: VALID. Committing to DB.")
             vdb.add(combined_diagnosis, llm_response)
             refinement_history[0]["is_valid"] = True
        else:
             print(f"Unified Refinement Result: INVALID. Not committing to DB.")
        
        # Cleanup
        if os.path.exists(temp_domain_path): os.remove(temp_domain_path)
        if os.path.exists(temp_prob_path): os.remove(temp_prob_path)

    new_domain = _preprocess_pddl(new_domain)
    new_problem = _preprocess_pddl(new_problem)
    return new_domain, new_problem, refinement_history


#####################
# NL OBS GROUNDING  #
#####################

def ground_nl_observation(
    nl_observation: str,
    domain_file_path: str,
    goal: str,
    environment: str,
    agent: Agent,
    logs_dir: str = None,
    workflow_iteration=0,
    refinement_iteration=0,
):  # -> tuple[Optional[str], Optional[str]]
    """
    LLM-mediated grounding: map a natural-language perceptual observation to a
    PDDL :init block that uses ONLY the predicates declared in the generated domain.

    Returns (init_str, error_msg).
      - init_str: raw predicate lines (no (:init ...) wrapper) on success
      - error_msg: description of the mismatch/failure for refinement feedback, or None
    """
    with open(domain_file_path, "r") as f:
        domain_pddl = f.read()

    system_prompt = """\
You are an expert PDDL problem modeller.
Your task: given a natural-language observation of the current world state and a PDDL domain, \
produce the PDDL :init block that faithfully represents the observation.

STRICT RULES:
1. Use ONLY predicate names that are declared in (:predicates ...) of the provided domain.
2. Use ONLY type names declared in (:types ...) of the provided domain.
3. Invent short, descriptive object names that match the domain types (e.g. "red_ball", "room_a").
   Do NOT copy names from outside this task.
4. Output ONLY the raw predicate lines — no (:init ...) wrapper, no (:objects ...) block.
5. Each line must be a single grounded predicate, e.g.: (on-table orange_block)
6. If the observation mentions something that has no matching predicate in the domain, \
   OMIT it and note the mismatch at the end after the line "GROUNDING_ISSUES:".
7. If you cannot ground the observation AT ALL (domain is missing critical predicates), \
   output only: GROUNDING_FAILED: <reason>
"""

    question = f"""\
<domain>
{domain_pddl}
</domain>

<task>
{goal}
</task>

<environment_description>
{environment}
</environment_description>

<observation>
{nl_observation}
</observation>

Produce the grounded :init predicates now (raw lines only, one predicate per line).
"""

    answer = agent.llm_call(system_prompt, question)

    if logs_dir:
        _save_prompt_response(
            prompt=f"{system_prompt}\n\n{question}",
            response=answer,
            prefix="NL_obs_grounding",
            suffix=f"_{workflow_iteration}_{refinement_iteration}",
            output_dir=logs_dir,
        )

    # Parse answer
    if "GROUNDING_FAILED:" in answer:
        reason = answer.split("GROUNDING_FAILED:", 1)[1].strip().split("\n")[0]
        return None, f"GROUNDING_FAILED: {reason}"

    issues = None
    if "GROUNDING_ISSUES:" in answer:
        parts = answer.split("GROUNDING_ISSUES:", 1)
        answer = parts[0]
        issues = parts[1].strip()

    # Strip markdown fences if present
    init_lines = answer.replace("```", "").strip()
    error_msg = f"GROUNDING_ISSUES: {issues}" if issues else None
    return init_lines, error_msg

