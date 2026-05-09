"""
PDDL Domain Parser - Generic domain constraint extraction
Supports PDDL 1.2: hierarchical typing, derived predicates, quantifiers
"""

import re
from typing import Dict, List, Tuple, Set
import logging

logger = logging.getLogger(__name__)


def extract_domain_constraints(domain_pddl: str) -> Dict:
    """
    Extract ALL structural constraints from domain using PDDL-compliant parser.
    Works for PDDL 1.2 including hierarchical typing, derived predicates, quantifiers.
    
    Args:
        domain_pddl: String containing PDDL domain definition
        
    Returns:
        {
            'name': str,
            'predicates': [(name, arity, param_types)],
            'types': {type_name: parent_type},
            'requirements': [':strips', ':typing', ...],
            'constants': [const_name],
            'derived_predicates': [name]
        }
    """
    constraints = {
        'name': '',
        'predicates': [],
        'types': {},
        'requirements': [],
        'constants': [],
        'derived_predicates': []
    }
    
    # Extract domain name
    name_match = re.search(r'\(define\s+\(domain\s+([a-zA-Z0-9\-_]+)\)', domain_pddl)
    if name_match:
        constraints['name'] = name_match.group(1)
    
    # Extract requirements (always present in well-formed PDDL)
    req_match = re.search(r':requirements\s+(.*?)\)', domain_pddl, re.DOTALL)
    if req_match:
        constraints['requirements'] = req_match.group(1).split()
    
    # Extract types (support hierarchical: block - movable, movable - object)
    types_match = re.search(r':types\s+(.*?)\)', domain_pddl, re.DOTALL)
    if types_match:
        type_text = types_match.group(1).strip()
        # Parse hierarchical types: "a b c - parent_type"
        for line in type_text.split('\n'):
            line = line.strip()
            if '-' in line:
                parts = line.split('-')
                if len(parts) == 2:
                    children_str, parent = parts
                    parent = parent.strip()
                    for child in children_str.split():
                        child = child.strip()
                        if child:
                            constraints['types'][child] = parent
            else:
                # No parent specified, assume 'object'
                for t in line.split():
                    t = t.strip()
                    if t and t != '-':
                        constraints['types'][t] = 'object'
    
    # Extract predicates with arity and parameter types
    pred_block = re.search(
        r':predicates\s+(.*?)(?:\n\s*\)\s*\n\s*\(:action|\n\s*\)\s*\n\s*\(:derived|\n\s*\)\s*\Z)',
        domain_pddl,
        re.DOTALL
    )
    if pred_block:
        predicate_text = pred_block.group(1)
        # Match: (predicate-name ?param1 - type1 ?param2 - type2)
        for pred_match in re.finditer(r'\(([a-zA-Z0-9\-_]+)(.*?)\)', predicate_text, re.DOTALL):
            pred_name = pred_match.group(1)
            params_text = pred_match.group(2)
            
            # Extract parameter types
            param_types = []
            # Match patterns: ?var - type
            for param in re.finditer(r'\?[a-zA-Z0-9_]+ - ([a-zA-Z0-9\-_]+)', params_text):
                param_types.append(param.group(1))
            
            arity = len(param_types)
            constraints['predicates'].append((pred_name, arity, tuple(param_types)))
    
    # Extract derived predicates (if :derived-predicates requirement)
    if ':derived-predicates' in constraints['requirements']:
        # Match all :derived blocks
        for derived_match in re.finditer(r':derived\s+\(([a-zA-Z0-9\-_]+)', domain_pddl):
            derived_name = derived_match.group(1)
            constraints['derived_predicates'].append(derived_name)
    
    # Extract constants
    const_match = re.search(r':constants\s+(.*?)\)', domain_pddl, re.DOTALL)
    if const_match:
        constants_text = const_match.group(1)
        # Handle typed constants: const1 const2 - type
        current_consts = []
        for token in constants_text.split():
            if token == '-':
                continue  # Skip type separator
            elif token in constraints['types'].values():
                continue  # Skip type names
            else:
                constraints['constants'].append(token)
    
    logger.debug(f"Extracted constraints from domain '{constraints['name']}':")
    logger.debug(f"  Predicates: {len(constraints['predicates'])}")
    logger.debug(f"  Types: {len(constraints['types'])}")
    logger.debug(f"  Requirements: {constraints['requirements']}")
    
    return constraints


def format_predicate_guide(constraints: Dict) -> str:
    """
    Generate human-readable predicate usage guide.
    
    Returns formatted string showing available predicates with parameter types.
    """
    guide = "Available Predicates (use EXACTLY as shown):\n"
    
    for pred_name, arity, param_types in constraints['predicates']:
        if arity == 0:
            guide += f"  ({pred_name})\n"
        else:
            param_example = ' '.join([
                f'?{chr(97+i)} - {param_types[i]}' 
                for i in range(min(arity, len(param_types)))
            ])
            guide += f"  ({pred_name} {param_example})\n"
    
    return guide


def format_type_hierarchy(constraints: Dict) -> str:
    """
    Generate human-readable type hierarchy guide.
    
    Returns formatted string showing type inheritance.
    """
    if not constraints['types']:
        return "Type Hierarchy: (none defined)\n"
    
    guide = "Type Hierarchy:\n"
    for child, parent in sorted(constraints['types'].items()):
        guide += f"  {child} - {parent}\n"
    
    return guide


def extract_problem_objects(problem_pddl: str) -> List[str]:
    """
    Extract object names declared in the (:objects ...) block of a PDDL problem.
    Skips type-separator '-' tokens and type names.
    """
    objs: List[str] = []
    obj_match = re.search(r':objects\s+(.*?)\)', problem_pddl, re.DOTALL)
    if not obj_match:
        return objs
    text = obj_match.group(1)
    skip_next = False
    for token in text.split():
        if skip_next:
            skip_next = False
            continue
        if token == '-':
            skip_next = True
            continue
        objs.append(token)
    return objs


def structural_problem_issues(
    domain_pddl: str,
    problem_pddl: str,
) -> List[Dict]:
    """
    Deterministic structural pre-check of a problem against its domain.

    Walks the problem's (:init) and (:goal) blocks and emits one synthetic
    diagnosis entry per offending predicate / constant — pre-classified with
    the appropriate TaxonomyEntry. Avoids relying on the LLM diagnosis to
    notice these (which has been a recurrent miss in v5 traces).

    Returns a list of dicts with keys:
        issue, solution, missing_init_facts, _taxonomy_entries (list[TaxonomyEntry])
    """
    from src.lapis.planner.low.pddl_taxonomy import TAXONOMY, ErrorClass

    constraints = extract_domain_constraints(domain_pddl)
    valid_predicates = {p[0] for p in constraints['predicates']}
    valid_predicates |= set(constraints.get('derived_predicates', []))
    declared_objects = set(extract_problem_objects(problem_pddl))
    declared_objects |= set(constraints.get('constants', []))

    pddl_keywords = {
        'define', 'domain', 'problem', 'objects', 'init', 'goal',
        'and', 'or', 'not', 'exists', 'forall', 'when', 'imply',
        ':domain', ':objects', ':init', ':goal', ':requirements', ':metric',
    }

    invented_pred_entry = next(
        (e for e in TAXONOMY
         if e.error_class == ErrorClass.P2_1_HALLUCINATION
         and e.subtype == "invented_predicate"),
        None,
    )
    constant_drift_entry = next(
        (e for e in TAXONOMY
         if e.error_class == ErrorClass.P2_1_HALLUCINATION
         and e.subtype == "constant_name_drift"),
        None,
    )

    issues: List[Dict] = []
    seen_preds: set = set()
    seen_consts: set = set()

    def _slice(block_re: str) -> str:
        m = re.search(block_re, problem_pddl, re.DOTALL)
        return m.group(0) if m else ""

    init_text = _slice(r'\(:init\s.*?\)\s*\(:goal') or _slice(r'\(:init\s.*$')
    goal_text = _slice(r'\(:goal\s.*$')
    body_text = init_text + "\n" + goal_text
    if not body_text.strip():
        body_text = problem_pddl

    if valid_predicates and invented_pred_entry is not None:
        from difflib import SequenceMatcher
        for m in re.finditer(r'\(([a-zA-Z0-9\-_:]+)(?:\s|\))', body_text):
            pred = m.group(1)
            if pred in pddl_keywords or pred.startswith(':') or pred in seen_preds:
                continue
            if pred in valid_predicates:
                continue
            seen_preds.add(pred)
            best, best_score = None, 0.0
            for valid in valid_predicates:
                s = SequenceMatcher(None, pred, valid).ratio()
                if s > best_score:
                    best, best_score = valid, s
            suggestion = (
                f" Closest valid predicate in the domain: '{best}'."
                if best and best_score >= 0.6 else ""
            )
            issues.append({
                "issue": (
                    f"Predicate '{pred}' is used in the problem but is not "
                    f"declared in the domain (:predicates) block.{suggestion}"
                ),
                "solution": (
                    f"Replace every occurrence of '({pred} ...)' with a "
                    "predicate that is declared in the domain. Use ONLY "
                    "predicate names from the domain's (:predicates) block."
                ),
                "correctness_check": (
                    f"Re-grep the problem for '({pred}' — there must be zero matches."
                ),
                "missing_init_facts": [],
                "_taxonomy_entries": [invented_pred_entry],
                "_source": "structural_pre_check",
            })

    if declared_objects and constant_drift_entry is not None:
        from difflib import SequenceMatcher
        for m in re.finditer(r'\(([a-zA-Z0-9\-_:]+)([^()]*)\)', body_text):
            head = m.group(1)
            args_text = m.group(2)
            if head in pddl_keywords or head.startswith(':'):
                continue
            for token in args_text.split():
                if token.startswith('?') or token in pddl_keywords:
                    continue
                if token in declared_objects or token in seen_consts:
                    continue
                seen_consts.add(token)
                best, best_score = None, 0.0
                for obj in declared_objects:
                    s = SequenceMatcher(None, token, obj).ratio()
                    if s > best_score:
                        best, best_score = obj, s
                suggestion = (
                    f" Closest declared object: '{best}'."
                    if best and best_score >= 0.6 else ""
                )
                issues.append({
                    "issue": (
                        f"Constant '{token}' is referenced in (:init) or "
                        f"(:goal) but is not declared in (:objects) and is "
                        f"not a domain :constants entry.{suggestion}"
                    ),
                    "solution": (
                        f"Either declare '{token}' under (:objects) with the "
                        f"correct type, or replace it with a declared object "
                        f"name from the domain's GROUND TRUTH OBJECT NAMES."
                    ),
                    "correctness_check": (
                        f"Every token used in (:init) and (:goal) must appear "
                        f"in (:objects) or be a domain :constants entry."
                    ),
                    "missing_init_facts": [],
                    "_taxonomy_entries": [constant_drift_entry],
                    "_source": "structural_pre_check",
                })

    return issues


def validate_predicate_usage(
    pddl: str,
    constraints: Dict
) -> Tuple[bool, List[str]]:
    """
    Validate that all predicates used in PDDL match domain constraints.
    
    Args:
        pddl: PDDL problem/domain content to validate
        constraints: Domain constraints from extract_domain_constraints()
        
    Returns:
        (is_valid, error_messages)
    """
    errors = []
    valid_predicate_names = {p[0] for p in constraints['predicates']}
    
    # Extract all predicate uses
    used_predicates = set()
    for match in re.finditer(r'\(([a-zA-Z0-9\-_]+)\s', pddl):
        pred = match.group(1)
        # Filter out PDDL keywords
        if pred not in [
            'define', 'domain', 'problem', 'objects', 'init', 'goal',
            'and', 'or', 'not', 'exists', 'forall', 'when', 'imply'
        ]:
            used_predicates.add(pred)
    
    # Check each used predicate
    for pred in used_predicates:
        if pred not in valid_predicate_names:
            # Find similar valid predicates (typo detection)
            from difflib import SequenceMatcher
            similar = []
            for valid_pred in valid_predicate_names:
                similarity = SequenceMatcher(None, pred, valid_pred).ratio()
                if similarity > 0.6:
                    similar.append((valid_pred, similarity))
            
            if similar:
                similar.sort(key=lambda x: x[1], reverse=True)
                suggestion = similar[0][0]
                errors.append(
                    f"Invalid predicate '{pred}'. Did you mean '{suggestion}'?"
                )
            else:
                errors.append(
                    f"Invalid predicate '{pred}' (not defined in domain)"
                )
    
    return len(errors) == 0, errors
