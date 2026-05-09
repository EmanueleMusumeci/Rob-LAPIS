"""
Semantic Verification for PDDL Domains and Problems

Lightweight structural analyzer that catches semantically broken PDDL files that VAL approves.
VAL only checks syntactic correctness; Planetarium (NAACL 2025) showed that 96% of LLM-generated
PDDL is syntactically valid but only 25% is semantically correct.

This module implements three core checks:
1. Predicate coverage: Can all goal predicates be achieved via action effects?
2. Action reachability: Can actions fire given the initial state?
3. Type grounding: Do all action parameters have at least one valid object?

ARCHITECTURE:
- Pluggable extraction backends (regex or UP parser)
- Easy to disable/enable different extraction strategies
- Graceful fallback when parsers fail
"""

import re
import logging
from abc import ABC, abstractmethod
from typing import Dict, List, Set, Tuple, Optional
from pathlib import Path
import tempfile

logger = logging.getLogger(__name__)


# ============================================================================
# PLUGGABLE EXTRACTION STRATEGIES
# ============================================================================
# This allows easy switching between different PDDL extraction backends
# Implement PDDLExtractor to add new extraction methods

class PDDLExtractor(ABC):
    """Abstract base class for PDDL extraction strategies."""

    @abstractmethod
    def extract_predicates(self, domain_pddl: str) -> Set[str]:
        """Extract predicate names from domain."""
        pass

    @abstractmethod
    def extract_action_effects(self, domain_pddl: str) -> Dict[str, Set[str]]:
        """Extract predicates from action effects."""
        pass

    @abstractmethod
    def extract_action_preconditions(self, domain_pddl: str) -> Dict[str, Set[str]]:
        """Extract predicates from action preconditions."""
        pass

    @abstractmethod
    def extract_action_parameters(self, domain_pddl: str) -> Dict[str, List[Tuple[str, str]]]:
        """Extract action parameters with types."""
        pass

    @abstractmethod
    def extract_goal_predicates(self, problem_pddl: str) -> Set[str]:
        """Extract goal predicates from problem."""
        pass

    @abstractmethod
    def extract_init_predicates(self, problem_pddl: str) -> Set[str]:
        """Extract init state predicates from problem."""
        pass

    @abstractmethod
    def extract_types(self, domain_pddl: str) -> Dict[str, str]:
        """Extract type hierarchy."""
        pass

    @abstractmethod
    def extract_objects(self, problem_pddl: str) -> Dict[str, Set[str]]:
        """Extract objects by type from problem."""
        pass


class RegexPDDLExtractor(PDDLExtractor):
    """PDDL extraction using regex and balanced parenthesis matching.

    Advantages: Works with any text, no external dependencies
    Disadvantages: Less semantic awareness, potential edge cases
    """

    def __init__(self):
        """Initialize the regex-based extractor."""
        pass

    def extract_predicates(self, domain_pddl: str) -> Set[str]:
        """Extract all predicate names from domain."""
        pred_section = self._extract_predicates_section(domain_pddl)
        predicates = set()
        for match in re.finditer(r'\(([a-zA-Z][a-zA-Z0-9_-]*)', pred_section):
            predicates.add(match.group(1).lower())
        return predicates

    def extract_action_effects(self, domain_pddl: str) -> Dict[str, Set[str]]:
        """Extract predicates from action effects."""
        # Delegated to module-level function
        return _extract_action_effects_regex(domain_pddl)

    def extract_action_preconditions(self, domain_pddl: str) -> Dict[str, Set[str]]:
        """Extract predicates from action preconditions."""
        return _extract_action_preconditions_regex(domain_pddl)

    def extract_action_parameters(self, domain_pddl: str) -> Dict[str, List[Tuple[str, str]]]:
        """Extract action parameters with types."""
        return _extract_action_parameters_regex(domain_pddl)

    def extract_goal_predicates(self, problem_pddl: str) -> Set[str]:
        """Extract goal predicates from problem."""
        return _extract_goal_predicates_regex(problem_pddl)

    def extract_init_predicates(self, problem_pddl: str) -> Set[str]:
        """Extract init state predicates from problem."""
        return _extract_init_predicates_regex(problem_pddl)

    def extract_types(self, domain_pddl: str) -> Dict[str, str]:
        """Extract type hierarchy."""
        return _extract_types_regex(domain_pddl)

    def extract_objects(self, problem_pddl: str) -> Dict[str, Set[str]]:
        """Extract objects by type from problem."""
        return _extract_objects_regex(problem_pddl)

    def _extract_predicates_section(self, domain_pddl: str) -> str:
        """Extract the :predicates section content using balanced parenthesis matching."""
        domain_clean = _remove_comments(domain_pddl)

        # Find :predicates position
        pred_start = domain_clean.lower().find(':predicates')
        if pred_start == -1:
            return ""

        # In PDDL, (:predicates ...) has the opening ( BEFORE :predicates
        paren_start = domain_clean.rfind('(', 0, pred_start)
        if paren_start == -1 or domain_clean[paren_start:pred_start].strip() != '(':
            paren_start = domain_clean.rfind('(', max(0, pred_start - 100), pred_start)
        if paren_start == -1:
            return ""

        # Find matching closing ) using balanced parenthesis counting
        paren_count = 1
        pos = paren_start + 1
        while pos < len(domain_clean) and paren_count > 0:
            if domain_clean[pos] == '(':
                paren_count += 1
            elif domain_clean[pos] == ')':
                paren_count -= 1
            pos += 1

        if paren_count == 0:
            return domain_clean[paren_start + 1:pos - 1]

        return ""


class UPParserExtractor(PDDLExtractor):
    """PDDL extraction using Unified Planning parser.

    Advantages: Semantic-aware, type-safe, no false positives
    Disadvantages: Requires valid PDDL, external dependency
    """

    def __init__(self):
        """Initialize the UP parser-based extractor."""
        try:
            from unified_planning import io
            self.reader = io.PDDLReader()
            self.available = True
        except ImportError:
            logger.warning("Unified Planning not available; UP extractor disabled")
            self.available = False
            self.reader = None

    def _parse_problem(self, domain_pddl: str, problem_pddl: str):
        """Parse domain and problem using UP parser."""
        if not self.available:
            raise RuntimeError("UP parser not available")

        try:
            # UP parser requires files or uses temp files
            with tempfile.TemporaryDirectory() as tmpdir:
                domain_file = Path(tmpdir) / "domain.pddl"
                problem_file = Path(tmpdir) / "problem.pddl"

                domain_file.write_text(domain_pddl)
                problem_file.write_text(problem_pddl)

                problem = self.reader.parse_problem_string(domain_pddl, problem_pddl)
                return problem
        except Exception as e:
            logger.warning(f"UP parser failed: {e}. Falling back to regex.")
            return None

    def extract_predicates(self, domain_pddl: str) -> Set[str]:
        """Extract all predicate names from domain."""
        # We don't have a problem here, so parse with a dummy problem
        dummy_problem = "(define (problem dummy) (:domain tmp) (:objects) (:init) (:goal (and)))"
        # Extract domain name from PDDL
        domain_match = re.search(r'\(define\s+\(domain\s+([a-zA-Z0-9_-]+)', domain_pddl)
        domain_name = domain_match.group(1) if domain_match else "tmp"
        dummy_problem = f"(define (problem dummy) (:domain {domain_name}) (:objects) (:init) (:goal (and)))"

        problem = self._parse_problem(domain_pddl, dummy_problem)
        if problem is None:
            return RegexPDDLExtractor().extract_predicates(domain_pddl)

        return {fluent.name for fluent in problem.fluents}

    def extract_action_effects(self, domain_pddl: str) -> Dict[str, Set[str]]:
        """Extract predicates from action effects."""
        # Need a problem to parse domain with UP
        try:
            domain_match = re.search(r'\(define\s+\(domain\s+([a-zA-Z0-9_-]+)', domain_pddl)
            domain_name = domain_match.group(1) if domain_match else "tmp"
            dummy_problem = f"(define (problem dummy) (:domain {domain_name}) (:objects) (:init) (:goal (and)))"

            problem = self._parse_problem(domain_pddl, dummy_problem)
            if problem is None:
                return RegexPDDLExtractor().extract_action_effects(domain_pddl)

            effects_dict = {}
            for action in problem.actions:
                preds = set()
                for effect in action.effects:
                    if hasattr(effect, 'fluent'):
                        preds.add(effect.fluent.name)
                effects_dict[action.name] = preds
            return effects_dict
        except Exception as e:
            logger.debug(f"UP extractor failed for effects: {e}")
            return RegexPDDLExtractor().extract_action_effects(domain_pddl)

    def extract_action_preconditions(self, domain_pddl: str) -> Dict[str, Set[str]]:
        """Extract predicates from action preconditions."""
        try:
            domain_match = re.search(r'\(define\s+\(domain\s+([a-zA-Z0-9_-]+)', domain_pddl)
            domain_name = domain_match.group(1) if domain_match else "tmp"
            dummy_problem = f"(define (problem dummy) (:domain {domain_name}) (:objects) (:init) (:goal (and)))"

            problem = self._parse_problem(domain_pddl, dummy_problem)
            if problem is None:
                return RegexPDDLExtractor().extract_action_preconditions(domain_pddl)

            preconds_dict = {}
            for action in problem.actions:
                preds = set()
                for prec in action.preconditions:
                    if hasattr(prec, 'fluent'):
                        preds.add(prec.fluent.name)
                preconds_dict[action.name] = preds
            return preconds_dict
        except Exception as e:
            logger.debug(f"UP extractor failed for preconditions: {e}")
            return RegexPDDLExtractor().extract_action_preconditions(domain_pddl)

    def extract_action_parameters(self, domain_pddl: str) -> Dict[str, List[Tuple[str, str]]]:
        """Extract action parameters with types."""
        try:
            domain_match = re.search(r'\(define\s+\(domain\s+([a-zA-Z0-9_-]+)', domain_pddl)
            domain_name = domain_match.group(1) if domain_match else "tmp"
            dummy_problem = f"(define (problem dummy) (:domain {domain_name}) (:objects) (:init) (:goal (and)))"

            problem = self._parse_problem(domain_pddl, dummy_problem)
            if problem is None:
                return RegexPDDLExtractor().extract_action_parameters(domain_pddl)

            params_dict = {}
            for action in problem.actions:
                params = [(param.name, param.type.basename) for param in action.parameters]
                params_dict[action.name] = params
            return params_dict
        except Exception as e:
            logger.debug(f"UP extractor failed for parameters: {e}")
            return RegexPDDLExtractor().extract_action_parameters(domain_pddl)

    def extract_goal_predicates(self, problem_pddl: str) -> Set[str]:
        """Extract goal predicates from problem."""
        # Can't parse without domain
        return RegexPDDLExtractor().extract_goal_predicates(problem_pddl)

    def extract_init_predicates(self, problem_pddl: str) -> Set[str]:
        """Extract init predicates from problem."""
        # Can't parse without domain
        return RegexPDDLExtractor().extract_init_predicates(problem_pddl)

    def extract_types(self, domain_pddl: str) -> Dict[str, str]:
        """Extract type hierarchy."""
        # Can't easily extract from UP without problem
        return RegexPDDLExtractor().extract_types(domain_pddl)

    def extract_objects(self, problem_pddl: str) -> Dict[str, Set[str]]:
        """Extract objects by type from problem."""
        # Can't parse without domain
        return RegexPDDLExtractor().extract_objects(problem_pddl)


class ExtractorFactory:
    """Factory for creating PDDL extractors."""

    _extractors = {
        "regex": RegexPDDLExtractor,
        "up": UPParserExtractor,
    }

    @classmethod
    def create(cls, extractor_type: str = "auto") -> PDDLExtractor:
        """Create an extractor of the specified type.

        Args:
            extractor_type: "regex" (always works), "up" (better but requires UP),
                           or "auto" (tries UP, falls back to regex)

        Returns:
            PDDLExtractor instance
        """
        if extractor_type == "auto":
            # Try UP first, fall back to regex
            try:
                up_extractor = UPParserExtractor()
                if up_extractor.available:
                    logger.info("Using UP parser for PDDL extraction")
                    return up_extractor
            except Exception:
                pass
            logger.info("Falling back to regex-based PDDL extraction")
            return RegexPDDLExtractor()

        elif extractor_type in cls._extractors:
            extractor_class = cls._extractors[extractor_type]
            instance = extractor_class()
            logger.info(f"Using {extractor_type} extractor for PDDL extraction")
            return instance

        else:
            raise ValueError(f"Unknown extractor type: {extractor_type}")

    @classmethod
    def register(cls, name: str, extractor_class: type):
        """Register a new extractor type."""
        if not issubclass(extractor_class, PDDLExtractor):
            raise TypeError(f"{extractor_class} must inherit from PDDLExtractor")
        cls._extractors[name] = extractor_class


# ============================================================================
# PDDL Parsing Helpers
# ============================================================================

def _remove_comments(pddl: str) -> str:
    """Remove PDDL comments (lines starting with ;)."""
    return '\n'.join(
        line for line in pddl.split('\n')
        if not line.strip().startswith(';')
    )


def _extract_predicates_section(domain_pddl: str) -> str:
    """Extract the :predicates section content using balanced parenthesis matching."""
    domain_clean = _remove_comments(domain_pddl)

    # Find :predicates position
    pred_start = domain_clean.lower().find(':predicates')
    if pred_start == -1:
        return ""

    # In PDDL, (:predicates ...) has the opening ( BEFORE :predicates
    # Find the opening ( that precedes :predicates
    paren_start = domain_clean.rfind('(', 0, pred_start)
    if paren_start == -1 or domain_clean[paren_start:pred_start].strip() != '(':
        # Fallback: just look for ( before :predicates on the same logical line
        paren_start = domain_clean.rfind('(', max(0, pred_start - 100), pred_start)
    if paren_start == -1:
        return ""

    # Find matching closing ) using balanced parenthesis counting
    paren_count = 1
    pos = paren_start + 1
    while pos < len(domain_clean) and paren_count > 0:
        if domain_clean[pos] == '(':
            paren_count += 1
        elif domain_clean[pos] == ')':
            paren_count -= 1
        pos += 1

    if paren_count == 0:
        # Extract content between the outer parentheses
        return domain_clean[paren_start + 1:pos - 1]

    return ""


def _extract_action_effects_regex(domain_pddl: str) -> Dict[str, Set[str]]:
    """
    Extract predicates from action effects.

    Returns:
        Dict mapping action names to sets of predicate names in their effects
    """
    action_effects = {}
    domain_clean = _remove_comments(domain_pddl)

    # Find all action blocks
    action_pattern = re.compile(
        r':action\s+([a-zA-Z0-9_-]+)(.*?)(?=\(:action|\Z)',
        re.DOTALL | re.IGNORECASE
    )

    for match in action_pattern.finditer(domain_clean):
        action_name = match.group(1).lower()
        action_body = match.group(2)

        # Find effect section
        effect_start = action_body.lower().find(':effect')

        predicates = set()
        if effect_start != -1:
            effect_section = action_body[effect_start:]

            # Match predicates: (name args...) or (name)
            for pred_match in re.finditer(r'\(([a-zA-Z][a-zA-Z0-9_-]*)(?:\s|\))', effect_section):
                pred = pred_match.group(1).lower()
                # Filter out PDDL keywords
                if pred not in {'and', 'or', 'not', 'when', 'forall', 'exists', 'increase', 'decrease', 'assign', 'effect'}:
                    predicates.add(pred)

        action_effects[action_name] = predicates

    return action_effects


def _extract_action_preconditions_regex(domain_pddl: str) -> Dict[str, Set[str]]:
    """
    Extract predicates from action preconditions.

    Returns:
        Dict mapping action names to sets of predicate names in their preconditions
    """
    action_preconditions = {}
    domain_clean = _remove_comments(domain_pddl)

    action_pattern = re.compile(
        r':action\s+([a-zA-Z0-9_-]+)(.*?)(?=\(:action|\Z)',
        re.DOTALL | re.IGNORECASE
    )

    for match in action_pattern.finditer(domain_clean):
        action_name = match.group(1).lower()
        action_body = match.group(2)

        # Find precondition section - from :precondition to :effect
        precond_start = action_body.lower().find(':precondition')
        effect_start = action_body.lower().find(':effect')

        predicates = set()
        if precond_start != -1:
            end_pos = effect_start if effect_start != -1 else len(action_body)
            precond_section = action_body[precond_start:end_pos]

            # Match predicates: (name args...) or (name)
            for pred_match in re.finditer(r'\(([a-zA-Z][a-zA-Z0-9_-]*)(?:\s|\))', precond_section):
                pred = pred_match.group(1).lower()
                if pred not in {'and', 'or', 'not', 'imply', 'forall', 'exists', 'precondition'}:
                    predicates.add(pred)

        action_preconditions[action_name] = predicates

    return action_preconditions


def _extract_action_parameters_regex(domain_pddl: str) -> Dict[str, List[Tuple[str, str]]]:
    """
    Extract typed parameters from actions.

    Returns:
        Dict mapping action names to list of (param_name, type_name) tuples
    """
    action_params = {}
    domain_clean = _remove_comments(domain_pddl)

    action_pattern = re.compile(
        r':action\s+([a-zA-Z0-9_-]+)\s*:parameters\s*\(([^)]*)\)',
        re.DOTALL | re.IGNORECASE
    )

    for match in action_pattern.finditer(domain_clean):
        action_name = match.group(1).lower()
        params_text = match.group(2).strip()

        params = []
        if params_text:
            # Parse typed parameters: ?x - type ?y - type
            # Also handle: ?x ?y - type (multiple params, same type)
            current_vars = []
            tokens = params_text.split()
            i = 0
            while i < len(tokens):
                token = tokens[i]
                if token.startswith('?'):
                    current_vars.append(token)
                elif token == '-':
                    # Next token is the type
                    if i + 1 < len(tokens):
                        ptype = tokens[i + 1].lower()
                        for var in current_vars:
                            params.append((var, ptype))
                        current_vars = []
                        i += 1
                i += 1

            # Handle untyped parameters (assume 'object')
            for var in current_vars:
                params.append((var, 'object'))

        action_params[action_name] = params

    return action_params


def _extract_goal_predicates_regex(problem_pddl: str) -> Set[str]:
    """Extract predicate names from the goal section."""
    problem_clean = _remove_comments(problem_pddl)

    # Find goal section - match :goal followed by balanced parentheses
    # Use simpler approach: find :goal and extract content until end or next section
    goal_start = problem_clean.lower().find(':goal')
    if goal_start == -1:
        return set()

    # Extract from :goal onwards and find predicates
    goal_section = problem_clean[goal_start:]
    predicates = set()

    # Match predicates: (name args...) where name is not a keyword
    # Also match (name) for 0-arity predicates
    for pred_match in re.finditer(r'\(([a-zA-Z][a-zA-Z0-9_-]*)(?:\s|\))', goal_section):
        pred = pred_match.group(1).lower()
        if pred not in {'and', 'or', 'not', 'imply', 'forall', 'exists', 'goal', 'define', 'problem'}:
            predicates.add(pred)

    return predicates


def _extract_init_predicates_regex(problem_pddl: str) -> Set[str]:
    """Extract predicate names from the init section."""
    problem_clean = _remove_comments(problem_pddl)

    # Find init section - from :init to :goal or :metric
    init_start = problem_clean.lower().find(':init')
    if init_start == -1:
        return set()

    # Find end of init section (start of :goal or :metric or end)
    goal_start = problem_clean.lower().find(':goal')
    metric_start = problem_clean.lower().find(':metric')

    end_pos = len(problem_clean)
    if goal_start != -1:
        end_pos = min(end_pos, goal_start)
    if metric_start != -1:
        end_pos = min(end_pos, metric_start)

    init_section = problem_clean[init_start:end_pos]
    predicates = set()

    # Match predicates: (name args...) or (name)
    for pred_match in re.finditer(r'\(([a-zA-Z][a-zA-Z0-9_-]*)(?:\s|\))', init_section):
        pred = pred_match.group(1).lower()
        # Filter out numeric functions and PDDL keywords
        if pred not in {'and', '=', '<', '>', '<=', '>=', 'init', 'define', 'problem'}:
            predicates.add(pred)

    return predicates


def _extract_types_regex(domain_pddl: str) -> Dict[str, str]:
    """
    Extract type hierarchy from domain.

    Returns:
        Dict mapping type names to their parent type
    """
    domain_clean = _remove_comments(domain_pddl)
    types = {}

    types_match = re.search(r':types\s+(.*?)(?=\(:|\)\s*\(:|\)\s*$)', domain_clean, re.DOTALL)
    if not types_match:
        return types

    types_text = types_match.group(1).strip()

    # Parse hierarchical types: "child1 child2 - parent"
    current_types = []
    tokens = types_text.split()
    i = 0
    while i < len(tokens):
        token = tokens[i].strip()
        if not token or token == ')':
            i += 1
            continue
        if token == '-':
            # Next token is the parent type
            if i + 1 < len(tokens):
                parent = tokens[i + 1].lower().rstrip(')')
                for t in current_types:
                    types[t] = parent
                current_types = []
                i += 1
        else:
            current_types.append(token.lower())
        i += 1

    # Remaining types without explicit parent -> assume 'object'
    for t in current_types:
        types[t] = 'object'

    return types


def _extract_objects_regex(problem_pddl: str) -> Dict[str, Set[str]]:
    """
    Extract typed objects from problem.

    Returns:
        Dict mapping type names to sets of object names
    """
    problem_clean = _remove_comments(problem_pddl)
    objects_by_type = {}

    # Find objects section
    objects_match = re.search(r':objects\s+(.*?)(?=\(:|\)\s*\(:|\)\s*$)', problem_clean, re.DOTALL)
    if not objects_match:
        return objects_by_type

    objects_text = objects_match.group(1).strip()

    # Parse typed objects: "obj1 obj2 - type"
    current_objs = []
    tokens = objects_text.split()
    i = 0
    while i < len(tokens):
        token = tokens[i].strip()
        if not token or token == ')':
            i += 1
            continue
        if token == '-':
            if i + 1 < len(tokens):
                obj_type = tokens[i + 1].lower().rstrip(')')
                if obj_type not in objects_by_type:
                    objects_by_type[obj_type] = set()
                objects_by_type[obj_type].update(current_objs)
                current_objs = []
                i += 1
        else:
            current_objs.append(token.lower())
        i += 1

    # Untyped objects -> type 'object'
    if current_objs:
        if 'object' not in objects_by_type:
            objects_by_type['object'] = set()
        objects_by_type['object'].update(current_objs)

    return objects_by_type


def _get_subtypes_regex(base_type: str, type_hierarchy: Dict[str, str]) -> Set[str]:
    """Get all types that are subtypes of base_type (including base_type itself)."""
    subtypes = {base_type}
    changed = True
    while changed:
        changed = False
        for child, parent in type_hierarchy.items():
            if parent in subtypes and child not in subtypes:
                subtypes.add(child)
                changed = True
    return subtypes


# ============================================================================
# Global Extractor Configuration
# ============================================================================
# Control which extraction backend is used
_extractor_type = "auto"  # "auto", "regex", or "up"
_extractor = None


def set_extractor_type(extractor_type: str):
    """Set the global PDDL extractor type.

    Args:
        extractor_type: "auto" (try UP, fall back to regex),
                       "regex" (always use regex),
                       "up" (always use UP parser)
    """
    global _extractor, _extractor_type
    _extractor_type = extractor_type
    _extractor = ExtractorFactory.create(extractor_type)
    logger.info(f"Switched to {extractor_type} extractor")


def get_extractor() -> PDDLExtractor:
    """Get the current PDDL extractor instance."""
    global _extractor, _extractor_type
    if _extractor is None:
        _extractor = ExtractorFactory.create(_extractor_type)
    return _extractor


# ============================================================================
# Core Verification Functions
# ============================================================================

def check_predicate_coverage(domain_pddl: str, problem_pddl: str) -> dict:
    """
    Check if all goal predicates can potentially be achieved.

    A goal predicate is achievable if it appears in at least one action's effects,
    OR if it's already true in the initial state.

    Returns:
        {
            "passed": bool,
            "goal_predicates": list[str],
            "achievable_predicates": list[str],  # predicates in action effects
            "init_predicates": list[str],  # predicates true in init
            "unreachable_goals": list[str],  # goal predicates not achievable
            "diagnosis": str
        }
    """
    goal_predicates = _extract_goal_predicates_regex(problem_pddl)
    init_predicates = _extract_init_predicates_regex(problem_pddl)
    action_effects = _extract_action_effects_regex(domain_pddl)

    # All predicates that can be made true
    achievable = set()
    for action_name, preds in action_effects.items():
        achievable.update(preds)

    # A goal predicate is reachable if it's in init or achievable via effects
    reachable_goals = goal_predicates.intersection(achievable.union(init_predicates))
    unreachable_goals = goal_predicates - reachable_goals

    passed = len(unreachable_goals) == 0

    diagnosis = ""
    if not passed:
        diagnosis = f"SEMANTIC ERROR: {len(unreachable_goals)} goal predicate(s) cannot be achieved.\n"
        diagnosis += f"Unreachable goals: {sorted(unreachable_goals)}\n"
        diagnosis += "These predicates are not produced by any action effect and not in init state.\n"
        diagnosis += "Fix: Add actions that produce these predicates, or correct the goal specification."

    return {
        "passed": passed,
        "goal_predicates": sorted(goal_predicates),
        "achievable_predicates": sorted(achievable),
        "init_predicates": sorted(init_predicates),
        "unreachable_goals": sorted(unreachable_goals),
        "diagnosis": diagnosis
    }


def check_action_reachability(domain_pddl: str, problem_pddl: str) -> dict:
    """
    Check if actions can fire given init state.

    A simple reachability check: an action is potentially reachable if at least
    one of its precondition predicates is either:
    - In the initial state, OR
    - Produced by some action effect (transitive closure approximation)

    This is a conservative approximation - it may report some unreachable
    actions as reachable, but won't miss truly unreachable ones.

    Returns:
        {
            "passed": bool,
            "total_actions": int,
            "reachable_actions": list[str],
            "unreachable_actions": list[str],  # actions that can never fire
            "diagnosis": str
        }
    """
    action_preconditions = _extract_action_preconditions_regex(domain_pddl)
    action_effects = _extract_action_effects_regex(domain_pddl)
    init_predicates = _extract_init_predicates_regex(problem_pddl)

    # Compute reachable predicates via fixpoint iteration
    reachable_predicates = set(init_predicates)
    changed = True
    while changed:
        changed = False
        for action_name, preconds in action_preconditions.items():
            # If all preconditions are reachable, add effects to reachable
            if preconds.issubset(reachable_predicates):
                effects = action_effects.get(action_name, set())
                before = len(reachable_predicates)
                reachable_predicates.update(effects)
                if len(reachable_predicates) > before:
                    changed = True

    # Now check each action
    reachable_actions = []
    unreachable_actions = []

    for action_name, preconds in action_preconditions.items():
        if not preconds:
            # No preconditions = always potentially applicable
            reachable_actions.append(action_name)
        elif preconds.issubset(reachable_predicates):
            reachable_actions.append(action_name)
        else:
            # Check which preconditions are missing
            missing = preconds - reachable_predicates
            unreachable_actions.append((action_name, sorted(missing)))

    passed = len(unreachable_actions) == 0

    diagnosis = ""
    if not passed:
        diagnosis = f"SEMANTIC WARNING: {len(unreachable_actions)} action(s) may never be applicable.\n"
        for action_name, missing in unreachable_actions:
            diagnosis += f"  - '{action_name}' requires predicates not reachable: {missing}\n"
        diagnosis += "This may indicate missing actions or incorrect init state."

    # Flatten unreachable_actions for return
    unreachable_names = [a[0] if isinstance(a, tuple) else a for a in unreachable_actions]

    return {
        "passed": passed,
        "total_actions": len(action_preconditions),
        "reachable_actions": sorted(reachable_actions),
        "unreachable_actions": sorted(unreachable_names),
        "diagnosis": diagnosis
    }


def check_type_grounding(domain_pddl: str, problem_pddl: str) -> dict:
    """
    Check if all action parameters have at least one valid object.

    Returns:
        {
            "passed": bool,
            "ungrounded_params": list[tuple[str, str]],  # (action, param_type)
            "diagnosis": str
        }
    """
    action_params = _extract_action_parameters_regex(domain_pddl)
    type_hierarchy = _extract_types_regex(domain_pddl)
    objects_by_type = _extract_objects_regex(problem_pddl)

    ungrounded_params = []

    for action_name, params in action_params.items():
        for param_name, param_type in params:
            # Get all subtypes of param_type
            valid_types = _get_subtypes_regex(param_type, type_hierarchy)

            # Check if any objects exist for these types
            has_objects = False
            for t in valid_types:
                if t in objects_by_type and objects_by_type[t]:
                    has_objects = True
                    break

            if not has_objects:
                ungrounded_params.append((action_name, param_type))

    passed = len(ungrounded_params) == 0

    diagnosis = ""
    if not passed:
        diagnosis = f"SEMANTIC ERROR: {len(ungrounded_params)} parameter type(s) have no matching objects.\n"
        for action_name, param_type in ungrounded_params:
            diagnosis += f"  - Action '{action_name}' has parameter of type '{param_type}' with no objects.\n"
        diagnosis += "Fix: Add objects of the required types to the :objects section."

    return {
        "passed": passed,
        "ungrounded_params": ungrounded_params,
        "diagnosis": diagnosis
    }


def check_init_predicates_defined(domain_pddl: str, problem_pddl: str) -> dict:
    """
    Check if all predicates used in init are defined in the domain.

    Returns:
        {
            "passed": bool,
            "undefined_predicates": list[str],
            "diagnosis": str
        }
    """
    # Extract defined predicates from domain
    pred_section = _extract_predicates_section(domain_pddl)
    defined_predicates = set()
    for match in re.finditer(r'\(([a-zA-Z][a-zA-Z0-9_-]*)', pred_section):
        defined_predicates.add(match.group(1).lower())

    # Also include predicates from action effects (for robustness)
    action_effects = _extract_action_effects_regex(domain_pddl)
    for preds in action_effects.values():
        defined_predicates.update(preds)

    init_predicates = _extract_init_predicates_regex(problem_pddl)
    undefined = init_predicates - defined_predicates

    passed = len(undefined) == 0

    diagnosis = ""
    if not passed:
        diagnosis = f"SEMANTIC ERROR: {len(undefined)} init predicate(s) not defined in domain.\n"
        diagnosis += f"Undefined: {sorted(undefined)}\n"
        diagnosis += "Fix: Define these predicates in :predicates or remove from :init."

    return {
        "passed": passed,
        "undefined_predicates": sorted(undefined),
        "diagnosis": diagnosis
    }


def check_goal_predicates_defined(domain_pddl: str, problem_pddl: str) -> dict:
    """
    Check if all predicates used in goal are defined in the domain.

    Returns:
        {
            "passed": bool,
            "undefined_predicates": list[str],
            "diagnosis": str
        }
    """
    # Extract defined predicates from domain
    pred_section = _extract_predicates_section(domain_pddl)
    defined_predicates = set()
    for match in re.finditer(r'\(([a-zA-Z][a-zA-Z0-9_-]*)', pred_section):
        defined_predicates.add(match.group(1).lower())

    # Also include predicates from action effects
    action_effects = _extract_action_effects_regex(domain_pddl)
    for preds in action_effects.values():
        defined_predicates.update(preds)

    goal_predicates = _extract_goal_predicates_regex(problem_pddl)
    undefined = goal_predicates - defined_predicates

    passed = len(undefined) == 0

    diagnosis = ""
    if not passed:
        diagnosis = f"SEMANTIC ERROR: {len(undefined)} goal predicate(s) not defined in domain.\n"
        diagnosis += f"Undefined: {sorted(undefined)}\n"
        diagnosis += "Fix: Define these predicates in :predicates or correct the goal."

    return {
        "passed": passed,
        "undefined_predicates": sorted(undefined),
        "diagnosis": diagnosis
    }


# ============================================================================
# Domain-Level Semantic Checks (independent of problem)
# ============================================================================

def check_domain_goal_producibility(domain_pddl: str, goal_predicates: Set[str]) -> dict:
    """
    Check if goal predicates can possibly be produced by domain actions.
    This is a domain-level check that's independent of the problem instance.

    Returns:
        {
            "passed": bool,
            "unproducible_goals": list[str],  # goals no action can produce
            "diagnosis": str
        }
    """
    action_effects = _extract_action_effects_regex(domain_pddl)

    # All predicates that can be produced by any action
    producible = set()
    for action_name, preds in action_effects.items():
        producible.update(preds)

    # Goals that no action can produce
    unproducible = goal_predicates - producible

    passed = len(unproducible) == 0

    diagnosis = ""
    if not passed:
        diagnosis = f"DOMAIN-LEVEL ERROR: {len(unproducible)} goal predicate(s) cannot be produced.\n"
        diagnosis += f"Unproducible goals: {sorted(unproducible)}\n"
        diagnosis += "No action in the domain produces these predicates.\n"
        diagnosis += "Fix: Add or modify actions to produce these predicates."

    return {
        "passed": passed,
        "unproducible_goals": sorted(unproducible),
        "diagnosis": diagnosis
    }


def check_domain_action_viability(domain_pddl: str) -> dict:
    """
    Check if actions can potentially fire (preconditions can be satisfied).
    This identifies actions whose preconditions require predicates no other action produces.

    Returns:
        {
            "passed": bool,
            "problem_actions": list[(action, missing_preds)],  # actions with unfixable preconditions
            "diagnosis": str
        }
    """
    action_effects = _extract_action_effects_regex(domain_pddl)
    action_preconditions = _extract_action_preconditions_regex(domain_pddl)

    # All predicates producible by any action
    producible = set()
    for preds in action_effects.values():
        producible.update(preds)

    problem_actions = []

    for action_name, preconds in action_preconditions.items():
        # Check if all preconditions can be satisfied
        satisfiable = preconds.intersection(producible)
        missing = preconds - producible

        # If action requires predicates that NO action produces, it's problematic
        # (unless those predicates might be in init, which we check in problem-level checks)
        if missing:
            problem_actions.append((action_name, sorted(missing)))

    passed = len(problem_actions) == 0

    diagnosis = ""
    if not passed:
        diagnosis = f"DOMAIN-LEVEL WARNING: {len(problem_actions)} action(s) have unfixable preconditions.\n"
        for action_name, missing in problem_actions:
            diagnosis += f"  - '{action_name}' requires predicates no action produces: {missing}\n"
        diagnosis += "These may be init predicates, but the domain has structural issues.\n"
        diagnosis += "Verify: Are these predicates supposed to be in the initial state?"

    return {
        "passed": passed,
        "problem_actions": problem_actions,
        "diagnosis": diagnosis
    }


# ============================================================================
# Main Entry Point
# ============================================================================

def run_semantic_checks(
    domain_pddl: str,
    problem_pddl: str,
    strict: bool = False,
    extractor_type: str = "auto"
) -> dict:
    """
    Run all semantic checks (domain-level and problem-level) and return aggregated result.

    Args:
        domain_pddl: PDDL domain content
        problem_pddl: PDDL problem content
        strict: If True, all checks must pass. If False (default), only critical checks must pass.
                Allows tuning to reduce false positives.
        extractor_type: Type of PDDL extractor to use: "auto" (default), "regex", or "up"

    Returns:
        {
            "passed": bool,  # all (or critical) checks passed
            "checks": {
                # Problem-level checks
                "predicate_coverage": {...},
                "action_reachability": {...},
                "type_grounding": {...},
                "init_defined": {...},
                "goal_defined": {...},
                # Domain-level checks
                "domain_goal_producibility": {...},
                "domain_action_viability": {...}
            },
            "domain_level_errors": list[str],  # errors at domain level (hard to fix)
            "problem_level_errors": list[str],  # errors at problem level (might be fixable)
            "combined_diagnosis": str  # for LLM refinement prompt
        }
    """
    # Set the extractor type if specified
    if extractor_type != _extractor_type:
        set_extractor_type(extractor_type)

    # Extract goal predicates for domain-level checks
    goal_predicates = _extract_goal_predicates_regex(problem_pddl)

    checks = {
        # Domain-level checks (independent of problem instance)
        "domain_goal_producibility": check_domain_goal_producibility(domain_pddl, goal_predicates),
        "domain_action_viability": check_domain_action_viability(domain_pddl),
        # Problem-level checks (dependent on problem instance)
        "predicate_coverage": check_predicate_coverage(domain_pddl, problem_pddl),
        "action_reachability": check_action_reachability(domain_pddl, problem_pddl),
        "type_grounding": check_type_grounding(domain_pddl, problem_pddl),
        "init_defined": check_init_predicates_defined(domain_pddl, problem_pddl),
        "goal_defined": check_goal_predicates_defined(domain_pddl, problem_pddl),
    }

    # Categorize errors by level
    domain_level_errors = []
    problem_level_errors = []

    if not checks["domain_goal_producibility"]["passed"]:
        domain_level_errors.append("domain_goal_producibility")
    if not checks["domain_action_viability"]["passed"]:
        domain_level_errors.append("domain_action_viability")

    problem_level = ["predicate_coverage", "action_reachability", "type_grounding", "init_defined", "goal_defined"]
    for name in problem_level:
        if not checks[name]["passed"]:
            problem_level_errors.append(name)

    # Determine pass/fail based on strictness
    # In non-strict mode: domain errors are critical (unfixable), problem errors might be fixable
    domain_errors_ok = len(domain_level_errors) == 0
    problem_errors_ok = len(problem_level_errors) == 0

    if strict:
        passed = domain_errors_ok and problem_errors_ok
    else:
        # Non-strict: only domain errors are blocking
        # Problem-level errors can potentially be fixed by refinement
        passed = domain_errors_ok

    # Build combined diagnosis
    diagnoses = []

    # Domain-level errors first (these are serious)
    if not checks["domain_goal_producibility"]["passed"]:
        diagnoses.append("CRITICAL (Domain): " + checks["domain_goal_producibility"]["diagnosis"])
    if not checks["domain_action_viability"]["passed"]:
        diagnoses.append("WARNING (Domain): " + checks["domain_action_viability"]["diagnosis"])

    # Problem-level errors (these might be fixable)
    for name in problem_level:
        if not checks[name]["passed"] and checks[name]["diagnosis"]:
            prefix = "PROBLEM-LEVEL" if name not in domain_level_errors else "ISSUE"
            diagnoses.append(f"{prefix}: {checks[name]['diagnosis']}")

    combined_diagnosis = "\n\n".join(diagnoses) if diagnoses else "All semantic checks passed."

    # Log results
    for name, result in checks.items():
        status = "PASS" if result["passed"] else "FAIL"
        logger.debug(f"Semantic check [{name}]: {status}")

    return {
        "passed": passed,
        "checks": checks,
        "domain_level_errors": domain_level_errors,
        "problem_level_errors": problem_level_errors,
        "combined_diagnosis": combined_diagnosis
    }


def format_semantic_report(results: dict) -> str:
    """
    Format semantic check results as a human-readable report.

    Args:
        results: Output from run_semantic_checks()

    Returns:
        Formatted string for display
    """
    lines = ["=" * 50, "SEMANTIC VERIFICATION REPORT", "=" * 50, ""]

    overall = "PASSED" if results["passed"] else "FAILED"
    lines.append(f"Overall Status: {overall}")
    lines.append("")

    for check_name, check_result in results["checks"].items():
        status = "PASS" if check_result["passed"] else "FAIL"
        lines.append(f"[{status}] {check_name.replace('_', ' ').title()}")

        if not check_result["passed"]:
            # Indent diagnosis
            for diag_line in check_result["diagnosis"].split('\n'):
                if diag_line.strip():
                    lines.append(f"      {diag_line}")
        lines.append("")

    lines.append("=" * 50)
    return "\n".join(lines)
