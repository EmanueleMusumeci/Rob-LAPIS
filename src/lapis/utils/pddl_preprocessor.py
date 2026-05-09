"""
PDDL Preprocessor for Unified Planning (UP) incompatible constructs.

NOTE: This preprocessing is ONLY required when using the unified-planning library
for simulation/validation. Other PDDL tools (VAL, FastDownward, pyperplan) handle
these constructs natively. The preprocessing is applied automatically by
PDDLSimulator.setup() and plan_renderer.simulate_plan() which use UP's
SequentialSimulator.

IPC benchmark domains require the following transformations for UP compatibility:

1. **Tyreworld** (IPC-5):
   - Action/predicate name collision: 'open'/'close' are both predicates and actions
   - Fix: Rename actions to 'open-container'/'close-container'
   - Undeclared constants: 'wrench', 'jack', 'pump' used in preconditions
   - Fix: Add :constants declaration, remove duplicates from problem :objects

2. **Storage** (IPC-5):
   - Type union: '(either storearea crate)' not supported by UP parser
   - Fix: Replace with common ancestor type 'surface'
   - Multi-inheritance: 'area' declared as child of both 'object' and 'surface'
   - Fix: Remove redundant 'area - object' declaration

3. **Floortile** (IPC-2011):
   - Action/predicate name collision: 'up'/'down'/'right'/'left'
   - Fix: Rename actions to 'move-up'/'move-down'/'move-right'/'move-left'

These are syntactic transformations that preserve semantic equivalence.
Plans generated with preprocessed domains use the renamed action names.
"""

import re
import tempfile
import logging
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)


def preprocess_tyreworld_domain(domain_content: str) -> str:
    """
    Fix tyreworld domain for UP compatibility:
    1. Rename 'open'/'close' actions to avoid predicate name collision
    2. Add :constants declaration for wrench, jack, pump (used in preconditions)
    """
    # Rename action open -> open-container
    domain_content = re.sub(
        r'\(:action\s+open\b',
        '(:action open-container',
        domain_content,
        flags=re.IGNORECASE
    )
    # Rename action close -> close-container
    domain_content = re.sub(
        r'\(:action\s+close\b',
        '(:action close-container',
        domain_content,
        flags=re.IGNORECASE
    )

    # Add :constants declaration for tool objects used in preconditions
    # These are referenced as constants like (have wrench) but not declared
    if ':constants' not in domain_content.lower():
        # Insert constants after :types
        domain_content = re.sub(
            r'(\(:types[^)]+\))',
            r'\1\n(:constants wrench jack pump - tool)',
            domain_content,
            flags=re.IGNORECASE
        )

    return domain_content


def preprocess_storage_domain(domain_content: str) -> str:
    """
    Fix storage domain for UP compatibility:

    Original type hierarchy:
        (:types hoist surface place area - object
            container depot - place
            storearea transitarea - area
            area crate - surface)

    Issues:
    1. '(either storearea crate)' type union in predicate not supported
    2. 'area' declared twice (child of object AND surface) - multi-inheritance

    Fix:
    1. Replace '(either storearea crate)' with 'surface' (common ancestor)
    2. Remove 'area' from first line (it's already child of surface → object)
    """
    # Fix 1: Replace (either storearea crate) with surface
    domain_content = re.sub(
        r'\(either\s+storearea\s+crate\)',
        'surface',
        domain_content,
        flags=re.IGNORECASE
    )

    # Fix 2: Remove 'area' from the first types line to avoid duplicate declaration
    # Original: (:types hoist surface place area - object
    # Fixed:    (:types hoist surface place - object
    domain_content = re.sub(
        r'(\(:types\s+hoist\s+surface\s+place)\s+area(\s+-\s+object)',
        r'\1\2',
        domain_content,
        flags=re.IGNORECASE
    )

    return domain_content


def preprocess_floortile_domain(domain_content: str) -> str:
    """
    Fix floortile domain for UP compatibility:
    Rename movement actions 'up'/'down'/'right'/'left' to avoid predicate name collision.
    """
    # Rename directional actions to avoid collision with predicates
    for direction in ['up', 'down', 'right', 'left']:
        domain_content = re.sub(
            rf'\(:action\s+{direction}\b',
            f'(:action move-{direction}',
            domain_content,
            flags=re.IGNORECASE
        )
    return domain_content


def preprocess_tyreworld_problem(problem_content: str) -> str:
    """
    Remove wrench, jack, pump from problem objects since they're now domain constants.
    """
    # Original: wrench jack pump - tool
    # Remove the line or the specific objects
    problem_content = re.sub(
        r'wrench\s+jack\s+pump\s+-\s+tool\s*\n?',
        '',
        problem_content,
        flags=re.IGNORECASE
    )
    return problem_content


def preprocess_pddl_for_up(
    domain_path: str,
    problem_path: str,
    domain_name: Optional[str] = None
) -> tuple[str, str]:
    """
    Preprocess PDDL files to make them compatible with Unified Planning.

    Args:
        domain_path: Path to domain.pddl
        problem_path: Path to problem.pddl
        domain_name: Optional domain name hint (auto-detected if not provided)

    Returns:
        Tuple of (preprocessed_domain_path, preprocessed_problem_path)
        Returns original paths if no preprocessing needed.
    """
    domain_content = Path(domain_path).read_text()
    problem_content = Path(problem_path).read_text()

    # Auto-detect domain name from content
    if domain_name is None:
        match = re.search(r'\(define\s+\(domain\s+(\S+)\)', domain_content, re.IGNORECASE)
        if match:
            domain_name = match.group(1).lower()

    modified = False

    # Apply domain-specific preprocessing
    if domain_name and 'tyreworld' in domain_name.lower():
        logger.info("Preprocessing tyreworld domain: renaming open/close actions, adding constants")
        domain_content = preprocess_tyreworld_domain(domain_content)
        problem_content = preprocess_tyreworld_problem(problem_content)
        modified = True

    if domain_name and 'storage' in domain_name.lower():
        logger.info("Preprocessing storage domain: expanding 'either' type unions")
        domain_content = preprocess_storage_domain(domain_content)
        modified = True

    if domain_name and ('floortile' in domain_name.lower() or 'floor-tile' in domain_name.lower()):
        logger.info("Preprocessing floortile domain: renaming movement actions")
        domain_content = preprocess_floortile_domain(domain_content)
        modified = True

    if not modified:
        # No preprocessing needed
        return domain_path, problem_path

    # Write preprocessed files to temp directory
    temp_dir = Path(tempfile.mkdtemp(prefix="pddl_preprocessed_"))

    preprocessed_domain = temp_dir / "domain.pddl"
    preprocessed_problem = temp_dir / "problem.pddl"

    preprocessed_domain.write_text(domain_content)
    preprocessed_problem.write_text(problem_content)

    logger.info(f"Preprocessed PDDL written to {temp_dir}")

    return str(preprocessed_domain), str(preprocessed_problem)


def preprocess_plan_for_up(plan_content: str, domain_name: str) -> str:
    """
    Rename actions in the plan string to match the preprocessed domain for UP compatibility.
    """
    if not domain_name:
        return plan_content

    # Tyreworld renames
    if 'tyreworld' in domain_name.lower():
        plan_content = re.sub(r'\(open\s', '(open-container ', plan_content, flags=re.IGNORECASE)
        plan_content = re.sub(r'\(close\s', '(close-container ', plan_content, flags=re.IGNORECASE)

    # Floortile renames
    if 'floortile' in domain_name.lower() or 'floor-tile' in domain_name.lower():
        for direction in ['up', 'down', 'right', 'left']:
            plan_content = re.sub(rf'\({direction}\s', f'(move-{direction} ', plan_content, flags=re.IGNORECASE)

    return plan_content


def needs_preprocessing(domain_path: str) -> bool:
    """Check if a domain file needs preprocessing for UP compatibility."""
    content = Path(domain_path).read_text().lower()

    # Check for tyreworld action/predicate name collision
    if 'tyreworld' in content:
        if '(:action open' in content and '(open ' in content:
            return True

    # Check for storage 'either' type unions
    if 'storage' in content:
        if '(either ' in content:
            return True

    return False
