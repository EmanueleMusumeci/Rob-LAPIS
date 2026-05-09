"""pddl_simulator.py — Generic PDDL simulator using unified_planning SequentialSimulator.

Subclasses only need to implement get_image() for domain-specific rendering.
All other simulation logic (setup, reset, step, observation) is domain-agnostic.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Optional

from .base_simulator import BaseSimulator
from .scenario import Scenario

logger = logging.getLogger(__name__)


class PDDLScenario(Scenario):
    """Generic scenario with unitary action costs."""

    def compute_cost(self, action: Any, state: Any = None) -> float:
        return 1.0


class PDDLSimulator(BaseSimulator):
    """Domain-agnostic PDDL simulator backed by unified_planning SequentialSimulator.

    Usage::

        sim = PDDLSimulator("blocksworld")
        ok = sim.setup(domain_path, problem_path)
        obs, info = sim.reset()
        obs, reward, done, truncated, info = sim.step(action_instance)
    """

    def __init__(
        self,
        domain_name: str,
        seed: Optional[int] = None,
        scenario: Optional[Scenario] = None,
    ):
        self.domain_name = domain_name
        self.seed = seed
        self.problem = None
        self.simulator = None
        self.current_state = None
        if scenario is None:
            scenario = PDDLScenario()
        super().__init__(None, scenario=scenario)

    # ── setup / reset ────────────────────────────────────────────────────────

    def setup(self, domain_path: "str | Path", problem_path: "str | Path", **kwargs) -> bool:
        """Parse PDDL files and initialise the UP SequentialSimulator."""
        try:
            from unified_planning.io import PDDLReader
            from unified_planning.shortcuts import SequentialSimulator
            from unified_planning.model import Problem

            domain_path = Path(domain_path)
            problem_path = Path(problem_path)

            if not domain_path.exists():
                logger.error("Domain file not found: %s", domain_path)
                return False
            if not problem_path.exists():
                logger.error("Problem file not found: %s", problem_path)
                return False

            # Preprocess for UP-specific incompatibilities (action/predicate name collisions,
            # type unions). Only needed for UP's SequentialSimulator; VAL/FastDownward work natively.
            from ..utils.pddl_preprocessor import preprocess_pddl_for_up
            try:
                preprocessed_domain, preprocessed_problem = preprocess_pddl_for_up(
                    str(domain_path), str(problem_path), self.domain_name
                )
            except Exception as preproc_err:
                logger.warning("UP preprocessing failed: %s - using original files", preproc_err)
                preprocessed_domain, preprocessed_problem = str(domain_path), str(problem_path)

            reader = PDDLReader()
            raw = reader.parse_problem(preprocessed_domain, preprocessed_problem)

            # Strip constraints/metrics that break SequentialSimulator
            clean = Problem(raw.name)
            for f in raw.fluents:
                clean.add_fluent(f)
            clean.add_objects(raw.all_objects)
            clean.add_actions(raw.actions)
            for f, v in raw.initial_values.items():
                clean.set_initial_value(f, v)
            for goal in raw.goals:
                clean.add_goal(goal)

            self.problem = clean
            self.simulator = SequentialSimulator(clean)
            self.current_state = self.simulator.get_initial_state()
            return True

        except Exception:
            logger.error("Setup failed for %s", self.domain_name, exc_info=True)
            return False

    def reset(self, seed: Optional[int] = None, **kwargs):
        if self.simulator is None:
            raise RuntimeError("Call setup() before reset()")
        self.current_state = self.simulator.get_initial_state()
        return self.observation(self.current_state), {"problem": self.problem}

    # ── step ─────────────────────────────────────────────────────────────────

    def step(self, action):
        if not self.simulator.is_applicable(self.current_state, action):
            raise ValueError(f"Action not applicable in current state: {action}")
        self.current_state = self.simulator.apply(self.current_state, action)
        is_goal = self.simulator.is_goal(self.current_state)
        reward = 1.0 if is_goal else 0.0
        return self.observation(self.current_state), reward, is_goal, False, {}

    # ── observation helpers ───────────────────────────────────────────────────

    def observation(self, state) -> dict:
        return {"state": state}

    def map_pddl2simulator(self, action):
        return action

    def map_plan2simulator(self, plan):
        return plan.actions if hasattr(plan, "actions") else plan

    # ── fluent helpers ────────────────────────────────────────────────────────

    def get_fluent_value(self, state, fluent_name: str, *args) -> Any:
        """Check a fluent's value in the given state."""
        try:
            fluent = self.problem.fluent(fluent_name)
            val = state.get_value(fluent(*args))
            return val.constant_value() if val.is_bool_constant() else val
        except Exception:
            return False

    def get_true_fluents(self, state) -> list[str]:
        """Return human-readable list of fluents that are True in *state*."""
        truths = []
        for fluent in self.problem.fluents:
            if fluent.type.is_bool_type():
                for obj_combo in _all_ground_combinations(self.problem, fluent):
                    try:
                        val = state.get_value(fluent(*obj_combo))
                        if val.bool_constant_value():
                            args_str = " ".join(o.name for o in obj_combo)
                            truths.append(
                                f"({fluent.name} {args_str})" if obj_combo
                                else f"({fluent.name})"
                            )
                    except Exception:
                        pass
        return truths

    # ── visualisation (override in subclasses) ────────────────────────────────

    def get_image(self, action_text: Optional[str] = None, **kwargs):
        """Return a PIL.Image for the current state, or None if not supported."""
        return None

    def render(self, *args, **kwargs):
        return self.get_image()


# ─── helpers ──────────────────────────────────────────────────────────────────

def _all_ground_combinations(problem, fluent):
    """Generate all ground argument combinations for a fluent."""
    from itertools import product

    if not fluent.signature:
        yield ()
        return

    all_objs = list(problem.all_objects)
    param_domains: list[list] = []
    for param in fluent.signature:
        # Check if object type matches or is a subtype of parameter type
        compatible = [o for o in all_objs if o.type == param.type or
                      o.type.is_subtype(param.type)]
        if not compatible:
            compatible = all_objs
        param_domains.append(compatible)

    for combo in product(*param_domains):
        yield combo
