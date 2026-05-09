from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional
import logging

from src.lapis.planner.low.semantic_verification import RegexPDDLExtractor, run_semantic_checks

logger = logging.getLogger(__name__)


@dataclass
class VerificationResult:
    valid: bool
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    details: Dict[str, Any] = field(default_factory=dict)

    def to_text(self) -> str:
        lines: List[str] = []
        lines.append(f"valid={self.valid}")
        if self.errors:
            lines.append("errors:")
            for item in self.errors:
                lines.append(f"  - {item}")
        if self.warnings:
            lines.append("warnings:")
            for item in self.warnings:
                lines.append(f"  - {item}")
        return "\n".join(lines)


class VerificationService:
    def __init__(self, asp_rules_path: Optional[str] = None) -> None:
        base_dir = Path(__file__).resolve().parent
        self.asp_rules_path = Path(asp_rules_path) if asp_rules_path else (base_dir / "pddl_rules.lp")
        self.extractor = RegexPDDLExtractor()

    @staticmethod
    def _quote(value: str) -> str:
        escaped = value.replace("\\", "\\\\").replace('"', '\\"')
        return f'"{escaped}"'

    def parse_pddl(self, domain_str: str, problem_str: str) -> Dict[str, Any]:
        predicates = sorted(self.extractor.extract_predicates(domain_str))
        goal_predicates = sorted(self.extractor.extract_goal_predicates(problem_str))
        init_predicates = sorted(self.extractor.extract_init_predicates(problem_str))
        action_effects = self.extractor.extract_action_effects(domain_str)
        action_preconditions = self.extractor.extract_action_preconditions(domain_str)
        action_parameters = self.extractor.extract_action_parameters(domain_str)
        objects_by_type = self.extractor.extract_objects(problem_str)

        non_strips_issues: List[str] = []
        if ":conditional-effects" in domain_str.lower() or "(when " in domain_str.lower():
            non_strips_issues.append("conditional_effect")
        if "(or " in domain_str.lower():
            non_strips_issues.append("disjunctive_construct")

        return {
            "predicates": predicates,
            "goal_predicates": goal_predicates,
            "init_predicates": init_predicates,
            "action_effects": action_effects,
            "action_preconditions": action_preconditions,
            "action_parameters": action_parameters,
            "objects_by_type": objects_by_type,
            "non_strips_issues": non_strips_issues,
        }

    def _facts_to_program(self, parsed: Dict[str, Any]) -> str:
        lines: List[str] = []

        for pred in parsed["predicates"]:
            lines.append(f"pred({self._quote(pred)}).")

        for pred in parsed["goal_predicates"]:
            lines.append(f"goal_pred({self._quote(pred)}).")

        for pred in parsed["init_predicates"]:
            lines.append(f"init_pred({self._quote(pred)}).")

        for action_name, predicates in parsed["action_preconditions"].items():
            for pred in sorted(predicates):
                lines.append(f"action_precond({self._quote(action_name)}, {self._quote(pred)}).")

        for action_name, predicates in parsed["action_effects"].items():
            for pred in sorted(predicates):
                lines.append(f"action_effect({self._quote(action_name)}, {self._quote(pred)}).")

        for action_name, params in parsed["action_parameters"].items():
            for param_name, param_type in params:
                lines.append(
                    f"param_type({self._quote(action_name)}, {self._quote(param_name)}, {self._quote(param_type)})."
                )

        for obj_type, objects in parsed["objects_by_type"].items():
            for obj_name in sorted(objects):
                lines.append(f"obj_type({self._quote(obj_name)}, {self._quote(obj_type)}).")

        for issue in parsed["non_strips_issues"]:
            lines.append(f"non_strips({self._quote(issue)}).")

        return "\n".join(lines)

    @staticmethod
    def _format_issue(symbol: Any) -> str:
        if not hasattr(symbol, "name"):
            return str(symbol)

        issue_name = symbol.name
        args = [str(arg).strip('"') for arg in getattr(symbol, "arguments", [])]

        if issue_name == "undefined_goal_pred" and len(args) == 1:
            return f"Goal uses undefined predicate: {args[0]}"
        if issue_name == "undefined_init_pred" and len(args) == 1:
            return f"Initial state uses undefined predicate: {args[0]}"
        if issue_name == "unachievable_goal" and len(args) == 1:
            return f"Goal predicate appears unachievable: {args[0]}"
        if issue_name == "undefined_precondition_pred" and len(args) == 2:
            return f"Action '{args[0]}' precondition uses undefined predicate: {args[1]}"
        if issue_name == "undefined_effect_pred" and len(args) == 2:
            return f"Action '{args[0]}' effect uses undefined predicate: {args[1]}"
        if issue_name == "no_object_for_type" and len(args) == 3:
            return f"No object found for parameter '{args[1]}' of type '{args[2]}' in action '{args[0]}'"
        if issue_name == "non_strips_issue" and len(args) == 1:
            return f"Potential non-STRIPS construct detected: {args[0]}"
        if issue_name == "unused_predicate" and len(args) == 1:
            return f"Predicate appears unused: {args[0]}"

        if args:
            return f"{issue_name}({', '.join(args)})"
        return issue_name

    def _run_asp_checks(self, parsed: Dict[str, Any]) -> VerificationResult:
        try:
            import clingo  # type: ignore
        except ImportError:
            logger.warning("clingo is not installed; skipping ASP checks.")
            return VerificationResult(valid=True, warnings=["clingo is not installed; skipped ASP checks"], details={"asp": "skipped"})

        if not self.asp_rules_path.exists():
            return VerificationResult(valid=True, warnings=[f"ASP rules not found: {self.asp_rules_path}"], details={"asp": "skipped"})

        ctl = clingo.Control([])
        facts = self._facts_to_program(parsed)
        ctl.add("base", [], facts)
        ctl.load(str(self.asp_rules_path))
        ctl.ground([("base", [])])

        errors: List[str] = []
        warnings: List[str] = []

        with ctl.solve(yield_=True) as handle:
            for model in handle:
                for sym in model.symbols(shown=True):
                    if sym.name == "error" and sym.arguments:
                        errors.append(self._format_issue(sym.arguments[0]))
                    elif sym.name == "warning" and sym.arguments:
                        warnings.append(self._format_issue(sym.arguments[0]))
                break

        unique_errors = sorted(set(errors))
        unique_warnings = sorted(set(warnings))
        return VerificationResult(valid=len(unique_errors) == 0, errors=unique_errors, warnings=unique_warnings, details={"asp": "completed"})

    def verify(self, domain_str: str, problem_str: str, run_asp: bool = True) -> VerificationResult:
        parsed = self.parse_pddl(domain_str, problem_str)

        semantic = run_semantic_checks(domain_str, problem_str, strict=False, extractor_type="auto")

        errors: List[str] = []
        warnings: List[str] = []

        for check_name in semantic.get("domain_level_errors", []):
            check = semantic.get("checks", {}).get(check_name, {})
            diagnosis = check.get("diagnosis", "")
            if diagnosis:
                errors.append(f"{check_name}: {diagnosis}")

        for check_name in semantic.get("problem_level_errors", []):
            check = semantic.get("checks", {}).get(check_name, {})
            diagnosis = check.get("diagnosis", "")
            if diagnosis:
                warnings.append(f"{check_name}: {diagnosis}")

        asp_details: Dict[str, Any] = {"asp": "disabled"}
        if run_asp:
            asp_result = self._run_asp_checks(parsed)
            errors.extend(asp_result.errors)
            warnings.extend(asp_result.warnings)
            asp_details = asp_result.details

        errors = sorted(set(filter(None, errors)))
        warnings = sorted(set(filter(None, warnings)))

        return VerificationResult(
            valid=len(errors) == 0,
            errors=errors,
            warnings=warnings,
            details={
                "semantic": semantic,
                "parsed": parsed,
                "asp": asp_details.get("asp", "disabled"),
            },
        )
