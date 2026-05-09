import re
import logging

logger = logging.getLogger("my_logger")

def check_pddl_heuristics(domain_pddl: str, problem_pddl: str) -> list[str]:
    """
    Statically analyze Domain and Problem PDDL content for common errors
    and inconsistencies. Returns a list of WARNING strings.
    """
    warnings = []
    
    # Check 1: Predicate consistency
    # Simple regex to find defined predicates in Domain
    # Pattern: Look for (pred-name ?arg ...) inside (:predicates ...) block
    # This is a heuristic and might not be perfect for complex PDDL but works for common cases.
    
    # Extract predicates block from domain
    pred_block_match = re.search(r'\(:predicates(.*?)\)', domain_pddl, re.DOTALL)
    domain_preds = set()
    if pred_block_match:
        content = pred_block_match.group(1)
        # Find all Words starting with (
        found = re.findall(r'\(\s*([a-zA-Z0-9_\-]+)', content)
        for f in found:
            domain_preds.add(f.lower())
    
    # Extract init and goal from problem
    # Heuristic: check usage in entire problem file for simplicity
    # but excluding the (:domain ...) declaration
    
    # We grep for "ontable" vs "on-table" suspicious pairs
    # Logic: If domain defines 'on-table' but problem uses 'ontable', warn.
    # Logic: If domain defines 'ontable' but problem uses 'on-table', warn.
    
    problem_words = set(re.findall(r'\(\s*([a-zA-Z0-9_\-]+)', problem_pddl))
    
    for d_pred in domain_preds:
        # Check for hyphen variation
        if "-" in d_pred:
            no_hyphen = d_pred.replace("-", "")
            if no_hyphen in problem_words and no_hyphen not in domain_preds:
                warnings.append(f"WARNING: Domain defines '{d_pred}' but Problem uses '{no_hyphen}'. Did you mean '{d_pred}'?")
        else:
            # Check if hyphenated version exists in problem but not in domain
            # This is harder to guess, e.g. "ontable" -> "on-table"? "on_table"?
            pass

    # Check 2: Unbound variables in Goal
    # Pattern: (:goal ...) block. Look for ?var inside it that is NOT inside (exists ...) or (forall ...)
    # This is tricky with Regex.
    # Simplified Heuristic: If we find ?[a-z] inside :goal, check if "exists" or "forall" precedes it closely.
    
    goal_match = re.search(r'\(:goal(.*?)\)\s*\)', problem_pddl, re.DOTALL) # Capture until end of goal block? Hard to match parens with regex.
    if goal_match:
        goal_content = goal_match.group(1)
        
        # Find all variables ?x
        vars_found = re.findall(r'\?[a-zA-Z0-9_]+', goal_content)
        
        # Check if they look quantified.
        # If "exists" or "forall" is NOT present in the goal content but variables ARE, it's definitely partially unbound.
        # If present, it's harder to say without full parsing. 
        # But we can look for basic mistake: (on ?b x) directly in goal.
        
        if vars_found:
            has_quantifier = "exists" in goal_content.lower() or "forall" in goal_content.lower()
            if not has_quantifier:
                 warnings.append(f"WARNING: Found variables {list(set(vars_found))} in (:goal ...) without any 'exists' or 'forall' quantifier. Goal targets must be ground objects or explicitly quantified.")
            else:
                 # Even with quantifiers, warn to be safe, as scope checking is hard with regex
                 warnings.append(f"WARNING: Variables {list(set(vars_found))} usage detected in goal. Verify that ALL variables are properly quantified (scope check).")

    return warnings
