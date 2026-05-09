import re
import logging

logger = logging.getLogger("my_logger")

def split_conjuncts(formula):
    """
    Split a formula into its top-level conjuncts by balancing parentheses.
    Example: (F(p) & G(q -> F(r))) -> ['F(p)', 'G(q -> F(r))']
    """
    if not formula:
        return []
        
    formula = formula.strip()
    
    # Remove outer parentheses if they wrap the entire formula
    # but only if they are matching pairs
    def strip_outer(f):
        f = f.strip()
        if f.startswith('(') and f.endswith(')'):
            depth = 0
            match = True
            for i in range(len(f) - 1):
                if f[i] == '(': depth += 1
                elif f[i] == ')': depth -= 1
                if depth == 0:
                    match = False
                    break
            if match:
                return strip_outer(f[1:-1])
        return f

    formula = strip_outer(formula)

    conjuncts = []
    current = ""
    depth = 0
    i = 0
    while i < len(formula):
        char = formula[i]
        if char == '(':
            depth += 1
            current += char
        elif char == ')':
            depth -= 1
            current += char
        elif char == '&' and depth == 0:
            conjuncts.append(strip_outer(current.strip()))
            current = ""
        else:
            current += char
        i += 1
    if current:
        conjuncts.append(strip_outer(current.strip()))
        
    return conjuncts

def extract_ltl_info(ltl_formula):
    """
    Extract fluents and individual constraints from an LTL formula string.
    Returns:
        fluents_map: dict {id: fluent_string}
        constraints_data: list of dicts {'formula': str, 'fluent_ids': list[int]}
    """
    try:
        if not ltl_formula:
            return {}, []
            
        # Find all things that look like predicates (e.g. on(a,b) or clear(a))
        matches = set(re.findall(r'[a-zA-Z_][a-zA-Z0-9_\-]*\([^)]*\)|[a-zA-Z_][a-zA-Z0-9_\-]*', ltl_formula))
        
        operators = {"F", "G", "X", "U", "R", "W", "M", "not", "NOT", "and", "or", "implies", "true", "false", "→", "∧", "∨", "¬", "&", "|", "->"}
        fluents_set = {m for m in matches if m not in operators and not (len(m) == 1 and m.upper() in ["G", "F", "X", "U", "R"])}
        
        sorted_fluents = sorted(list(fluents_set))
        fluents_map = {i: f for i, f in enumerate(sorted_fluents)}
        fluent_to_id = {f: i for i, f in fluents_map.items()}
        
        # Split formula into top-level conjuncts
        conjunct_strings = split_conjuncts(ltl_formula)
        constraints_data = []

        for conj in conjunct_strings:
            # For each conjunct, find which of the global fluents it refers to
            conj_matches = set(re.findall(r'[a-zA-Z_][a-zA-Z0-9_\-]*\([^)]*\)|[a-zA-Z_][a-zA-Z0-9_\-]*', conj))
            conj_fluents = {m for m in conj_matches if m in fluent_to_id}
            conj_fluent_ids = sorted([fluent_to_id[f] for f in conj_fluents])
            
            constraints_data.append({
                "formula": conj,
                "fluent_ids": conj_fluent_ids
            })
        
        if not constraints_data:
            constraints_data = [{
                "formula": ltl_formula,
                "fluent_ids": sorted(list(fluent_to_id.values()))
            }]
            
        return fluents_map, constraints_data
        
    except Exception as e:
        logger.warning(f"Failed to extract LTL info: {e}")
        return {}, [{"formula": ltl_formula, "fluent_ids": []}]

def parse_ltl_to_up(formula_str):
    """
    Parse an LTL formula string into a unified_planning symbolic expression.
    (Stubbed out during transition to external verifier)
    """
    logger.warning("parse_ltl_to_up is stubbed out. Use ltl_verifier submodule instead.")
    return None


def convert_trace_to_strings(trace):
    """
    Convert a trace of UP States to a list of sets of predicate strings.
    """
    string_trace = []
    for i, state in enumerate(trace):
        if state is None:
            # logger.info(f"DEBUG: Trace state {i} is None")
            continue
        
        # Check if state is a UP state object (has _values)
        if not hasattr(state, '_values'):
             # logger.info(f"DEBUG: Trace state {i} (type {type(state)}) has no _values attribute")
             continue
             
        state_fluents = set()
        # Double check state is not None before accessing _values
        # (Paranoid check for rare race conditions or corrupted objects)
        if state is not None and hasattr(state, '_values'):
            for fluent, value in state._values.items():
                if value.is_bool_constant() and value.constant_value():
                    fluent_name = fluent.fluent().name
                    args = [str(arg).replace("'", "") for arg in fluent.args]
                    if args:
                        # Use pred(arg1, arg2) format to match LTL formula likely format
                        pred_str = f"{fluent_name}({', '.join(args)})"
                    else:
                        pred_str = f"{fluent_name}"
                    state_fluents.add(pred_str)
        string_trace.append(state_fluents)
    return string_trace

def extract_constraints(problem_file):
    try:
        with open(problem_file, 'r') as f:
            content = f.read()
        
        # Look for (:constraints ...) block
        match = re.search(r'\(:constraints\s+(.*?)\)\s*\)', content, re.DOTALL)
        if match:
                return match.group(1).strip()
    except Exception as e:
        logger.error(f"Failed to extract constraints from {problem_file}: {e}")
    return ""
