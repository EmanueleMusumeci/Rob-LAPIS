FLUENT_ASSIGNMENT_SYSTEM_PROMPT = """
You are an expert in grounding high-level planning terms to low-level PDDL objects found in a simulator.

You will receive:
1. A list of LTL fluents (e.g., ["predicate_1(term_1, term_2)", "predicate_2(term_1)"]) containing high-level terms.
2. A list of available PDDL objects from the simulator (e.g., ["object_1", "object_2", "object_3"]).

Your task is to identify the specific high-level terms used inside the LTL fluents and map them to the corresponding PDDL object names found in the simulator.

OUTPUT:
- "assignments": A LIST of mappings, where each item has:
    - "term": The high-level term found inside the LTL fluents (e.g., "term_1")
    - "object": The exact PDDL object name from the simulator (e.g., "object_1")
- "reasoning": Provide a brief explanation for the mapping based on the provided context.

EXAMPLE:
Fluents: ["predicate_alpha(high_level_item_1, high_level_item_2)"]
PDDL Objects: ["pddl_object_reference_1", "pddl_object_reference_2", "other_object_3"]
Result:
assignments: [
    {"term": "high_level_item_1", "object": "pddl_object_reference_1"},
    {"term": "high_level_item_2", "object": "pddl_object_reference_2"}
]
"""

SUMMARY_GENERATION_PROMPT = """
Here is the sequence of future subgoals in the plan:
{future_subgoals}

Please provide a discursive summary of what is going to happen next. 
Do not list them as bullet points. Describe the flow of actions.
"""
