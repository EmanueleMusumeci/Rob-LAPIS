"""
Prose barrier for VirtualHome NL scene descriptions.

Rewrites bijective predicate-to-sentence NL (from _pddl_init_to_nl) into
genuine natural prose using an LLM call, breaking the bijective encoding so
LAPIS must do real NL understanding rather than structured PDDL reconstruction.
"""

import warnings


_SYSTEM_PROMPT = """\
You are a scene describer that emulates the output of a visual observer (e.g. a VLM) \
in a household robotics pipeline. Given a structured state report about a home \
environment, produce a concise natural-language description of the scene as a human \
observer would describe it.

RULES:
- Do NOT reuse predicate-like or capability-style phrasing \
(e.g. do NOT write "the bed can be lain on", "the lamp has an on/off switch", \
"the fridge can be opened" — these are internal properties, not observable facts).
- Describe spatial relationships and observable states in plain English \
(e.g. "there is a bed in the bedroom", "the fridge is closed", \
"the television is switched off").
- Do NOT invent facts not present in the report.
- Do NOT use parentheses, predicate syntax, variable names, \
or the phrase "is True / is False".
- Do NOT mention object identifiers as machine names; use natural names \
(e.g. "a glass of water", not "water_glass").
"""

_USER_PROMPT = """\
STRUCTURED STATE REPORT (from a perception module):
{structured_nl}

Rewrite the above as a plain-English scene description in 1-6 short sentences.
"""


def rewrite_as_prose(structured_nl: str, agent, enabled: bool = True) -> str:
    if not enabled:
        return structured_nl
    try:
        user_msg = _USER_PROMPT.format(structured_nl=structured_nl)
        return agent.llm_call(_SYSTEM_PROMPT, user_msg)
    except Exception as exc:
        warnings.warn(f"nl_barrier.rewrite_as_prose failed ({exc}); using structured NL")
        return structured_nl
