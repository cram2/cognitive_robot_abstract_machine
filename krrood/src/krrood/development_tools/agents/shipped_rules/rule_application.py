"""
The rules on when and how far the other rules apply.
"""

from __future__ import annotations


from krrood.development_tools.agents.rule import Section


class RuleApplicationRules(Section):
    """
    The rules on when and how far the other rules apply.
    """

    LEAVE_UNTOUCHED_CODE_ALONE = (
        "Existing code that breaks these rules stays as it is unless a change "
        "touches it or the user asks for the cleanup. When you modify a function "
        "or class, bring that function or class in line with these rules as part "
        "of the change, and leave code the change does not otherwise modify alone. "
        "If the cleanup renames a method and so changes the API, ask the user "
        "first"
    )

    ASK_BEFORE_DEPARTING_FROM_A_RULE = (
        "When a rule seems wrong for the case at hand - a failing test that is "
        "itself wrong, an exception that has to be caught, a value that genuinely "
        "has to live at module level - ask the user before departing from it; "
        "never decide the exception on your own"
    )


class RuleModellingRules(Section):
    """
    The rules on writing the rules in this file.
    """

    SPLIT_OFF_THE_CHECKABLE_PART = (
        "When a script can check part of a rule, split the rule: the checkable "
        "part becomes a rule of its own that carries its checker, and the part "
        "only a person can judge stays a rule without one"
    )
