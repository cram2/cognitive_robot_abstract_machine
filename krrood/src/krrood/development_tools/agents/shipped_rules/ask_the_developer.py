"""
The rules on what to ask the developer instead of deciding.
"""

from __future__ import annotations


from krrood.development_tools.agents.rule import Section


class AskTheDeveloperRules(Section):
    """
    The rules on what to ask the developer instead of deciding.
    """

    ASK_INSTEAD_OF_INVENTING_REASONS = (
        "When you are unsure why something was done or why a specific number was "
        "chosen, ask; never invent a reason and write it down as a comment"
    )

    ASK_ABOUT_TEST_ONLY_METHODS = (
        "When a method is never used outside of tests, ask whether it can be removed"
    )

    ASK_BEFORE_ADDING_DEPENDENCIES = "Before adding a new dependency, ask"

    SUGGEST_MORE_POWERFUL_PACKAGES = (
        "If a package could be replaced by a more powerful one, tell us"
    )
