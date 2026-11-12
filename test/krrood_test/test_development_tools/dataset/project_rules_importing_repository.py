"""
A repository's own rules module that builds a rule from another module of the
repository.
"""

from __future__ import annotations

from repository_wording import REVIEW_STATEMENT

from krrood.development_tools.agents.rule import Section


class ReviewRules(Section):
    """
    The repository's rules on reviewing.
    """

    REVIEW_BEFORE_MERGING = REVIEW_STATEMENT
