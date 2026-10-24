"""
A small rule set standing in for the ones krrood ships.
"""

from __future__ import annotations

from krrood.development_tools.agent_rule import RuleSet, Section
from krrood.ormatic.utils import classproperty

# %% rule set


class DatasetRules(RuleSet):
    """
    A rule set standing in for the ones krrood ships.
    """


# %% sections


class DesignPrinciples(Section):
    """
    A section listing rules and holding a subsection.
    """

    @classproperty
    def parent(cls) -> type[Section]:
        return DatasetRules

    KEEP_DESIGNS_SIMPLE = "A rule about design"


class Errors(Section):
    """
    A subsection whose first rule is spelled out by details.
    """

    @classproperty
    def parent(cls) -> type[Section]:
        return DesignPrinciples

    RAISE_ERRORS = ("A rule about errors", ("A first detail", "A second detail"))

    NAME_ERRORS = "Another rule about errors"


class Testing(Section):
    """
    A section listing rules only in its subsection.
    """

    @classproperty
    def parent(cls) -> type[Section]:
        return DatasetRules


class Assertions(Section):
    """
    The subsection of a section without rules of its own.
    """

    @classproperty
    def parent(cls) -> type[Section]:
        return Testing

    ASSERT_EXACTLY = "A rule about assertions"


class Deployment(Section):
    """
    A section without rules.
    """

    @classproperty
    def parent(cls) -> type[Section]:
        return DatasetRules


class ReleasesThroughCI(Section):
    """
    A subsection whose name ends in an acronym.
    """

    @classproperty
    def parent(cls) -> type[Section]:
        return Deployment
