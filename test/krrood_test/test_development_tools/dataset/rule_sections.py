"""
A few sections standing in for the ones krrood ships.
"""

from __future__ import annotations

from krrood.development_tools.agents.rule import Section
from krrood.ormatic.utils import classproperty

# %% sections


class DesignPrinciplesRules(Section):
    """
    A section listing rules and holding a subsection.
    """

    KEEP_DESIGNS_SIMPLE = "A rule about design"


class ErrorsRules(Section):
    """
    A subsection whose first rule is spelled out by details.
    """

    @classproperty
    def parent(cls) -> type[Section]:
        return DesignPrinciplesRules

    RAISE_ERRORS = "A rule about errors", ("A first detail", "A second detail")

    NAME_ERRORS = "Another rule about errors"


class VerificationRules(Section):
    """
    A section listing rules only in its subsection.
    """


class AssertionsRules(Section):
    """
    The subsection of a section without rules of its own.
    """

    @classproperty
    def parent(cls) -> type[Section]:
        return VerificationRules

    ASSERT_EXACTLY = "A rule about assertions"


class ReleasesThroughCIRules(Section):
    """
    A section whose name ends in an acronym.
    """
