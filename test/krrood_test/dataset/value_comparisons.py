"""
Predicates comparing plain values.
"""

from __future__ import annotations

from dataclasses import dataclass

from typing_extensions import Mapping

from krrood.entity_query_language.predicate import Predicate
from krrood.entity_query_language.verbalization.fragments.base import (
    VerbalizationFragment,
)
from krrood.entity_query_language.verbalization.vocabulary.parts_of_speech import (
    clause,
    Noun,
    Verb,
)


@dataclass
class IsGreaterThan(Predicate):
    """
    Whether a value exceeds a bound.
    """

    value: float
    """
    The value to compare.
    """

    bound: float
    """
    The value it has to exceed.
    """

    def __call__(self) -> bool:
        return self.value > self.bound

    @classmethod
    def _verbalization_fragment_(
        cls, fields: Mapping[str, VerbalizationFragment]
    ) -> VerbalizationFragment:
        return clause(Noun(fields["value"]), Verb("exceed"), Noun(fields["bound"]))


@dataclass
class IsGreaterThanByMargin(Predicate):
    """
    Whether a value exceeds a bound by more than a margin.
    """

    value: float
    """
    The value to compare.
    """

    bound: float
    """
    The value it has to exceed.
    """

    margin: float = 1.0
    """
    How far above the bound the value has to be.
    """

    def __call__(self) -> bool:
        return self.value > self.bound + self.margin

    @classmethod
    def _verbalization_fragment_(
        cls, fields: Mapping[str, VerbalizationFragment]
    ) -> VerbalizationFragment:
        return clause(Noun(fields["value"]), Verb("exceed"), Noun(fields["bound"]))
