"""
What every question of an experiment's catalogue is made of: a query for an example with
some parts listed open, a cause, confounders and an effect marked on it, and the words
to describe them.

Every query lists at least one part of every kind with every attribute left open: that
is what makes grounding retain the example's aggregation counts as variables instead of
computing them from an empty part list, and the flat table ignores parts a query says
nothing about.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, fields

from krrood.entity_query_language.factories import a
from krrood.entity_query_language.query.match import Match
from typing_extensions import Any, Dict, List, Tuple, Type

from experiments.causal_reasoning.comparison.domain import RelationalDomain


def part_query(part_class: Type, **specified: Any) -> Match:
    """
    A query for one part with every attribute left open but the given ones.

    :param part_class: The part's class.
    :param specified: Attribute markers or values to set instead of leaving open.
    :return: The query.
    """
    return a(part_class)(
        **{
            part_field.name: specified.get(part_field.name, ...)
            for part_field in fields(part_class)
        }
    )


def example_query(
    domain: RelationalDomain, parts: Dict[str, List[Match]], **specified: Any
) -> Match:
    """
    A query for an example with the given parts and every scalar attribute left open but
    the given ones.

    :param domain: The example and its parts.
    :param parts: Per exchangeable-part field, one query per listed part.
    :param specified: Scalar attribute markers or values to set instead of leaving open;
        an aggregation count's name is accepted too.
    :return: The query.
    """
    scalar_fields = domain.scalar_fields
    return a(domain.example_class)(
        **{name: specified.get(name, ...) for name in scalar_fields},
        **{
            name: value
            for name, value in specified.items()
            if name not in scalar_fields
        },
        **parts,
    )


@dataclass(frozen=True)
class Confounder:
    """
    One example attribute or aggregation count a question adjusts for, as a variable and
    in words.
    """

    name: str
    """
    The example attribute or aggregation count.
    """

    noun: str
    """
    The confounder in words, such as ``the number of objects``.
    """


@dataclass(frozen=True, kw_only=True)
class CausalQueryCase(ABC):
    """
    One question, as a query and as words.
    """

    open_part_count: int = 1
    """
    How many parts of every kind the query lists with every attribute open.
    """

    @property
    @abstractmethod
    def domain(self) -> RelationalDomain:
        """
        The example the question is about.
        """

    @property
    @abstractmethod
    def name(self) -> str:
        """
        A short identifier for tables.
        """

    @property
    @abstractmethod
    def question(self) -> str:
        """
        The question in words.
        """

    @abstractmethod
    def build(self) -> Match:
        """
        :return: The query, freshly built, with its cause, confounders and effect
            marked.
        """

    @abstractmethod
    def describe_cause(self, region: str) -> str:
        """
        :param region: A region of the cause, written out.
        :return: The cause set to that region, in words.
        """

    @property
    @abstractmethod
    def effect(self) -> str:
        """
        The effect in words.
        """

    def open_parts(self) -> Dict[str, List[Match]]:
        """
        :return: Per exchangeable-part field, :attr:`open_part_count` fully open
            queries.
        """
        return {
            part_field: [
                part_query(self.domain.part_class(part_field))
                for _ in range(self.open_part_count)
            ]
            for part_field in self.domain.part_fields
        }

    def example_query(self, parts: Dict[str, List[Match]], **specified: Any) -> Match:
        """
        :param parts: Per exchangeable-part field, one query per listed part.
        :param specified: Scalar attribute markers or values to set instead of leaving
            open.
        :return: The query for the example.
        """
        return example_query(self.domain, parts, **specified)


@dataclass(frozen=True, kw_only=True)
class AdjustedCountCase(CausalQueryCase, ABC):
    """
    A question whose cause is one of the example's aggregation counts, asked under a set
    of confounders; the same count asked under several sets is one family in the report.
    """

    statistic_name: str
    """
    The aggregation count that is the cause.
    """

    confounders: Tuple[Confounder, ...] = ()
    """
    What the question adjusts for.
    """
