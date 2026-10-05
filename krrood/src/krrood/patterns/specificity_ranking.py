"""
Generic specificity-ranked selection: pick the single most-specific candidate from a
set, with class-hierarchy depth as the usual ranking key.

These are domain-agnostic. :class:`SpecificityRule` builds a family of guarded
alternatives on them; the verbalization grammar and ORMatic's field storage rules are
such families, and the verbalization ``PhraseRule`` registry ranks its rules with the
same functions.
"""

from __future__ import annotations

import inspect
from abc import ABC, abstractmethod

from typing_extensions import Any, Callable, List, Optional, Sequence, Type, TypeVar

from krrood.patterns.exceptions import AmbiguousRuleError
from krrood.utils import recursive_subclasses

_T = TypeVar("_T")


def maxima(candidates: Sequence[_T], key: Callable[[_T], Any]) -> List[_T]:
    """
    :param candidates: Items already filtered to those that apply.
    :param key: Specificity key; the highest value wins.
    :return: Every candidate sharing the maximum *key* (more than one ⇒ a tie); empty when there
        are no candidates.

    >>> maxima(["a", "abc", "ab"], key=len)
    ['abc']
    >>> maxima(["ab", "cd", "a"], key=len)
    ['ab', 'cd']
    >>> maxima([], key=len)
    []
    """
    if not candidates:
        return []
    best = max(key(candidate) for candidate in candidates)
    return [candidate for candidate in candidates if key(candidate) == best]


def sole_maximum(
    candidates: Sequence[_T],
    key: Callable[[_T], Any],
    collision_error: Callable[[List[_T]], Exception],
) -> Optional[_T]:
    """
    :param candidates: Items already filtered to those that apply.
    :param key: Specificity key; the highest value wins.
    :param collision_error: Builds the exception to raise when several candidates tie, given the
        tied candidates. Injected so this stays decoupled from any one caller's exception type.
    :return: The single most-specific candidate by *key*, or ``None`` when empty.
    :raises Exception: The *collision_error* result when two or more candidates are equally specific.

    >>> sole_maximum(["a", "abc", "ab"], key=len, collision_error=AssertionError)
    'abc'
    >>> sole_maximum([], key=len, collision_error=AssertionError) is None
    True
    """
    winners = maxima(candidates, key)
    if len(winners) > 1:
        raise collision_error(winners)
    return winners[0] if winners else None


def mro_depth(cls: type) -> int:
    """
    :param cls: A class.
    :return: Its specificity — deeper in the hierarchy ⇒ more specific (a subclass outranks the
        alternative it refines).

    >>> mro_depth(object)
    1
    >>> mro_depth(bool) > mro_depth(int)
    True
    """
    return len(cls.__mro__)


def concrete_subclasses(base: Type[_T]) -> List[Type[_T]]:
    """
    Every concrete (instantiable) transitive subclass of *base*, abstract intermediates
    excluded — the single subclass-discovery primitive.

    :param base: The family / rule base class.
    :return: Its concrete transitive subclasses.

    >>> from abc import ABC, abstractmethod
    >>> class Shape(ABC):
    ...     @abstractmethod
    ...     def area(self): ...
    >>> class Polygon(Shape, ABC):  # still abstract
    ...     pass
    >>> class Square(Polygon):
    ...     def area(self): return 1
    >>> [cls.__name__ for cls in concrete_subclasses(Shape)]
    ['Square']
    """
    return [
        subclass
        for subclass in recursive_subclasses(base)
        if not inspect.isabstract(subclass)
    ]


class SpecificityRule(ABC):
    """
    A guarded alternative selected by specificity: the base of a family of rules of
    which at most one decides for any subject.

    An alternative is a concrete subclass that implements the ``applies(...)`` guard (its
    signature is the family's concern) and whatever payload the family needs. Alternatives
    register themselves by subclassing and are ranked by class specificity: a more-derived
    alternative, one that subclasses another and refines its guard, outranks the alternative it
    refines. Alternatives that are not in a subclass relationship must have mutually exclusive
    guards, so at most one applies; a tie raises
    :class:`~krrood.patterns.exceptions.AmbiguousRuleError`.

    Precedence comes from the class hierarchy (genuine subsumption) or from disjoint guards,
    never from a hand-assigned number or the order in which rules are checked.

    Reference: production-rule selection; the systemic-functional "most delicate system wins"
    principle (:cite:t:`halliday2014functional`).
    """

    @classmethod
    @abstractmethod
    def applies(cls, *args: Any) -> bool:
        """
        :return: True when this alternative decides for the given subject.
        """

    @classmethod
    def alternatives(cls) -> List[Type[SpecificityRule]]:
        """
        :return: The concrete alternative subclasses of this family, transitively, with abstract
            family bases excluded.
        """
        return concrete_subclasses(cls)

    @classmethod
    def most_applicable(cls, *args: Any) -> Optional[Type[SpecificityRule]]:
        """
        :param args: Forwarded to each alternative's ``applies``.
        :return: The most specific alternative whose ``applies(*args)`` holds, or ``None``.
        :raises AmbiguousRuleError: When several alternatives are equally specific.
        """
        applicable = [
            alternative
            for alternative in cls.alternatives()
            if alternative.applies(*args)
        ]
        return sole_maximum(
            applicable,
            key=mro_depth,
            collision_error=lambda tied: AmbiguousRuleError(
                subject=args, candidates=tied
            ),
        )
