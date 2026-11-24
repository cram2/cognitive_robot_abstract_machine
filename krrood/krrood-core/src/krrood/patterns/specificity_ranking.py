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

from typing_extensions import Any, Callable, Sequence, TypeVar

from krrood.patterns.classproperty import classproperty
from krrood.patterns.exceptions import AmbiguousRuleError
from krrood.utils import recursive_subclasses

_T = TypeVar("_T")

# %% ranking primitives


def maxima(candidates: Sequence[_T], key: Callable[[_T], Any]) -> list[_T]:
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
    collision_error: Callable[[list[_T]], Exception],
) -> _T | None:
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


def concrete_subclasses(base: type[_T]) -> list[type[_T]]:
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


# %% specificity-ranked alternatives


class SpecificityRankedAlternative(ABC):
    """
    An alternative of its family: its :attr:`guard` decides whether it applies to a
    subject and its :attr:`specificity` ranks it; equally specific alternatives that
    both apply raise :class:`~krrood.patterns.exceptions.AmbiguousRuleError`.

    .. seealso:: Production-rule selection; the systemic-functional "most delicate system
        wins" principle (:cite:t:`halliday2014functional`).
    """

    @classproperty
    @abstractmethod
    def guard(cls) -> Callable[..., bool]:
        """
        The test that is true when this alternative decides for a subject; it takes the
        subject as the keyword arguments given to :meth:`most_applicable`.
        """

    @classproperty
    def specificity(cls) -> int:
        """
        The rank of this alternative among those that apply; the highest wins.

        A more-derived alternative outranks the alternative it refines.
        """
        return mro_depth(cls)

    @classmethod
    def alternatives(cls) -> list[type[SpecificityRankedAlternative]]:
        """
        :return: The concrete alternative subclasses of this family, transitively, with abstract
            family bases excluded.
        """
        return concrete_subclasses(cls)

    @classmethod
    def most_applicable(
        cls, **subject: Any
    ) -> type[SpecificityRankedAlternative] | None:
        """
        :param subject: Forwarded to each alternative's :attr:`guard`.
        :return: The most specific alternative whose guard holds for ``subject``, or
            ``None``.
        :raises AmbiguousRuleError: When several alternatives are equally specific.
        """
        applicable = [
            alternative
            for alternative in cls.alternatives()
            if alternative.guard(**subject)
        ]
        return sole_maximum(
            applicable,
            key=lambda alternative: alternative.specificity,
            collision_error=lambda tied: AmbiguousRuleError(
                subject=subject, candidates=tied
            ),
        )


class SpecificityRule(SpecificityRankedAlternative, ABC):
    """
    A specificity-ranked alternative whose guard is :meth:`applies`, ranked by class
    depth: an alternative that subclasses another and refines its guard outranks it.
    """

    @classproperty
    def guard(cls) -> Callable[..., bool]:
        """
        :meth:`applies`.
        """
        return cls.applies

    @classmethod
    @abstractmethod
    def applies(cls, **subject: Any) -> bool:
        """
        :param subject: The subject, keyed by the parameters of the family's guard.
        :return: True when this alternative decides for the given subject.
        """
