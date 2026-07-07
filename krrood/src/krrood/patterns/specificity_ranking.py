"""
Generic specificity-ranked selection: pick the single most-specific candidate from a set, with
class-hierarchy depth as the usual ranking key.

These are domain-agnostic — they know nothing about verbalization. The grammar's
:class:`~krrood.entity_query_language.verbalization.grammar.framework.specificity.SpecificityRule`
families and the ``PhraseRule`` registry build on them, but so could any other consumer that ranks
guarded alternatives by how specific they are.
"""

from __future__ import annotations

import inspect

from typing_extensions import Any, Callable, List, Optional, Sequence, Type, TypeVar

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
    Every concrete (instantiable) transitive subclass of *base*, abstract intermediates excluded —
    the single subclass-discovery primitive.

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
