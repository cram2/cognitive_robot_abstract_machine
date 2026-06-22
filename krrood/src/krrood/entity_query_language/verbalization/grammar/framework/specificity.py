from __future__ import annotations

import inspect
from abc import ABC

from typing_extensions import (
    Any,
    Callable,
    List,
    Optional,
    Sequence,
    Type,
    TypeVar,
)

from krrood.utils import recursive_subclasses

_T = TypeVar("_T")


def most_specific(candidates: Sequence[_T], key: Callable[[_T], Any]) -> Optional[_T]:
    """
    :param candidates: Items already filtered to those that apply.
    :param key: Specificity key; the maximum wins.
    :return: The single most-specific candidate by *key*, or ``None`` when empty.
    """
    return max(candidates, key=key, default=None)


def mro_depth(cls: type) -> int:
    """
    :param cls: A class.
    :return: Its specificity — deeper in the hierarchy ⇒ more specific (a subclass outranks the
        alternative it refines).
    """
    return len(cls.__mro__)


class SpecificityRule(ABC):
    """
    A guarded alternative selected by specificity: the shared base of the small rule
    registries (restriction folding, restriction-subject resolution, navigation forms).

    An alternative is a **subclass** that implements an ``applies(...)`` guard (its
    signature is the subfamily's concern) and a payload method; alternatives
    *self-register* as the family's concrete subclasses and are ranked by class specificity —
    a more-derived alternative (one that subclasses another and refines its guard) outranks the
    alternative it refines. Alternatives that are not in a subclass relationship must have
    mutually exclusive guards, so at most one applies (there is no other ordering between them).

    This mirrors ``select`` for :class:`PhraseRule`: precedence comes from the class hierarchy
    (genuine subsumption) or from disjoint guards, never from a hand-assigned number.

    Reference: production-rule selection; the systemic-functional "most delicate system wins"
    principle.
    """

    @classmethod
    def alternatives(cls) -> List[Type[SpecificityRule]]:
        """:return: The concrete alternative subclasses of this family (transitive; abstract
        family bases are excluded)."""
        return [
            subclass
            for subclass in recursive_subclasses(cls)
            if not inspect.isabstract(subclass)
        ]

    @classmethod
    def most_applicable(cls, *args: Any) -> Optional[Type[SpecificityRule]]:
        """
        *args* are forwarded verbatim to each alternative's ``applies`` classmethod, so
        the subfamily fixes that signature (e.g. ``(item, subject)``).

        :return: The most-specific alternative whose ``applies(*args)`` holds, or ``None``.
        """
        applicable = [alt for alt in cls.alternatives() if alt.applies(*args)]
        return most_specific(applicable, key=mro_depth)

