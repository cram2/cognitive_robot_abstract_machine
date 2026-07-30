"""
Frozen dataclasses used to verify that ORMatic can persist and reconstruct
``frozen=True`` dataclasses (its reconstruction path uses ``object.__new__`` +
``object.__setattr__``, which is frozen-safe).

Kept in their own module so the generated interface can import them by name.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from typing_extensions import List, Set


@dataclass(frozen=True)
class FrozenInner:
    """
    A flat frozen value object (hashable, so it can live in a ``Set``).
    """

    label: str
    """
    The label of the inner value.
    """

    weight: int
    """
    The weight of the inner value.
    """


@dataclass(frozen=True)
class FrozenOuter:
    """
    A frozen object with a nested frozen relationship, a list, and a set collection.
    """

    name: str
    """
    The name of the outer value.
    """

    inner: FrozenInner
    """
    The inner value.
    """

    values: List[int] = field(default_factory=list)
    """
    The list of values in the outer value.
    """

    members: Set[FrozenInner] = field(default_factory=set)
    """
    The members of the outer value.
    """
