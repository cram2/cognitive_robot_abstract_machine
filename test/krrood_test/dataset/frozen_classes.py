"""
Frozen dataclasses used to verify that ORMatic can persist and reconstruct ``frozen=True``
dataclasses (its reconstruction path uses ``object.__new__`` + ``object.__setattr__``, which is
frozen-safe).  Kept in their own module so the generated interface can import them by name.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from typing_extensions import List, Set


@dataclass(frozen=True)
class FrozenInner:
    """A flat frozen value object (hashable, so it can live in a ``Set``)."""

    label: str
    weight: int


@dataclass(frozen=True)
class FrozenOuter:
    """A frozen object with a nested frozen relationship, a list, and a set collection."""

    name: str
    inner: FrozenInner
    values: List[int] = field(default_factory=list)
    members: Set[FrozenInner] = field(default_factory=set)
