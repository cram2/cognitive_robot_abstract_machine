"""
Classes used to verify that ORMatic can persist and reconstruct a many-to-many
relationship declared as a :class:`~sortedcontainers.SortedSet` of a mapped class whose
order only exists once its objects are initialised.

Kept in their own module so the generated interface can import them by name.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from sortedcontainers import SortedSet
from typing_extensions import Self


# %% members compared by state their initialisation creates


@dataclass(kw_only=True, eq=False)
class SortedSetCollectionMember:
    """
    A value that orders itself by a key :meth:`__post_init__` derives from its label,
    which ORMatic does not store.
    """

    label: str
    """
    The label of the member.
    """

    _sort_key: str = field(init=False, repr=False)
    """
    The key the members are ordered by, created when the member is initialised.
    """

    def __post_init__(self) -> None:
        self._sort_key = self.label

    def __lt__(self, other: Self) -> bool:
        return self._sort_key < other._sort_key

    def __eq__(self, other: object) -> bool:
        return (
            isinstance(other, SortedSetCollectionMember) and self.label == other.label
        )

    def __hash__(self) -> int:
        return hash(self.label)


# %% owner of a sorted set


@dataclass(kw_only=True)
class SortedSetCollectionOwner:
    """
    An object whose collection of mapped members is a sorted set.
    """

    members: SortedSet[SortedSetCollectionMember]
    """
    The members of the owner, in the order of their labels.
    """
