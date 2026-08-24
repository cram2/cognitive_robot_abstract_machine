"""
Classes used to verify that ORMatic can persist and reconstruct a ``tuple``-valued many-
to-many relationship (a field typed as ``tuple[SomeMappedClass, ...]``).

Kept in their own module so the generated interface can import them by name.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class TupleCollectionMember:
    """
    A flat value object referenced from a tuple-valued collection field.
    """

    label: str
    """
    The label of the member.
    """


@dataclass
class TupleCollectionOwner:
    """
    An object whose collection of mapped members is a ``tuple`` rather than a ``list``.
    """

    members: tuple[TupleCollectionMember, ...]
    """
    The members of the owner, in a fixed order.
    """
