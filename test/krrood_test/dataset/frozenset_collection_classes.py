"""
Classes used to verify that ORMatic persists and reconstructs fields declared as a
``frozenset``, both of a mapped class and of a builtin.

Kept in their own module so the generated interface can import them by name.
"""

from __future__ import annotations

from dataclasses import dataclass

# %% members


@dataclass(kw_only=True, frozen=True)
class FrozensetCollectionMember:
    """
    A flat value object referenced from a frozenset-valued collection field.
    """

    label: str
    """
    The label of the member.
    """


# %% owner of frozensets


@dataclass(kw_only=True)
class FrozensetCollectionOwner:
    """
    An object whose collections are frozensets: one of mapped members, one of strings.
    """

    members: frozenset[FrozensetCollectionMember]
    """
    The members of the owner, in no particular order.
    """

    tags: frozenset[str]
    """
    The tags of the owner, in no particular order.
    """
