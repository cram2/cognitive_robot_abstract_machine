"""
Dataclasses whose instances are hashed by their identity.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from krrood.patterns.hashed_by_identity import HashedByIdentity

# %% identity-hashed dataclasses


@dataclass(kw_only=True, eq=False)
class Visitor(HashedByIdentity):
    """
    A visitor that stays a distinct visitor whatever its fields are.
    """

    name: str
    """
    The name the visitor gives.
    """

    visited_rooms: list[str] = field(default_factory=list)
    """
    The rooms the visitor went to, which change while the visitor is kept in a set.
    """
