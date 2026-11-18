"""
Objects hashed by their identity.
"""

from __future__ import annotations

from krrood.patterns.mixin import Mixin

# %% hashing by identity


class HashedByIdentity(Mixin):
    """
    Mixin for objects that are equal only to themselves, so they stay distinct and
    findable in sets and as keys whatever their fields are or become.

    .. note::
        A dataclass inheriting it passes ``eq=False``: with ``eq=True`` the decorator
        replaces the hash with ``None`` and compares the fields instead.
    """

    def __hash__(self) -> int:
        """
        :return: A hash of the object's identity.
        """
        return id(self)
