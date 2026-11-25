"""
The value a path reaches when it reads an attribute through a collection that holds no
elements.
"""

from __future__ import annotations

from enum import Enum, auto

# %% absence


class Absence(Enum):
    """
    A value that stands for an element that does not exist.

    It lets a row survive an empty collection: every mapping of it is again absent, and
    every comparison on it is false - except the negated ones, ``!=`` and *not in*, which
    are true - so a condition on an element that does not exist does not hold.
    """

    NO_ELEMENT = auto()
    """
    The element of a collection that holds no elements, or of one that is itself absent.
    """
