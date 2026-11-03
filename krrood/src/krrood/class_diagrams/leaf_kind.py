"""
The kinds of leaf values krrood recognises.
"""

from __future__ import annotations

from enum import Enum
from types import NoneType

# %% leaf kinds


class LeafKind(Enum):
    """
    A kind of leaf value, each member holding the classes of that kind.
    """

    JSON_SCALAR = (int, float, str, bool, NoneType)
    """
    Scalars that the built-in JSON module serializes as they are.
    """
