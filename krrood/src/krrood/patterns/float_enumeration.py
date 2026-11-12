from __future__ import annotations

from enum import ReprEnum

# %% enumeration of floats


class FloatEnum(float, ReprEnum):
    """
    An enumeration whose members are floats.

    A member is the number it names: it compares, computes and converts to text as that
    number does, like the members of :class:`enum.IntEnum` and :class:`enum.StrEnum`.
    """
