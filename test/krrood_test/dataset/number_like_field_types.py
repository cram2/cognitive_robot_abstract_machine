"""
Classes whose fields cover the types a number-like field check must tell apart.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum, IntEnum

from krrood.symbol_graph.symbol_graph import Symbol

# %% field types


class Shade(Enum):
    """
    An enumeration whose members are not numbers.
    """

    LIGHT = "light"
    DARK = "dark"


class Priority(IntEnum):
    """
    An enumeration whose members are integers.
    """

    LOW = 1
    HIGH = 2


class Count(int):
    """
    An integer subclass that is not an enumeration.
    """


# %% classes with fields of each type


@dataclass(kw_only=True)
class Measurement(Symbol):
    """
    A record with one field of every type a number-like field check tells apart.
    """

    reading: float
    """
    A floating-point field.
    """

    sample_size: int
    """
    An integer field.
    """

    tally: Count
    """
    A field of an integer subclass.
    """

    is_calibrated: bool
    """
    A boolean field, which is an integer subclass but not a number.
    """

    priority: Priority
    """
    An integer enumeration field.
    """

    shade: Shade
    """
    A plain enumeration field.
    """
