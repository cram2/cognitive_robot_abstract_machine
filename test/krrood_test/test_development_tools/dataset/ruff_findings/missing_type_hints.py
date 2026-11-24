"""
A module leaving each kind of parameter and return value unannotated once, next to a
function annotated with ``Any``, which counts as annotated.
"""

from __future__ import annotations

from typing_extensions import Any


def scale(value, factor: float) -> float:  # reported
    """
    A parameter without a type hint.
    """
    return value * factor


def describe(value: float):  # reported
    """
    A public function without a return type.
    """
    return str(value)


def total(*values) -> float:  # reported
    """
    Variadic positional parameters without a type hint.
    """
    return sum(values)


def configure(**options) -> None:  # reported
    """
    Variadic keyword parameters without a type hint.
    """


def pass_through(value: Any) -> Any:
    """
    Fully annotated, with ``Any``.
    """
    return value


class Gauge:
    """
    A class whose methods leave their return types out.
    """

    def __init__(self, reading: float):  # reported
        self.reading = reading

    @staticmethod
    def zero():  # reported
        """
        A static method without a return type.
        """
        return Gauge(reading=0.0)

    @classmethod
    def calibrated(cls):  # reported
        """
        A class method without a return type.
        """
        return cls(reading=1.0)

    def _offset(self):  # reported
        """
        A private method without a return type.
        """
        return 0.0
