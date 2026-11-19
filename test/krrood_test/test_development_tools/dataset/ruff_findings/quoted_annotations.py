"""
A module postponing its annotations and still quoting some.
"""

from __future__ import annotations


def scale(value: "float", factor: float) -> float:  # reported
    """
    A quoted parameter type.
    """
    return value * factor


def describe(value: float) -> "str":  # reported
    """
    A quoted return type.
    """
    return str(value)
