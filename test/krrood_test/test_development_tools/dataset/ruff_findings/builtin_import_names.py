"""
A module binding the names of builtins through an import and a lambda parameter.
"""

from __future__ import annotations

from fractions import Fraction as round  # reported


def halve_all(values: list[int]) -> list[round]:
    """
    Halves each value with a lambda whose parameter takes a builtin's name.
    """
    return list(map(lambda input: round(input, 2), values))  # reported
