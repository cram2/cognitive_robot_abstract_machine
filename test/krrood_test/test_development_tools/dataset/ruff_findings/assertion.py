"""
A module that checks a precondition with ``assert``.
"""

from __future__ import annotations


def halve(number: int) -> int:
    """
    Halves an even number.
    """
    assert number % 2 == 0  # reported
    return number // 2
