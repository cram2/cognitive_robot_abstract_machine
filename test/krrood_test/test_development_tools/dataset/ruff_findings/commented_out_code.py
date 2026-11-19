"""
A module leaving code commented out next to a comment that explains.
"""

from __future__ import annotations


def doubled(value: int) -> int:
    """
    Twice *value*.
    """
    # result = value + value  # reported
    # Shifting would be faster but reads worse.
    return value * 2
