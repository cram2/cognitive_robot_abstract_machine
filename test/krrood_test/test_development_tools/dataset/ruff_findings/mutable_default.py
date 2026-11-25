"""
A module whose function defaults a parameter to a list, and which asserts.
"""

from __future__ import annotations


def collect(item: int, collected: list[int] = []) -> list[int]:  # reported
    """
    Appends *item* to a list every call shares.
    """
    assert item >= 0
    collected.append(item)
    return collected
