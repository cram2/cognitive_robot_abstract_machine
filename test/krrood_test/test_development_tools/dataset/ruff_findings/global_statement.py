"""
A module whose function rebinds a module variable.
"""

from __future__ import annotations

calls = 0


def count_call() -> int:
    """
    Counts how often it ran.
    """
    global calls  # reported
    calls += 1
    return calls
