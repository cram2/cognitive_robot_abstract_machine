"""
A module whose mutable default is a deliberate exception, marked for ruff.
"""

from __future__ import annotations


def remembered(item: int, seen: list[int] = []) -> list[int]:  # noqa: B006
    """
    Remembers every item across calls on purpose.
    """
    seen.append(item)
    return seen
