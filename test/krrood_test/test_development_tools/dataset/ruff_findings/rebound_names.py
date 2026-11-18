"""
A module binding names again before using them, inside a loop and over a parameter.
"""

from __future__ import annotations

from os import linesep
from os import sep
from os import sep  # reported


def count_lines(text: str) -> int:
    """
    A loop variable overwritten inside its loop, and a loop rebinding a parameter.
    """
    total = 0
    for line in text.split(linesep):
        line = line.strip()  # reported
        total += len(line)
    for text in ("a", "b"):  # reported
        total += len(text)
    return total


def join_names(names: list[str]) -> str:
    """
    A loop variable rebinding an imported name.
    """
    joined = sep
    for linesep in names:  # reported
        joined += linesep
    return joined
