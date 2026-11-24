"""
A module importing under TYPE_CHECKING a class it uses when it runs.
"""

from __future__ import annotations

from typing_extensions import TYPE_CHECKING

if TYPE_CHECKING:
    from pathlib import Path  # reported
    from decimal import Decimal


def working_directory() -> Path:
    """
    Calls the class only imported for type checkers.
    """
    return Path.cwd()


def half(value: Decimal) -> Decimal:
    """
    Uses its import only in annotations.
    """
    return value / 2
