"""
A module whose dataclass defaults a field to a list.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(kw_only=True)
class Basket:
    """
    A basket of items.
    """

    items: list[str] = []  # reported
    """
    The items every basket shares.
    """

    labels: list[str] = field(default_factory=list)
    """
    The labels of this basket alone.
    """
