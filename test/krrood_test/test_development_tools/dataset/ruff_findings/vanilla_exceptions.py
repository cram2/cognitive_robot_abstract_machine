"""
A module raising :class:`Exception` itself and wording a message where it raises.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(kw_only=True)
class EmptyBasket(Exception):
    """
    Raised when a basket holds nothing.
    """


def first_item(items: list[str]) -> str:
    """
    The first of *items*.
    """
    if not items:
        raise EmptyBasket()
    if len(items) > len(set(items)):
        raise Exception("duplicate")  # reported
    if not all(items):
        raise ValueError("Every item needs a name, but at least one is empty")  # reported
    return items[0]
