"""
What every entity an EQL query can range over has in common.
"""

from __future__ import annotations

from abc import ABC
from dataclasses import dataclass


@dataclass
class NamedEntity(ABC):
    """
    One entity of the recorded episode or the scanned architecture.

    Every EQL variable ranges over these, and the viewer highlights a result by its
    name, so the name is what they all share.
    """

    name: str
    """
    Identifier the viewer highlights this entity by.
    """
