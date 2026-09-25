"""
What every entity an EQL query can range over has in common.
"""

from __future__ import annotations

from abc import ABC
from dataclasses import dataclass

from krrood.symbol_graph.symbol_graph import Symbol


@dataclass
class NamedEntity(Symbol, ABC):
    """
    One entity of the recorded episode or the scanned architecture.

    Every EQL variable ranges over these, and the viewer highlights a result by its
    name, so the name is what they all share. As a :class:`Symbol`, every instance is
    tracked in the SymbolGraph.
    """

    name: str
    """
    Identifier the viewer highlights this entity by.
    """
