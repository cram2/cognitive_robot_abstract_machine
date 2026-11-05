"""
The class a ``TYPE_CHECKING``-only annotation of the annotation resolution helper module
names.
"""

from __future__ import annotations

from dataclasses import dataclass

from krrood.symbol_graph.symbol_graph import Symbol


@dataclass(eq=False)
class FallbackTarget(Symbol):
    """
    A class that no module binds at runtime under the name its annotation uses.
    """

    name: str = ""
