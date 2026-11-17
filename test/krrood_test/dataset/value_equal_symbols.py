"""
Mimic symbols that compare equal by value, so two distinct instances of one class can be
equal in the symbol graph.
"""

from __future__ import annotations

from dataclasses import dataclass

from krrood.symbol_graph.symbol_graph import Symbol

# %% value-equal symbols


@dataclass(kw_only=True)
class Label(Symbol):
    """
    A symbol that equals every other label with the same text.
    """

    text: str
    """
    The text of the label.
    """
