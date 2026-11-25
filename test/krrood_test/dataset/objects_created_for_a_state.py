"""
Mimics of objects created for a hypothetical state, which the live
:class:`~krrood.symbol_graph.symbol_graph.SymbolGraph` never holds.
"""

from __future__ import annotations

from dataclasses import dataclass

from typing_extensions import ClassVar

from .semantic_world_like_classes import FixedConnection

# %% connections planned but never built


@dataclass(kw_only=True, unsafe_hash=True)
class PlannedFixedConnection(FixedConnection):
    """
    A fixed connection that exists only in the states that add it, as one an effect
    would create.
    """

    _cache_instances_: ClassVar[bool] = False
    """
    Constructing one leaves the live symbol graph unchanged.
    """
