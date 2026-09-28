"""
A role whose taker is kept as a value in the role's own row rather than in a table of
its own, and whose type is a generic left unparameterized.
"""

from __future__ import annotations

from dataclasses import dataclass

from krrood.patterns.role import Role
from krrood.symbol_graph.symbol_graph import Symbol
from test.krrood_test.dataset.example_classes import GenericJSONSerializableClass


@dataclass(eq=False)
class RoleOverAValueStoredAsJson(Role[GenericJSONSerializableClass], Symbol):
    """
    Plays a part that the value it is about cannot play by itself.
    """

    note: str = ""
    """
    What this role records about its taker.
    """
