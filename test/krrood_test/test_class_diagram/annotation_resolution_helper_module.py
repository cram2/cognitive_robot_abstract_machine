"""
Helper classes for the annotation resolution tests.

This module binds the name ``Occupant`` to its own class, and declares a field whose
type is imported only under ``TYPE_CHECKING``. The test module that imports it binds the
same name ``Occupant`` differently, so a string annotation naming ``Occupant`` means a
different class in each of the two modules.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from typing_extensions import TYPE_CHECKING

from krrood.symbol_graph.symbol_graph import Symbol

if TYPE_CHECKING:
    # Imported for type checkers only, so the name is absent from this module's globals.
    from .annotation_resolution_fallback_target import FallbackTarget


@dataclass(eq=False)
class Occupant(Symbol):
    """
    The class the name ``Occupant`` is bound to in this module.
    """

    name: str = ""


@dataclass(eq=False)
class OccupiedBase(Symbol):
    """
    A base class declaring a field typed by this module's ``Occupant``.
    """

    occupants: set[Occupant] = field(default_factory=set)


@dataclass(eq=False)
class FallbackHolder(Symbol):
    """
    Declares a field whose type name is bound in no runtime namespace of this module.
    """

    targets: set[FallbackTarget] = field(default_factory=set)
