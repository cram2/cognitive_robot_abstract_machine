"""
A module defining a ``Person`` that is neither the university dataset's nor the one the
annotation resolution tests declare.

The tests load it after their own classes are defined, so the class diagram lists this
``Person`` after theirs: a name lookup in the class diagram that lets the later class
win would find this one.
"""

from __future__ import annotations

from dataclasses import dataclass

from krrood.symbol_graph.symbol_graph import Symbol


@dataclass(eq=False)
class Person(Symbol):
    """
    Deliberately named like the ``Person`` of the annotation resolution tests.
    """

    label: str = ""
