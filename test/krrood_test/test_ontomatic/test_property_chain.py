"""
Statements implied by property chains.
"""

from __future__ import annotations

import pytest

from krrood.symbol_graph.symbol_graph import SymbolGraph
from ..dataset.role_and_ontology.property_chains import PersonWithAncestors


@pytest.fixture(autouse=True)
def fresh_symbol_graph():
    """
    Start every test from an empty symbol graph.
    """
    SymbolGraph().clear()
    SymbolGraph()


# %% property chain axioms


def test_a_chain_of_two_properties_implies_the_chained_property():
    """
    A chain of two properties implies the chained property.
    """
    grandparent = PersonWithAncestors("grandparent")
    parent = PersonWithAncestors("parent")
    child = PersonWithAncestors("child")

    grandparent.parent_of.add(parent)
    parent.parent_of.add(child)

    assert child in grandparent.ancestor_of


def test_a_chain_through_the_implied_property_implies_it_again():
    """
    A chain through the implied property implies it again.
    """
    great_grandparent = PersonWithAncestors("great grandparent")
    grandparent = PersonWithAncestors("grandparent")
    parent = PersonWithAncestors("parent")
    child = PersonWithAncestors("child")

    great_grandparent.parent_of.add(grandparent)
    grandparent.parent_of.add(parent)
    parent.parent_of.add(child)

    assert great_grandparent.ancestor_of == {parent, child}
