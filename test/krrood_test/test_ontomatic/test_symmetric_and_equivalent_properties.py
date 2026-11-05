"""
Statements implied by symmetric and by equivalent properties.
"""

from __future__ import annotations

import pytest

from krrood.symbol_graph.symbol_graph import SymbolGraph
from ..dataset.role_and_ontology.symmetric_and_equivalent_properties import (
    PersonWithFriends,
)


@pytest.fixture(autouse=True)
def fresh_symbol_graph():
    """
    Start every test from an empty symbol graph.
    """
    SymbolGraph().clear()
    SymbolGraph()


# %% symmetric properties


def test_a_symmetric_property_relates_the_object_back_to_the_subject():
    """
    A symmetric property relates the object back to the subject.
    """
    alice = PersonWithFriends("alice")
    bob = PersonWithFriends("bob")

    alice.friend_of.add(bob)

    assert bob.friend_of == {alice}


# %% equivalent properties


def test_a_statement_of_a_property_is_a_statement_of_its_equivalent_property():
    """
    A statement of a property is a statement of its equivalent property.
    """
    alice = PersonWithFriends("alice")
    bob = PersonWithFriends("bob")

    alice.knows.add(bob)

    assert alice.is_acquainted_with == {bob}


def test_equivalence_holds_in_both_directions():
    """
    Equivalence holds in both directions.
    """
    alice = PersonWithFriends("alice")
    bob = PersonWithFriends("bob")

    alice.is_acquainted_with.add(bob)

    assert alice.knows == {bob}
