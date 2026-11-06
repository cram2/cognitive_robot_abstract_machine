"""
The positions a property takes in the property chains that imply other properties.
"""

from __future__ import annotations

from krrood.ontomatic.property_descriptor.property_descriptor import (
    ChainPosition,
    PropertyChain,
)
from ..dataset.role_and_ontology.property_chains import AncestorOf, ParentOf

# %% chain positions


def test_a_property_takes_a_position_in_every_chain_it_is_part_of():
    """
    A property takes one position for each place it has in a chain that implies another
    property.
    """
    child_of_child = PropertyChain((ParentOf, ParentOf))
    child_of_descendant = PropertyChain((ParentOf, AncestorOf))

    assert set(ParentOf.chain_positions()) == {
        ChainPosition(AncestorOf, child_of_child, 0),
        ChainPosition(AncestorOf, child_of_child, 1),
        ChainPosition(AncestorOf, child_of_descendant, 0),
    }


def test_the_implied_property_takes_a_position_in_a_chain_it_is_part_of():
    """
    The property a chain implies can itself be part of the chain.
    """
    assert AncestorOf.chain_positions() == [
        ChainPosition(AncestorOf, PropertyChain((ParentOf, AncestorOf)), 1)
    ]


def test_a_position_knows_the_properties_before_and_after_it():
    """
    A position splits its chain into the properties before it and those after it.
    """
    position = ChainPosition(
        AncestorOf, PropertyChain((ParentOf, AncestorOf, ParentOf)), 1
    )

    assert position.descriptors_before == (ParentOf,)
    assert position.descriptors_after == (ParentOf,)
