"""
The positions a property takes in the property chains that imply other properties, and
descriptors made from an attribute of a symbolic variable.
"""

from __future__ import annotations

from krrood.ontomatic.property_descriptor.property_descriptor import (
    PropertyChainPosition,
    PropertyChain,
)
from ..dataset.role_and_ontology.property_chains import (
    AncestorOf,
    ParentOf,
    PersonWithAncestors,
)

# %% chain positions


def test_a_property_takes_a_position_in_every_chain_it_is_part_of():
    """
    A property takes one position for each place it has in a chain that implies another
    property.
    """
    child_of_child = PropertyChain((ParentOf, ParentOf))
    child_of_descendant = PropertyChain((ParentOf, AncestorOf))

    assert set(ParentOf.chain_positions()) == {
        PropertyChainPosition(AncestorOf, child_of_child, 0),
        PropertyChainPosition(AncestorOf, child_of_child, 1),
        PropertyChainPosition(AncestorOf, child_of_descendant, 0),
    }


def test_the_implied_property_takes_a_position_in_a_chain_it_is_part_of():
    """
    The property a chain implies can itself be part of the chain.
    """
    assert AncestorOf.chain_positions() == [
        PropertyChainPosition(AncestorOf, PropertyChain((ParentOf, AncestorOf)), 1)
    ]


def test_a_position_knows_the_properties_before_and_after_it():
    """
    A position splits its chain into the properties before it and those after it.
    """
    position = PropertyChainPosition(
        AncestorOf, PropertyChain((ParentOf, ParentOf, AncestorOf)), 1
    )

    assert position.descriptors_before == (ParentOf,)
    assert position.descriptors_after == (AncestorOf,)


def test_a_chain_finds_every_place_of_a_property_in_it():
    """
    A property can take several places in one chain.
    """
    chain = PropertyChain((ParentOf, AncestorOf, ParentOf))

    assert chain.positions_of(ParentOf) == [0, 2]
    assert chain.positions_of(AncestorOf) == [1]


# %% descriptors from attributes


def test_a_descriptor_made_from_an_attribute_manages_that_field_of_its_class():
    """
    The attribute of a symbolic variable names both the domain and the field.
    """
    descriptor = PersonWithAncestors.__dict__["parent_of"]

    assert (descriptor.domain, descriptor.field_name) == (
        PersonWithAncestors,
        "parent_of",
    )
