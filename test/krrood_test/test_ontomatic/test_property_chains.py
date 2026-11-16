"""
The positions a property takes in the property chains that imply other properties, the
statements the chains imply, and descriptors made from an attribute of a symbolic
variable.
"""

from __future__ import annotations

from krrood.ontomatic.property_descriptor.property_descriptor import (
    PropertyChainPosition,
    PropertyChain,
)
from krrood.symbol_graph.symbol_graph import SymbolGraph
from krrood.ontomatic.property_descriptor.property_descriptor_relation import (
    InferredThrough,
    PropertyDescriptorRelation,
    RelationExplanation,
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
    child_of_child = PropertyChain(properties=(ParentOf, ParentOf))
    child_of_descendant = PropertyChain(properties=(ParentOf, AncestorOf))

    assert set(ParentOf.chain_positions()) == {
        PropertyChainPosition(
            implied_descriptor=AncestorOf, chain=child_of_child, index=0
        ),
        PropertyChainPosition(
            implied_descriptor=AncestorOf, chain=child_of_child, index=1
        ),
        PropertyChainPosition(
            implied_descriptor=AncestorOf, chain=child_of_descendant, index=0
        ),
    }


def test_the_implied_property_takes_a_position_in_a_chain_it_is_part_of():
    """
    The property a chain implies can itself be part of the chain.
    """
    assert AncestorOf.chain_positions() == [
        PropertyChainPosition(
            implied_descriptor=AncestorOf,
            chain=PropertyChain(properties=(ParentOf, AncestorOf)),
            index=1,
        )
    ]


def test_a_position_knows_the_properties_before_and_after_it():
    """
    A position splits its chain into the properties before it and those after it.
    """
    position = PropertyChainPosition(
        implied_descriptor=AncestorOf,
        chain=PropertyChain(properties=(ParentOf, ParentOf, AncestorOf)),
        index=1,
    )

    assert position.descriptors_before == (ParentOf,)
    assert position.descriptors_after == (AncestorOf,)


def test_a_chain_finds_every_place_of_a_property_in_it():
    """
    A property can take several places in one chain.
    """
    chain = PropertyChain(properties=(ParentOf, AncestorOf, ParentOf))

    assert chain.positions_of(descriptor_class=ParentOf) == [0, 2]
    assert chain.positions_of(descriptor_class=AncestorOf) == [1]


# %% statements implied by chains


def test_a_chain_of_two_properties_implies_the_chained_property():
    """
    A chain of two properties implies the chained property.
    """
    grandparent = PersonWithAncestors(name="grandparent")
    parent = PersonWithAncestors(name="parent")
    child = PersonWithAncestors(name="child")

    grandparent.parent_of.add(parent)
    parent.parent_of.add(child)

    assert child in grandparent.ancestor_of


def test_a_chain_through_the_implied_property_implies_it_again():
    """
    A chain through the implied property implies it again.
    """
    great_grandparent = PersonWithAncestors(name="great grandparent")
    grandparent = PersonWithAncestors(name="grandparent")
    parent = PersonWithAncestors(name="parent")
    child = PersonWithAncestors(name="child")

    great_grandparent.parent_of.add(grandparent)
    grandparent.parent_of.add(parent)
    parent.parent_of.add(child)

    assert great_grandparent.ancestor_of == {parent, child}


def test_an_implied_statement_is_explained_by_the_statements_of_its_chain():
    """
    The statement a chain implies is explained by the statements of the chain, in the
    order of the chain.
    """
    grandparent = PersonWithAncestors(name="grandparent")
    parent = PersonWithAncestors(name="parent")
    child = PersonWithAncestors(name="child")

    grandparent.parent_of.add(parent)
    parent.parent_of.add(child)

    assert PropertyDescriptorRelation.find_relation_for_triple(
        source_instance=grandparent,
        property_descriptor_class=AncestorOf,
        target_instance=child,
    ).explanation == RelationExplanation(
        rule=InferredThrough.CHAIN,
        premises=(
            PropertyDescriptorRelation.find_relation_for_triple(
                source_instance=grandparent,
                property_descriptor_class=ParentOf,
                target_instance=parent,
            ),
            PropertyDescriptorRelation.find_relation_for_triple(
                source_instance=parent,
                property_descriptor_class=ParentOf,
                target_instance=child,
            ),
        ),
    )


def test_an_explanation_shows_a_fact_above_the_facts_it_was_inferred_from():
    """
    An explanation names each fact by the symbol graph's names of its subject and value,
    and the rule it was inferred through, with its premises indented below it.
    """
    grandparent = PersonWithAncestors(name="grandparent")
    parent = PersonWithAncestors(name="parent")
    child = PersonWithAncestors(name="child")

    grandparent.parent_of.add(parent)
    parent.parent_of.add(child)

    names = {
        person.name: SymbolGraph().get_wrapped_instance(instance=person).name
        for person in (grandparent, parent, child)
    }
    assert PropertyDescriptorRelation.find_relation_for_triple(
        source_instance=grandparent,
        property_descriptor_class=AncestorOf,
        target_instance=child,
    ).explain() == (
        f"{names['grandparent']} ancestor_of {names['child']} (chain)\n"
        f"  {names['grandparent']} parent_of {names['parent']} (asserted)\n"
        f"  {names['parent']} parent_of {names['child']} (asserted)"
    )


def test_an_explanation_shows_premises_only_down_to_its_depth():
    """
    An explanation of depth zero shows only the fact and its rule.
    """
    grandparent = PersonWithAncestors(name="grandparent")
    parent = PersonWithAncestors(name="parent")
    child = PersonWithAncestors(name="child")

    grandparent.parent_of.add(parent)
    parent.parent_of.add(child)

    explanation = PropertyDescriptorRelation.find_relation_for_triple(
        source_instance=grandparent,
        property_descriptor_class=AncestorOf,
        target_instance=child,
    ).explain(depth=0)
    assert explanation.endswith("(chain)")
    assert "\n" not in explanation


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
