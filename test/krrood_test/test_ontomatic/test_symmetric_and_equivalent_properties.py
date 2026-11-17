"""
Statements implied by symmetric, by equivalent and by transitive properties.
"""

from __future__ import annotations

from krrood.ontomatic.property_descriptor.property_descriptor_relation import (
    InferredThrough,
    PropertyDescriptorRelation,
    RelationExplanation,
)
from ..dataset.role_and_ontology.symmetric_and_equivalent_properties import (
    Area,
    Building,
    FriendOf,
    IsPartOf,
    PersonWithFriends,
)


# %% symmetric properties


def test_a_symmetric_property_relates_the_object_back_to_the_subject():
    """
    A symmetric property relates the object back to the subject.
    """
    alice = PersonWithFriends(name="alice")
    bob = PersonWithFriends(name="bob")

    alice.friend_of.add(bob)

    assert bob.friend_of == {alice}


def test_the_statement_back_is_explained_by_the_statement_it_reverses():
    """
    The statement back from the object is inferred through symmetry from the statement
    made, which is asserted and has no explanation.
    """
    alice = PersonWithFriends(name="alice")
    bob = PersonWithFriends(name="bob")

    alice.friend_of.add(bob)

    asserted = PropertyDescriptorRelation.find_relation_for_triple(
        source_instance=alice, property_descriptor_class=FriendOf, target_instance=bob
    )
    assert asserted.explanation is None
    assert PropertyDescriptorRelation.find_relation_for_triple(
        source_instance=bob, property_descriptor_class=FriendOf, target_instance=alice
    ).explanation == RelationExplanation(
        rule=InferredThrough.SYMMETRY, premises=(asserted,)
    )


# %% equivalent properties


def test_a_statement_of_a_property_is_a_statement_of_its_equivalent_property():
    """
    A statement of a property is a statement of its equivalent property.
    """
    alice = PersonWithFriends(name="alice")
    bob = PersonWithFriends(name="bob")

    alice.knows.add(bob)

    assert alice.is_acquainted_with == {bob}


def test_equivalence_holds_in_both_directions():
    """
    Equivalence holds in both directions.
    """
    alice = PersonWithFriends(name="alice")
    bob = PersonWithFriends(name="bob")

    alice.is_acquainted_with.add(bob)

    assert alice.knows == {bob}


# %% transitive properties with an equivalent property


def test_a_statement_through_an_equivalent_property_is_composed_with_the_statements_after_it():
    """
    A statement of a transitive property implied by its equivalent property is composed
    with the statements of the transitive property already made about its object.
    """
    kitchen = Area(name="kitchen")
    flat = Area(name="flat")
    city = Area(name="city")
    flat.is_part_of.add(city)

    kitchen.lies_within.add(flat)

    assert kitchen.is_part_of == {flat, city}


# %% transitive properties held by different fields


def test_a_composed_statement_is_held_by_the_field_of_its_subject():
    """
    A statement composed with a statement made before it is held by the field of the
    subject of the earlier statement, also when the two subjects hold the property in
    fields of different classes.
    """
    house = Building(name="house")
    city = Area(name="city")
    country = Area(name="country")
    house.stands_in.add(city)

    city.is_part_of.add(country)

    composed = PropertyDescriptorRelation.find_relation_for_triple(
        source_instance=house,
        property_descriptor_class=IsPartOf,
        target_instance=country,
    )
    assert composed.explanation == RelationExplanation(
        rule=InferredThrough.TRANSITIVE,
        premises=(
            PropertyDescriptorRelation.find_relation_for_triple(
                source_instance=house,
                property_descriptor_class=IsPartOf,
                target_instance=city,
            ),
            PropertyDescriptorRelation.find_relation_for_triple(
                source_instance=city,
                property_descriptor_class=IsPartOf,
                target_instance=country,
            ),
        ),
    )
