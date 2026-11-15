"""
A fact about an individual is stored on the object of the individual, its root role
taker or one of its roles, whose declaration of the property admits the value.
"""

from __future__ import annotations

import pytest

from krrood.ontomatic.property_descriptor.property_descriptor_relation import (
    PropertyDescriptorRelation,
)
from ..dataset.role_and_ontology.hobbies import (
    BasketBall,
    BasketBallLover,
    Likes,
    Loves,
    PeopleWithHobby,
    PersonWithHobbies,
    PersonWithTwoRoles,
    Sports,
    SportsLover,
    VideoGames,
)


@pytest.fixture
def individual() -> PersonWithTwoRoles:
    """
    :return: A person with a basketball lover role and a sports lover role.
    """
    return PersonWithTwoRoles.named(name="Bass")


# %% declarations of a property


def test_a_class_declares_a_property_through_the_nearest_class_declaring_it():
    """
    A class declares a property through its own field managed by exactly that property,
    or else through the nearest class it inherits such a field from.
    """
    assert (
        Loves.declared_by(domain_type=BasketBallLover)
        is BasketBallLover.__dict__["loves"]
    )
    assert (
        Likes.declared_by(domain_type=BasketBallLover)
        is PeopleWithHobby.__dict__["likes"]
    )
    assert Loves.declared_by(domain_type=PersonWithHobbies) is None


# %% the objects of an individual


def test_objects_of_individual_lists_the_root_first_and_tells_roles_apart(
    individual: PersonWithTwoRoles,
):
    """
    The objects of an individual are its root followed by its roles in the order they
    were created.

    Roles compare equal to their taker, so the objects are compared by identity.
    """
    for individual_object in individual.objects:
        objects = PropertyDescriptorRelation.objects_of_individual(
            instance=individual_object
        )
        assert list(map(id, objects)) == list(map(id, individual.objects))


# %% declaring objects


def test_declaring_object_chooses_the_first_declaration_whose_range_admits_the_value(
    individual: PersonWithTwoRoles,
):
    """
    The declaring object is the first object whose declaration admits the value.
    """
    football = Sports(name="Football")
    basketball = BasketBall(name="BasketBall")
    for individual_object in individual.objects:
        declaration = PropertyDescriptorRelation.declaring_object(
            instance=individual_object, property_descriptor_class=Loves, value=football
        )
        assert declaration.holder is individual.sports_lover
        assert declaration.descriptor.domain is SportsLover
        declaration = PropertyDescriptorRelation.declaring_object(
            instance=individual_object,
            property_descriptor_class=Loves,
            value=basketball,
        )
        assert declaration.holder is individual.basketball_lover
        assert declaration.descriptor.domain is BasketBallLover


def test_declaring_object_falls_back_to_the_first_declaration(
    individual: PersonWithTwoRoles,
):
    """
    Without a declaration admitting the value, the declaring object is the first
    declaring the property.
    """
    game = VideoGames(name="Tetris")
    declaration = PropertyDescriptorRelation.declaring_object(
        instance=individual.person, property_descriptor_class=Loves, value=game
    )
    assert declaration.holder is individual.basketball_lover
    assert declaration.descriptor.domain is BasketBallLover


def test_declaring_object_is_none_without_a_declaration():
    """
    No object declares the property of an individual without a role declaring it.
    """
    person = PersonWithHobbies(name="Bass")
    assert (
        PropertyDescriptorRelation.declaring_object(
            instance=person,
            property_descriptor_class=Loves,
            value=Sports(name="Football"),
        )
        is None
    )
