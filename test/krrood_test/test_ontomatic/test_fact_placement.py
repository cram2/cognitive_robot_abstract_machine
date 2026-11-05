"""
A fact about an individual is stored on the object of the individual, its root role
taker or one of its roles, whose declaration of the property admits the value.
"""

from __future__ import annotations

from krrood.ontomatic.property_descriptor.property_descriptor_relation import (
    PropertyDescriptorRelation,
)
from krrood.symbol_graph.symbol_graph import SymbolGraph
from ..dataset.role_and_ontology.hobbies import (
    BasketBall,
    BasketBallLover,
    Loves,
    PersonWithHobbies,
    Sports,
    SportsLover,
    VideoGames,
)

# %% helpers


def make_individual() -> tuple[PersonWithHobbies, BasketBallLover, SportsLover]:
    """
    :return: A person with a basketball lover role and a sports lover role, created in that order.
    """
    SymbolGraph().clear()
    SymbolGraph()
    person = PersonWithHobbies(name="Bass")
    basketball_lover = BasketBallLover(role_taker=person)
    sports_lover = SportsLover(role_taker=person)
    return person, basketball_lover, sports_lover


# %% the objects of an individual


def test_objects_of_individual_lists_the_root_first_and_tells_roles_apart():
    """
    The objects of an individual are its root followed by its roles in the order they
    were created.
    """
    person, basketball_lover, sports_lover = make_individual()
    for individual_object in (person, basketball_lover, sports_lover):
        objects = PropertyDescriptorRelation.objects_of_individual(individual_object)
        assert [id(o) for o in objects] == [
            id(person),
            id(basketball_lover),
            id(sports_lover),
        ]


# %% declaring objects


def test_declaring_object_chooses_the_first_declaration_whose_range_admits_the_value():
    """
    The declaring object is the first object whose declaration admits the value.
    """
    person, basketball_lover, sports_lover = make_individual()
    football = Sports(name="Football")
    basketball = BasketBall(name="BasketBall")
    for individual_object in (person, basketball_lover, sports_lover):
        holder, descriptor = PropertyDescriptorRelation.declaring_object(
            individual_object, Loves, football
        )
        assert holder is sports_lover and descriptor.domain is SportsLover
        holder, descriptor = PropertyDescriptorRelation.declaring_object(
            individual_object, Loves, basketball
        )
        assert holder is basketball_lover and descriptor.domain is BasketBallLover


def test_declaring_object_falls_back_to_the_first_declaration():
    """
    Without a declaration admitting the value, the declaring object is the first
    declaring the property.
    """
    person, basketball_lover, _ = make_individual()
    game = VideoGames(name="Tetris")
    holder, descriptor = PropertyDescriptorRelation.declaring_object(
        person, Loves, game
    )
    assert holder is basketball_lover and descriptor.domain is BasketBallLover


def test_declaring_object_is_none_without_a_declaration():
    """
    No object declares the property of an individual without a role declaring it.
    """
    SymbolGraph().clear()
    SymbolGraph()
    person = PersonWithHobbies(name="Bass")
    assert (
        PropertyDescriptorRelation.declaring_object(
            person, Loves, Sports(name="Football")
        )
        is None
    )
