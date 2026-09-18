"""
The structural key that decides which selected column a query is ranked by.

``QueryAssembler._expression_signature`` must give different signatures to two chains
describing different navigations, whichever kind of step they differ in, and one
signature to two describing the same navigation.
"""

import pytest

from krrood.entity_query_language.factories import flat_variable, variable
from krrood.entity_query_language.verbalization.context import MicroplanningServices
from krrood.entity_query_language.verbalization.engine import root_context
from krrood.entity_query_language.verbalization.grammar.framework.registry import RULES
from krrood.entity_query_language.verbalization.grammar.query.assembler import (
    QueryAssembler,
)

from ...dataset.semantic_world_like_classes import Body, Cabinet, World


@pytest.fixture
def signature():
    """
    :return: The structural signature of an expression, taken from a query assembler
        built on the standard grammar.
    """
    world = variable(World, domain=[])
    assembler = QueryAssembler(
        root_context(MicroplanningServices.from_expression(world), RULES)
    )
    return assembler._expression_signature


# %% chains that differ in one step


def test_chains_differing_in_the_key_they_are_indexed_by_differ(signature):
    """
    An index step names the element it reaches, so two chains through different elements
    of one collection are different navigations.
    """
    world = variable(World, domain=[])

    assert signature(world.bodies[0].name) != signature(world.bodies[1].name)


def test_chains_differing_in_the_expression_they_are_indexed_by_differ(signature):
    world = variable(World, domain=[])
    first_position, second_position = variable(int, domain=[]), variable(int, domain=[])

    assert signature(world.bodies[first_position].name) != signature(
        world.bodies[second_position].name
    )


def test_chains_differing_in_the_arguments_a_call_is_made_with_differ(signature):
    name = variable(Body, domain=[]).name

    assert signature(name.startswith("Handle")) != signature(name.startswith("Drawer"))


def test_chains_through_two_flattenings_of_one_attribute_differ(signature):
    """
    Each flattening is an iteration of its own, so two of them range over the drawers
    independently and the chains taken from them are different navigations.
    """
    cabinet = variable(Cabinet, domain=[])

    assert signature(flat_variable(cabinet.drawers).handle.name) != signature(
        flat_variable(cabinet.drawers).handle.name
    )


# %% chains that describe the same navigation


def test_two_statements_of_one_chain_match(signature):
    """
    The signature is structural rather than identity-based, so a chain restated
    somewhere else in the query still matches the one it repeats.
    """
    world = variable(World, domain=[])

    assert signature(world.bodies[0].name) == signature(world.bodies[0].name)


def test_chains_differing_in_the_attribute_they_read_differ(signature):
    """
    The attribute case the signature already told apart is unchanged.
    """
    body = variable(Body, domain=[])

    assert signature(body.name) != signature(body.size)
