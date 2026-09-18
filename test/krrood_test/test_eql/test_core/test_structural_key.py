"""
The key each mapping states for itself, ``MappedVariable._structural_key_``.

It is what tells one step of a mapping chain apart from another step of its kind. Every
kind of mapping is written with its own arguments, so every kind states its own key;
these pin one kind per test, and the identity a flattening carries in place of
arguments.
"""

from krrood.entity_query_language.core.mapped_variable import (
    Attribute,
    Call,
    FlatVariable,
    IndexByExpression,
    IndexByValue,
)
from krrood.entity_query_language.factories import flat_variable, variable

from ...dataset.semantic_world_like_classes import Body, Cabinet, World

# %% the arguments each kind of step is written with


def test_an_attribute_names_the_field_and_the_class_it_is_read_from():
    world = variable(World, domain=[])

    assert world.bodies._structural_key_ == (Attribute, "bodies", World)


def test_an_index_by_a_value_names_the_key_its_element_is_stored_under():
    world = variable(World, domain=[])

    assert world.bodies[1]._structural_key_ == (IndexByValue, 1)


def test_an_index_by_an_expression_identifies_that_expression():
    """
    The key is an expression, and comparing two expressions builds a comparison rather
    than answering whether they are the same, so the key names it by identifier.
    """
    world = variable(World, domain=[])
    position = variable(int, domain=[])

    assert world.bodies[position]._structural_key_ == (
        IndexByExpression,
        position._id_,
    )


def test_a_call_names_the_arguments_it_is_called_with():
    name = variable(Body, domain=[]).name

    assert name.format(prefix="Handle")._structural_key_ == (
        Call,
        (),
        (("prefix", "Handle"),),
    )


def test_a_flattening_is_named_by_its_own_identity():
    """
    A flattening takes no argument naming which element it means -- the iteration
    belongs to the node itself -- so the node is what its key names.
    """
    cabinet = variable(Cabinet, domain=[])
    flattening = flat_variable(cabinet.drawers)

    assert flattening._structural_key_ == (FlatVariable, flattening._id_)


# %% two steps of one kind


def test_two_indexings_by_different_keys_are_different_steps():
    world = variable(World, domain=[])

    assert world.bodies[0]._structural_key_ != world.bodies[1]._structural_key_


def test_two_flattenings_of_one_attribute_are_different_steps():
    """
    Each flattening written ranges over the elements on its own, which is the identity
    :meth:`FlatVariable._rebuild_on_` keeps by constructing outside the mapped-variable
    cache.
    """
    cabinet = variable(Cabinet, domain=[])

    assert (
        flat_variable(cabinet.drawers)._structural_key_
        != flat_variable(cabinet.drawers)._structural_key_
    )
