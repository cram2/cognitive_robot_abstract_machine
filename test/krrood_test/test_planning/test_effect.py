"""
Tests for the effects in :mod:`krrood.planning.effect`: what each changes in a state,
under which bindings, and which changes they reject.
"""

from __future__ import annotations

import pytest

from krrood.entity_query_language.factories import a, literal, variable
from krrood.entity_query_language.ground_term import AttributeTerm
from krrood.planning.effect import (
    Assignment,
    CollectionInsertion,
    ElementInsertion,
    ObjectRemoval,
)
from krrood.planning.exceptions import AssignedValueHasWrongType, ValueIsNotACollection
from krrood.planning.state import State

from ..dataset.stacking_blocks import StackingBlock, Tray

# %% conditional effects


def test_conditional_effect_changes_only_the_bindings_its_condition_holds_for(
    stacking_world,
):
    """
    Only the blocks on the table are cleared, which leaves ``c`` as it was.
    """
    block = variable(StackingBlock)
    state = State().apply_effects(
        Assignment(
            fact=block.clear == True, condition=block.support == stacking_world.table
        )
    )
    assert a(StackingBlock).clear.tolist(state=state) == [True, True, True]


def test_conditional_effect_whose_condition_fails_changes_nothing(stacking_world):
    """
    No block satisfies the condition, so nothing is assigned.
    """
    block = variable(StackingBlock)
    effect = Assignment(fact=block.clear == False, condition=block.name == "none")
    assert State().apply_effects(effect) == State()


def test_effect_value_depends_on_the_binding(stacking_world):
    """
    Each block takes the name of its own support, read before any is renamed.
    """
    block = variable(StackingBlock)
    state = State().apply_effects(Assignment(fact=block.name == block.support.name))
    assert a(StackingBlock).name.tolist(state=state) == ["table", "table", "a"]


def test_conditional_removal_removes_only_the_matching_objects(stacking_world):
    """
    The blocks on the table are removed, and the block on ``a`` stays.
    """
    block = variable(StackingBlock)
    state = State().apply_effects(
        ObjectRemoval(objects=block, condition=block.support == stacking_world.table)
    )
    assert a(StackingBlock).name.tolist(state=state) == ["c"]


# %% rejected changes


def test_assigning_a_value_of_the_wrong_type_is_rejected(stacking_world):
    """
    A support is a surface, never a string.
    """
    with pytest.raises(AssignedValueHasWrongType):
        State().apply_effects(
            Assignment(fact=literal(stacking_world.b).support == "the floor")
        )


def test_inserting_into_a_value_that_is_not_a_collection_is_rejected(stacking_world):
    """
    A name is a string, which holds no elements to insert next to.
    """
    with pytest.raises(ValueIsNotACollection):
        State().apply_effects(
            ElementInsertion(
                collection=literal(stacking_world.tray).name, element=literal("x")
            )
        )


# %% insertions


def test_insertion_is_recorded_as_the_inserted_elements(stacking_world):
    """
    The state keeps what was inserted into the tray, not a copy of its whole list.
    """
    state = State().apply_effects(
        ElementInsertion(
            collection=literal(stacking_world.tray).blocks,
            element=literal(stacking_world.b),
        )
    )
    tray_blocks = AttributeTerm(instance=stacking_world.tray, attribute_name="blocks")
    assert (state.assignments, state.insertions) == (
        {},
        {tray_blocks: CollectionInsertion(elements=(stacking_world.b,))},
    )


def test_assignment_replaces_what_was_inserted_before(stacking_world):
    """
    Assigning the collection drops the earlier insertion.
    """
    state = State().apply_effects(
        ElementInsertion(
            collection=literal(stacking_world.tray).blocks,
            element=literal(stacking_world.b),
        )
    )
    state = state.add_facts(literal(stacking_world.tray).blocks == [stacking_world.c])
    assert a(Tray).blocks.tolist(state=state) == [[stacking_world.c]]


@pytest.mark.parametrize(
    "collection, inserted",
    [
        ([1], [1, 2]),
        ((1,), (1, 2)),
        ({1}, {1, 2}),
        (frozenset({1}), frozenset({1, 2})),
    ],
)
def test_insertion_keeps_the_type_of_the_collection(collection, inserted):
    """
    Lists and sets take the element through their own method, and the immutable
    collections are rebuilt with it; the original is left unchanged.
    """
    original = type(collection)(collection)
    result = CollectionInsertion(elements=(2,)).insert_into(collection=collection)
    assert (result, type(result), collection) == (inserted, type(inserted), original)
