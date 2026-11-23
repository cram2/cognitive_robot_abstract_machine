"""
Tests for the effects in :mod:`krrood.planning.effect`: what each changes in a state,
under which bindings, and which changes they reject.
"""

from __future__ import annotations

import pytest

from krrood.entity_query_language.factories import a, literal, variable
from krrood.planning.effect import Assignment, Effect, ElementInsertion, ObjectRemoval
from krrood.planning.exceptions import AssignedValueHasWrongType, ValueIsNotACollection
from krrood.planning.state import State

from ..dataset.stacking_blocks import StackingBlock

# %% helpers


def apply(*effects: Effect) -> State:
    """
    :param effects: The effects to apply.
    :return: The empty state with the change of every effect applied, in order.
    """
    state = State()
    for effect in effects:
        state = state.apply_change(change=effect.change(state=state))
    return state


# %% conditional effects


def test_conditional_effect_changes_only_the_bindings_its_condition_holds_for(
    stacking_world,
):
    """
    Only the blocks on the table are cleared, which leaves ``c`` as it was.
    """
    block = variable(StackingBlock)
    state = apply(
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
    assert effect.change(state=State()).assignments == {}


def test_effect_value_depends_on_the_binding(stacking_world):
    """
    Each block takes the name of its own support, read before any is renamed.
    """
    block = variable(StackingBlock)
    state = apply(Assignment(fact=block.name == block.support.name))
    assert a(StackingBlock).name.tolist(state=state) == ["table", "table", "a"]


def test_conditional_removal_removes_only_the_matching_objects(stacking_world):
    """
    The blocks on the table are removed, and the block on ``a`` stays.
    """
    block = variable(StackingBlock)
    state = apply(
        ObjectRemoval(objects=block, condition=block.support == stacking_world.table)
    )
    assert a(StackingBlock).name.tolist(state=state) == ["c"]


# %% rejected changes


def test_assigning_a_value_of_the_wrong_type_is_rejected(stacking_world):
    """
    A support is a surface, never a string.
    """
    with pytest.raises(AssignedValueHasWrongType):
        Assignment(fact=literal(stacking_world.b).support == "the floor").change(
            state=State()
        )


def test_inserting_into_a_value_that_is_not_a_collection_is_rejected(stacking_world):
    """
    A name is a string, which holds no elements to insert next to.
    """
    with pytest.raises(ValueIsNotACollection):
        ElementInsertion(
            collection=literal(stacking_world.tray).name, element=literal("x")
        ).change(state=State())
