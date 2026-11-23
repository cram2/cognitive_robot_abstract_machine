"""
Tests for :class:`~krrood.planning.action_schema.ActionSchema`: grounding a schema in a
state, checking a ground action's precondition, and applying its effects to a state.
"""

from __future__ import annotations

import pytest

from krrood.entity_query_language.factories import a
from krrood.planning.exceptions import ActionIsNotApplicable
from krrood.planning.state import State

from ..dataset.stacking_blocks import (
    BuildBlock,
    Discard,
    MoveToTable,
    PlannedBlock,
    PutOnTray,
    Stack,
    StackingBlock,
    Tray,
)

# %% helpers


def stacked_pairs(actions: list[Stack]) -> set[tuple[str, str]]:
    """
    :param actions: Ground stack actions.
    :return: The names of the block and the target of each action.
    """
    return {(action.block.name, action.target.name) for action in actions}


# %% grounding a schema


def test_applicable_actions_bind_every_parameter_that_satisfies_the_precondition(
    stacking_world,
):
    """
    Only the clear blocks ``b`` and ``c`` can be stacked, each onto the other.
    """
    actions = Stack.applicable_actions(state=State())
    assert stacked_pairs(actions) == {("b", "c"), ("c", "b")}


def test_applicable_actions_read_the_state(stacking_world):
    """
    Once ``b`` is on ``c``, only ``b`` is clear, and a block cannot be stacked onto
    itself.
    """
    state = Stack(block=stacking_world.b, target=stacking_world.c).apply(state=State())
    assert stacked_pairs(Stack.applicable_actions(state=state)) == set()


def test_applicable_actions_are_ground_actions_of_the_schema(stacking_world):
    """
    Grounding builds instances of the schema whose fields hold the bound objects.
    """
    actions = MoveToTable.applicable_actions(state=State())
    assert actions == [MoveToTable(block=stacking_world.c, table=stacking_world.table)]


def test_parameters_range_over_objects_the_state_adds(stacking_world):
    """
    A block a state builds can be a parameter of a later action.
    """
    state = BuildBlock(base=stacking_world.b).apply(state=State())
    (built,) = state.added_objects.values()
    assert built in {action.block for action in Discard.applicable_actions(state=state)}


def test_parameters_never_range_over_objects_the_state_removes(stacking_world):
    """
    A block a state discards is never a parameter again.
    """
    state = Discard(block=stacking_world.c).apply(state=State())
    assert stacking_world.c not in {
        action.block for action in Discard.applicable_actions(state=state)
    }


# %% checking a ground action


def test_ground_action_whose_precondition_holds_is_applicable(stacking_world):
    """
    Both blocks are clear and distinct.
    """
    assert Stack(block=stacking_world.b, target=stacking_world.c).is_applicable(
        state=State()
    )


def test_ground_action_whose_precondition_fails_is_not_applicable(stacking_world):
    """
    Block ``a`` carries ``c``, so it is not clear.
    """
    assert not Stack(block=stacking_world.a, target=stacking_world.b).is_applicable(
        state=State()
    )


def test_applying_an_action_that_is_not_applicable_is_rejected(stacking_world):
    """
    A ground action is never applied where its precondition fails.
    """
    with pytest.raises(ActionIsNotApplicable):
        Stack(block=stacking_world.a, target=stacking_world.b).apply(state=State())


# %% applying effects


def test_applying_an_action_assigns_its_effects(stacking_world):
    """
    Stacking ``b`` onto ``c`` moves only ``b``.
    """
    state = Stack(block=stacking_world.b, target=stacking_world.c).apply(state=State())
    assert a(StackingBlock).support.tolist(state=state) == [
        stacking_world.table,
        stacking_world.c,
        stacking_world.a,
    ]


def test_effects_read_the_state_the_action_is_applied_in(stacking_world):
    """
    Moving ``c`` away clears the block it rested on before the move.
    """
    state = MoveToTable(block=stacking_world.c, table=stacking_world.table).apply(
        state=State()
    )
    assert a(StackingBlock).clear.tolist(state=state) == [True, True, True]


def test_every_effect_reads_the_state_before_the_action(stacking_world):
    """
    The effect that clears the old support reads it before another effect moves the
    block.
    """
    state = Stack(block=stacking_world.c, target=stacking_world.b).apply(state=State())
    assert a(StackingBlock).clear.tolist(state=state) == [True, False, True]


def test_applying_an_action_leaves_the_state_and_live_objects_unchanged(
    stacking_world,
):
    """
    Applying returns a new state.
    """
    state = State()
    Stack(block=stacking_world.b, target=stacking_world.c).apply(state=state)
    assert (state, stacking_world.b.support) == (State(), stacking_world.table)


def test_successive_actions_build_on_each_other(stacking_world):
    """
    The second action reads the state the first one leaves.
    """
    state = MoveToTable(block=stacking_world.c, table=stacking_world.table).apply(
        state=State()
    )
    state = Stack(block=stacking_world.a, target=stacking_world.b).apply(state=state)
    assert a(StackingBlock).support.tolist(state=state) == [
        stacking_world.b,
        stacking_world.table,
        stacking_world.table,
    ]


def test_insertion_effect_appends_to_the_collection(stacking_world):
    """
    Each insertion appends to the elements the state already gives the tray.
    """
    state = PutOnTray(block=stacking_world.b, tray=stacking_world.tray).apply(
        state=State()
    )
    state = PutOnTray(block=stacking_world.c, tray=stacking_world.tray).apply(
        state=state
    )
    assert a(Tray).blocks.tolist(state=state) == [[stacking_world.b, stacking_world.c]]


def test_creation_effect_adds_the_objects_it_builds(stacking_world):
    """
    The built block is built from the bound parameters.
    """
    state = BuildBlock(base=stacking_world.b).apply(state=State())
    (built,) = state.added_objects.values()
    assert (type(built), built.name, built.support) == (
        PlannedBlock,
        stacking_world.b.name,
        stacking_world.b,
    )


def test_removal_effect_removes_the_object(stacking_world):
    """
    The discarded block leaves the blocks the state has.
    """
    state = Discard(block=stacking_world.c).apply(state=State())
    assert a(StackingBlock).tolist(state=state) == [stacking_world.a, stacking_world.b]
