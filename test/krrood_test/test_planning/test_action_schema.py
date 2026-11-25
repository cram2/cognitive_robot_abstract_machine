"""
Tests for :class:`~krrood.planning.action_schema.ActionSchema`: grounding a schema in a
state, checking a ground action's precondition, and applying its effects to a state.
"""

from __future__ import annotations

import pytest

from krrood.entity_query_language.factories import a, evaluate_condition, literal
from krrood.entity_query_language.ground_term import AttributeTerm
from krrood.entity_query_language.query.query import Entity
from krrood.planning.exceptions import ActionIsNotApplicable
from krrood.planning.fact import Fact
from krrood.planning.state import State

from ..dataset.stacking_blocks import (
    BuildBlock,
    Discard,
    MoveToTable,
    PlannedBlock,
    PutOnTray,
    Stack,
    StackingBlock,
    StackingWorld,
    Tray,
)

# %% helpers


def stacked_pairs(actions: list[Stack]) -> set[tuple[str, str]]:
    """
    :param actions: Ground stack actions.
    :return: The names of the block and the target of each action.
    """
    return {(action.block.name, action.target.name) for action in actions}


def is_clear(*, block: StackingBlock, state: State) -> bool:
    """
    :param block: The block to read.
    :param state: The state to read it in.
    :return: Whether *block* is clear in *state*.
    """
    return literal(block).clear.first(state=state)


def support_in(*, block: StackingBlock, state: State) -> StackingBlock:
    """
    :param block: The block to read.
    :param state: The state to read it in.
    :return: What *block* rests on in *state*.
    """
    return literal(block).support.first(state=state)


def discardable_blocks(*, state: State) -> list[StackingBlock]:
    """
    :param state: The state the actions are grounded in.
    :return: The block of every applicable :class:`Discard`, in order.
    """
    return [action.block for action in Discard.applicable_actions().tolist(state=state)]


# %% grounding a schema


def test_applicable_actions_is_a_query(stacking_world):
    """
    Grounding builds a query and leaves its evaluation to the caller.
    """
    assert isinstance(Stack.applicable_actions(), Entity)


def test_applicable_actions_bind_every_parameter_that_satisfies_the_precondition(
    stacking_world,
):
    """
    Only the clear blocks ``b`` and ``c`` can be stacked, each onto the other.
    """
    actions = Stack.applicable_actions().tolist(state=State())
    assert stacked_pairs(actions) == {("b", "c"), ("c", "b")}


def test_applicable_actions_are_evaluated_lazily(stacking_world):
    """
    The first ground action is available before the others are computed.
    """
    first = next(Stack.applicable_actions().evaluate(state=State()))
    assert (first.block.name, first.target.name) in {("b", "c"), ("c", "b")}


def test_applicable_actions_read_the_state(stacking_world):
    """
    Once ``b`` is on ``c``, only ``b`` is clear, and a block cannot be stacked onto
    itself.
    """
    state = State().apply_action(
        action=Stack(block=stacking_world.b, target=stacking_world.c)
    )
    assert stacked_pairs(Stack.applicable_actions().tolist(state=state)) == set()


def test_applicable_actions_are_ground_actions_of_the_schema(stacking_world):
    """
    Grounding builds instances of the schema whose fields hold the bound objects.
    """
    actions = MoveToTable.applicable_actions().tolist(state=State())
    assert actions == [MoveToTable(block=stacking_world.c, table=stacking_world.table)]


def test_parameters_range_over_objects_the_state_adds(stacking_world):
    """
    Before, only ``b`` and ``c`` can be discarded; once a block is built on ``b``, the
    built block and ``c`` can.
    """
    before = discardable_blocks(state=State())
    state = State().apply_action(action=BuildBlock(base=stacking_world.b))
    (built,) = state.added_objects.values()
    assert (before, built.name, built.support, discardable_blocks(state=state)) == (
        [stacking_world.b, stacking_world.c],
        stacking_world.b.name,
        stacking_world.b,
        [stacking_world.c, built],
    )


def test_parameters_never_range_over_objects_the_state_removes(stacking_world):
    """
    A block a state discards is never a parameter again.
    """
    state = State().apply_action(action=Discard(block=stacking_world.c))
    assert stacking_world.c not in discardable_blocks(state=state)


# %% checking a ground action


def test_ground_action_whose_precondition_holds_is_applicable(stacking_world):
    """
    Both blocks are clear and distinct.
    """
    action = Stack(block=stacking_world.b, target=stacking_world.c)
    assert evaluate_condition(action.to_symbolic().precondition(), state=State())


def test_ground_action_whose_precondition_fails_is_not_applicable(stacking_world):
    """
    Block ``a`` carries ``c``, so it is not clear.
    """
    action = Stack(block=stacking_world.a, target=stacking_world.b)
    assert not evaluate_condition(action.to_symbolic().precondition(), state=State())


def test_applying_an_action_that_is_not_applicable_is_rejected(stacking_world):
    """
    A ground action is never applied where its precondition fails.
    """
    with pytest.raises(ActionIsNotApplicable):
        State().apply_action(
            action=Stack(block=stacking_world.a, target=stacking_world.b)
        )


def test_rejected_action_names_the_conditions_that_fail(stacking_world):
    """
    Only the block's clearness fails: the target is clear and differs from it.
    """
    action = Stack(block=stacking_world.a, target=stacking_world.b)
    with pytest.raises(ActionIsNotApplicable) as raised:
        State().apply_action(action=action)
    (failed,) = raised.value.failed_conditions
    fact = Fact.from_condition(condition=failed)
    assert (fact.ground_term, fact.value) == (
        AttributeTerm(instance=stacking_world.a, attribute_name="clear"),
        True,
    )


def test_rejected_action_names_the_conditions_that_fail_in_the_state(stacking_world):
    """
    Block ``b`` is clear on the live objects, but the state makes it carry something.
    """
    state = State().add_facts(literal(stacking_world.b).clear == False)
    with pytest.raises(ActionIsNotApplicable) as raised:
        state.apply_action(
            action=Stack(block=stacking_world.b, target=stacking_world.c)
        )
    (failed,) = raised.value.failed_conditions
    assert Fact.from_condition(condition=failed).ground_term == AttributeTerm(
        instance=stacking_world.b, attribute_name="clear"
    )


# %% applying effects


def test_applying_an_action_assigns_its_effects(stacking_world):
    """
    Stacking ``b`` onto ``c`` moves only ``b``.
    """
    state = State().apply_action(
        action=Stack(block=stacking_world.b, target=stacking_world.c)
    )
    assert support_in(block=stacking_world.b, state=state) is stacking_world.c
    assert a(StackingBlock).support.tolist(state=state) == [
        stacking_world.table,
        stacking_world.c,
        stacking_world.a,
    ]


def test_effects_read_the_state_the_action_is_applied_in(stacking_world):
    """
    Moving ``c`` away clears ``a``, the block it rested on before the move.
    """
    before = is_clear(block=stacking_world.a, state=State())
    state = State().apply_action(
        action=MoveToTable(block=stacking_world.c, table=stacking_world.table)
    )
    assert (before, is_clear(block=stacking_world.a, state=state)) == (False, True)
    assert a(StackingBlock).clear.tolist(state=state) == [True, True, True]


def test_every_effect_reads_the_state_before_the_action(stacking_world):
    """
    The effect that clears the old support reads it before another effect moves the
    block, and the target ``b`` stops being clear.
    """
    before = is_clear(block=stacking_world.b, state=State())
    state = State().apply_action(
        action=Stack(block=stacking_world.c, target=stacking_world.b)
    )
    assert (before, is_clear(block=stacking_world.b, state=state)) == (True, False)
    assert a(StackingBlock).clear.tolist(state=state) == [True, False, True]


def test_applying_an_action_leaves_the_state_and_live_objects_unchanged(
    stacking_world,
):
    """
    Applying returns a new state.
    """
    state = State()
    state.apply_action(action=Stack(block=stacking_world.b, target=stacking_world.c))
    assert (state, stacking_world.b.support) == (State(), stacking_world.table)


def test_successive_actions_build_on_each_other(stacking_world):
    """
    The second action reads the state the first one leaves.
    """
    state = State().apply_action(
        action=MoveToTable(block=stacking_world.c, table=stacking_world.table)
    )
    state = state.apply_action(
        action=Stack(block=stacking_world.a, target=stacking_world.b)
    )
    assert a(StackingBlock).support.tolist(state=state) == [
        stacking_world.b,
        stacking_world.table,
        stacking_world.table,
    ]


def test_insertion_effect_appends_to_the_collection(stacking_world):
    """
    Each insertion appends to the elements the state already gives the tray.
    """
    state = State().apply_action(
        action=PutOnTray(block=stacking_world.b, tray=stacking_world.tray)
    )
    state = state.apply_action(
        action=PutOnTray(block=stacking_world.c, tray=stacking_world.tray)
    )
    assert a(Tray).blocks.tolist(state=state) == [[stacking_world.b, stacking_world.c]]


def test_creation_effect_adds_the_objects_it_builds(stacking_world):
    """
    The built block is built from the bound parameters.
    """
    state = State().apply_action(action=BuildBlock(base=stacking_world.b))
    (built,) = state.added_objects.values()
    assert (type(built), built.name, built.support) == (
        PlannedBlock,
        stacking_world.b.name,
        stacking_world.b,
    )


def test_created_objects_stay_out_of_the_live_symbol_graph(
    stacking_world: StackingWorld,
):
    """
    Outside the state that builds it, the built block does not exist.
    """
    State().apply_action(action=BuildBlock(base=stacking_world.b))
    assert a(StackingBlock).tolist() == [
        stacking_world.a,
        stacking_world.b,
        stacking_world.c,
    ]


def test_removal_effect_removes_the_object(stacking_world):
    """
    The discarded block leaves the blocks the state has.
    """
    state = State().apply_action(action=Discard(block=stacking_world.c))
    assert a(StackingBlock).tolist(state=state) == [stacking_world.a, stacking_world.b]
