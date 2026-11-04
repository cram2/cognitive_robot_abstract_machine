"""
Tests for how the states of a statechart are presented.
"""

import pytest

from krrood.symbolic_math.symbolic_math import FloatVariable, Scalar
from cramph.data_types import (
    LifeCyclePredicate,
    LifeCycleValues,
    ObservationStateValues,
    TransitionKind,
)
from cramph.exceptions import TransitionHasNoOutcomeError
from semantic_digital_twin.world_description.geometry import Color

# %% every state has to be drawable


@pytest.mark.parametrize("life_cycle_state", list(LifeCycleValues))
def test_every_life_cycle_state_has_a_color_and_a_badge(life_cycle_state) -> None:
    """
    A visualization asks a state how it is drawn, so a state without an answer only
    shows up as a failure while drawing.
    """
    assert isinstance(life_cycle_state.color, Color)
    assert life_cycle_state.badge


@pytest.mark.parametrize("observation_state", list(ObservationStateValues))
def test_every_observation_state_has_a_color_and_a_badge(observation_state) -> None:
    assert isinstance(observation_state.color, Color)
    assert observation_state.badge


# %% states have to be told apart in a drawing


def test_life_cycle_states_are_drawn_differently_from_each_other() -> None:
    """
    Two states sharing a color or a symbol would be indistinguishable to whoever reads
    the drawing.
    """
    assert len({state.color for state in LifeCycleValues}) == len(LifeCycleValues)
    assert len({state.badge for state in LifeCycleValues}) == len(LifeCycleValues)


def test_observation_states_are_drawn_differently_from_each_other() -> None:
    assert len({state.color for state in ObservationStateValues}) == len(
        ObservationStateValues
    )
    assert len({state.badge for state in ObservationStateValues}) == len(
        ObservationStateValues
    )


# %% a state is still the number it stands for


def test_a_life_cycle_state_is_found_by_the_number_it_stands_for() -> None:
    """
    Every read of a life cycle state looks the member up by the number stored for it, so
    that number has to stay the member's value.
    """
    assert LifeCycleValues(int(LifeCycleValues.FAILED)) is LifeCycleValues.FAILED


def test_an_observation_state_is_found_by_the_number_it_stands_for() -> None:
    """
    The compiled updater writes the trinary constants, and reading a node's observation
    turns one back into the member it stands for.
    """
    assert (
        ObservationStateValues(float(Scalar.const_true()))
        is ObservationStateValues.TRUE
    )


# %% transition triggerability rules


@pytest.mark.parametrize(
    "transition_kind, expected_states",
    [
        (TransitionKind.START, frozenset({LifeCycleValues.NOT_STARTED})),
        (
            TransitionKind.PAUSE,
            frozenset({LifeCycleValues.RUNNING, LifeCycleValues.PAUSED}),
        ),
        (
            TransitionKind.SUCCEED,
            frozenset({LifeCycleValues.RUNNING, LifeCycleValues.PAUSED}),
        ),
        (
            TransitionKind.INTERRUPT,
            frozenset({LifeCycleValues.RUNNING, LifeCycleValues.PAUSED}),
        ),
        (
            TransitionKind.FAIL,
            frozenset({LifeCycleValues.RUNNING, LifeCycleValues.PAUSED}),
        ),
        (TransitionKind.RESET, frozenset(LifeCycleValues)),
    ],
)
def test_transition_kind_source_states(transition_kind, expected_states) -> None:
    """
    Each transition kind declares the exact lifecycle states from which it can legally
    trigger.
    """
    assert transition_kind.source_states == expected_states


@pytest.mark.parametrize("transition_kind", list(TransitionKind))
@pytest.mark.parametrize("life_cycle_state", list(LifeCycleValues))
def test_transition_kind_can_trigger_from(transition_kind, life_cycle_state) -> None:
    """
    The triggerability check matches membership in the transition kind's source states.
    """
    expected = life_cycle_state in transition_kind.source_states
    assert transition_kind.can_trigger_from(life_cycle_state) is expected


# %% the outcome an ending transition yields


@pytest.mark.parametrize(
    "transition_kind, expected_outcome",
    [
        (TransitionKind.SUCCEED, LifeCycleValues.SUCCEEDED),
        (TransitionKind.FAIL, LifeCycleValues.FAILED),
        (TransitionKind.INTERRUPT, LifeCycleValues.INTERRUPTED),
    ],
)
def test_an_ending_transition_yields_its_own_outcome(
    transition_kind: TransitionKind, expected_outcome: LifeCycleValues
) -> None:
    """
    An outcome is declared by the condition that ended a node, never read off what the
    node observed.
    """
    assert transition_kind.outcome is expected_outcome


@pytest.mark.parametrize(
    "transition_kind",
    [kind for kind in TransitionKind if kind not in TransitionKind.ending_kinds()],
)
def test_a_transition_that_does_not_end_a_node_has_no_outcome(
    transition_kind: TransitionKind,
) -> None:
    with pytest.raises(TransitionHasNoOutcomeError):
        transition_kind.outcome


# %% life cycle predicates read off an expression


@pytest.mark.parametrize("predicate", list(LifeCyclePredicate))
@pytest.mark.parametrize("life_cycle_state", list(LifeCycleValues))
def test_a_life_cycle_predicate_expression_matches_its_truth_table(
    predicate: LifeCyclePredicate, life_cycle_state: LifeCycleValues
) -> None:
    life_cycle = FloatVariable(name="life_cycle")

    expression = predicate.expression(life_cycle).substitute(
        [life_cycle], [Scalar(life_cycle_state)]
    )

    assert float(expression) == predicate.truth_value(life_cycle_state)
