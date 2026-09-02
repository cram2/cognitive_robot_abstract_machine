"""
Tests for how the states of a motion statechart are presented.
"""

import pytest

from krrood.symbolic_math.symbolic_math import Scalar
from giskardpy.motion_statechart.data_types import (
    LifeCycleValues,
    ObservationStateValues,
)
from semantic_digital_twin.world_description.geometry import Color

# %% every state has to be drawable


@pytest.mark.parametrize("life_cycle_state", list(LifeCycleValues))
def test_every_life_cycle_state_has_a_color_and_a_badge(life_cycle_state):
    """
    A visualization asks a state how it is drawn, so a state without an answer only
    shows up as a failure while drawing.
    """
    assert isinstance(life_cycle_state.color, Color)
    assert life_cycle_state.badge


@pytest.mark.parametrize("observation_state", list(ObservationStateValues))
def test_every_observation_state_has_a_color_and_a_badge(observation_state):
    assert isinstance(observation_state.color, Color)
    assert observation_state.badge


# %% states have to be told apart in a drawing


def test_life_cycle_states_are_drawn_differently_from_each_other():
    """
    Two states sharing a color or a symbol would be indistinguishable to whoever reads
    the drawing.
    """
    assert len({state.color for state in LifeCycleValues}) == len(LifeCycleValues)
    assert len({state.badge for state in LifeCycleValues}) == len(LifeCycleValues)


def test_observation_states_are_drawn_differently_from_each_other():
    assert len({state.color for state in ObservationStateValues}) == len(
        ObservationStateValues
    )
    assert len({state.badge for state in ObservationStateValues}) == len(
        ObservationStateValues
    )


# %% a state is still the number it stands for


def test_a_life_cycle_state_is_found_by_the_number_it_stands_for():
    """
    Every read of a life cycle state looks the member up by the number stored for it, so
    that number has to stay the member's value.
    """
    assert LifeCycleValues(int(LifeCycleValues.FAILED)) is LifeCycleValues.FAILED


def test_an_observation_state_is_found_by_the_number_it_stands_for():
    """
    The compiled updater writes the trinary constants, and reading a node's observation
    turns one back into the member it stands for.
    """
    assert (
        ObservationStateValues(float(Scalar.const_true()))
        is ObservationStateValues.TRUE
    )
