from dataclasses import dataclass

from semantic_digital_twin.datastructures.prefixed_name import PrefixedName
from semantic_digital_twin.world_description.effects import (
    Effect,
    MonotoneDecreasingEffect,
    MonotoneIncreasingEffect,
)

# %% test target


@dataclass
class MeasuredObject:
    """
    Effect target whose observed property value is set directly by the test.
    """

    name: PrefixedName
    """Name of the object."""

    measured_value: float
    """
    The property value the effect reads.
    """


def build_effect(effect_class: type, current_value: float) -> Effect:
    """
    Build an effect of the given class around a goal of 1.0 and tolerance 0.25.
    """
    return effect_class(
        target_object=MeasuredObject(
            name=PrefixedName("cup"), measured_value=current_value
        ),
        property_getter=lambda target: target.measured_value,
        goal_value=1.0,
        tolerance=0.25,
    )


# %% symmetric tolerance boundaries


def test_is_achieved_at_upper_tolerance_boundary():
    assert build_effect(Effect, 1.25).is_achieved() is True


def test_is_achieved_at_lower_tolerance_boundary():
    assert build_effect(Effect, 0.75).is_achieved() is True


def test_is_not_achieved_beyond_upper_tolerance():
    assert build_effect(Effect, 1.3).is_achieved() is False


def test_is_not_achieved_beyond_lower_tolerance():
    assert build_effect(Effect, 0.7).is_achieved() is False


# %% monotone increasing


def test_monotone_increasing_is_achieved_at_lower_boundary():
    assert build_effect(MonotoneIncreasingEffect, 0.75).is_achieved() is True


def test_monotone_increasing_is_not_achieved_below_lower_boundary():
    assert build_effect(MonotoneIncreasingEffect, 0.7).is_achieved() is False


def test_monotone_increasing_is_achieved_on_overshoot():
    """
    An overshoot past the goal counts as achieved for an increasing effect, unlike the
    symmetric base effect.
    """
    assert build_effect(MonotoneIncreasingEffect, 1.3).is_achieved() is True
    assert build_effect(Effect, 1.3).is_achieved() is False


# %% monotone decreasing


def test_monotone_decreasing_is_achieved_at_upper_boundary():
    assert build_effect(MonotoneDecreasingEffect, 1.25).is_achieved() is True


def test_monotone_decreasing_is_not_achieved_above_upper_boundary():
    assert build_effect(MonotoneDecreasingEffect, 1.3).is_achieved() is False


def test_monotone_decreasing_is_achieved_on_undershoot():
    """
    An undershoot past the goal counts as achieved for a decreasing effect, unlike the
    symmetric base effect.
    """
    assert build_effect(MonotoneDecreasingEffect, 0.7).is_achieved() is True
    assert build_effect(Effect, 0.7).is_achieved() is False
