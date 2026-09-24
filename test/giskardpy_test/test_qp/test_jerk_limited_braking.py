import math
from itertools import product

import pytest

from giskardpy.qp.jerk_limited_braking import (
    BRAKING_RELATIVE_TOLERANCE,
    JerkLimitedBraking,
)

TIME_STEPS = [1 / 20, 1 / 25, 1 / 50, 1 / 80, 1 / 100]
VELOCITY_LIMITS = [0.013, 0.2, 1.0, 2.5]
JERK_LIMITS = [1.0, 44.4, 1111.0]

# %% removable velocity


@pytest.mark.parametrize("number_of_steps", range(41))
def test_removable_velocity_is_a_triangular_acceleration_ramp(number_of_steps):
    """
    Braking in m steps removes the most velocity when the acceleration climbs by one
    jerk step per time step and falls back to zero by the end, i.e. follows a tent.
    """
    tent = sum(
        min(step, number_of_steps + 1 - step) for step in range(1, number_of_steps + 1)
    )

    assert JerkLimitedBraking._removable_velocity_in_jerk_steps(number_of_steps) == tent


# %% number of braking steps


@pytest.mark.parametrize(
    "velocity_limit, jerk_limit, time_step",
    list(product(VELOCITY_LIMITS, JERK_LIMITS, TIME_STEPS)),
)
def test_number_of_steps_is_the_fewest_that_remove_the_velocity(
    velocity_limit, jerk_limit, time_step
):
    braking = JerkLimitedBraking(
        velocity_limit=velocity_limit, jerk_limit=jerk_limit, time_step=time_step
    )
    jerk_step = jerk_limit * time_step**2

    removable = JerkLimitedBraking._removable_velocity_in_jerk_steps(
        braking.number_of_steps
    )
    removable_with_one_step_less = JerkLimitedBraking._removable_velocity_in_jerk_steps(
        braking.number_of_steps - 1
    )

    assert removable * jerk_step >= velocity_limit * (1 - BRAKING_RELATIVE_TOLERANCE)
    assert removable_with_one_step_less * jerk_step < velocity_limit


@pytest.mark.parametrize("number_of_steps", [1, 2, 5, 8, 13, 28, 29])
@pytest.mark.parametrize("time_step", TIME_STEPS)
def test_braking_that_exactly_fills_its_steps_needs_no_extra_step(
    number_of_steps, time_step
):
    """
    A jerk limit that removes the velocity in exactly m steps is not pushed to m + 1 by
    rounding.
    """
    velocity_limit = 1.0
    jerk_limit = velocity_limit / (
        time_step**2
        * JerkLimitedBraking._removable_velocity_in_jerk_steps(number_of_steps)
    )
    braking = JerkLimitedBraking(
        velocity_limit=velocity_limit, jerk_limit=jerk_limit, time_step=time_step
    )

    assert braking.number_of_steps == number_of_steps


# %% braking time


@pytest.mark.parametrize("velocity_limit", VELOCITY_LIMITS)
def test_jerk_limit_from_braking_time_does_not_depend_on_the_time_step(
    velocity_limit,
):
    braking_time = 0.3
    jerk_limits = {
        JerkLimitedBraking.from_braking_time(
            velocity_limit=velocity_limit,
            braking_time=braking_time,
            time_step=time_step,
        ).jerk_limit
        for time_step in TIME_STEPS
    }

    assert jerk_limits == {4 * velocity_limit / braking_time**2}


@pytest.mark.parametrize("braking_time", [0.05, 0.3, 0.36, 1.16])
@pytest.mark.parametrize("time_step", TIME_STEPS)
def test_every_velocity_limit_needs_the_same_steps_for_one_braking_time(
    braking_time, time_step
):
    """
    The jerk limit derived from a braking time scales with the velocity limit, so all
    degrees of freedom brake in the same number of steps.
    """
    expected = JerkLimitedBraking.number_of_steps_for_braking_time(
        braking_time=braking_time, time_step=time_step
    )
    for velocity_limit in VELOCITY_LIMITS:
        braking = JerkLimitedBraking.from_braking_time(
            velocity_limit=velocity_limit,
            braking_time=braking_time,
            time_step=time_step,
        )
        assert braking.number_of_steps == expected


@pytest.mark.parametrize("number_of_steps", [2, 5, 8, 28])
@pytest.mark.parametrize("time_step", TIME_STEPS)
def test_braking_time_of_a_whole_number_of_steps_needs_exactly_those_steps(
    number_of_steps, time_step
):
    """
    The braking time whose jerk limit removes the velocity in exactly m steps, which is
    what a horizon of m + 2 steps used to derive, needs exactly m steps.
    """
    braking_time = (
        2
        * time_step
        * math.sqrt(
            JerkLimitedBraking._removable_velocity_in_jerk_steps(number_of_steps)
        )
    )

    assert (
        JerkLimitedBraking.number_of_steps_for_braking_time(
            braking_time=braking_time, time_step=time_step
        )
        == number_of_steps
    )


# %% acceleration limit


def _peak_acceleration_of_a_reversal(braking: JerkLimitedBraking) -> float:
    """
    Largest acceleration of a jerk-limited change from one velocity limit to the other,
    starting and ending with zero acceleration.

    :param braking: Braking whose velocity and jerk limit the reversal uses.
    :return: Largest acceleration magnitude of the reversal.
    """
    return math.sqrt(2 * braking.velocity_limit * braking.jerk_limit)


@pytest.mark.parametrize(
    "velocity_limit, time_step", list(product(VELOCITY_LIMITS, TIME_STEPS))
)
def test_acceleration_limit_lowers_a_jerk_limit_that_would_exceed_it(
    velocity_limit, time_step
):
    braking = JerkLimitedBraking(
        velocity_limit=velocity_limit, jerk_limit=max(JERK_LIMITS), time_step=time_step
    )
    acceleration_limit = _peak_acceleration_of_a_reversal(braking) / 2

    limited = braking.limited_to_acceleration(acceleration_limit)

    assert _peak_acceleration_of_a_reversal(limited) == pytest.approx(
        acceleration_limit
    )
    assert limited.velocity_limit == braking.velocity_limit
    assert limited.time_step == braking.time_step


def test_acceleration_limit_keeps_a_jerk_limit_that_stays_within_it():
    braking = JerkLimitedBraking(
        velocity_limit=max(VELOCITY_LIMITS),
        jerk_limit=min(JERK_LIMITS),
        time_step=max(TIME_STEPS),
    )
    acceleration_limit = _peak_acceleration_of_a_reversal(braking) * 2

    assert braking.limited_to_acceleration(acceleration_limit) == braking
