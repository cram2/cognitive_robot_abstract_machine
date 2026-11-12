"""
Direct unit tests for :class:`giskardpy.qp.pos_in_vel_limits.BrakingProfile`.
"""

from datetime import timedelta
from itertools import product

import numpy as np
import pytest

import krrood.symbolic_math.symbolic_math as sm
from giskardpy.qp.jerk_limited_braking import JerkLimitedBraking
from giskardpy.qp.pos_in_vel_limits import BrakingProfile

DELTA_TIME = timedelta(seconds=0.05)
RESIDUE = 1e-10


def _shifted_velocities(velocity_profile: np.ndarray) -> np.ndarray:
    """
    Evaluates the shifted velocity profile for a braking profile that has already
    covered one time step worth of its own distance.
    """
    braking_profile = BrakingProfile(velocity=velocity_profile, time_step=DELTA_TIME)
    return braking_profile.shifted_by(
        distance=sm.Scalar(DELTA_TIME.total_seconds())
    ).evaluate()


# %% residue in the braking profile tail


def test_shifted_velocity_profile_treats_negligible_velocities_as_rest():
    """
    A braking profile whose tail is solver residue rather than exact zero must shift the
    same way as one that ends at rest.
    """
    with_residue = _shifted_velocities(np.array([1.0, 0.5, RESIDUE, RESIDUE]))
    at_rest = _shifted_velocities(np.array([1.0, 0.5, 0.0, 0.0]))

    np.testing.assert_array_equal(with_residue, at_rest)


def test_shifted_velocity_profile_already_treats_negative_residue_as_rest():
    """
    Negative residue was already clamped, which is why only positive residue reached the
    velocity bounds.
    """
    with_residue = _shifted_velocities(np.array([1.0, 0.5, -RESIDUE, -RESIDUE]))
    at_rest = _shifted_velocities(np.array([1.0, 0.5, 0.0, 0.0]))

    np.testing.assert_array_equal(with_residue, at_rest)


# %% distance covered by the shifted profile


def test_shifted_velocity_profile_covers_exactly_the_remaining_distance():
    """
    Between the distances that whole tails of the braking profile cover, the shifted
    profile still covers the remaining distance exactly, so a degree of freedom braking
    along it comes to rest on the position limit instead of short of it.
    """
    braking_profile = BrakingProfile.fastest(
        braking=JerkLimitedBraking.from_braking_time(
            velocity_limit=1.0,
            braking_time=timedelta(seconds=0.3),
            time_step=DELTA_TIME,
        ),
        prediction_horizon=7,
    )
    time_step = DELTA_TIME.total_seconds()
    braking_distance = time_step * np.sum(braking_profile.velocity)

    for distance in np.linspace(0, braking_distance, 41):
        shifted = braking_profile.shifted_by(distance=sm.Scalar(distance)).evaluate()

        assert time_step * np.sum(shifted) == pytest.approx(distance)


# %% zero_negligible_velocities
NEGLIGIBLE_VELOCITY = 1e-4


def test_zero_negligible_velocities_clears_small_positive_velocities():
    cleared = BrakingProfile.zero_negligible_velocities(
        np.array([NEGLIGIBLE_VELOCITY / 2])
    )

    np.testing.assert_array_equal(cleared, np.array([0.0]))


def test_zero_negligible_velocities_keeps_velocities_at_the_threshold():
    kept = BrakingProfile.zero_negligible_velocities(np.array([NEGLIGIBLE_VELOCITY]))

    np.testing.assert_array_equal(kept, np.array([NEGLIGIBLE_VELOCITY]))


def test_zero_negligible_velocities_clears_negative_velocities():
    cleared = BrakingProfile.zero_negligible_velocities(np.array([-0.5]))

    np.testing.assert_array_equal(cleared, np.array([0.0]))


def test_zero_negligible_velocities_leaves_the_input_unchanged():
    """
    The braking profile is a view into a memoized array, so clearing must not write
    through to it.
    """
    velocity_profile = np.array([1.0, NEGLIGIBLE_VELOCITY / 2])

    BrakingProfile.zero_negligible_velocities(velocity_profile)

    np.testing.assert_array_equal(
        velocity_profile, np.array([1.0, NEGLIGIBLE_VELOCITY / 2])
    )


# %% fastest braking
VELOCITY_LIMITS = [0.1, 1.0, 2.0]
ACCELERATION_LIMITS = [0.5, 2.0, 20.0]
BRAKING_TIMES = [timedelta(seconds=0.1), timedelta(seconds=0.3), timedelta(seconds=1)]


@pytest.mark.parametrize(
    "velocity_limit, acceleration_limit, braking_time",
    list(product(VELOCITY_LIMITS, ACCELERATION_LIMITS, BRAKING_TIMES)),
)
def test_fastest_braking_stays_within_the_acceleration_limit_of_its_braking(
    velocity_limit, acceleration_limit, braking_time
):
    """
    A braking limited to an acceleration keeps that acceleration without the profile
    bounding it separately.
    """
    braking = JerkLimitedBraking.from_braking_time(
        velocity_limit=velocity_limit, braking_time=braking_time, time_step=DELTA_TIME
    ).limited_to_acceleration(acceleration_limit)

    profile = BrakingProfile.fastest(
        braking=braking, prediction_horizon=braking.number_of_steps + 2
    )

    accelerations = (
        np.diff(np.concatenate([[velocity_limit], profile.velocity]))
        / DELTA_TIME.total_seconds()
    )
    assert np.max(np.abs(accelerations)) <= acceleration_limit
