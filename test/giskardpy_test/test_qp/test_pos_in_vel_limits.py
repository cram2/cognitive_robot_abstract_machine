"""
Direct unit tests for :mod:`giskardpy.qp.pos_in_vel_limits`.
"""

import numpy as np

import krrood.symbolic_math.symbolic_math as sm
from giskardpy.qp.pos_in_vel_limits import (
    shifted_velocity_profile,
    zero_negligible_velocities,
)

DELTA_TIME = 0.05
RESIDUE = 1e-10


def _shifted_velocities(velocity_profile: np.ndarray) -> np.ndarray:
    """
    Evaluates the shifted velocity profile for a braking profile that has already
    covered one time step worth of its own distance.
    """
    velocity_bound, _ = shifted_velocity_profile(
        velocity_profile=velocity_profile,
        acceleration_profile=np.zeros_like(velocity_profile),
        distance=sm.Scalar(DELTA_TIME),
        delta_time=DELTA_TIME,
    )
    return velocity_bound.evaluate()


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


# %% zero_negligible_velocities
NEGLIGIBLE_VELOCITY = 1e-4


def test_zero_negligible_velocities_clears_small_positive_velocities():
    cleared = zero_negligible_velocities(np.array([NEGLIGIBLE_VELOCITY / 2]))

    np.testing.assert_array_equal(cleared, np.array([0.0]))


def test_zero_negligible_velocities_keeps_velocities_at_the_threshold():
    kept = zero_negligible_velocities(np.array([NEGLIGIBLE_VELOCITY]))

    np.testing.assert_array_equal(kept, np.array([NEGLIGIBLE_VELOCITY]))


def test_zero_negligible_velocities_clears_negative_velocities():
    cleared = zero_negligible_velocities(np.array([-0.5]))

    np.testing.assert_array_equal(cleared, np.array([0.0]))


def test_zero_negligible_velocities_leaves_the_input_unchanged():
    """
    The braking profile is a view into a memoized array, so clearing must not write
    through to it.
    """
    velocity_profile = np.array([1.0, NEGLIGIBLE_VELOCITY / 2])

    zero_negligible_velocities(velocity_profile)

    np.testing.assert_array_equal(
        velocity_profile, np.array([1.0, NEGLIGIBLE_VELOCITY / 2])
    )
