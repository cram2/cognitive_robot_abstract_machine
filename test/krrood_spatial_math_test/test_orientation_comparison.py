"""
The closeness checks behind the shared orientation assertions.
"""

from __future__ import annotations

import numpy as np

from krrood_spatial_math.testing.orientation_comparison import (
    is_almost_equal,
    is_angle_almost_equal,
)


# %% shapes


def test_a_scalar_is_not_almost_equal_to_a_vector() -> None:
    """
    A single number is not close to a vector of zeros, although it is close to each
    element.
    """
    assert not is_almost_equal(actual=0.0, desired=np.zeros(3), decimal=6)


def test_a_column_is_not_almost_equal_to_a_flat_vector() -> None:
    """
    A quaternion as a column differs from the same quaternion as a flat vector.
    """
    quaternion = np.array([0.0, 0.0, 0.0, 1.0])
    assert not is_almost_equal(
        actual=quaternion.reshape(4, 1), desired=quaternion, decimal=6
    )


def test_equal_vectors_are_almost_equal() -> None:
    """
    Two copies of one quaternion are close.
    """
    quaternion = np.array([0.0, 0.0, 0.0, 1.0])
    assert is_almost_equal(actual=quaternion.copy(), desired=quaternion, decimal=6)


# %% angles


def test_angles_a_full_turn_apart_are_almost_equal() -> None:
    """
    Angles that differ by a full turn describe the same direction.
    """
    assert is_angle_almost_equal(actual=0.5 + 2 * np.pi, desired=0.5, decimal=6)


def test_angles_on_either_side_of_pi_are_almost_equal() -> None:
    """
    Half a turn either way describes the same direction.
    """
    assert is_angle_almost_equal(actual=np.pi, desired=-np.pi, decimal=6)


def test_different_angles_are_not_almost_equal() -> None:
    """
    Angles that describe different directions are not close.
    """
    assert not is_angle_almost_equal(actual=0.5, desired=-0.5, decimal=6)
