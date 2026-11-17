"""
Assertions that two orientations describe the same rotation.

A quaternion and its negation, or an axis-angle pair and its flipped counterpart, are
written differently but count as the same.
"""

from __future__ import annotations

import numpy as np
from typing_extensions import Any

from .semantic_digital_twin_test.test_spatial_types.reference_implementations import (
    shortest_angular_distance,
)

# %% numerical closeness


def is_almost_equal(*, actual: Any, desired: Any, decimal: int) -> bool:
    """
    :param actual: Numbers, or a constant symbolic expression, to check.
    :param desired: Numbers, or a constant symbolic expression, to check against.
    :param decimal: Number of decimal places that have to agree, as in
        :func:`numpy.testing.assert_almost_equal`.
    :return: Whether every element of ``actual`` agrees with the one in ``desired``.
    """
    difference = np.asarray(actual).astype(float) - np.asarray(desired).astype(float)
    return bool(np.all(np.abs(difference) < 1.5 * 10.0**-decimal))


def is_angle_almost_equal(*, actual: Any, desired: Any, decimal: int) -> bool:
    """
    :param actual: Angle to check, in radians.
    :param desired: Angle to check against, in radians.
    :param decimal: Number of decimal places that have to agree.
    :return: Whether both angles describe the same direction.
    """
    actual_angle = np.asarray(actual).astype(float).item()
    desired_angle = np.asarray(desired).astype(float).item()
    return is_almost_equal(
        actual=shortest_angular_distance(actual_angle, desired_angle),
        desired=0.0,
        decimal=decimal,
    )


# %% orientation assertions


def assert_same_orientation(
    actual_orientation: Any, desired_orientation: Any, decimal: int = 6
) -> None:
    """
    Asserts that two quaternions describe the same rotation, which a quaternion and its
    negation both do.

    :param actual_orientation: Quaternion to check, as x, y, z, w.
    :param desired_orientation: Quaternion to check against, as x, y, z, w.
    :param decimal: Number of decimal places that have to agree.
    """
    desired = np.asarray(desired_orientation).astype(float)
    assert is_almost_equal(
        actual=actual_orientation, desired=desired, decimal=decimal
    ) or is_almost_equal(
        actual=actual_orientation, desired=-desired, decimal=decimal
    ), f"{actual_orientation} and {desired_orientation} are different rotations"


def assert_same_axis_angle(
    actual_angle: Any,
    actual_axis: Any,
    expected_angle: Any,
    expected_axis: Any,
    decimal: int = 6,
) -> None:
    """
    Asserts that two axis-angle pairs describe the same rotation: either equal, or with
    the axis flipped and the angle mirrored, or both a rotation by zero about any axis.

    :param actual_angle: Angle to check, in radians.
    :param actual_axis: Rotation axis to check.
    :param expected_angle: Angle to check against, in radians.
    :param expected_axis: Rotation axis to check against.
    :param decimal: Number of decimal places that have to agree.
    """
    expected_axis = np.asarray(expected_axis).astype(float)
    actual_axis = np.asarray(actual_axis).astype(float)
    expected_angle = np.asarray(expected_angle).astype(float).item()
    is_equal = is_almost_equal(
        actual=actual_axis, desired=expected_axis, decimal=decimal
    ) and is_angle_almost_equal(
        actual=actual_angle, desired=expected_angle, decimal=decimal
    )
    is_flipped = is_almost_equal(
        actual=actual_axis, desired=-expected_axis, decimal=decimal
    ) and is_angle_almost_equal(
        actual=actual_angle, desired=-expected_angle, decimal=decimal
    )
    is_zero_rotation = (
        is_angle_almost_equal(actual=actual_angle, desired=0.0, decimal=decimal)
        and is_angle_almost_equal(actual=expected_angle, desired=0.0, decimal=decimal)
        and not np.any(np.isnan(actual_axis))
        and not np.any(np.isnan(expected_axis))
    )
    assert is_equal or is_flipped or is_zero_rotation, (
        f"angle {actual_angle} about {actual_axis} is a different rotation than "
        f"angle {expected_angle} about {expected_axis}"
    )
