from typing import Union, Iterable

import numpy as np
from numpy import pi

import krrood.symbolic_math.symbolic_math as sm
from .reference_implementations import shortest_angular_distance

all_expressions_float_np = Union[
    sm.SymbolicMathType, float, np.ndarray, Iterable[float], Iterable[Iterable[float]]
]


def compare_axis_angle(
    actual_angle: all_expressions_float_np,
    actual_axis: all_expressions_float_np,
    expected_angle: sm.ScalarData,
    expected_axis: all_expressions_float_np,
):
    try:
        np.allclose(actual_axis, expected_axis)
        np.allclose(shortest_angular_distance(actual_angle, expected_angle), 0.0)
    except AssertionError:
        try:
            np.allclose(actual_axis, -expected_axis)
            np.allclose(
                shortest_angular_distance(actual_angle, abs(expected_angle - 2 * pi)),
                0.0,
            )
        except AssertionError:
            np.allclose(shortest_angular_distance(actual_angle, 0), 0.0)
            np.allclose(shortest_angular_distance(0, expected_angle), 0.0)
            assert not np.any(np.isnan(actual_axis))
            assert not np.any(np.isnan(expected_axis))


def compare_orientations(
    actual_orientation: all_expressions_float_np,
    desired_orientation: all_expressions_float_np,
) -> None:
    try:
        assert np.allclose(actual_orientation, desired_orientation)
    except:
        assert np.allclose(actual_orientation, -desired_orientation)
