import keyword
from typing import Union, Iterable

import hypothesis.strategies as st
import numpy as np
from hypothesis import assume
from hypothesis.strategies import composite
from numpy import pi

import semantic_digital_twin.spatial_types.spatial_types as cas
from .reference_implementations import shortest_angular_distance

BIG_NUMBER = 1e100
SMALL_NUMBER = 1e-100

all_expressions_float_np = Union[
    cas.SymbolicType, float, np.ndarray, Iterable[float], Iterable[Iterable[float]]
]


def to_float_or_np(x: all_expressions_float_np) -> Union[float, np.ndarray]:
    if isinstance(x, cas.SymbolicType):
        return x.to_np()
    return x


def assert_allclose(
    a: all_expressions_float_np,
    b: all_expressions_float_np,
    atol: float = 1e-3,
    rtol: float = 1e-3,
    equal_nan: bool = False,
):
    a = to_float_or_np(a)
    b = to_float_or_np(b)
    assert np.allclose(a, b, atol=atol, rtol=rtol, equal_nan=equal_nan)


def vector(x):
    return st.lists(float_no_nan_no_inf(), min_size=x, max_size=x)


def angle_positive():
    return st.floats(0, 2 * np.pi)


def random_angle():
    return st.floats(-np.pi, np.pi)


def compare_axis_angle(
    actual_angle: all_expressions_float_np,
    actual_axis: all_expressions_float_np,
    expected_angle: cas.ScalarData,
    expected_axis: all_expressions_float_np,
):
    actual_angle = to_float_or_np(actual_angle)
    actual_axis = to_float_or_np(actual_axis)
    expected_angle = to_float_or_np(expected_angle)
    expected_axis = to_float_or_np(expected_axis)
    try:
        assert_allclose(actual_axis, expected_axis)
        assert_allclose(shortest_angular_distance(actual_angle, expected_angle), 0.0)
    except AssertionError:
        try:
            assert_allclose(actual_axis, -expected_axis)
            assert_allclose(
                shortest_angular_distance(actual_angle, abs(expected_angle - 2 * pi)),
                0.0,
            )
        except AssertionError:
            assert_allclose(shortest_angular_distance(actual_angle, 0), 0.0)
            assert_allclose(shortest_angular_distance(0, expected_angle), 0.0)
            assert not np.any(np.isnan(actual_axis))
            assert not np.any(np.isnan(expected_axis))


def compare_orientations(
    actual_orientation: all_expressions_float_np,
    desired_orientation: all_expressions_float_np,
) -> None:
    try:
        assert_allclose(actual_orientation, desired_orientation)
    except:
        assert_allclose(actual_orientation, -desired_orientation)


@composite
def variable_name(draw):
    variable = draw(st.text("qwertyuiopasdfghjklzxcvbnm", min_size=1))
    assume(variable not in keyword.kwlist)
    return variable


@composite
def lists_of_same_length(
    draw, data_types=(), min_length=1, max_length=10, unique=False
):
    length = draw(st.integers(min_value=min_length, max_value=max_length))
    lists = []
    for elements in data_types:
        lists.append(
            draw(st.lists(elements, min_size=length, max_size=length, unique=unique))
        )
    return lists


@composite
def rnd_joint_state(draw, joint_limits):
    return {
        jn: draw(st.floats(ll, ul, allow_nan=False, allow_infinity=False))
        for jn, (ll, ul) in joint_limits.items()
    }


@composite
def rnd_joint_state2(draw, joint_limits):
    muh = draw(joint_limits)
    muh = {
        jn: ((ll if ll is not None else pi * 2), (ul if ul is not None else pi * 2))
        for (jn, (ll, ul)) in muh.items()
    }
    return {
        jn: draw(st.floats(ll, ul, allow_nan=False, allow_infinity=False))
        for jn, (ll, ul) in muh.items()
    }


def float_no_nan_no_inf(outer_limit=1e5):
    return float_no_nan_no_inf_min_max(-outer_limit, outer_limit)


def float_no_nan_no_inf_min_max(min_value=-1e5, max_value=1e5):
    return st.floats(
        allow_nan=False,
        allow_infinity=False,
        max_value=max_value,
        min_value=min_value,
        allow_subnormal=False,
    )


@composite
def sq_matrix(draw):
    i = draw(st.integers(min_value=1, max_value=10))
    i_sq = i**2
    l = draw(
        st.lists(float_no_nan_no_inf(outer_limit=1000), min_size=i_sq, max_size=i_sq)
    )
    return np.array(l).reshape((i, i))


def unit_vector(length, elements=None):
    if elements is None:
        elements = float_no_nan_no_inf()
    vector = st.lists(elements, min_size=length, max_size=length).filter(
        lambda x: SMALL_NUMBER < np.linalg.norm(x) < BIG_NUMBER
    )

    def normalize(v):
        v = [round(x, 4) for x in v]
        l = np.linalg.norm(v)
        if l == 0:
            return np.array([0] * (length - 1) + [1])
        return np.array([x / l for x in v])

    return st.builds(normalize, vector)


def quaternion():
    return unit_vector(4, float_no_nan_no_inf(outer_limit=1))
