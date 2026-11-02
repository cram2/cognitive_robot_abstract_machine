from __future__ import annotations

import numpy as np
import pytest
from scipy.special import logsumexp as scipy_logsumexp

from probabilistic_model.utils import logsumexp

# %% agreement with scipy


@pytest.mark.parametrize(
    "values",
    [
        np.array([-3.0, 0.5, 2.0, 7.25]),
        [-1.0, -2.0, -0.5],
        np.array([-np.inf, 0.0, 1.0]),
    ],
    ids=["array", "list", "one negative infinity"],
)
def test_full_reduction_matches_scipy(values):
    result = logsumexp(values)
    assert np.ndim(result) == 0
    assert result == pytest.approx(scipy_logsumexp(values))


@pytest.mark.parametrize(
    "values",
    [
        np.array([[0.0, 1.0, 2.0], [3.0, 4.0, 5.0]]),
        np.array([[-np.inf, -np.inf], [0.0, 0.0]]),
    ],
    ids=["finite", "row of negative infinity"],
)
@pytest.mark.parametrize("axis", [0, 1])
def test_reduction_over_an_axis_matches_scipy(values, axis):
    expected = scipy_logsumexp(values, axis=axis)
    result = logsumexp(values, axis=axis)
    assert result.shape == expected.shape
    np.testing.assert_allclose(result, expected)


# %% numerical edge cases


def test_is_numerically_stable_for_large_values():
    assert logsumexp(np.array([1000.0, 1000.0])) == pytest.approx(1000.0 + np.log(2.0))


def test_all_negative_infinity_reduces_to_negative_infinity():
    assert logsumexp(np.array([-np.inf, -np.inf])) == -np.inf
