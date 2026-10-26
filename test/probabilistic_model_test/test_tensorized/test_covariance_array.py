"""
The covariances of the nodes of a layer behave like the full matrices they stand for.
"""

from __future__ import annotations

import numpy as np
import pytest

from probabilistic_model.distributions.multivariate_gaussian import Covariance
from probabilistic_model.exceptions import ShapeMismatchError
from probabilistic_model.probabilistic_circuit.tensorized.input_layer.multivariate_gaussian.covariance_array import (
    CovarianceArray,
)

# %% fixtures


@pytest.fixture
def matrices() -> np.ndarray:
    """
    Two symmetric positive definite 3x3 matrices.
    """
    return np.array(
        [
            [[1.0, 0.6, 0.1], [0.6, 2.0, -0.3], [0.1, -0.3, 0.5]],
            [[0.5, -0.2, 0.0], [-0.2, 0.8, 0.4], [0.0, 0.4, 3.0]],
        ]
    )


@pytest.fixture
def covariances(matrices) -> CovarianceArray:
    """
    The covariances of :func:`matrices`.
    """
    return CovarianceArray.from_matrices(matrices)


# %% storage


def test_only_the_lower_triangles_are_stored(covariances):
    assert covariances.lower_triangles.shape == (2, 6)
    np.testing.assert_array_equal(
        covariances.lower_triangles[0], [1.0, 0.6, 2.0, 0.1, -0.3, 0.5]
    )


def test_matrices_are_the_full_symmetric_matrices(covariances, matrices):
    assert covariances.number_of_matrices == 2
    assert covariances.dimension == 3
    np.testing.assert_array_equal(covariances.matrices, matrices)


def test_variances_are_the_diagonals(covariances, matrices):
    np.testing.assert_array_equal(
        covariances.variances, np.diagonal(matrices, axis1=1, axis2=2)
    )


def test_covariance_at_is_the_covariance_of_one_matrix(covariances, matrices):
    covariance = covariances.covariance_at(1)
    assert isinstance(covariance, Covariance)
    np.testing.assert_array_equal(covariance.matrix, matrices[1])


def test_from_covariances_stacks_single_covariances(matrices):
    stacked = CovarianceArray.from_covariances(
        [Covariance.from_matrix(matrix) for matrix in matrices]
    )
    np.testing.assert_array_equal(stacked.matrices, matrices)


def test_entries_that_are_no_lower_triangles_are_rejected():
    with pytest.raises(ShapeMismatchError):
        CovarianceArray(np.zeros((2, 5)))
    with pytest.raises(ShapeMismatchError):
        CovarianceArray.from_matrices(np.zeros((2, 3, 2)))


# %% operations


def test_between_reads_the_entries_at_rows_and_columns(covariances, matrices):
    rows, columns = np.array([2, 0]), np.array([1])
    np.testing.assert_array_equal(
        covariances.between(rows, columns), matrices[:, rows][:, :, columns]
    )


def test_marginal_keeps_the_indices_in_the_given_order(covariances, matrices):
    order = np.array([2, 0])
    np.testing.assert_array_equal(
        covariances.marginal(order).matrices, matrices[:, order][:, :, order]
    )


def test_scaled_multiplies_every_entry_once_per_index(covariances, matrices):
    factors = np.array([2.0, -1.0, 0.5])
    np.testing.assert_allclose(
        covariances.scaled(factors).matrices,
        matrices * factors[:, None] * factors[None, :],
    )


def test_select_and_concatenate(covariances, matrices):
    selected = covariances.select(np.array([False, True]))
    np.testing.assert_array_equal(selected.matrices, matrices[1:])
    joined = CovarianceArray.concatenate([selected, covariances])
    np.testing.assert_array_equal(
        joined.matrices, np.concatenate([matrices[1:], matrices])
    )


def test_copy_shares_no_array(covariances, matrices):
    copy = covariances.copy()
    copy.lower_triangles[:] = 0.0
    np.testing.assert_array_equal(covariances.matrices, matrices)
