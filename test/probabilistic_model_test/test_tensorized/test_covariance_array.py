import unittest

import numpy as np

from probabilistic_model.distributions.multivariate_gaussian import Covariance
from probabilistic_model.exceptions import ShapeMismatchError
from probabilistic_model.probabilistic_circuit.tensorized.input_layer.multivariate_gaussian.covariance_array import (
    CovarianceArray,
)

MATRICES = np.array(
    [
        [[1.0, 0.6, 0.1], [0.6, 2.0, -0.3], [0.1, -0.3, 0.5]],
        [[0.5, -0.2, 0.0], [-0.2, 0.8, 0.4], [0.0, 0.4, 3.0]],
    ]
)


class CovarianceArrayTestCase(unittest.TestCase):
    """
    The covariances of the nodes of a layer behave like the full matrices they stand
    for.
    """

    def setUp(self):
        self.covariances = CovarianceArray.from_matrices(MATRICES)

    def test_only_the_lower_triangles_are_stored(self):
        self.assertEqual(self.covariances.lower_triangles.shape, (2, 6))
        np.testing.assert_array_equal(
            self.covariances.lower_triangles[0], [1.0, 0.6, 2.0, 0.1, -0.3, 0.5]
        )

    def test_matrices_are_the_full_symmetric_matrices(self):
        self.assertEqual(self.covariances.number_of_matrices, 2)
        self.assertEqual(self.covariances.dimension, 3)
        np.testing.assert_array_equal(self.covariances.matrices, MATRICES)

    def test_variances_are_the_diagonals(self):
        np.testing.assert_array_equal(
            self.covariances.variances, np.diagonal(MATRICES, axis1=1, axis2=2)
        )

    def test_covariance_at_is_the_covariance_of_one_matrix(self):
        covariance = self.covariances.covariance_at(1)
        self.assertIsInstance(covariance, Covariance)
        np.testing.assert_array_equal(covariance.matrix, MATRICES[1])

    def test_from_covariances_stacks_single_covariances(self):
        stacked = CovarianceArray.from_covariances(
            [Covariance.from_matrix(matrix) for matrix in MATRICES]
        )
        np.testing.assert_array_equal(stacked.matrices, MATRICES)

    def test_between_reads_the_entries_at_rows_and_columns(self):
        rows, columns = np.array([2, 0]), np.array([1])
        np.testing.assert_array_equal(
            self.covariances.between(rows, columns), MATRICES[:, rows][:, :, columns]
        )

    def test_marginal_keeps_the_indices_in_the_given_order(self):
        order = np.array([2, 0])
        np.testing.assert_array_equal(
            self.covariances.marginal(order).matrices,
            MATRICES[:, order][:, :, order],
        )

    def test_scaled_multiplies_every_entry_once_per_index(self):
        factors = np.array([2.0, -1.0, 0.5])
        np.testing.assert_allclose(
            self.covariances.scaled(factors).matrices,
            MATRICES * factors[:, None] * factors[None, :],
        )

    def test_select_and_concatenate(self):
        selected = self.covariances.select(np.array([False, True]))
        np.testing.assert_array_equal(selected.matrices, MATRICES[1:])
        joined = CovarianceArray.concatenate([selected, self.covariances])
        np.testing.assert_array_equal(
            joined.matrices, np.concatenate([MATRICES[1:], MATRICES])
        )

    def test_copy_shares_no_array(self):
        copy = self.covariances.copy()
        copy.lower_triangles[:] = 0.0
        np.testing.assert_array_equal(self.covariances.matrices, MATRICES)

    def test_entries_that_are_no_lower_triangles_are_rejected(self):
        with self.assertRaises(ShapeMismatchError):
            CovarianceArray(np.zeros((2, 5)))
        with self.assertRaises(ShapeMismatchError):
            CovarianceArray.from_matrices(np.zeros((2, 3, 2)))


if __name__ == "__main__":
    unittest.main()
