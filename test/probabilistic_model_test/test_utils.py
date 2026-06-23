import unittest

import numpy as np
from scipy.special import logsumexp as scipy_logsumexp

from probabilistic_model.utils import logsumexp


class LogSumExpTestCase(unittest.TestCase):
    """
    High level tests for :func:`probabilistic_model.utils.logsumexp`.

    The function is a lightweight drop-in for :func:`scipy.special.logsumexp`, so the
    tests pin its behaviour against scipy on the cases used in the circuit inference
    loops and additionally cover the numerical edge cases (``-inf`` rows) that the
    circuit relies on.
    """

    def test_matches_scipy_on_one_dimensional_input(self):
        values = np.array([-3.0, 0.5, 2.0, 7.25])
        self.assertAlmostEqual(logsumexp(values), scipy_logsumexp(values))

    def test_accepts_plain_python_list(self):
        values = [-1.0, -2.0, -0.5]
        self.assertAlmostEqual(logsumexp(values), scipy_logsumexp(values))

    def test_reduces_over_axis(self):
        values = np.array([[0.0, 1.0, 2.0], [3.0, 4.0, 5.0]])
        for axis in (0, 1):
            np.testing.assert_allclose(
                logsumexp(values, axis=axis),
                scipy_logsumexp(values, axis=axis),
            )

    def test_axis_reduction_returns_expected_shape(self):
        values = np.zeros((4, 6))
        self.assertEqual(logsumexp(values, axis=0).shape, (6,))
        self.assertEqual(logsumexp(values, axis=1).shape, (4,))

    def test_full_reduction_returns_scalar(self):
        result = logsumexp(np.array([1.0, 2.0, 3.0]))
        self.assertTrue(np.isscalar(result) or np.ndim(result) == 0)

    def test_is_numerically_stable_for_large_values(self):
        # naive exp() would overflow to inf here; the result must stay finite
        values = np.array([1000.0, 1000.0])
        self.assertAlmostEqual(logsumexp(values), 1000.0 + np.log(2.0))

    def test_known_value(self):
        # log(exp(0) + exp(0)) == log(2)
        self.assertAlmostEqual(logsumexp([0.0, 0.0]), np.log(2.0))

    def test_all_negative_infinity_reduces_to_negative_infinity(self):
        # an all -inf reduction must yield -inf rather than NaN (a zero-probability
        # branch during truncation), matching scipy
        values = np.array([-np.inf, -np.inf])
        self.assertEqual(logsumexp(values), -np.inf)

    def test_single_negative_infinity_is_ignored(self):
        values = np.array([-np.inf, 0.0, 1.0])
        self.assertAlmostEqual(logsumexp(values), scipy_logsumexp(values))

    def test_negative_infinity_row_over_axis(self):
        values = np.array([[-np.inf, -np.inf], [0.0, 0.0]])
        np.testing.assert_allclose(
            logsumexp(values, axis=1),
            scipy_logsumexp(values, axis=1),
        )


if __name__ == "__main__":
    unittest.main()
