import unittest

import numpy as np
from random_events.interval import closed, open_closed, reals

from probabilistic_model.probabilistic_circuit.tensorized.input_layer.multivariate_gaussian.hyperrectangle_array import (
    HyperrectangleArray,
)


def hyperrectangle_of(*intervals) -> HyperrectangleArray:
    """
    :return: The single hyperrectangle that one simple interval per variable spans.
    """
    return HyperrectangleArray.of_simple_intervals(
        [interval.simple_sets[0] for interval in intervals]
    )


class HyperrectangleArrayTestCase(unittest.TestCase):
    """
    Hyperrectangles stacked per node behave like the simple intervals they were made of.
    """

    def setUp(self):
        self.first = hyperrectangle_of(closed(-1.0, 1.0), open_closed(0.0, 2.0))
        self.second = hyperrectangle_of(closed(0.0, 3.0), reals())
        self.per_node = HyperrectangleArray.stack([self.first, self.second])

    def test_stack_puts_one_hyperrectangle_per_node(self):
        self.assertEqual(self.per_node.interval.shape, (2, 2, 2))
        np.testing.assert_array_equal(self.per_node.interval[0], self.first.interval)
        np.testing.assert_array_equal(self.per_node.bounds[1], self.second.bounds)

    def test_lower_and_upper_are_the_ends_of_the_intervals(self):
        np.testing.assert_array_equal(
            self.per_node.lower, [[-1.0, 0.0], [0.0, -np.inf]]
        )
        np.testing.assert_array_equal(self.per_node.upper, [[1.0, 2.0], [3.0, np.inf]])

    def test_concatenate_joins_the_nodes_in_order(self):
        joined = HyperrectangleArray.concatenate(
            [self.per_node, self.per_node.select(np.array([1]))]
        )
        np.testing.assert_array_equal(
            joined.interval,
            np.concatenate([self.per_node.interval, self.per_node.interval[1:]]),
        )
        np.testing.assert_array_equal(
            joined.bounds,
            np.concatenate([self.per_node.bounds, self.per_node.bounds[1:]]),
        )

    def test_over_keeps_the_variables_in_the_given_order(self):
        swapped = self.per_node.over(np.array([1, 0]))
        np.testing.assert_array_equal(
            swapped.interval, self.per_node.interval[:, [1, 0]]
        )
        np.testing.assert_array_equal(swapped.bounds, self.per_node.bounds[:, [1, 0]])

    def test_translated_moves_both_ends_of_every_interval(self):
        offsets = np.array([1.0, -2.0])
        moved = self.per_node.translated(offsets)
        np.testing.assert_array_equal(moved.lower, self.per_node.lower + offsets)
        np.testing.assert_array_equal(moved.upper, self.per_node.upper + offsets)
        np.testing.assert_array_equal(moved.bounds, self.per_node.bounds)

    def test_scaled_multiplies_both_ends_of_every_interval(self):
        factors = np.array([2.0, 0.5])
        scaled = self.per_node.scaled(factors)
        np.testing.assert_array_equal(scaled.lower, self.per_node.lower * factors)
        np.testing.assert_array_equal(scaled.upper, self.per_node.upper * factors)

    def test_contains_respects_open_ends(self):
        points = np.array([[0.0, 0.0], [0.0, 2.0]])
        np.testing.assert_array_equal(
            self.per_node.contains(points), [[False, True], [True, True]]
        )


if __name__ == "__main__":
    unittest.main()
