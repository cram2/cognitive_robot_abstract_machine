"""
Hyperrectangles stacked per node behave like the simple intervals they were made of.
"""

from __future__ import annotations

import numpy as np
import pytest
from random_events.interval import Interval, closed, open_closed, reals

from probabilistic_model.probabilistic_circuit.tensorized.input_layer.multivariate_gaussian.hyperrectangle_array import (
    HyperrectangleArray,
)


def hyperrectangle_of(*intervals: Interval) -> HyperrectangleArray:
    """
    :return: The single hyperrectangle that one simple interval per variable spans.
    """
    return HyperrectangleArray.of_simple_intervals(
        [interval.simple_sets[0] for interval in intervals]
    )


# %% fixtures


@pytest.fixture
def first() -> HyperrectangleArray:
    """
    The hyperrectangle [-1, 1] x (0, 2].
    """
    return hyperrectangle_of(closed(-1.0, 1.0), open_closed(0.0, 2.0))


@pytest.fixture
def second() -> HyperrectangleArray:
    """
    The hyperrectangle [0, 3] x the reals.
    """
    return hyperrectangle_of(closed(0.0, 3.0), reals())


@pytest.fixture
def per_node(first, second) -> HyperrectangleArray:
    """
    :func:`first` and :func:`second`, one per node.
    """
    return HyperrectangleArray.stack([first, second])


# %% hyperrectangles per node


def test_stack_puts_one_hyperrectangle_per_node(per_node, first, second):
    assert per_node.interval.shape == (2, 2, 2)
    np.testing.assert_array_equal(per_node.interval[0], first.interval)
    np.testing.assert_array_equal(per_node.bounds[1], second.bounds)


def test_lower_and_upper_are_the_ends_of_the_intervals(per_node):
    np.testing.assert_array_equal(per_node.lower, [[-1.0, 0.0], [0.0, -np.inf]])
    np.testing.assert_array_equal(per_node.upper, [[1.0, 2.0], [3.0, np.inf]])


def test_concatenate_joins_the_nodes_in_order(per_node):
    joined = HyperrectangleArray.concatenate([per_node, per_node.select(np.array([1]))])
    np.testing.assert_array_equal(
        joined.interval, np.concatenate([per_node.interval, per_node.interval[1:]])
    )
    np.testing.assert_array_equal(
        joined.bounds, np.concatenate([per_node.bounds, per_node.bounds[1:]])
    )


def test_over_keeps_the_variables_in_the_given_order(per_node):
    swapped = per_node.over(np.array([1, 0]))
    np.testing.assert_array_equal(swapped.interval, per_node.interval[:, [1, 0]])
    np.testing.assert_array_equal(swapped.bounds, per_node.bounds[:, [1, 0]])


def test_translated_moves_both_ends_of_every_interval(per_node):
    offsets = np.array([1.0, -2.0])
    moved = per_node.translated(offsets)
    np.testing.assert_array_equal(moved.lower, per_node.lower + offsets)
    np.testing.assert_array_equal(moved.upper, per_node.upper + offsets)
    np.testing.assert_array_equal(moved.bounds, per_node.bounds)


def test_scaled_multiplies_both_ends_of_every_interval(per_node):
    factors = np.array([2.0, 0.5])
    scaled = per_node.scaled(factors)
    np.testing.assert_array_equal(scaled.lower, per_node.lower * factors)
    np.testing.assert_array_equal(scaled.upper, per_node.upper * factors)


def test_contains_respects_open_ends(per_node):
    points = np.array([[0.0, 0.0], [0.0, 2.0]])
    np.testing.assert_array_equal(
        per_node.contains(points), [[False, True], [True, True]]
    )
