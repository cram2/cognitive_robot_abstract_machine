"""
Tests for :class:`~semantic_digital_twin.world_description.geometry.Bounds` read off a
point cloud, and for the points a region contains.
"""

from __future__ import annotations

import numpy as np

from semantic_digital_twin.world_description.geometry import Bounds

# %% reading bounds off a point cloud


def test_bounds_enclose_every_point_they_were_read_from():
    points = np.array([[1.0, -2.0, 3.0], [-1.0, 2.0, 0.5], [0.0, 0.0, 4.0]])

    bounds = Bounds.from_points(points)

    assert bounds.lower.tolist() == [-1.0, -2.0, 0.5]
    assert bounds.upper.tolist() == [1.0, 2.0, 4.0]


# %% points a region contains


def test_bounds_contain_the_points_inside_them():
    bounds = Bounds(np.array([0.0, 0.0, 0.0]), np.array([1.0, 1.0, 1.0]))
    points = np.array([[0.5, 0.5, 0.5], [2.0, 0.5, 0.5], [0.5, 0.5, -0.1]])

    assert bounds.contains(points).tolist() == [True, False, False]


def test_bounds_contain_a_point_on_their_boundary():
    bounds = Bounds(np.array([0.0, 0.0, 0.0]), np.array([1.0, 1.0, 1.0]))

    assert bounds.contains(np.array([[0.5, 0.5, 1.0]])).tolist() == [True]


def test_bounds_with_no_volume_contain_nothing():
    """
    A region flattened onto a plane encloses no volume, so nothing is inside it -- not
    even a point lying on it.
    """
    bounds = Bounds(np.array([0.0, 0.0, 1.0]), np.array([1.0, 1.0, 1.0]))

    assert bounds.contains(np.array([[0.5, 0.5, 1.0]])).tolist() == [False]


def test_empty_bounds_contain_no_point():
    """
    What absent geometry reads back as must not hold even the origin.
    """
    assert Bounds.empty().contains(np.zeros((1, 3))).tolist() == [False]
