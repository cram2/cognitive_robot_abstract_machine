"""
Tests for measuring a detected outline and naming the shape it belongs to.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from experiments.montessori.perception.footprint import (
    CrossSectionClassifier,
    Footprint,
)
from experiments.montessori.semantics import MontessoriShapeCategory

# %% measuring an outline


def _contour_of(boundary: np.ndarray, resolution: float) -> np.ndarray:
    """
    Turn a metric polygon into the pixel contour a rectified image would yield.
    """
    return np.round(boundary / resolution).astype(np.int32).reshape(-1, 1, 2)


def test_square_footprint_measures_its_own_side_length():
    side = 0.032
    corners = np.array([[0, 0], [side, 0], [side, side], [0, side]])

    footprint = Footprint.from_contour(_contour_of(corners, 0.001), 0.001)

    assert footprint.width == pytest.approx(side, abs=0.002)
    assert footprint.length == pytest.approx(side, abs=0.002)
    assert footprint.area == pytest.approx(side * side, rel=0.1)


def test_footprint_fill_ratio_separates_the_shape_families():
    side = 0.04
    square = np.array([[0, 0], [side, 0], [side, side], [0, side]])
    triangle = np.array([[0, 0], [side, 0], [side / 2, side]])
    circle = np.array(
        [
            [side / 2 * (1 + math.cos(angle)), side / 2 * (1 + math.sin(angle))]
            for angle in np.linspace(0, 2 * math.pi, 64, endpoint=False)
        ]
    )

    measured = {
        name: Footprint.from_contour(_contour_of(boundary, 0.0005), 0.0005).fill_ratio
        for name, boundary in (
            ("square", square),
            ("triangle", triangle),
            ("circle", circle),
        )
    }

    assert measured["triangle"] == pytest.approx(0.5, abs=0.05)
    assert measured["circle"] == pytest.approx(math.pi / 4, abs=0.05)
    assert measured["square"] == pytest.approx(1.0, abs=0.05)


@pytest.mark.parametrize(
    "fill_ratio, aspect_ratio, expected",
    [
        (0.5, 1.15, MontessoriShapeCategory.TRIANGULAR_PRISM),
        (math.pi / 4, 1.0, MontessoriShapeCategory.CYLINDER),
        (1.0, 1.0, MontessoriShapeCategory.CUBE),
        (1.0, 1.9, MontessoriShapeCategory.RECTANGULAR_PRISM),
        (1.0, 9.6, MontessoriShapeCategory.DISK),
    ],
)
def test_classifier_names_each_shape_from_its_proportions(
    fill_ratio: float, aspect_ratio: float, expected: MontessoriShapeCategory
):
    width = 0.02
    footprint = Footprint(
        area=fill_ratio * width * width * aspect_ratio,
        width=width,
        length=width * aspect_ratio,
        fill_ratio=fill_ratio,
        corner_count=4,
        yaw=0.0,
    )

    assert CrossSectionClassifier().classify(footprint) is expected


def test_footprint_yaw_follows_a_rotated_rectangle():
    angle = math.radians(30.0)
    rotation = np.array(
        [[math.cos(angle), -math.sin(angle)], [math.sin(angle), math.cos(angle)]]
    )
    rectangle = np.array([[-0.03, -0.01], [0.03, -0.01], [0.03, 0.01], [-0.03, 0.01]])
    rotated = rectangle @ rotation.T + np.array([0.06, 0.06])

    footprint = Footprint.from_contour(_contour_of(rotated, 0.0005), 0.0005)

    assert footprint.yaw == pytest.approx(angle, abs=math.radians(3.0))
