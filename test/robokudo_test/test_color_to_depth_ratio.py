"""
Tests for mapping between color and depth image pixels.
"""

from __future__ import annotations

import numpy as np
import pytest

from robokudo.types.cv import ColorToDepthRatio, Point2D, Rect


@pytest.fixture
def half_resolution() -> ColorToDepthRatio:
    """
    A depth image with half the width and a quarter of the height of the color image.
    """
    return ColorToDepthRatio(horizontal=0.5, vertical=0.25)


# %% identity


def test_default_ratio_is_identity():
    assert ColorToDepthRatio().is_identity


def test_ratio_scaling_one_axis_is_not_identity():
    assert not ColorToDepthRatio(horizontal=1.0, vertical=0.5).is_identity


# %% color to depth


def test_depth_pixel_truncates_to_containing_pixel(half_resolution: ColorToDepthRatio):
    assert half_resolution.depth_pixel(9.0, 7.0) == Point2D(x=4, y=1)


def test_depth_region_scales_each_corner(half_resolution: ColorToDepthRatio):
    region = Rect.from_corner_points(x1=4, y1=8, x2=10, y2=20)

    assert half_resolution.depth_region(region) == Rect.from_corner_points(
        x1=2, y1=2, x2=5, y2=5
    )


# %% depth to color


def test_color_pixels_scale_each_axis(half_resolution: ColorToDepthRatio):
    depth_pixels = np.array([[2, 3], [5, 1]])

    np.testing.assert_array_equal(
        half_resolution.color_pixels(depth_pixels), [[4, 12], [10, 4]]
    )
