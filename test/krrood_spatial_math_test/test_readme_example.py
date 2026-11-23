"""
The README's example of adding a space, run as written, and the refusals it describes.
"""

from __future__ import annotations

import sys
from collections.abc import Iterator
from dataclasses import dataclass
from types import ModuleType

import pytest

from krrood_spatial_math.exceptions import (
    FrameHasNoCrossProductError,
    FrameMeasuresNoLengthsError,
    MissingAxisCoordinatesError,
    ReferenceFrameMismatchError,
)
from krrood_spatial_math.point import Point

# %% the example as the README gives it


@pytest.fixture(scope="module")
def adding_a_space_example(adding_a_space_example_code) -> Iterator[ModuleType]:
    """
    :return: The README's *Adding a space* example as a module, after running all of
        its code blocks in order, its own assertions included.
    """
    example = ModuleType("adding_a_space_example")
    with pytest.MonkeyPatch.context() as patch:
        patch.setitem(sys.modules, example.__name__, example)
        exec(
            compile(adding_a_space_example_code, "README.md: Adding a space", "exec"),
            example.__dict__,
        )
        yield example


def test_the_examples_change_composed_with_its_inverse_leaves_a_keypoint_in_place(
    adding_a_space_example,
):
    """
    The example's crop and resize, composed with its inverse, is no change at all.
    """
    change = adding_a_space_example.thumbnail_from_camera_image

    keypoint = adding_a_space_example.keypoint.apply_transformation(
        transformation=change.inverse() @ change
    )

    assert keypoint == adding_a_space_example.keypoint


# %% what the example says the interfaces refuse


def test_points_of_two_pixel_frames_have_no_difference(adding_a_space_example):
    """
    A keypoint of the camera image and one of the thumbnail are not subtracted.
    """
    with pytest.raises(ReferenceFrameMismatchError):
        adding_a_space_example.keypoint - adding_a_space_example.thumbnail_keypoint


def test_a_pixel_vector_has_no_length(adding_a_space_example):
    """
    Pixels may not be square, so pixel coordinates alone give no lengths.
    """
    with pytest.raises(FrameMeasuresNoLengthsError):
        adding_a_space_example.box_width.norm()


def test_a_pixel_vector_has_no_cross_product(adding_a_space_example):
    """
    A pixel frame is neither Cartesian nor three-dimensional.
    """
    box_width = adding_a_space_example.box_width

    with pytest.raises(FrameHasNoCrossProductError):
        box_width.cross(other=box_width)


def test_a_pixel_point_named_after_other_axes_is_refused_where_it_is_defined(
    adding_a_space_example,
):
    """
    A pixel point holding x and y instead of the pixel axes u and v has no coordinates.
    """
    pixel_vector = adding_a_space_example.PixelVector

    with pytest.raises(MissingAxisCoordinatesError):

        @dataclass(kw_only=True)
        class MisnamedPixelPoint(Point[pixel_vector]):
            """
            A pixel point whose fields are not named after the pixel axes.
            """

            x: float
            """
            Meant as the column.
            """

            y: float
            """
            Meant as the row.
            """
