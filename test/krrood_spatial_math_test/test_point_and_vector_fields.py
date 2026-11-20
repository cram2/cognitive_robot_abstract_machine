"""
Points and vectors whose coordinates are the fields their class declares, with the
arithmetic and frame checks :class:`Point` and :class:`Vector` implement on them.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import pytest

from krrood_spatial_math.coordinate_system import FrameAxis, GenericCoordinate
from krrood_spatial_math.exceptions import (
    MissingAxisCoordinatesError,
    ReferenceFrameMismatchError,
)
from krrood_spatial_math.point import Point

from .dataset.spaces import (
    NamedScalablePlaneFrame,
    ScalablePlaneFrame,
    ScalablePlanePoint,
    ScalablePlaneVector,
)

# %% fixtures


@pytest.fixture
def scalable_plane_frame() -> ScalablePlaneFrame:
    """
    :return: A frame on a scalable plane.
    """
    return NamedScalablePlaneFrame(name="sheet")


@pytest.fixture
def other_scalable_plane_frame() -> ScalablePlaneFrame:
    """
    :return: Another frame on a scalable plane, told apart from the first.
    """
    return NamedScalablePlaneFrame(name="other sheet")


# %% coordinates are the fields named after the axes


def test_the_coordinates_of_a_vector_are_its_declared_fields_in_order(
    scalable_plane_frame,
):
    """
    The coordinates come from the fields the vector class names after the axes.
    """
    vector = ScalablePlaneVector(x=3.0, y=4.0, coordinate_system=scalable_plane_frame)

    assert vector.coordinates == (3.0, 4.0)


def test_the_coordinates_of_a_point_are_its_declared_fields_in_order(
    scalable_plane_frame,
):
    """
    The coordinates come from the fields the point class names after the axes.
    """
    point = ScalablePlanePoint(x=1.0, y=2.0, coordinate_system=scalable_plane_frame)

    assert point.coordinates == (1.0, 2.0)


def test_a_vectors_coordinates_are_floats_unless_its_class_says_otherwise():
    """
    A vector type that names no coordinate type stores floats.
    """
    assert (
        ScalablePlaneVector.get_type_of_generic_parameter(parameter=GenericCoordinate)
        is float
    )


# %% arithmetic the base implements


def test_two_vectors_add_up_coordinate_by_coordinate(scalable_plane_frame):
    """
    Sums and differences of vectors of one frame are taken per coordinate.
    """
    first = ScalablePlaneVector(x=1.0, y=2.0, coordinate_system=scalable_plane_frame)
    second = ScalablePlaneVector(x=3.0, y=5.0, coordinate_system=scalable_plane_frame)

    assert first + second == ScalablePlaneVector(
        x=4.0, y=7.0, coordinate_system=scalable_plane_frame
    )
    assert second - first == ScalablePlaneVector(
        x=2.0, y=3.0, coordinate_system=scalable_plane_frame
    )


def test_a_vector_is_stretched_by_a_factor(scalable_plane_frame):
    """
    Multiplying a vector by a number multiplies every coordinate.
    """
    vector = ScalablePlaneVector(x=1.0, y=-2.0, coordinate_system=scalable_plane_frame)

    assert vector * 3.0 == ScalablePlaneVector(
        x=3.0, y=-6.0, coordinate_system=scalable_plane_frame
    )


def test_the_difference_of_two_points_is_their_vector_type(
    scalable_plane_frame,
):
    """
    Subtracting two points gives the vector type the point type is bound to.
    """
    start = ScalablePlanePoint(x=1.0, y=2.0, coordinate_system=scalable_plane_frame)
    end = ScalablePlanePoint(x=4.0, y=6.0, coordinate_system=scalable_plane_frame)

    assert end - start == ScalablePlaneVector(
        x=3.0, y=4.0, coordinate_system=scalable_plane_frame
    )


def test_a_point_moved_by_a_vector_stays_a_point(scalable_plane_frame):
    """
    Adding a vector to a point shifts each coordinate and keeps the point type.
    """
    point = ScalablePlanePoint(x=1.0, y=2.0, coordinate_system=scalable_plane_frame)
    vector = ScalablePlaneVector(x=3.0, y=4.0, coordinate_system=scalable_plane_frame)

    assert point + vector == ScalablePlanePoint(
        x=4.0, y=6.0, coordinate_system=scalable_plane_frame
    )


# %% values of two frames are never combined


def test_vectors_of_two_frames_do_not_add_up(
    scalable_plane_frame, other_scalable_plane_frame
):
    """
    Vectors of two frames are never added or subtracted.
    """
    first = ScalablePlaneVector(x=1.0, y=2.0, coordinate_system=scalable_plane_frame)
    second = ScalablePlaneVector(
        x=3.0, y=5.0, coordinate_system=other_scalable_plane_frame
    )

    with pytest.raises(ReferenceFrameMismatchError):
        first + second
    with pytest.raises(ReferenceFrameMismatchError):
        first - second


def test_points_of_two_frames_have_no_difference(
    scalable_plane_frame, other_scalable_plane_frame
):
    """
    Points of two frames have no vector between them.
    """
    start = ScalablePlanePoint(x=1.0, y=2.0, coordinate_system=scalable_plane_frame)
    end = ScalablePlanePoint(x=4.0, y=6.0, coordinate_system=other_scalable_plane_frame)

    with pytest.raises(ReferenceFrameMismatchError):
        end - start


def test_a_point_is_not_moved_by_a_vector_of_another_frame(
    scalable_plane_frame, other_scalable_plane_frame
):
    """
    A point is only moved by a vector of its own frame.
    """
    point = ScalablePlanePoint(x=1.0, y=2.0, coordinate_system=scalable_plane_frame)
    vector = ScalablePlaneVector(
        x=3.0, y=4.0, coordinate_system=other_scalable_plane_frame
    )

    with pytest.raises(ReferenceFrameMismatchError):
        point + vector


# %% coordinates are the attributes named after the frame's axes


def test_a_field_not_named_after_an_axis_is_no_coordinate(scalable_plane_frame):
    """
    A field a point type declares besides its axes is kept out of its coordinates.
    """

    @dataclass(kw_only=True)
    class LabelledPoint(Point[ScalablePlaneVector]):
        """
        A point on a scalable plane that also carries a label.
        """

        x: float
        """
        The first coordinate.
        """

        label: str
        """
        What the point marks.
        """

        y: float
        """
        The second coordinate.
        """

    point = LabelledPoint(
        x=1.0, label="corner", y=2.0, coordinate_system=scalable_plane_frame
    )

    assert point.coordinates == (1.0, 2.0)


def test_the_coordinates_follow_the_order_of_the_frames_axes(scalable_plane_frame):
    """
    The coordinates are in the order of the axes, whatever order the fields have.
    """

    @dataclass(kw_only=True)
    class ReorderedPoint(Point[ScalablePlaneVector]):
        """
        A point on a scalable plane declaring its second coordinate first.
        """

        y: float
        """
        The second coordinate.
        """

        x: float
        """
        The first coordinate.
        """

    point = ReorderedPoint(y=2.0, x=1.0, coordinate_system=scalable_plane_frame)

    assert point.coordinates == (1.0, 2.0)


def test_a_point_type_without_an_attribute_per_axis_is_refused_where_it_is_defined():
    """
    A value type must hold a coordinate for every axis of its kind of frame.
    """
    with pytest.raises(MissingAxisCoordinatesError) as raised:

        @dataclass(kw_only=True)
        class HalfPoint(Point[ScalablePlaneVector]):
            """
            A point on a scalable plane that declares only its first coordinate.
            """

            x: float
            """
            The first coordinate.
            """

    assert raised.value.axes == (FrameAxis.Y,)


def test_a_value_built_from_coordinates_holds_them_under_the_axes_names(
    scalable_plane_frame,
):
    """
    Building from coordinates gives the value one coordinate per axis, in axis order.
    """
    vector = ScalablePlaneVector.from_coordinates(
        coordinates=(3.0, 4.0), coordinate_system=scalable_plane_frame
    )

    assert vector == ScalablePlaneVector(
        x=3.0, y=4.0, coordinate_system=scalable_plane_frame
    )
