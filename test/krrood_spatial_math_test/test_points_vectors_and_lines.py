"""
The point, vector and line interfaces of an affine space, and the stored mimics that
implement them.
"""

from __future__ import annotations

import math
from dataclasses import fields

import numpy as np
import pytest

from krrood.symbolic_math import symbolic_math as sm

from krrood.patterns.exceptions import InterfaceInstantiationError
from krrood_spatial_math.cartesian_frame_3_dimensional import CartesianFrame3Dimensional
from krrood_spatial_math.coordinate_system import (
    GenericAffineFrame,
    CoordinateRepresentation,
    GenericCoordinate,
    GenericCoordinateSystem,
)
from krrood_spatial_math.exceptions import (
    ReferenceFrameMismatchError,
    SpaceMeasuresNoLengthsError,
)
from krrood_spatial_math.line import Line, LineSegment
from krrood_spatial_math.point import Point, GenericPoint
from krrood_spatial_math.spatial_types import (
    Point3Dimensional,
    Vector3Dimensional,
)
from krrood_spatial_math.space import EuclideanSpace
from krrood_spatial_math.spatial_types_2_dimensional import Vector2Dimensional
from krrood_spatial_math.vector import Vector, GenericVector

from .dataset.frames import NamedFrame
from .dataset.spaces import (
    AxisScalingChange,
    FloatCoordinateVector3Dimensional,
    OpenCoordinateScalablePlanePoint,
    OpenCoordinateScalablePlaneVector,
    ScalablePlaneFrame,
    StoredScalablePlaneLine,
    StoredScalablePlaneLineSegment,
    ScalablePlanePoint,
    ScalablePlaneVector,
    StretchedAxesFrame,
    StretchedAxesVector,
)

# %% points and vectors given in an affine frame


@pytest.mark.parametrize("affine_value", [Point, Vector])
def test_a_point_or_vector_is_a_coordinate_representation(affine_value):
    """
    Every point and vector is given in a coordinate system.
    """
    assert issubclass(affine_value, CoordinateRepresentation)


def test_a_point_is_not_a_vector():
    assert not issubclass(Point, Vector)


def test_a_point_has_a_position_and_a_vector_has_none():
    """
    Only a point is shifted by an affine change; a vector is only turned or scaled.
    """
    assert (ScalablePlanePoint.has_position, ScalablePlaneVector.has_position) == (
        True,
        False,
    )


def test_a_point_is_moved_by_a_vector():
    """
    Adding a vector shifts each coordinate of the point by the vector's.
    """
    point = ScalablePlanePoint(x=1.0, y=2.0, coordinate_system=None)
    vector = ScalablePlaneVector(x=3.0, y=4.0, coordinate_system=None)

    assert point + vector == ScalablePlanePoint(x=4.0, y=6.0, coordinate_system=None)


def test_a_point_type_binds_only_the_vector_type_it_is_moved_by():
    """
    A point's only type parameter is its vector type; its frame is the vector's.
    """
    (vector_parameter,) = Point.__parameters__

    assert vector_parameter.__bound__ is Vector


def test_a_vector_of_a_space_that_is_not_euclidean_has_no_length():
    """
    Only a Euclidean space has lengths; a vector of any other space is refused.
    """
    vector = ScalablePlaneVector(x=3.0, y=4.0, coordinate_system=None)

    with pytest.raises(SpaceMeasuresNoLengthsError):
        vector.norm()


def test_points_of_a_space_that_is_not_euclidean_have_no_distance():
    """
    Only a Euclidean space has distances; points of any other space are refused.
    """
    point = ScalablePlanePoint(x=1.0, y=2.0, coordinate_system=None)

    with pytest.raises(SpaceMeasuresNoLengthsError):
        point.euclidean_distance(point)


def test_a_point_is_given_in_the_kind_of_frame_of_its_vector_type():
    """
    The kind of frame a point type reports is the one its vector type binds.
    """
    assert ScalablePlanePoint.get_type_of_generic_parameter(
        parameter=GenericCoordinateSystem
    ) is ScalablePlaneVector.get_type_of_generic_parameter(parameter=GenericAffineFrame)


def test_a_point_reports_what_its_coordinates_are_stored_as():
    """
    The coordinate type a point binds is read back from the class, without an instance.
    """
    assert (
        ScalablePlanePoint.get_type_of_generic_parameter(parameter=GenericCoordinate)
        is float
    )


def test_a_point_reports_the_kind_of_vector_it_is_moved_by():
    """
    The vector type a point type binds is read back from the class.
    """
    assert (
        ScalablePlanePoint.get_type_of_generic_parameter(parameter=GenericVector)
        is ScalablePlaneVector
    )


def test_a_points_frame_field_is_narrowed_to_the_bound_frame():
    """
    The frame a point type binds becomes the type of its coordinate_system field.
    """
    point_types = {
        point_field.name: point_field.type for point_field in fields(ScalablePlanePoint)
    }

    assert point_types["coordinate_system"] == ScalablePlaneFrame | None


# %% how points and vectors combine


def test_the_difference_of_two_points_is_the_vector_between_them():
    """
    Subtracting points gives the vector type the point binds, not another point.
    """
    start = ScalablePlanePoint(x=1.0, y=2.0, coordinate_system=None)
    end = ScalablePlanePoint(x=4.0, y=6.0, coordinate_system=None)

    assert end - start == ScalablePlaneVector(x=3.0, y=4.0, coordinate_system=None)


def test_a_point_moved_by_the_difference_to_another_point_reaches_it():
    """
    Point subtraction and point-vector addition undo each other.
    """
    start = ScalablePlanePoint(x=1.0, y=2.0, coordinate_system=None)
    end = ScalablePlanePoint(x=4.0, y=6.0, coordinate_system=None)

    assert start + (end - start) == end


# %% points and vectors with symbolic coordinates


def test_the_difference_of_two_points_with_symbolic_coordinates_is_symbolic():
    """
    A point type that leaves its coordinate type open subtracts symbolic coordinates
    into its vector type, as it does numbers.
    """
    x = sm.FloatVariable(name="x")
    y = sm.FloatVariable(name="y")
    start = OpenCoordinateScalablePlanePoint(x=x, y=y, coordinate_system=None)
    end = OpenCoordinateScalablePlanePoint(x=x + 3, y=y * 2, coordinate_system=None)

    difference = end - start

    assert type(difference) is OpenCoordinateScalablePlaneVector
    assert difference.x.equivalent(other=3)
    assert difference.y.equivalent(other=y)


def test_a_point_with_symbolic_coordinates_is_moved_by_a_change():
    """
    A change moves a point's symbolic coordinates with its position, and a vector's
    without.
    """
    x = sm.FloatVariable(name="x")
    y = sm.FloatVariable(name="y")
    change = AxisScalingChange(x_scale=0.5, y_scale=2.0, x_shift=-50.0, y_shift=10.0)

    point = OpenCoordinateScalablePlanePoint(
        x=x, y=y, coordinate_system=None
    ).apply_transformation(transformation=change)
    vector = OpenCoordinateScalablePlaneVector(
        x=x, y=y, coordinate_system=None
    ).apply_transformation(transformation=change)

    assert point.x.equivalent(other=0.5 * x - 50.0)
    assert point.y.equivalent(other=2.0 * y + 10.0)
    assert vector.x.equivalent(other=0.5 * x)
    assert vector.y.equivalent(other=2.0 * y)


# %% straight lines given by a point and a direction


def test_a_line_cannot_be_instantiated():
    with pytest.raises(InterfaceInstantiationError):
        Line()


def test_a_line_binds_only_its_point_type_and_runs_along_its_vector_type():
    """
    A line's only type parameter is its point type, whose vector type it runs along.
    """
    assert Line.__parameters__ == (GenericPoint,)
    assert GenericPoint.__bound__ is Point
    assert (
        StoredScalablePlaneLine.get_type_of_generic_parameter(parameter=GenericVector)
        is ScalablePlaneVector
    )


def test_a_lines_fields_are_narrowed_to_the_bound_types():
    """
    A line type that binds its point type gives its fields that type and its vector
    type.
    """
    line_types = {
        line_field.name: line_field.type
        for line_field in fields(StoredScalablePlaneLine)
    }

    assert line_types == {
        "point": ScalablePlanePoint,
        "direction": ScalablePlaneVector,
    }


@pytest.mark.parametrize(
    "position_along_line, expected_x, expected_y",
    [(0.0, 1.0, 2.0), (1.0, 4.0, 6.0), (0.5, 2.5, 4.0), (-1.0, -2.0, -2.0)],
)
def test_a_line_reaches_the_point_at_a_position_along_it(
    position_along_line, expected_x, expected_y
):
    """
    A line's point at a position is its point moved by its direction that many times,
    behind it for a negative position.
    """
    line = StoredScalablePlaneLine(
        point=ScalablePlanePoint(x=1.0, y=2.0, coordinate_system=None),
        direction=ScalablePlaneVector(x=3.0, y=4.0, coordinate_system=None),
    )

    assert line.point_at_position(
        position_along_line=position_along_line
    ) == ScalablePlanePoint(x=expected_x, y=expected_y, coordinate_system=None)


# %% straight lines between two points


def test_a_line_segment_cannot_be_instantiated():
    """
    The segment interface needs an implementation that binds its point and line types.
    """
    with pytest.raises(InterfaceInstantiationError):
        LineSegment()


def test_a_line_segment_lies_on_the_line_from_its_start_to_its_end():
    """
    A segment's line passes through its start along the vector to its end, reaching the
    end at position 1.
    """
    start = ScalablePlanePoint(x=1.0, y=2.0, coordinate_system=None)
    end = ScalablePlanePoint(x=4.0, y=6.0, coordinate_system=None)

    line = StoredScalablePlaneLineSegment(start=start, end=end).line

    assert (line, line.point_at_position(position_along_line=1.0)) == (
        StoredScalablePlaneLine(point=start, direction=end - start),
        end,
    )


# %% the inner product of a Euclidean space


def test_a_vectors_length_is_the_square_root_of_its_inner_product_with_itself():
    """
    A vector is as long as the norm its inner product with itself induces.
    """
    vector = Vector3Dimensional(3, 4, 0)
    assert float(vector.norm()) == float(vector.dot(vector)) ** 0.5 == 5.0


def test_a_euclidean_space_multiplies_the_coordinates_of_a_cartesian_frame():
    """
    In a Cartesian frame, whose Gram matrix is the identity, the inner product is the
    sum of the products of the coordinates.
    """
    product = EuclideanSpace.inner_product(
        first=FloatCoordinateVector3Dimensional(x=1, y=2, z=3, coordinate_system=None),
        second=FloatCoordinateVector3Dimensional(x=4, y=5, z=6, coordinate_system=None),
    )
    assert product == 32.0


def test_a_euclidean_space_measures_a_frame_with_long_axes_by_its_gram_matrix():
    """
    One unit along each axis of a frame whose axes are two and three units long is the
    diagonal of a two by three rectangle.
    """
    vector = StretchedAxesVector(x=1.0, y=1.0, coordinate_system=None)

    assert vector.norm() == math.sqrt(
        sum(entry for row in StretchedAxesFrame.gram_matrix for entry in row)
    )


def test_a_symbolic_vectors_inner_product_is_its_euclidean_spaces():
    """
    A symbolic vector computes the same inner product its Euclidean space does.
    """
    first, second = Vector3Dimensional(1, 0, 2), Vector3Dimensional(3, 1, 1)
    assert float(first.dot(second)) == float(
        EuclideanSpace.inner_product(first=first, second=second)
    )


@pytest.mark.parametrize("product_name", ["inner_product", "cross_product"])
def test_a_frame_computes_no_product_of_vectors(product_name):
    """
    A frame only supplies how its axes measure; products of vectors are not its.
    """
    assert not any(
        product_name in vars(frame_kind)
        for frame_kind in CartesianFrame3Dimensional.__mro__
    )


def test_vectors_of_two_frames_have_no_inner_product():
    first = Vector3Dimensional(1, 0, 0, reference_frame=NamedFrame(name="table"))
    second = Vector3Dimensional(1, 0, 0, reference_frame=NamedFrame(name="gripper"))
    with pytest.raises(ReferenceFrameMismatchError):
        first.dot(second)


def test_the_distance_between_two_points_is_the_length_of_their_difference():
    """
    Two points are as far apart as the vector between them is long.
    """
    first, second = Point3Dimensional(4, 6, 3), Point3Dimensional(1, 2, 3)
    assert (
        float(first.euclidean_distance(second)) == float((first - second).norm()) == 5.0
    )


# %% the cross product of three-dimensional vectors


def test_two_three_dimensional_vectors_are_crossed_right_handed():
    """
    The cross product of two three-dimensional vectors is the right-handed one.
    """
    first, second = (1.0, 2.0, 3.0), (4.0, 5.0, 6.0)

    product = Vector3Dimensional(*first).cross(other=Vector3Dimensional(*second))

    assert [float(coordinate) for coordinate in product.coordinates] == list(
        np.cross(first, second)
    )


def test_the_cross_product_is_given_in_the_frame_of_the_crossed_vectors():
    """
    Crossing two vectors of one frame gives a vector of that frame.
    """
    table = NamedFrame(name="table")

    product = Vector3Dimensional(1, 0, 0, reference_frame=table).cross(
        other=Vector3Dimensional(0, 1, 0, reference_frame=table)
    )

    assert product.reference_frame is table


@pytest.mark.parametrize(
    "vector_type", [Vector, Vector2Dimensional, ScalablePlaneVector]
)
def test_only_a_three_dimensional_vector_has_a_cross_product(vector_type):
    """
    A cross product exists only in three dimensions, so no other vector has one.
    """
    assert not hasattr(vector_type, "cross")


def test_vectors_of_two_frames_have_no_cross_product():
    """
    Crossing vectors of two frames is refused, as their inner product is.
    """
    first = Vector3Dimensional(1, 0, 0, reference_frame=NamedFrame(name="table"))
    second = Vector3Dimensional(0, 1, 0, reference_frame=NamedFrame(name="gripper"))

    with pytest.raises(ReferenceFrameMismatchError):
        first.cross(other=second)
