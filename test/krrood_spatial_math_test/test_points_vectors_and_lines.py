"""
The point, vector and line interfaces of an affine space, and the stored mimics that
implement them.
"""

from __future__ import annotations

from dataclasses import fields

import pytest

from krrood.patterns.exceptions import InterfaceInstantiationError
from krrood_spatial_math.cartesian_frame_3_dimensional import CartesianFrame3Dimensional
from krrood_spatial_math.coordinate_system import (
    AffineFrame,
    AffineFrameT,
    CartesianFrame,
    CartesianFrameT,
    CoordinateRepresentation,
    CoordinateT,
)
from krrood_spatial_math.exceptions import ReferenceFrameMismatchError
from krrood_spatial_math.line import Line
from krrood_spatial_math.point import MetricPoint, Point, PointT
from krrood_spatial_math.spatial_types import (
    CartesianPoint3Dimensional,
    Vector3Dimensional,
)
from krrood_spatial_math.vector import MetricVector, Vector, VectorT

from .dataset.frames import NamedFrame
from .dataset.spaces import (
    ScalablePlaneFrame,
    StoredScalablePlaneLine,
    StoredScalablePlanePoint,
    StoredScalablePlaneVector,
)

# %% points and vectors given in an affine frame


@pytest.mark.parametrize("affine_value", [Point, Vector])
def test_a_point_or_vector_cannot_be_instantiated(affine_value):
    with pytest.raises(InterfaceInstantiationError):
        affine_value()


@pytest.mark.parametrize("affine_value", [Point, MetricPoint, Vector, MetricVector])
def test_a_point_or_vector_is_a_coordinate_representation(affine_value):
    assert issubclass(affine_value, CoordinateRepresentation)


def test_a_metric_point_is_a_point():
    assert issubclass(MetricPoint, Point)


def test_a_metric_vector_is_a_vector():
    assert issubclass(MetricVector, Vector)


def test_a_point_is_not_a_vector():
    assert not issubclass(Point, Vector)


def test_a_point_is_moved_by_a_vector():
    """
    Adding a vector shifts each coordinate of the point by the vector's.
    """
    point = StoredScalablePlanePoint(x=1.0, y=2.0, coordinate_system=None)
    vector = StoredScalablePlaneVector(x=3.0, y=4.0, coordinate_system=None)

    assert point + vector == StoredScalablePlanePoint(
        x=4.0, y=6.0, coordinate_system=None
    )


def test_a_point_is_given_in_an_affine_frame():
    """
    A point's frame may be any affine frame, and it is moved by a vector of that frame.
    """
    frame_parameter, _, vector_parameter = Point.__parameters__

    assert frame_parameter.__bound__ is AffineFrame
    assert vector_parameter.__bound__ is Vector


def test_a_metric_point_is_given_in_a_cartesian_frame():
    """
    Only a Cartesian frame measures the distances a metric point has.
    """
    frame_parameter, _, _ = MetricPoint.__parameters__

    assert frame_parameter.__bound__ is CartesianFrame


def test_a_point_reports_the_kind_of_frame_it_is_given_in():
    assert (
        StoredScalablePlanePoint.get_type_of_generic_parameter(parameter=AffineFrameT)
        is ScalablePlaneFrame
    )


def test_a_point_reports_what_its_coordinates_are_stored_as():
    """
    The coordinate type a point binds is read back from the class, without an instance.
    """
    assert (
        StoredScalablePlanePoint.get_type_of_generic_parameter(parameter=CoordinateT)
        is float
    )


def test_a_point_reports_the_kind_of_vector_it_is_moved_by():
    assert (
        StoredScalablePlanePoint.get_type_of_generic_parameter(parameter=VectorT)
        is StoredScalablePlaneVector
    )


def test_a_points_fields_are_narrowed_to_the_bound_types():
    point_types = {
        point_field.name: point_field.type
        for point_field in fields(StoredScalablePlanePoint)
    }

    assert point_types == {
        "x": float,
        "y": float,
        "coordinate_system": ScalablePlaneFrame | None,
    }


# %% how points and vectors combine


def test_the_difference_of_two_points_is_the_vector_between_them():
    """
    Subtracting points gives the vector type the point binds, not another point.
    """
    start = StoredScalablePlanePoint(x=1.0, y=2.0, coordinate_system=None)
    end = StoredScalablePlanePoint(x=4.0, y=6.0, coordinate_system=None)

    assert end - start == StoredScalablePlaneVector(
        x=3.0, y=4.0, coordinate_system=None
    )


def test_a_point_moved_by_the_difference_to_another_point_reaches_it():
    """
    Point subtraction and point-vector addition undo each other.
    """
    start = StoredScalablePlanePoint(x=1.0, y=2.0, coordinate_system=None)
    end = StoredScalablePlanePoint(x=4.0, y=6.0, coordinate_system=None)

    assert start + (end - start) == end


# %% straight lines given by a point and a direction


def test_a_line_cannot_be_instantiated():
    with pytest.raises(InterfaceInstantiationError):
        Line()


def test_a_line_runs_through_a_point_along_a_vector():
    assert Line.__parameters__ == (PointT, VectorT)
    assert PointT.__bound__ is Point


def test_a_lines_fields_are_narrowed_to_the_bound_types():
    line_types = {
        line_field.name: line_field.type
        for line_field in fields(StoredScalablePlaneLine)
    }

    assert line_types == {
        "point": StoredScalablePlanePoint,
        "direction": StoredScalablePlaneVector,
    }


# %% the metric of a Cartesian frame kind


def test_a_cartesian_frame_kind_measures_a_vectors_length_without_a_frame():
    assert (
        float(CartesianFrame3Dimensional.norm(vector=Vector3Dimensional(3, 4, 0)))
        == 5.0
    )


def test_a_cartesian_frame_kind_measures_the_angle_between_two_vectors():
    product = CartesianFrame3Dimensional.inner_product(
        first=Vector3Dimensional(1, 2, 3), second=Vector3Dimensional(4, 5, 6)
    )
    assert float(product) == 32.0


def test_a_cartesian_frame_kind_measures_the_distance_between_two_points():
    distance = CartesianFrame3Dimensional.distance(
        first=CartesianPoint3Dimensional(1, 2, 3),
        second=CartesianPoint3Dimensional(4, 6, 3),
    )
    assert float(distance) == 5.0


def test_a_vectors_length_is_the_one_its_frame_kind_measures():
    vector = Vector3Dimensional(1, 2, 2)
    assert float(vector.norm()) == float(CartesianFrame3Dimensional.norm(vector=vector))


def test_a_vectors_inner_product_is_the_one_its_frame_kind_measures():
    first, second = Vector3Dimensional(1, 0, 2), Vector3Dimensional(3, 1, 1)
    assert float(first.dot(second)) == float(
        CartesianFrame3Dimensional.inner_product(first=first, second=second)
    )


def test_a_points_distance_is_the_one_its_frame_kind_measures():
    first, second = (
        CartesianPoint3Dimensional(0, 0, 0),
        CartesianPoint3Dimensional(1, 2, 2),
    )
    assert float(first.euclidean_distance(second)) == float(
        CartesianFrame3Dimensional.distance(first=first, second=second)
    )


def test_vectors_of_two_frames_have_no_inner_product():
    first = Vector3Dimensional(1, 0, 0, reference_frame=NamedFrame(name="table"))
    second = Vector3Dimensional(1, 0, 0, reference_frame=NamedFrame(name="gripper"))
    with pytest.raises(ReferenceFrameMismatchError):
        first.dot(second)
