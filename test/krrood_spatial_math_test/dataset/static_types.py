"""
The types a static type checker reads for spatial values, stated with
:func:`typing_extensions.assert_type`, which a type checker reports where the read type
differs.

Nothing here runs: the functions are only read by the type checker.
"""

from __future__ import annotations

from typing_extensions import assert_type

from krrood.symbolic_math import symbolic_math as sm
from krrood_spatial_math.coordinate_system import (
    AffineCoordinateChange,
    CoordinateSystem,
    FrameAxis,
)
from krrood_spatial_math.spatial_types import (
    HomogeneousTransformationMatrix3Dimensional,
    Point3Dimensional,
    Vector3Dimensional,
)
from krrood_spatial_math.spatial_types_2_dimensional import (
    Point2Dimensional,
    Vector2Dimensional,
)

from .spaces import (
    AxisScalingChange,
    OpenCoordinateScalablePlanePoint,
    OpenCoordinateScalablePlaneVector,
    ScalablePlaneFrame,
    ScalablePlanePoint,
    ScalablePlaneVector,
)

# %% values bound to their own kind of frame


def check_point_and_vector_arithmetic(
    *, point: ScalablePlanePoint, vector: ScalablePlaneVector
) -> None:
    """
    The difference of two points is the point type's vector type, and a point moved by a
    vector, like a sum or multiple of vectors, keeps its type.
    """
    assert_type(point - point, ScalablePlaneVector)
    assert_type(point + vector, ScalablePlanePoint)
    assert_type(vector + vector, ScalablePlaneVector)
    assert_type(vector - vector, ScalablePlaneVector)
    assert_type(vector * 2.0, ScalablePlaneVector)


def check_values_moved_by_a_change(
    *,
    point: ScalablePlanePoint,
    vector: ScalablePlaneVector,
    change: AxisScalingChange,
) -> None:
    """
    A moved value keeps its type, and a change inverts and composes to its own type.
    """
    assert_type(point.apply_transformation(transformation=change), ScalablePlanePoint)
    assert_type(vector.apply_transformation(transformation=change), ScalablePlaneVector)
    assert_type(change.inverse(), AxisScalingChange)
    assert_type(change @ change, AxisScalingChange)


def check_frame_of_a_vector(*, vector: ScalablePlaneVector) -> None:
    """
    A vector binds its kind of frame, so its coordinate system is read as that kind.
    """
    assert_type(vector.coordinate_system, ScalablePlaneFrame | None)


def check_frame_kind_of_a_value_type(*, vector: ScalablePlaneVector) -> None:
    """
    A value type's kind of frame is read as the kind it binds, on the class and on an
    instance.
    """
    assert_type(ScalablePlaneVector.frame_kind, type[ScalablePlaneFrame])
    assert_type(vector.frame_kind, type[ScalablePlaneFrame])


def check_axes_of_a_frame_kind() -> None:
    """
    A kind of frame's axes are read through its class property.
    """
    assert_type(ScalablePlaneFrame.axes, tuple[FrameAxis, ...])


# %% values whose coordinate type is left to their user


def check_values_with_symbolic_coordinates(
    *,
    point: OpenCoordinateScalablePlanePoint[sm.Scalar],
    vector: OpenCoordinateScalablePlaneVector[sm.Scalar],
    change: AffineCoordinateChange[ScalablePlaneFrame, sm.Scalar],
) -> None:
    """
    A value type that leaves its coordinate type open, given symbolic coordinates, reads
    them as symbols, subtracts into its vector type of symbols, and is moved by a change
    of symbolic coordinates.
    """
    assert_type(point.x, sm.Scalar)
    assert_type(vector.y, sm.Scalar)
    assert_type(point - point, OpenCoordinateScalablePlaneVector[sm.Scalar])
    assert_type(
        point.apply_transformation(transformation=change),
        OpenCoordinateScalablePlanePoint[sm.Scalar],
    )


# %% symbolic values


def check_symbolic_point_and_vector_arithmetic(
    *,
    point: Point3Dimensional,
    vector: Vector3Dimensional,
    planar_point: Point2Dimensional,
) -> None:
    """
    The symbolic points and vectors read their results through the vector type a point
    binds, in both dimensions.
    """
    assert_type(point - point, Vector3Dimensional)
    assert_type(point + vector, Point3Dimensional)
    assert_type(vector.cross(other=vector), Vector3Dimensional)
    assert_type(planar_point - planar_point, Vector2Dimensional)


def check_symbolic_point_moved_by_a_transformation(
    *,
    point: Point3Dimensional,
    transformation: HomogeneousTransformationMatrix3Dimensional,
) -> None:
    """
    A symbolic transformation is accepted by a point's
    :meth:`~krrood_spatial_math.coordinate_system.AxisCoordinates.apply_transformation`.
    """
    assert_type(
        point.apply_transformation(transformation=transformation), Point3Dimensional
    )


def check_coordinate_system_of_a_symbolic_value(*, point: Point3Dimensional) -> None:
    """
    A symbolic value's coordinate system is read as its bound, never as a type variable
    its class does not have.
    """
    assert_type(point.coordinate_system, CoordinateSystem | None)
