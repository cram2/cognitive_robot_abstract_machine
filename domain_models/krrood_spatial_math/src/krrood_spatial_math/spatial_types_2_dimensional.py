"""
The symbolic casadi spatial types on a plane: points, vectors and poses given in a
Cartesian frame on the x-y plane of a three-dimensional frame, the rigid changes
between such frames, and the projection of three-dimensional points onto a plane.
"""

from __future__ import annotations
from abc import ABC, abstractmethod
from dataclasses import dataclass
import casadi as ca
import numpy as np
from typing_extensions import Any, Self, TypeVar
import krrood.symbolic_math.symbolic_math as sm
from krrood.patterns.interface import Interface
from krrood.symbolic_math.exceptions import (
    WrongDimensionsError,
    UnsupportedOperationError,
)
from krrood_spatial_math.cartesian_frame_2_dimensional import CartesianFrame2Dimensional
from krrood_spatial_math.cartesian_frame_3_dimensional import (
    CartesianFrame3Dimensional,
    EmbeddedPlaneRelation,
)
from krrood_spatial_math.coordinate_system import (
    CoordinateRepresentation,
)
from krrood_spatial_math.point import Point
from krrood_spatial_math.vector import Vector
from krrood_spatial_math.embedded_plane import EmbeddedPlane
from krrood_spatial_math.exceptions import (
    FramesOnDifferentPlanesError,
    MirroredPlaneFramesError,
    ReferenceFrameMismatchError,
    XAxisNormalToPlaneError,
)
from krrood_spatial_math.space_map import OrthogonalProjection
from krrood_spatial_math.spatial_types import (
    Point3Dimensional,
    HomogeneousTransformationMatrix3Dimensional,
    Line3Dimensional,
    Pose3Dimensional,
    Quaternion,
    RotationMatrix,
    SpatialFrameKey,
    SymbolicSpatialChange,
    SymbolicSpatialType,
    Vector3Dimensional,
)


# %% how a planar change acts on a value


class PlanarSpecialEuclideanGroupAction(Interface):
    """
    A value or change on a plane the planar rigid motion group SE(2) acts on: a
    :class:`HomogeneousTransformationMatrix2Dimensional` moves it, giving one of the same type.
    """

    @abstractmethod
    def apply_planar_transformation(
        self, *, transformation: HomogeneousTransformationMatrix2Dimensional
    ) -> Self:
        """
        :param transformation: A transformation whose child frame is this value's
            reference frame.
        :return: This value, on the x-y plane of the reference frame of
            ``transformation``.
        """


GenericPlanarSpecialEuclideanGroupAction = TypeVar(
    "GenericPlanarSpecialEuclideanGroupAction", bound=PlanarSpecialEuclideanGroupAction
)
"""
The kind of value or change on a plane a planar transformation is applied to.
"""


# %% values on a plane


@dataclass(eq=False, repr=False, kw_only=True)
class SymbolicSpatialValue2Dimensional(
    SymbolicSpatialType,
    CoordinateRepresentation[CartesianFrame2Dimensional],
    ABC,
):
    """
    A value, as a symbolic matrix, such as a point or a pose, on the x-y plane of its
    :attr:`~SymbolicSpatialType.reference_frame`, given in the
    :class:`~krrood_spatial_math.cartesian_frame_2_dimensional.CartesianFrame2Dimensional`
    on that plane.
    """

    @property
    def coordinate_system(self) -> CartesianFrame2Dimensional | None:
        """
        The frame on the x-y plane of :attr:`~SymbolicSpatialType.reference_frame`;
        ``None`` for a value without a reference frame.
        """
        if self.reference_frame is None:
            return None
        return CartesianFrame2Dimensional(frame_3_dimensional=self.reference_frame)


@dataclass(eq=False, init=False, repr=False)
class Vector2Dimensional(
    SymbolicSpatialValue2Dimensional,
    Vector[CartesianFrame2Dimensional, sm.Scalar],
    PlanarSpecialEuclideanGroupAction,
):
    """
    A translation within the x-y plane of its reference frame, given in the
    :class:`~krrood_spatial_math.cartesian_frame_2_dimensional.CartesianFrame2Dimensional` of that frame.

    Stored as a bare 2×1 symbolic vector ``[x, y]``, like :class:`Point2Dimensional`.
    """

    def __init__(
        self,
        x: sm.ScalarData = 0,
        y: sm.ScalarData = 0,
        reference_frame: CartesianFrame3Dimensional | None = None,
    ):
        """
        :param x: The amount along the x-axis.
        :param y: The amount along the y-axis.
        :param reference_frame: The frame on whose x-y plane the vector lies.
        """
        self.casadi_sx = sm.to_sx([x, y])
        self.reference_frame = reference_frame
        super().__post_init__()

    @property
    def x(self) -> sm.Scalar:
        return self[0]

    @x.setter
    def x(self, value: sm.ScalarData):
        self[0] = value

    @property
    def y(self) -> sm.Scalar:
        return self[1]

    @y.setter
    def y(self, value: sm.ScalarData):
        self[1] = value

    @property
    def coordinates(self) -> tuple[sm.Scalar, sm.Scalar]:
        return self.x, self.y

    def _verify_type(self):
        if self.shape != (2, 1):
            raise WrongDimensionsError(
                expected_dimensions=(2, 1), actual_dimensions=self.shape
            )

    @classmethod
    def _from_constant_json(cls, data: dict[str, Any], **kwargs) -> Self:
        reference_frame = cls._parse_optional_frame_from_json(
            data, frame=SpatialFrameKey.REFERENCE, **kwargs
        )
        x, y = data["data"][:2]
        return cls(x=x, y=y, reference_frame=reference_frame)

    def _constant_to_json(self) -> dict[str, Any]:
        return {
            "data": self.to_np().tolist(),
        }

    def transform(
        self,
        target_frame_T_reference_frame: HomogeneousTransformationMatrix3Dimensional,
    ) -> Vector2Dimensional:
        """
        This vector re-expressed in ``target_frame_T_reference_frame``'s reference frame.

        .. warning::
            Between frames whose x-y planes are not parallel this is not a change of
            coordinates but a projection: the vector is turned in 3D space and projected
            onto the target plane. The height the rotation produced is lost.

        :param target_frame_T_reference_frame: The transformation from this vector's
            reference frame to the frame it should be expressed in.
        :return: The vector in the transformation's reference frame.
        """
        turned = target_frame_T_reference_frame @ Vector3Dimensional(
            self.x, self.y, 0, reference_frame=self.reference_frame
        )
        return Vector2Dimensional(
            turned.x, turned.y, reference_frame=turned.reference_frame
        )

    def apply_planar_transformation(
        self, *, transformation: HomogeneousTransformationMatrix2Dimensional
    ) -> Vector2Dimensional:
        turned = ca.mtimes(transformation.casadi_sx, ca.vertcat(self.casadi_sx, 0))
        return Vector2Dimensional(
            turned[0], turned[1], reference_frame=transformation.reference_frame
        )

    def __add__(self, other: Vector2Dimensional) -> Vector2Dimensional:
        """
        :raises ReferenceFrameMismatchError: If ``other`` is given in another frame.
        """
        if not isinstance(other, Vector2Dimensional):
            return NotImplemented
        self.ensure_same_reference_frame(other, "the added Vector2Dimensional")
        return Vector2Dimensional(
            self.x + other.x, self.y + other.y, reference_frame=self.reference_frame
        )

    def __sub__(self, other: Vector2Dimensional) -> Vector2Dimensional:
        """
        :raises ReferenceFrameMismatchError: If ``other`` is given in another frame.
        """
        if not isinstance(other, Vector2Dimensional):
            return NotImplemented
        self.ensure_same_reference_frame(other, "the subtracted Vector2Dimensional")
        return Vector2Dimensional(
            self.x - other.x, self.y - other.y, reference_frame=self.reference_frame
        )

    def __mul__(self, factor: sm.ScalarData) -> Vector2Dimensional:
        if not isinstance(factor, sm.ScalarData):
            return NotImplemented
        return Vector2Dimensional(
            self.x * factor, self.y * factor, reference_frame=self.reference_frame
        )


@dataclass(eq=False, init=False, repr=False)
class Point2Dimensional(
    SymbolicSpatialValue2Dimensional,
    Point[CartesianFrame2Dimensional, Vector2Dimensional, sm.Scalar],
    PlanarSpecialEuclideanGroupAction,
):
    """
    Represents a 2D point with reference frame handling.

    Stored as a bare 2×1 symbolic vector ``[x, y]``, with no homogeneous augmentation --
    the same convention :class:`Pose2Dimensional` uses. The point lies on its reference frame's
    own x-y plane and has no height; it relates to 3D space only through an
    :class:`EmbeddedPlaneProjection` onto that plane, which :meth:`from_point_3_dimensional`,
    :meth:`from_pose` and :meth:`to_point_3_dimensional` apply.
    """

    def __init__(
        self,
        x: sm.ScalarData = 0,
        y: sm.ScalarData = 0,
        reference_frame: CartesianFrame3Dimensional | None = None,
    ):
        """
        :param x: X-coordinate of the point. Defaults to 0.
        :param y: Y-coordinate of the point. Defaults to 0.
        :param reference_frame:
        """
        self.casadi_sx = sm.to_sx([x, y])
        self.reference_frame = reference_frame
        super().__post_init__()

    @property
    def x(self) -> sm.Scalar:
        return self[0]

    @x.setter
    def x(self, value: sm.ScalarData):
        self[0] = value

    @property
    def y(self) -> sm.Scalar:
        return self[1]

    @y.setter
    def y(self, value: sm.ScalarData):
        self[1] = value

    @property
    def coordinates(self) -> tuple[sm.Scalar, sm.Scalar]:
        return self.x, self.y

    def _verify_type(self):
        if self.shape != (2, 1):
            raise WrongDimensionsError(
                expected_dimensions=(2, 1), actual_dimensions=self.shape
            )

    @classmethod
    def from_point_3_dimensional(
        cls,
        point: Point3Dimensional,
        plane_frame: CartesianFrame3Dimensional | None = None,
    ) -> Point2Dimensional:
        """
        The :meth:`EmbeddedPlaneProjection.project` of ``point`` onto the x-y plane of
        ``plane_frame``.

        :param point: The point to project.
        :param plane_frame: The frame whose x-y plane to project onto. Defaults to
            ``point``'s reference frame.
        :return: The point on the plane directly below or above ``point``.
        """
        frame = plane_frame if plane_frame is not None else point.reference_frame
        return EmbeddedPlaneProjection(plane_frame=frame).project(point)

    @classmethod
    def from_pose(
        cls,
        pose: Pose3Dimensional,
        plane_frame: CartesianFrame3Dimensional | None = None,
    ) -> Point2Dimensional:
        """
        The :meth:`from_point_3_dimensional` of ``pose``'s position.

        :param pose: The pose whose position to project.
        :param plane_frame: The frame whose x-y plane to project onto. Defaults to
            ``pose``'s reference frame.
        :return: The point on the plane directly below or above ``pose``.
        """
        return cls.from_point_3_dimensional(pose.position, plane_frame)

    def to_point_3_dimensional(self, height: sm.ScalarData = 0) -> Point3Dimensional:
        """
        The :meth:`EmbeddedPlaneProjection.unproject` of this point into its own reference
        frame.

        :param height: The z-coordinate the point is placed at.
        :return: The 3D point at ``height`` above this point.
        """
        return EmbeddedPlaneProjection(plane_frame=self.reference_frame).unproject(
            self, height
        )

    @classmethod
    def _from_constant_json(cls, data: dict[str, Any], **kwargs) -> Self:
        reference_frame = cls._parse_optional_frame_from_json(
            data, frame=SpatialFrameKey.REFERENCE, **kwargs
        )
        x, y = data["data"][:2]
        return cls(x=x, y=y, reference_frame=reference_frame)

    def _constant_to_json(self) -> dict[str, Any]:
        return {
            "data": self.to_np().tolist(),
        }

    def transform(
        self,
        target_frame_T_reference_frame: HomogeneousTransformationMatrix3Dimensional,
    ) -> Point2Dimensional:
        """
        This point re-expressed in ``target_frame_T_reference_frame``'s reference frame.

        .. warning::
            Between frames whose x-y planes are not parallel this is not a change of
            coordinates but a projection: the point is placed at height 0 through an
            :class:`EmbeddedPlaneProjection`, moved, and projected onto the target plane.
            The height the rotation produced is lost.

        :param target_frame_T_reference_frame: The transformation from this point's
            reference frame to the frame it should be expressed in.
        :return: The point in the transformation's reference frame.
        """
        return Point2Dimensional.from_point_3_dimensional(
            target_frame_T_reference_frame @ self.to_point_3_dimensional()
        )

    def apply_planar_transformation(
        self, *, transformation: HomogeneousTransformationMatrix2Dimensional
    ) -> Point2Dimensional:
        moved = ca.mtimes(transformation.casadi_sx, ca.vertcat(self.casadi_sx, 1))
        return Point2Dimensional(
            moved[0], moved[1], reference_frame=transformation.reference_frame
        )

    def __add__(self, vector: Vector2Dimensional) -> Point2Dimensional:
        """
        :raises ReferenceFrameMismatchError: If ``vector`` is given in another frame.
        """
        if not isinstance(vector, Vector2Dimensional):
            return NotImplemented
        self.ensure_same_reference_frame(vector, "the added Vector2Dimensional")
        return Point2Dimensional(
            self.x + vector.x, self.y + vector.y, reference_frame=self.reference_frame
        )

    def __sub__(self, other: Point2Dimensional) -> Vector2Dimensional:
        """
        :raises ReferenceFrameMismatchError: If ``other`` is given in another frame.
        """
        if not isinstance(other, Point2Dimensional):
            return NotImplemented
        self.ensure_same_reference_frame(other, "the subtracted Point2Dimensional")
        return Vector2Dimensional(
            self.x - other.x, self.y - other.y, reference_frame=self.reference_frame
        )

    def __hash__(self):
        if self.is_constant():
            return hash((*self.to_np().tolist(), self.reference_frame))
        return super().__hash__()


# %% projecting three-dimensional points onto a plane


@dataclass(kw_only=True)
class EmbeddedPlaneProjection(
    OrthogonalProjection[
        Point3Dimensional,
        Point2Dimensional,
        Line3Dimensional,
        sm.ScalarData,
    ]
):
    """
    The orthogonal projection of 3D space onto the
    :class:`~krrood_spatial_math.embedded_plane.EmbeddedPlane` of :attr:`plane_frame`.

    A point's :meth:`projection_line` is the vertical line through it, and its
    position along that line is its height above the plane.
    """

    plane_frame: CartesianFrame3Dimensional | None
    """
    The frame whose x-y plane points are projected onto.

    ``None`` is the plane of points that have no reference frame. Their coordinates
    are taken as they are, so such a projection only relates points without a
    reference frame to each other.
    """

    def project(self, point: Point3Dimensional) -> Point2Dimensional:
        """
        :raises ReferenceFrameMismatchError: If exactly one of ``point`` and this
            projection has no reference frame.
        """
        if point.reference_frame is not self.plane_frame:
            if point.reference_frame is None or self.plane_frame is None:
                raise ReferenceFrameMismatchError(
                    expected_frame=self.plane_frame,
                    actual_frame=point.reference_frame,
                    context="the projected Point3Dimensional",
                )
            point = self.plane_frame.express_in_own_frame(value=point)
        return Point2Dimensional(point.x, point.y, reference_frame=self.plane_frame)

    @property
    def plane_normal(self) -> Vector3Dimensional:
        """
        The :attr:`~krrood_spatial_math.embedded_plane.EmbeddedPlane.normal` of the plane
        of :attr:`plane_frame`; the z-axis of no frame when there is none.
        """
        if self.plane_frame is None:
            return Vector3Dimensional.Z(reference_frame=None)
        return EmbeddedPlane(frame_3_dimensional=self.plane_frame).normal

    def embed_planar_point(self, *, point: Point2Dimensional) -> Point3Dimensional:
        """
        :raises ReferenceFrameMismatchError: If ``point`` lies on another frame's plane.
        """
        if point.reference_frame is not self.plane_frame:
            raise ReferenceFrameMismatchError(
                expected_frame=self.plane_frame,
                actual_frame=point.reference_frame,
                context="the unprojected Point2Dimensional",
            )
        return Point3Dimensional(point.x, point.y, 0, reference_frame=self.plane_frame)


# %% poses on a plane


@dataclass(eq=False, init=False, repr=False)
class Pose2Dimensional(
    SymbolicSpatialValue2Dimensional,
    PlanarSpecialEuclideanGroupAction,
):
    """
    Represents a 2D pose consisting of an x coordinate, a y coordinate, and a yaw angle.

    Internally stored as a 3×1 symbolic vector ``[x, y, yaw]``. It lies on its reference
    frame's x-y plane and has no height or tilt of its own, so it carries no ``z``,
    ``roll`` or ``pitch``. Whenever 3D calculations are required, use :attr:`pose` to
    obtain the equivalent 3D :class:`Pose3Dimensional`.
    """

    def __init__(
        self,
        position: Point2Dimensional | None = None,
        yaw: sm.ScalarData = 0,
        reference_frame: CartesianFrame3Dimensional | None = None,
    ) -> None:
        """
        :param position: Where the pose is. Defaults to the origin.
        :param yaw: Which way the pose faces. Defaults to 0.
        :param reference_frame: The reference frame. Defaults to *position*'s.
        """
        if position is None:
            position = Point2Dimensional()
        self.casadi_sx = sm.to_sx([position.x, position.y, yaw])
        self.reference_frame = (
            reference_frame if reference_frame is not None else position.reference_frame
        )
        super().__post_init__()

    def _verify_type(self):
        if self.shape != (3, 1):
            raise WrongDimensionsError(
                expected_dimensions=(3, 1), actual_dimensions=self.shape
            )

    # %% Properties

    @property
    def x(self) -> sm.Scalar:
        return self[0]

    @x.setter
    def x(self, value: sm.ScalarData):
        self[0] = value

    @property
    def y(self) -> sm.Scalar:
        return self[1]

    @y.setter
    def y(self, value: sm.ScalarData):
        self[1] = value

    @property
    def yaw(self) -> sm.Scalar:
        return self[2]

    @yaw.setter
    def yaw(self, value: sm.ScalarData):
        self[2] = value

    # %% Conversion to 3D

    @property
    def pose(self) -> Pose3Dimensional:
        """
        Convert to a 3D :class:`Pose3Dimensional` with z=0, roll=0, pitch=0.
        """
        return Pose3Dimensional.from_xyz_rpy(
            x=self.x,
            y=self.y,
            z=0,
            roll=0,
            pitch=0,
            yaw=self.yaw,
            reference_frame=self.reference_frame,
        )

    # %% Pose3Dimensional-like interface (delegates to pose)

    @property
    def position(self) -> Point2Dimensional:
        """
        :return: The position this pose is composed of, dropping the bearing.
        """
        return Point2Dimensional(self.x, self.y, reference_frame=self.reference_frame)

    @property
    def orientation(self) -> Quaternion:
        return self.quaternion

    @property
    def quaternion(self) -> Quaternion:
        return self.pose.quaternion

    @property
    def rotation_matrix(self) -> RotationMatrix:
        """
        :return: The rotation around the z-axis by the yaw of this pose.
        """
        return self.pose.rotation_matrix

    @property
    def homogeneous_matrix(self) -> HomogeneousTransformationMatrix3Dimensional:
        """
        :return: The transformation that moves the reference frame to this pose.
        """
        return self.pose.homogeneous_matrix

    # %% Factory methods

    @classmethod
    def from_pose(
        cls,
        pose: Pose3Dimensional,
        plane_frame: CartesianFrame3Dimensional | None = None,
    ) -> Pose2Dimensional:
        """
        Flatten a 3D Pose3Dimensional onto the x-y plane of a frame: its position is projected onto
        the plane, and it faces the way its x-axis, projected onto the plane, points.

        That heading, ``atan2(r21, r11)`` of the pose's rotation matrix, equals the yaw
        of its ZYX Euler angles.

        :param pose: The pose to flatten.
        :param plane_frame: The frame whose x-y plane to flatten the pose onto.
            Defaults to ``pose``'s reference frame; any other frame must belong to the
            same world as ``pose``'s.
        :return: The Pose2Dimensional on the x-y plane of ``plane_frame``.
        :raises XAxisNormalToPlaneError: If the pose's x-axis is constant and normal to
            the plane.
        """
        frame = plane_frame if plane_frame is not None else pose.reference_frame
        if frame is not pose.reference_frame:
            pose = frame.express_in_own_frame(value=pose)
        rotation = pose.rotation_matrix
        x_axis_x, x_axis_y = rotation[0, 0], rotation[1, 0]
        if (
            x_axis_x.is_constant()
            and x_axis_y.is_constant()
            and np.allclose([x_axis_x.to_np(), x_axis_y.to_np()], 0)
        ):
            raise XAxisNormalToPlaneError(pose=pose)
        return cls.from_xy_yaw(
            x=pose.x,
            y=pose.y,
            yaw=sm.atan2(x_axis_y, x_axis_x),
            reference_frame=frame,
        )

    def transform(
        self,
        target_frame_T_reference_frame: HomogeneousTransformationMatrix3Dimensional,
    ) -> Pose2Dimensional:
        """
        This pose re-expressed in ``target_frame_T_reference_frame``'s reference frame.

        .. warning::
            A Pose2Dimensional carries no z, roll or pitch, so it is transformed as the pose flat
            on its own plane and the result is projected back onto the target frame's
            x-y plane. Transforming into a frame tilted against this one therefore loses
            the height and tilt the rotation produced.

        :param target_frame_T_reference_frame: The transformation from this pose's
            reference frame to the frame it should be expressed in.
        :return: The pose in the transformation's reference frame.
        """
        return Pose2Dimensional.from_pose(target_frame_T_reference_frame @ self.pose)

    def apply_planar_transformation(
        self, *, transformation: HomogeneousTransformationMatrix2Dimensional
    ) -> Pose2Dimensional:
        flat_pose = HomogeneousTransformationMatrix2Dimensional.from_xy_yaw(
            x=self.x, y=self.y, yaw=self.yaw
        )
        return flat_pose.apply_planar_transformation(transformation=transformation).pose

    @classmethod
    def from_xy_yaw(
        cls,
        x: sm.ScalarData = 0,
        y: sm.ScalarData = 0,
        yaw: sm.ScalarData = 0,
        reference_frame: CartesianFrame3Dimensional | None = None,
    ) -> Pose2Dimensional:
        """
        Create a Pose2Dimensional from its coordinates and bearing.

        :param x: The x-coordinate of the position.
        :param y: The y-coordinate of the position.
        :param yaw: The rotation around the z-axis.
        :param reference_frame: The reference frame.
        :return: The Pose2Dimensional instance.
        """
        return cls(
            position=Point2Dimensional(x=x, y=y),
            yaw=yaw,
            reference_frame=reference_frame,
        )

    # %% JSON serialization

    @classmethod
    def _from_constant_json(cls, data: dict[str, Any], **kwargs) -> Self:
        reference_frame = cls._parse_optional_frame_from_json(
            data, frame=SpatialFrameKey.REFERENCE, **kwargs
        )
        return cls.from_xy_yaw(*data["data"], reference_frame=reference_frame)

    def _constant_to_json(self) -> dict[str, Any]:
        return {
            "data": self.to_np().tolist(),
        }

    def __hash__(self):
        if self.is_constant():
            return hash((*self.to_np().tolist(), self.reference_frame))
        return super().__hash__()


# %% rigid changes between frames on one plane


@dataclass(eq=False, init=False, repr=False)
class HomogeneousTransformationMatrix2Dimensional(
    SymbolicSpatialChange[CartesianFrame2Dimensional],
    PlanarSpecialEuclideanGroupAction,
):
    """
    A 3x3 homogeneous matrix of SE(2): the rigid change from the
    :class:`~krrood_spatial_math.cartesian_frame_2_dimensional.CartesianFrame2Dimensional` of its :attr:`~SymbolicSpatialChange.child_frame` into that of its
    :attr:`~SymbolicSpatialType.reference_frame`, two frames on one plane.

    It is only a change of coordinates, never a value given in a frame: the value it
    corresponds to is a :class:`Pose2Dimensional`.

    .. warning::
        The matrix must be rigid: its upper left 2x2 block a rotation and its last row
        ``[0, 0, 1]``.
    """

    def __init__(
        self,
        data: sm.MatrixData | None = None,
        reference_frame: CartesianFrame3Dimensional | None = None,
        child_frame: CartesianFrame3Dimensional | None = None,
    ):
        """
        :param data: A rigid 3x3 homogeneous matrix. Defaults to the identity.
        :param reference_frame: The frame on whose plane the change ends.
        :param child_frame: The frame on whose plane the change starts.
        :raises FramesOnDifferentPlanesError: If both frames are given and span
            different planes.
        :raises MirroredPlaneFramesError: If both frames are given and span one plane
            with z-axes pointing in opposite directions.
        """
        if reference_frame is not None and child_frame is not None:
            self._ensure_rigid_change_on_one_plane(
                relation=reference_frame.compare_embedded_plane(frame=child_frame),
                reference_frame=reference_frame,
                child_frame=child_frame,
            )
        self.reference_frame = reference_frame
        self.child_frame = child_frame
        if data is None:
            self._casadi_sx = ca.SX.eye(3)
        else:
            self.casadi_sx = sm.to_sx(data)
        super().__post_init__()

    def _verify_type(self):
        if self.shape != (3, 3):
            raise WrongDimensionsError(
                expected_dimensions=(3, 3), actual_dimensions=self.shape
            )
        self[2, :] = [0, 0, 1]

    @property
    def source_coordinate_system(self) -> CartesianFrame2Dimensional | None:
        """
        The frame on the x-y plane of :attr:`~SymbolicSpatialChange.child_frame`;
        ``None`` for a change without a child frame.
        """
        if self.child_frame is None:
            return None
        return CartesianFrame2Dimensional(frame_3_dimensional=self.child_frame)

    @property
    def target_coordinate_system(self) -> CartesianFrame2Dimensional | None:
        """
        The frame on the x-y plane of :attr:`~SymbolicSpatialType.reference_frame`;
        ``None`` for a change without a reference frame.
        """
        if self.reference_frame is None:
            return None
        return CartesianFrame2Dimensional(frame_3_dimensional=self.reference_frame)

    # %% construction

    @classmethod
    def from_xy_yaw(
        cls,
        x: sm.ScalarData = 0,
        y: sm.ScalarData = 0,
        yaw: sm.ScalarData = 0,
        reference_frame: CartesianFrame3Dimensional | None = None,
        child_frame: CartesianFrame3Dimensional | None = None,
    ) -> HomogeneousTransformationMatrix2Dimensional:
        """
        :param x: The x-coordinate of the child frame's origin in the reference frame.
        :param y: The y-coordinate of the child frame's origin in the reference frame.
        :param yaw: The angle from the reference frame's x-axis to the child frame's,
            counterclockwise.
        :param reference_frame: The frame on whose plane the change ends.
        :param child_frame: The frame on whose plane the change starts.
        :return: The change that rotates by ``yaw`` and then shifts by ``(x, y)``.
        """
        cosine, sine = sm.cos(yaw), sm.sin(yaw)
        return cls(
            [[cosine, -sine, x], [sine, cosine, y], [0, 0, 1]],
            reference_frame=reference_frame,
            child_frame=child_frame,
        )

    @classmethod
    def from_transformation(
        cls, transformation: HomogeneousTransformationMatrix3Dimensional
    ) -> HomogeneousTransformationMatrix2Dimensional:
        """
        :param transformation: A change between two frames that span one x-y plane and
            whose z-axes point the same way.
        :return: The same change, between the frames on that plane.
        :raises FramesOnDifferentPlanesError: If the transformation is constant, or its
            frames are known, and its frames span different planes.
        :raises MirroredPlaneFramesError: If the transformation is constant, or its
            frames are known, and its frames span one plane with z-axes pointing in
            opposite directions.

        .. note::
            A transformation that is not constant, between frames whose planes are not
            known, is taken to be planar unchecked.
        """
        cls._ensure_rigid_change_on_one_plane(
            relation=transformation.embedded_plane_relation,
            reference_frame=transformation.reference_frame,
            child_frame=transformation.child_frame,
        )
        rotation = transformation.rotation_matrix
        return cls.from_xy_yaw(
            x=transformation.x,
            y=transformation.y,
            yaw=sm.atan2(rotation[1, 0], rotation[0, 0]),
            reference_frame=transformation.reference_frame,
            child_frame=transformation.child_frame,
        )

    @staticmethod
    def _ensure_rigid_change_on_one_plane(
        *,
        relation: EmbeddedPlaneRelation,
        reference_frame: CartesianFrame3Dimensional | None,
        child_frame: CartesianFrame3Dimensional | None,
    ) -> None:
        """
        :param relation: How the x-y plane of ``child_frame`` lies relative to that of
            ``reference_frame``.
        :param reference_frame: The frame on whose plane the change ends.
        :param child_frame: The frame on whose plane the change starts.
        :raises FramesOnDifferentPlanesError: If the frames span different planes.
        :raises MirroredPlaneFramesError: If the frames span one plane with z-axes
            pointing in opposite directions.
        """
        if relation is EmbeddedPlaneRelation.DIFFERENT_PLANES:
            raise FramesOnDifferentPlanesError(
                reference_frame=reference_frame, child_frame=child_frame
            )
        if relation is EmbeddedPlaneRelation.SAME_PLANE_OPPOSITE_NORMAL:
            raise MirroredPlaneFramesError(
                reference_frame=reference_frame, child_frame=child_frame
            )

    # %% motion

    @property
    def x(self) -> sm.Scalar:
        """
        The x-coordinate of the child frame's origin in the reference frame.
        """
        return self[0, 2]

    @property
    def y(self) -> sm.Scalar:
        """
        The y-coordinate of the child frame's origin in the reference frame.
        """
        return self[1, 2]

    @property
    def yaw(self) -> sm.Scalar:
        """
        The angle from the reference frame's x-axis to the child frame's,
        counterclockwise.
        """
        return sm.atan2(self[1, 0], self[0, 0])

    @property
    def pose(self) -> Pose2Dimensional:
        """
        :return: The pose this change moves its reference frame's plane frame to.
        """
        return Pose2Dimensional.from_xy_yaw(
            x=self.x, y=self.y, yaw=self.yaw, reference_frame=self.reference_frame
        )

    # %% acting on values and changes

    def __matmul__(
        self, other: GenericPlanarSpecialEuclideanGroupAction
    ) -> GenericPlanarSpecialEuclideanGroupAction:
        """
        :param other: A change into this change's child frame, or a point or pose given
            in it.
        :return: The change applied after ``other``, or the point or pose given in this
            change's reference frame.
        :raises UnsupportedOperationError: If this change does not act on ``other``.
        :raises ReferenceFrameMismatchError: If ``other`` is given in a frame other than
            this change's child frame.
        """
        if not isinstance(other, PlanarSpecialEuclideanGroupAction):
            raise UnsupportedOperationError(operation="@", left=self, right=other)
        self.ensure_operand_in_child_frame(other)
        return other.apply_planar_transformation(transformation=self)

    def apply_planar_transformation(
        self, *, transformation: HomogeneousTransformationMatrix2Dimensional
    ) -> HomogeneousTransformationMatrix2Dimensional:
        return HomogeneousTransformationMatrix2Dimensional(
            ca.mtimes(transformation.casadi_sx, self.casadi_sx),
            reference_frame=transformation.reference_frame,
            child_frame=self.child_frame,
        )

    def inverse(self) -> HomogeneousTransformationMatrix2Dimensional:
        inverse = HomogeneousTransformationMatrix2Dimensional(
            reference_frame=self.child_frame, child_frame=self.reference_frame
        )
        inverse[:2, :2] = self[:2, :2].T
        inverse[:2, 2] = (-inverse[:2, :2]).dot(other=self[:2, 2])
        return inverse

    # %% JSON serialization

    @classmethod
    def _from_constant_json(cls, data: dict[str, Any], **kwargs) -> Self:
        return cls.from_xy_yaw(
            *data["data"],
            reference_frame=cls._parse_optional_frame_from_json(
                data, frame=SpatialFrameKey.REFERENCE, **kwargs
            ),
            child_frame=cls._parse_optional_frame_from_json(
                data, frame=SpatialFrameKey.CHILD, **kwargs
            ),
        )

    def _constant_to_json(self) -> dict[str, Any]:
        return {
            "data": [
                float(self.x.to_np()[0]),
                float(self.y.to_np()[0]),
                float(self.yaw.to_np()[0]),
            ]
        }

    def __hash__(self):
        if self.is_constant():
            return hash(
                (
                    *self.to_np().flatten().tolist(),
                    self.reference_frame,
                    self.child_frame,
                )
            )
        return super().__hash__()
