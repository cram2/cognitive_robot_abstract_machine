from __future__ import annotations

from copy import deepcopy, copy
from dataclasses import dataclass, field

import casadi as ca
import numpy as np
from typing_extensions import (
    Any,
    TYPE_CHECKING,
    overload,
    Optional,
    Dict,
    Self,
    List,
    Union,
    Tuple,
    Callable,
    TypeVar,
)

import krrood.symbolic_math.symbolic_math as sm
from krrood.adapters.json_serializer import SubclassJSONSerializer, from_json, to_json
from krrood.symbolic_math.exceptions import (
    WrongDimensionsError,
    UnsupportedOperationError,
)
from krrood.symbolic_math.symbolic_math import Matrix, to_sx
from ..adapters.world_entity_kwargs_tracker import (
    WorldEntityWithIDKwargsTracker,
)
from ..exceptions import (
    SpatialTypesError,
    SpatialTypeNotJsonSerializable,
)

if TYPE_CHECKING:
    from ..world_description.world_entity import KinematicStructureEntity


@dataclass(eq=False)
class SpatialType:
    """
    Provides functionality to associate a reference frame with an object.

    This mixin class allows the inclusion of a reference frame within objects that
    require spatial or kinematic context. The reference frame is represented by a
    `KinematicStructureEntity`, which provides the necessary structural and spatial
    information.

    """

    casadi_sx: ca.SX
    """
    Implement Symbolic Math protocol
    """

    reference_frame: Optional[KinematicStructureEntity] = field(
        kw_only=True, default=None
    )
    """
    The reference frame associated with the object. Can be None if no reference frame is required or applicable.
    """

    @classmethod
    def _parse_optional_frame_from_json(
        cls, data: Dict[str, Any], key: str, **kwargs
    ) -> Optional[KinematicStructureEntity]:
        """
        Resolve an optional kinematic structure entity from JSON by key.
        Raises KinematicStructureEntityNotInKwargs if the name cannot be resolved via the tracker/world.

        :param data: parsed JSON data
        :param key: name of the attribute in data that is a KinematicStructureEntity
        :param kwargs: addition kwargs of _from_json
        :return: None if the key is not present or its value is None.
        """

        frame_data = data.get(key, {})
        if not frame_data:
            return None
        tracker = WorldEntityWithIDKwargsTracker.from_kwargs(kwargs)
        return tracker.get_world_entity_with_id(id=from_json(frame_data))

    @staticmethod
    def _ensure_consistent_frame(
        spatial_objects: List[Optional[SpatialType]],
    ) -> Optional[KinematicStructureEntity]:
        """
        Ensures that all provided spatial objects have a consistent reference frame. If a mismatch
        in the reference frames is detected among the non-null spatial objects, an exception is
        raised. If the list contains only null objects, None is returned.

        This method is primarily used to validate the reference frames of spatial objects before
        proceeding with further operations.

        :param spatial_objects: A list containing zero or more spatial objects, which can either
            be instances of ReferenceFrameMixin or None.
        :return: The common reference frame of the spatial objects if consistent, or None if no
            valid reference frame exists.

        :raises SpatialTypesError: Raised when the reference frames of provided input spatial
            objects are inconsistent.
        """
        reference_frame = None
        for spatial_object in spatial_objects:
            if (
                spatial_object is not None
                and spatial_object.reference_frame is not None
            ):
                if reference_frame is None:
                    reference_frame = spatial_object.reference_frame
                    continue
                if reference_frame != spatial_object.reference_frame:
                    raise SpatialTypesError(
                        message=f"Reference frames of input parameters don't match ({reference_frame} != {spatial_object.reference_frame})."
                    )
        return reference_frame

    def __deepcopy__(self, memo) -> Self:
        """
        Even in a deep copy, we don't want to copy the reference and child frame, just the matrix itself,
        because are just references to kinematic structure entities.
        """
        if id(self) in memo:
            return memo[id(self)]
        result = type(self).from_casadi_sx(deepcopy(self.casadi_sx))
        result.reference_frame = self.reference_frame
        return result

    def __copy__(self, memo) -> Self:
        """
        Even in a deep copy, we don't want to copy the reference and child frame, just the matrix itself,
        because are just references to kinematic structure entities.
        """
        if id(self) in memo:
            return memo[id(self)]
        result = type(self).from_casadi_sx(copy(self.casadi_sx))
        result.reference_frame = self.reference_frame
        return result


@dataclass(eq=False, init=False)
class HomogeneousTransformationMatrix(
    sm.SymbolicMathType, SpatialType, SubclassJSONSerializer
):
    """
    Represents a 4x4 transformation matrix used in kinematics and transformations.

    A `TransformationMatrix` encapsulates relationships between a parent coordinate
    system (reference frame) and a child coordinate system through rotation and
    translation. It provides utilities to derive transformations, compute dot
    products, and create transformations from various inputs such as Euler angles or
    quaternions.
    """

    child_frame: Optional[KinematicStructureEntity] = field(kw_only=True, default=None)
    """
    child_frame of this transformation matrix.
    """

    def __init__(
        self,
        data: Optional[sm.MatrixData] = None,
        reference_frame: Optional[KinematicStructureEntity] = None,
        child_frame: Optional[KinematicStructureEntity] = None,
    ):
        """
        :param data: A 4x4 matrix of some form that represents the rotation matrix.
        """
        self.reference_frame = reference_frame
        self.child_frame = child_frame
        if data is None:
            self._casadi_sx = ca.SX.eye(4)
            return
        if isinstance(data, SpatialType):
            # create a copy if data is a spatial type, because they are often still being used
            casadi_sx = copy(data.casadi_sx)
        else:
            casadi_sx = sm.to_sx(data)
        self.casadi_sx = casadi_sx

    def _verify_type(self):
        if self.shape != (4, 4):
            raise WrongDimensionsError(
                expected_dimensions=(4, 4), actual_dimensions=self.shape
            )
        self[3, 0] = 0.0
        self[3, 1] = 0.0
        self[3, 2] = 0.0
        self[3, 3] = 1.0

    @classmethod
    def _from_json(cls, data: Dict[str, Any], **kwargs) -> Self:
        reference_frame = cls._parse_optional_frame_from_json(
            data, key="reference_frame_id", **kwargs
        )
        child_frame = cls._parse_optional_frame_from_json(
            data, key="child_frame_id", **kwargs
        )
        return cls.from_xyz_quaternion(
            *data["position"][:3],
            *data["rotation"],
            reference_frame=reference_frame,
            child_frame=child_frame,
        )

    def to_json(self) -> Dict[str, Any]:
        if not self.is_constant():
            raise SpatialTypeNotJsonSerializable(self)
        result = super().to_json()
        if self.reference_frame is not None:
            result["reference_frame_id"] = to_json(self.reference_frame.id)
        if self.child_frame is not None:
            result["child_frame_id"] = to_json(self.child_frame.id)
        result["position"] = self.to_position().to_np().tolist()
        result["rotation"] = self.to_quaternion().to_np().tolist()
        return result

    @classmethod
    def from_point_rotation_matrix(
        cls,
        point: Optional[Point3] = None,
        rotation_matrix: Optional[RotationMatrix] = None,
        reference_frame: Optional[KinematicStructureEntity] = None,
        child_frame: Optional[KinematicStructureEntity] = None,
    ) -> HomogeneousTransformationMatrix:
        """
        Constructs a TransformationMatrix object from a given point, a rotation matrix,
        a reference frame, and a child frame.

        :param point: The 3D point used to set the translation part of the
            transformation matrix. If None, no translation is applied.
        :param rotation_matrix: The rotation matrix defines the rotational component
            of the transformation. If None, the identity matrix is assumed.
        :param reference_frame: The reference frame for the transformation matrix.
            It specifies the parent coordinate system.
        :param child_frame: The child or target frame for the transformation. It
            specifies the target coordinate system.
        :return: A `TransformationMatrix` instance initialized with the provided
            parameters or default values.
        """
        if reference_frame is None:
            reference_frame = cls._ensure_consistent_frame([point, rotation_matrix])

        if rotation_matrix is None:
            a_T_b = cls(reference_frame=reference_frame, child_frame=child_frame)
        else:
            a_T_b = cls(
                data=rotation_matrix,
                reference_frame=reference_frame,
                child_frame=child_frame,
            )
        if point is not None:
            a_T_b._casadi_sx[0, 3] = point._casadi_sx[0]
            a_T_b._casadi_sx[1, 3] = point._casadi_sx[1]
            a_T_b._casadi_sx[2, 3] = point._casadi_sx[2]
        return a_T_b

    @classmethod
    def from_xyz_rpy(
        cls,
        x: sm.ScalarData = 0,
        y: sm.ScalarData = 0,
        z: sm.ScalarData = 0,
        roll: sm.ScalarData = 0,
        pitch: sm.ScalarData = 0,
        yaw: sm.ScalarData = 0,
        reference_frame: Optional[KinematicStructureEntity] = None,
        child_frame: Optional[KinematicStructureEntity] = None,
    ) -> HomogeneousTransformationMatrix:
        """
        Creates a TransformationMatrix object from position (x, y, z) and Euler angles
        (roll, pitch, yaw) values. The function also accepts optional reference and
        child frame parameters.

        :param x: The x-coordinate of the position
        :param y: The y-coordinate of the position
        :param z: The z-coordinate of the position
        :param roll: The rotation around the x-axis
        :param pitch: The rotation around the y-axis
        :param yaw: The rotation around the z-axis
        :param reference_frame: The reference frame for the transformation
        :param child_frame: The child frame associated with the transformation
        :return: A TransformationMatrix object created using the provided
            position and orientation values
        """
        p = Point3(x=x, y=y, z=z)
        r = RotationMatrix.from_rpy(roll, pitch, yaw)
        return cls.from_point_rotation_matrix(
            p, r, reference_frame=reference_frame, child_frame=child_frame
        )

    @classmethod
    def from_xyz_quaternion(
        cls,
        pos_x: sm.ScalarData = 0,
        pos_y: sm.ScalarData = 0,
        pos_z: sm.ScalarData = 0,
        quat_x: sm.ScalarData = 0,
        quat_y: sm.ScalarData = 0,
        quat_z: sm.ScalarData = 0,
        quat_w: sm.ScalarData = 1,
        reference_frame: Optional[KinematicStructureEntity] = None,
        child_frame: Optional[KinematicStructureEntity] = None,
    ) -> HomogeneousTransformationMatrix:
        """
        Creates a `TransformationMatrix` instance from the provided position coordinates and quaternion
        values representing rotation. This method constructs a 3D point for the position and a rotation
        matrix derived from the quaternion, and initializes the transformation matrix with these along
        with optional reference and child frame entities.

        :param pos_x: X coordinate of the position in space.
        :param pos_y: Y coordinate of the position in space.
        :param pos_z: Z coordinate of the position in space.
        :param quat_w: W component of the quaternion representing rotation.
        :param quat_x: X component of the quaternion representing rotation.
        :param quat_y: Y component of the quaternion representing rotation.
        :param quat_z: Z component of the quaternion representing rotation.
        :param reference_frame: Optional reference frame for the transformation matrix.
        :param child_frame: Optional child frame for the transformation matrix.
        :return: A `TransformationMatrix` object constructed from the given parameters.
        """
        p = Point3(x=pos_x, y=pos_y, z=pos_z)
        r = RotationMatrix.from_quaternion(
            q=Quaternion(w=quat_w, x=quat_x, y=quat_y, z=quat_z)
        )
        return cls.from_point_rotation_matrix(
            p, r, reference_frame=reference_frame, child_frame=child_frame
        )

    @classmethod
    def from_xyz_axis_angle(
        cls,
        x: sm.ScalarData = 0,
        y: sm.ScalarData = 0,
        z: sm.ScalarData = 0,
        axis: Vector3 | sm.NumericalVector = None,
        angle: sm.ScalarData = 0,
        reference_frame: Optional[KinematicStructureEntity] = None,
        child_frame: Optional[KinematicStructureEntity] = None,
    ) -> Self:
        """
        Creates an instance of the class from x, y, z coordinates, axis and angle.

        This class method generates an object using provided spatial coordinates and a
        rotation defined by an axis and angle. The resulting object is defined with
        a specified reference frame and child frame.

        :param x: Initial x-coordinate.
        :param y: Initial y-coordinate.
        :param z: Initial z-coordinate.
        :param axis: Vector defining the axis of rotation. Defaults to Vector3(0, 0, 1) if not specified.
        :param angle: Angle of rotation around the specified axis, in radians.
        :param reference_frame: Reference frame entity to be associated with the object.
        :param child_frame: Child frame entity associated with the object.
        :return: An instance of the class with the specified transformations applied.
        """
        if axis is None:
            axis = Vector3(0, 0, 1)
        rotation_matrix = RotationMatrix.from_axis_angle(axis=axis, angle=angle)
        point = Point3(x=x, y=y, z=z)
        return cls.from_point_rotation_matrix(
            point=point,
            rotation_matrix=rotation_matrix,
            reference_frame=reference_frame,
            child_frame=child_frame,
        )

    @property
    def x(self) -> sm.Scalar:
        return self[0, 3]

    @x.setter
    def x(self, value: sm.ScalarData):
        self[0, 3] = value

    @property
    def y(self) -> sm.Scalar:
        return self[1, 3]

    @y.setter
    def y(self, value: sm.ScalarData):
        self[1, 3] = value

    @property
    def z(self) -> sm.Scalar:
        return self[2, 3]

    @z.setter
    def z(self, value: sm.ScalarData):
        self[2, 3] = value

    def dot(
        self, other: GenericHomogeneousSpatialType
    ) -> GenericHomogeneousSpatialType:
        if isinstance(
            other,
            (Vector3, Point3, RotationMatrix, HomogeneousTransformationMatrix, Pose),
        ):
            result_sx = ca.mtimes(self.casadi_sx, other.casadi_sx)
            result = type(other).from_casadi_sx(casadi_sx=result_sx)
            result.reference_frame = self.reference_frame
            if isinstance(other, HomogeneousTransformationMatrix):
                result.child_frame = other.child_frame
            return result
        raise UnsupportedOperationError("dot", self, other)

    def __matmul__(self, other: GenericSpatialType) -> GenericSpatialType:
        return self.dot(other)

    def inverse(self) -> HomogeneousTransformationMatrix:
        inv = HomogeneousTransformationMatrix(
            child_frame=self.reference_frame, reference_frame=self.child_frame
        )
        inv[:3, :3] = self[:3, :3].T
        inv[:3, 3] = (-inv[:3, :3]).dot(self[:3, 3])
        return inv

    def to_position(self) -> Point3:
        result = Point3.from_iterable(
            self[:4, 3:], reference_frame=self.reference_frame
        )
        return result

    def to_translation_matrix(self) -> HomogeneousTransformationMatrix:
        """
        :return: sets the rotation part of a frame to identity
        """
        r = HomogeneousTransformationMatrix()
        r[0, 3] = self[0, 3]
        r[1, 3] = self[1, 3]
        r[2, 3] = self[2, 3]
        return HomogeneousTransformationMatrix(
            data=r, reference_frame=self.reference_frame, child_frame=None
        )

    def to_rotation_matrix(self) -> RotationMatrix:
        return RotationMatrix(data=self, reference_frame=self.reference_frame)

    def to_quaternion(self) -> Quaternion:
        return self.to_rotation_matrix().to_quaternion()

    def to_pose(self) -> Pose:
        result = Pose.from_casadi_sx(casadi_sx=self.casadi_sx)
        result.reference_frame = self.reference_frame
        return result

    def __deepcopy__(self, memo) -> HomogeneousTransformationMatrix:
        """
        Even in a deep copy, we don't want to copy the reference and child frame, just the matrix itself,
        because are just references to kinematic structure entities.
        """
        if id(self) in memo:
            return memo[id(self)]
        return HomogeneousTransformationMatrix(
            data=deepcopy(self.casadi_sx),
            reference_frame=self.reference_frame,
            child_frame=self.child_frame,
        )


@dataclass(eq=False, init=False)
class RotationMatrix(sm.SymbolicMathType, SpatialType, SubclassJSONSerializer):
    """
    Class to represent a 4x4 symbolic rotation matrix tied to kinematic references.

    This class provides methods for creating and manipulating rotation matrices within the context
    of kinematic structures. It supports initialization using data such as quaternions, axis-angle,
    other matrices, or directly through vector definitions. The primary purpose is to facilitate
    rotational transformations and computations in a symbolic context, particularly for applications
    like robotic kinematics or mechanical engineering.
    """

    def __init__(
        self,
        data: Optional[sm.MatrixData] = None,
        reference_frame: Optional[KinematicStructureEntity] = None,
    ):
        """
        :param data: A 4x4 matrix of some form that represents the rotation matrix.
        :param reference_frame:
        """
        self.reference_frame = reference_frame
        if data is None:
            self._casadi_sx = ca.SX.eye(4)
            return
        empty_data = to_sx(Matrix.eye(4))
        empty_data[:3, :3] = sm.to_sx(data)[:3, :3]
        self._casadi_sx = empty_data

    def _verify_type(self):
        if self.shape != (4, 4):
            raise WrongDimensionsError(
                expected_dimensions=(4, 4), actual_dimensions=self.shape
            )
        self[0, 3] = 0
        self[1, 3] = 0
        self[2, 3] = 0
        self[3, 0] = 0
        self[3, 1] = 0
        self[3, 2] = 0
        self[3, 3] = 1

    @classmethod
    def _from_json(cls, data: Dict[str, Any], **kwargs) -> Self:
        reference_frame = cls._parse_optional_frame_from_json(
            data, key="reference_frame_id", **kwargs
        )
        return Quaternion.from_iterable(
            data["quaternion"],
            reference_frame=reference_frame,
        ).to_rotation_matrix()

    def to_json(self) -> Dict[str, Any]:
        if not self.is_constant():
            raise SpatialTypeNotJsonSerializable(self)
        result = super().to_json()
        if self.reference_frame is not None:
            result["reference_frame_id"] = to_json(self.reference_frame.id)
        result["quaternion"] = self.to_quaternion().to_np().tolist()
        return result

    @classmethod
    def from_axis_angle(
        cls,
        axis: Union[Vector3, sm.NumericalVector],
        angle: sm.ScalarData,
        reference_frame: Optional[KinematicStructureEntity] = None,
    ) -> RotationMatrix:
        """
        Conversion of unit axis and angle to 4x4 rotation matrix according to:
        https://www.euclideanspace.com/maths/geometry/rotations/conversions/angleToMatrix/index.htm
        """
        # use casadi to prevent a bunch of sm.SymbolicMathType.__init__.py calls
        axis = sm.to_sx(axis)
        angle = sm.to_sx(angle)
        ct = ca.cos(angle)
        st = ca.sin(angle)
        vt = 1 - ct
        m_vt = axis * vt
        m_st = axis * st
        m_vt_0_ax = (m_vt[0] * axis)[1:]
        m_vt_1_2 = m_vt[1] * axis[2]
        s = ca.SX.eye(4)
        ct__m_vt__axis = ct + m_vt * axis
        s[0, 0] = ct__m_vt__axis[0]
        s[0, 1] = -m_st[2] + m_vt_0_ax[0]
        s[0, 2] = m_st[1] + m_vt_0_ax[1]
        s[1, 0] = m_st[2] + m_vt_0_ax[0]
        s[1, 1] = ct__m_vt__axis[1]
        s[1, 2] = -m_st[0] + m_vt_1_2
        s[2, 0] = -m_st[1] + m_vt_0_ax[1]
        s[2, 1] = m_st[0] + m_vt_1_2
        s[2, 2] = ct__m_vt__axis[2]
        return cls(s, reference_frame=reference_frame)

    @classmethod
    def from_quaternion(cls, q: Quaternion) -> RotationMatrix:
        """
        Unit quaternion to 4x4 rotation matrix according to:
        https://github.com/orocos/orocos_kinematics_dynamics/blob/master/orocos_kdl/src/frames.cpp#L167
        """
        x = q[0]
        y = q[1]
        z = q[2]
        w = q[3]
        x2 = x * x
        y2 = y * y
        z2 = z * z
        w2 = w * w
        return cls(
            data=[
                [
                    w2 + x2 - y2 - z2,
                    2 * x * y - 2 * w * z,
                    2 * x * z + 2 * w * y,
                    0,
                ],
                [
                    2 * x * y + 2 * w * z,
                    w2 - x2 + y2 - z2,
                    2 * y * z - 2 * w * x,
                    0,
                ],
                [
                    2 * x * z - 2 * w * y,
                    2 * y * z + 2 * w * x,
                    w2 - x2 - y2 + z2,
                    0,
                ],
                [0, 0, 0, 1],
            ],
            reference_frame=q.reference_frame,
        )

    def x_vector(self) -> Vector3:
        return Vector3(
            x=self[0, 0],
            y=self[1, 0],
            z=self[2, 0],
            reference_frame=self.reference_frame,
        )

    def y_vector(self) -> Vector3:
        return Vector3(
            x=self[0, 1],
            y=self[1, 1],
            z=self[2, 1],
            reference_frame=self.reference_frame,
        )

    def z_vector(self) -> Vector3:
        return Vector3(
            x=self[0, 2],
            y=self[1, 2],
            z=self[2, 2],
            reference_frame=self.reference_frame,
        )

    def dot(self, other: GenericRotatableSpatialType) -> GenericRotatableSpatialType:
        if isinstance(
            other, (Vector3, RotationMatrix, HomogeneousTransformationMatrix, Pose)
        ):
            result_sx = ca.mtimes(self.casadi_sx, other.casadi_sx)
            result = type(other).from_casadi_sx(casadi_sx=result_sx)
            result.reference_frame = self.reference_frame
            return result
        raise UnsupportedOperationError("dot", self, other)

    def __matmul__(
        self, other: GenericRotatableSpatialType
    ) -> GenericRotatableSpatialType:
        return self.dot(other)

    def to_axis_angle(self) -> Tuple[Vector3, sm.Scalar]:
        return self.to_quaternion().to_axis_angle()

    def to_angle(self, hint: Optional[Callable] = None) -> sm.Scalar:
        """
        :param hint: A function whose sign of the result will be used to determine if angle should be positive or
                        negative
        :return:
        """
        axis, angle = self.to_axis_angle()
        if hint is not None:
            return sm.normalize_angle(
                sm.if_greater_zero(hint(axis), if_result=angle, else_result=-angle)
            )
        else:
            return angle

    def to_generic_matrix(self) -> sm.Matrix:
        return sm.Matrix.from_casadi_sx(self.casadi_sx)

    @classmethod
    def from_vectors(
        cls,
        x: Optional[Vector3] = None,
        y: Optional[Vector3] = None,
        z: Optional[Vector3] = None,
        reference_frame: Optional[KinematicStructureEntity] = None,
    ) -> RotationMatrix:
        """
        Create a rotation matrix from 2 or 3 orthogonal vectors.

        If exactly two of x, y, z must be provided. The third will be computed using the cross product.

        Valid combinations:
        - x and y provided: z = x × y
        - x and z provided: y = z × x
        - y and z provided: x = y × z
        - x, y, and z provided: all three used directly
        """
        if x is None and y is None and z is None:
            raise SpatialTypesError(
                message="from_vectors requires at least two vectors"
            )
        if x is not None and y is not None and z is None:
            z = x.cross(y)
        elif x is not None and y is None and z is not None:
            y = z.cross(x)
        elif x is None and y is not None and z is not None:
            x = y.cross(z)
        x.scale(1)
        y.scale(1)
        z.scale(1)
        R = cls(
            data=[
                [x[0], y[0], z[0], 0],
                [x[1], y[1], z[1], 0],
                [x[2], y[2], z[2], 0],
                [0, 0, 0, 1],
            ],
            reference_frame=reference_frame,
        )
        return R

    @classmethod
    def from_rpy(
        cls,
        roll: Optional[sm.ScalarData] = None,
        pitch: Optional[sm.ScalarData] = None,
        yaw: Optional[sm.ScalarData] = None,
        reference_frame: Optional[KinematicStructureEntity] = None,
    ) -> RotationMatrix:
        """
        Conversion of roll, pitch, yaw to 4x4 rotation matrix according to:
        https://github.com/orocos/orocos_kinematics_dynamics/blob/master/orocos_kdl/src/frames.cpp#L167
        """
        roll = 0 if roll is None else roll
        pitch = 0 if pitch is None else pitch
        yaw = 0 if yaw is None else yaw
        roll = sm.to_sx(roll)
        pitch = sm.to_sx(pitch)
        yaw = sm.to_sx(yaw)

        s = ca.SX.eye(4)

        s[0, 0] = ca.cos(yaw) * ca.cos(pitch)
        s[0, 1] = (ca.cos(yaw) * ca.sin(pitch) * ca.sin(roll)) - (
            ca.sin(yaw) * ca.cos(roll)
        )
        s[0, 2] = (ca.sin(yaw) * ca.sin(roll)) + (
            ca.cos(yaw) * ca.sin(pitch) * ca.cos(roll)
        )
        s[1, 0] = ca.sin(yaw) * ca.cos(pitch)
        s[1, 1] = (ca.cos(yaw) * ca.cos(roll)) + (
            ca.sin(yaw) * ca.sin(pitch) * ca.sin(roll)
        )
        s[1, 2] = (ca.sin(yaw) * ca.sin(pitch) * ca.cos(roll)) - (
            ca.cos(yaw) * ca.sin(roll)
        )
        s[2, 0] = -ca.sin(pitch)
        s[2, 1] = ca.cos(pitch) * ca.sin(roll)
        s[2, 2] = ca.cos(pitch) * ca.cos(roll)
        result = cls(reference_frame=reference_frame)
        result.casadi_sx = s
        return result

    def inverse(self) -> RotationMatrix:
        return self.T

    def to_rpy(self) -> Tuple[sm.Scalar, sm.Scalar, sm.Scalar]:
        """
        :return: roll, pitch, yaw
        """
        i = 0
        j = 1
        k = 2

        cy = sm.sqrt(self[i, i] * self[i, i] + self[j, i] * self[j, i])
        if0 = cy - sm.EPS
        ax = sm.if_greater_zero(
            if0, sm.atan2(self[k, j], self[k, k]), sm.atan2(-self[j, k], self[j, j])
        )
        ay = sm.if_greater_zero(
            if0, sm.atan2(-self[k, i], cy), sm.atan2(-self[k, i], cy)
        )
        az = sm.if_greater_zero(if0, sm.atan2(self[j, i], self[i, i]), sm.Scalar(0))
        return ax, ay, az

    def to_quaternion(self) -> Quaternion:
        return Quaternion.from_rotation_matrix(self)

    def normalize(self) -> None:
        """Scales each of the axes to the length of one."""
        scale_v = 1.0
        self[:3, 0] = self[:3, 0].scale(scale_v)
        self[:3, 1] = self[:3, 1].scale(scale_v)
        self[:3, 2] = self[:3, 2].scale(scale_v)

    @property
    def T(self) -> RotationMatrix:
        return RotationMatrix(self.casadi_sx.T, reference_frame=self.reference_frame)

    def rotational_error(self, other: RotationMatrix) -> sm.Scalar:
        """
        Calculate the rotational error between two rotation matrices.

        This function computes the angular difference between two rotation matrices
        by computing the dot product of the first matrix and the inverse of the second.
        Subsequently, it generates the angle of the resulting rotation matrix.

        :param other: The second rotation matrix.
        :return: The angular error between the two rotation matrices as an expression.
        """
        r_distance = self.dot(other.inverse())
        return r_distance.to_angle()


@dataclass(eq=False, init=False)
class Point3(sm.SymbolicMathType, SpatialType, SubclassJSONSerializer):
    """
    Represents a 3D point with reference frame handling.

    This class provides a representation of a point in 3D space, including support
    for operations such as addition, subtraction, projection onto planes/lines, and
    distance calculations. It incorporates a reference frame for kinematic computations
    and facilitates mathematical operations essential for 3D geometry modeling.

    .. note:: this is represented as a 4d vector, where the last entry is always a 1.
    """

    def __init__(
        self,
        x: sm.ScalarData = 0,
        y: sm.ScalarData = 0,
        z: sm.ScalarData = 0,
        reference_frame: Optional[KinematicStructureEntity] = None,
    ):
        """

        :param x: X-coordinate of the point. Defaults to 0.
        :param y: Y-coordinate of the point. Defaults to 0.
        :param z: Z-coordinate of the point. Defaults to 0.
        :param reference_frame:
        """
        self.casadi_sx = sm.to_sx([x, y, z, 1])
        self.reference_frame = reference_frame

    def _verify_type(self):
        if self.shape == (3, 1):
            casadi_sx = ca.SX.zeros(4)
            casadi_sx[:3, 0] = self._casadi_sx
            self._casadi_sx = casadi_sx
        elif self.shape != (4, 1):
            raise WrongDimensionsError(
                expected_dimensions=(4, 1), actual_dimensions=self.shape
            )
        self._casadi_sx[3, 0] = 1

    @classmethod
    def from_iterable(
        cls,
        data: sm.VectorData,
        reference_frame: Optional[KinematicStructureEntity] = None,
    ) -> Point3:
        """
        Creates an instance of Point3 from provided iterable data.

        This class method is used to construct a Point3 object by processing the given
        data and optionally assigning a reference frame. The data can represent
        different array-like objects compatible with the desired format for a Point3
        instance. The provided iterable or array should follow a 1D structure to avoid
        raised errors.

        :param data: The array-like data or object such as a list, tuple, or numpy array
            used to initialize the Point3 instance.
        :param reference_frame: A reference to a `KinematicStructureEntity` object,
            representing the frame of reference for the Point3 instance. If the data
            has a `reference_frame` attribute, and this parameter is not specified,
            it will be taken from the data.
        :return: Returns an instance of Point3 initialized with the processed data
            and an optional reference frame.
        """
        if isinstance(data, SpatialType) and reference_frame is None:
            reference_frame = data.reference_frame
        result = cls(reference_frame=reference_frame)
        result.casadi_sx = sm.to_sx(data)
        return result

    @classmethod
    def _from_json(cls, data: Dict[str, Any], **kwargs) -> Self:
        reference_frame = cls._parse_optional_frame_from_json(
            data, key="reference_frame_id", **kwargs
        )
        return cls.from_iterable(
            data["data"][:3],
            reference_frame=reference_frame,
        )

    def to_json(self) -> Dict[str, Any]:
        if not self.is_constant():
            raise SpatialTypeNotJsonSerializable(self)
        result = super().to_json()
        if self.reference_frame is not None:
            result["reference_frame_id"] = to_json(self.reference_frame.id)
        result["data"] = self.to_np().tolist()
        return result

    def norm(self) -> sm.Scalar:
        return sm.Scalar.from_casadi_sx(ca.norm_2(self[:3].casadi_sx))

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
    def z(self) -> sm.Scalar:
        return self[2]

    @z.setter
    def z(self, value: sm.ScalarData):
        self[2] = value

    def __add__(self, other: Vector3) -> Point3:
        if isinstance(other, Vector3):
            result = Point3.from_iterable(self.casadi_sx.__add__(other.casadi_sx))
        else:
            return NotImplemented
        result.reference_frame = self.reference_frame
        return result

    @overload
    def __sub__(self, other: Point3) -> Vector3: ...

    @overload
    def __sub__(self, other: Vector3) -> Point3: ...

    def __sub__(self, other):
        if isinstance(other, Point3):
            result = Vector3.from_iterable(self.casadi_sx.__sub__(other.casadi_sx))
        elif isinstance(other, Vector3):
            result = Point3.from_iterable(self.casadi_sx.__sub__(other.casadi_sx))
        else:
            return NotImplemented
        result.reference_frame = self.reference_frame
        return result

    def __neg__(self) -> Point3:
        result = Point3.from_iterable(self.casadi_sx.__neg__())
        result.reference_frame = self.reference_frame
        return result

    def project_to_plane(
        self, frame_V_plane_vector1: Vector3, frame_V_plane_vector2: Vector3
    ) -> Tuple[Point3, sm.Scalar]:
        """
        Projects a point onto a plane defined by two vectors.
        This function assumes that all parameters are defined with respect to the same reference frame.

        :param frame_V_plane_vector1: First vector defining the plane
        :param frame_V_plane_vector2: Second vector defining the plane
        :return: Tuple of (projected point on the plane, signed distance from point to plane)
        """
        normal = frame_V_plane_vector1.cross(frame_V_plane_vector2)
        normal.scale(1)
        frame_V_current = self.to_vector3()
        d = normal @ frame_V_current
        v: Vector3 = normal * d
        projection = self - v
        return projection, d

    def project_to_line(
        self, line_point: Point3, line_direction: Vector3
    ) -> Tuple[Point3, sm.Scalar]:
        """
        :param line_point: a point that the line intersects, must have the same reference frame as self
        :param line_direction: the direction of the line, must have the same reference frame as self
        :return: tuple of (closest point on the line, shortest distance between self and the line)
        """
        lp_vector = self - line_point
        cross_product = lp_vector.cross(line_direction)
        distance = cross_product.norm() / line_direction.norm()

        line_direction_unit = line_direction / line_direction.norm()
        projection_length = lp_vector @ line_direction_unit
        closest_point = line_point + line_direction_unit * projection_length

        return closest_point, distance

    def distance_to_line_segment(
        self, line_start: Point3, line_end: Point3
    ) -> Tuple[sm.Scalar, Point3]:
        """
        All parameters must have the same reference frame as self.
        :param line_start: start of the approached line
        :param line_end: end of the approached line
        :return: distance to line, the nearest point on the line
        """
        frame_P_current = self
        frame_P_line_start = line_start
        frame_P_line_end = line_end
        frame_V_line_vec = frame_P_line_end - frame_P_line_start
        pnt_vec = frame_P_current - frame_P_line_start
        line_len = frame_V_line_vec.norm()
        line_unitvec = frame_V_line_vec / line_len
        pnt_vec_scaled = pnt_vec / line_len
        t = line_unitvec @ pnt_vec_scaled
        t = sm.limit(t, lower_limit=0.0, upper_limit=1.0)
        frame_V_offset = frame_V_line_vec * t
        dist = (frame_V_offset - pnt_vec).norm()
        frame_P_nearest = frame_P_line_start + frame_V_offset
        return dist, frame_P_nearest

    def to_vector3(self) -> Vector3:
        result = Vector3.from_casadi_sx(copy(self.casadi_sx))
        result.reference_frame = self.reference_frame
        return result

    def to_generic_vector(self) -> sm.Vector:
        return sm.Vector.from_casadi_sx(self.casadi_sx[:3])

    def euclidean_distance(self, other: Self) -> sm.Scalar:
        return self.to_generic_vector().euclidean_distance(other.to_generic_vector())


@dataclass(eq=False, init=False)
class Vector3(sm.SymbolicMathType, SpatialType, SubclassJSONSerializer):
    """
    Representation of a 3D vector with reference frame support for homogenous transformations.

    This class provides a structured representation of 3D vectors. It includes
    support for operations such as addition, subtraction, scaling, dot product,
    cross product, and more. It is compatible with symbolic computations and
    provides methods to define standard basis vectors, normalize a vector, and
    compute geometric properties such as the angle between vectors. The class
    also includes support for working in different reference frames.

    .. note:: this is represented as a 4d vector, where the last entry is always a 0.
    """

    visualisation_frame: KinematicStructureEntity = field(default=None)

    def __init__(
        self,
        x: sm.ScalarData = 0,
        y: sm.ScalarData = 0,
        z: sm.ScalarData = 0,
        reference_frame: Optional[KinematicStructureEntity] = None,
        visualisation_frame: Optional[KinematicStructureEntity] = None,
    ):
        """

        :param x: X-coordinate of the point. Defaults to 0.
        :param y: Y-coordinate of the point. Defaults to 0.
        :param z: Z-coordinate of the point. Defaults to 0.
        :param reference_frame:
        :param visualisation_frame: The reference frame associated with the vector, used for visualization purposes only. Optional.
        It will be visualized at the origin of the vis_frame
        """
        self.casadi_sx = sm.to_sx([x, y, z, 0])
        self.reference_frame = reference_frame
        self.visualisation_frame = visualisation_frame

    def _verify_type(self):
        if self.shape == (3, 1):
            casadi_sx = ca.SX.zeros(4)
            casadi_sx[:3, 0] = self._casadi_sx
            self._casadi_sx = casadi_sx
        elif self.shape != (4, 1):
            raise WrongDimensionsError(
                expected_dimensions=(4, 1), actual_dimensions=self.shape
            )
        self._casadi_sx[3] = 0

    @classmethod
    def _from_json(cls, data: Dict[str, Any], **kwargs) -> Self:
        reference_frame = cls._parse_optional_frame_from_json(
            data, key="reference_frame_id", **kwargs
        )
        return cls.from_iterable(
            data["data"][:3],
            reference_frame=reference_frame,
        )

    def to_json(self) -> Dict[str, Any]:
        if not self.is_constant():
            raise SpatialTypeNotJsonSerializable(self)
        result = super().to_json()
        if self.reference_frame is not None:
            result["reference_frame_id"] = to_json(self.reference_frame.id)
        result["data"] = self.to_np().tolist()
        return result

    @classmethod
    def from_iterable(
        cls,
        data: sm.VectorData,
        reference_frame: Optional[KinematicStructureEntity] = None,
    ) -> Vector3:
        """
        Creates an instance of Vector3 from provided iterable data.

        This class method is used to construct a Vector3 object by processing the given
        data and optionally assigning a reference frame. The data can represent
        different array-like objects compatible with the desired format for a Vector3
        instance. The provided iterable or array should follow a 1D structure to avoid
        raised errors.

        :param data: The array-like data or object such as a list, tuple, or numpy array
            used to initialize the Vector3 instance.
        :param reference_frame: A reference to a `KinematicStructureEntity` object,
            representing the frame of reference for the Vector3 instance. If the data
            has a `reference_frame` attribute, and this parameter is not specified,
            it will be taken from the data.
        :return: Returns an instance of Vector3 initialized with the processed data
            and an optional reference frame.
        """
        if isinstance(data, SpatialType) and reference_frame is None:
            reference_frame = data.reference_frame
        if isinstance(data, Vector3) and reference_frame is None:
            visualisation_frame = data.visualisation_frame
        else:
            visualisation_frame = None
        result = cls(
            reference_frame=reference_frame, visualisation_frame=visualisation_frame
        )
        result.casadi_sx = sm.to_sx(data)
        return result

    @classmethod
    def X(cls, reference_frame: Optional[KinematicStructureEntity] = None) -> Vector3:
        return cls(x=1, y=0, z=0, reference_frame=reference_frame)

    @classmethod
    def Y(cls, reference_frame: Optional[KinematicStructureEntity] = None) -> Vector3:
        return cls(x=0, y=1, z=0, reference_frame=reference_frame)

    @classmethod
    def Z(cls, reference_frame: Optional[KinematicStructureEntity] = None) -> Vector3:
        return cls(x=0, y=0, z=1, reference_frame=reference_frame)

    @classmethod
    def unit_vector(
        cls,
        x: sm.ScalarData = 0,
        y: sm.ScalarData = 0,
        z: sm.ScalarData = 0,
        reference_frame: Optional[KinematicStructureEntity] = None,
    ) -> Vector3:
        v = cls(x=x, y=y, z=z, reference_frame=reference_frame)
        v.scale(1, unsafe=True)
        return v

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
    def z(self) -> sm.Scalar:
        return self[2]

    @z.setter
    def z(self, value: sm.ScalarData):
        self[2] = value

    def __add__(self, other: Vector3) -> Vector3:
        if isinstance(other, Vector3):
            result = Vector3.from_iterable(self.casadi_sx.__add__(other.casadi_sx))
        else:
            return NotImplemented
        result.reference_frame = self.reference_frame
        return result

    def __sub__(self, other: Vector3) -> Vector3:
        if isinstance(other, Vector3):
            result = Vector3.from_iterable(self.casadi_sx.__sub__(other.casadi_sx))
        else:
            return NotImplemented
        result.reference_frame = self.reference_frame
        return result

    def __mul__(self, other: sm.ScalarData) -> Vector3:
        if isinstance(other, sm.ScalarData):
            result = Vector3.from_iterable(self.casadi_sx.__mul__(sm.to_sx(other)))
        else:
            return NotImplemented
        result.reference_frame = self.reference_frame
        return result

    def __rmul__(self, other: float) -> Vector3:
        if isinstance(other, (int, float)):
            result = Vector3.from_iterable(self.casadi_sx.__mul__(other))
        else:
            return NotImplemented
        result.reference_frame = self.reference_frame
        return result

    def __truediv__(self, other: sm.ScalarData) -> Vector3:
        if isinstance(other, sm.ScalarData):
            result = Vector3.from_iterable(self.casadi_sx.__truediv__(sm.to_sx(other)))
        else:
            return NotImplemented
        result.reference_frame = self.reference_frame
        return result

    def safe_division(
        self,
        other: sm.ScalarData,
        if_nan: Optional[Vector3] = None,
    ) -> sm.GenericSymbolicType:
        """
        A version of division where no sub-expression is ever NaN. The expression would evaluate to 'if_nan', but
        you should probably never work with the 'if_nan' result. However, if one sub-expressions is NaN, the whole expression
        evaluates to NaN, even if it is only in a branch of an if-else, that is not returned.
        This method is a workaround for such cases.
        """
        if if_nan is None:
            if_nan = Vector3()
        save_denominator = sm.if_eq_zero(
            condition=other, if_result=sm.Scalar(1), else_result=other
        )
        return sm.if_eq_zero(
            other, if_result=if_nan, else_result=self / save_denominator
        )

    def __neg__(self) -> Vector3:
        result = Vector3.from_iterable(self.casadi_sx.__neg__())
        result.reference_frame = self.reference_frame
        return result

    def dot(self, other: Vector3) -> sm.Scalar:
        if isinstance(other, Vector3):
            return sm.Scalar(ca.mtimes(sm.to_sx(self[:3]).T, sm.to_sx(other[:3])))
        raise UnsupportedOperationError("dot", self, other)

    def __matmul__(self, other: Vector3) -> sm.Scalar:
        return self.dot(other)

    def cross(self, other: Vector3) -> Vector3:
        result = ca.cross(self.casadi_sx[:3], other.casadi_sx[:3])
        result = self.__class__.from_iterable(result)
        result.reference_frame = self.reference_frame
        return result

    def norm(self) -> sm.Scalar:
        return sm.Scalar(ca.norm_2(sm.to_sx(self[:3])))

    def scale(self, a: sm.ScalarData, unsafe: bool = False):
        if unsafe:
            self.casadi_sx = ((self / self.norm()) * a).casadi_sx
        else:
            self.casadi_sx = (self.safe_division(self.norm()) * a).casadi_sx

    def project_to_cone(
        self,
        frame_V_cone_axis: Vector3,
        cone_theta: sm.ScalarData,
    ) -> Vector3:
        """
        Projects a given vector onto the boundary of a cone defined by its axis and angle.

        This function computes the projection of a vector onto the boundary of a
        cone specified by its axis and half-angle. It handles special cases where
        the input vector is collinear with the cone's axis. The projection ensures
        the resulting vector lies within the cone's boundary.

        :param frame_V_cone_axis: The axis of the cone.
        :param cone_theta: The half-angle of the cone in radians. Can be a symbolic value or a float.
        :return: The projection of the input vector onto the cone's boundary.
        """
        frame_V_current = self
        frame_V_cone_axis_normed = copy(frame_V_cone_axis)
        frame_V_cone_axis_normed.scale(1)
        beta = frame_V_current @ frame_V_cone_axis_normed
        norm_v = frame_V_current.norm()

        # Compute the perpendicular component.
        v_perp = frame_V_current - (frame_V_cone_axis_normed * beta)
        norm_v_perp = v_perp.norm()
        v_perp.scale(1)

        s = beta * sm.cos(cone_theta) + norm_v_perp * sm.sin(cone_theta)
        projected_vector = (
            (frame_V_cone_axis_normed * sm.cos(cone_theta))
            + (v_perp * sm.sin(cone_theta))
        ) * s
        # Handle the case when v is collinear with a.
        project_on_cone_boundary = sm.if_less(
            a=norm_v_perp,
            b=1e-8,
            if_result=frame_V_cone_axis_normed * norm_v * sm.cos(cone_theta),
            else_result=projected_vector,
        )

        return sm.if_greater_eq(
            a=beta,
            b=norm_v * np.cos(cone_theta),
            if_result=frame_V_current,
            else_result=project_on_cone_boundary,
        )

    def angle_between(self, other: Vector3) -> sm.Scalar:
        return sm.acos(
            sm.limit(
                self @ other / (self.norm() * other.norm()),
                lower_limit=-1,
                upper_limit=1,
            )
        )

    def slerp(self, other: Vector3, t: sm.ScalarData) -> Vector3:
        """
        spherical linear interpolation
        :param other: vector of same length as self
        :param t: value between 0 and 1. 0 is v1 and 1 is v2
        """
        angle = sm.safe_acos(self @ other)
        angle2 = sm.if_eq(angle, 0, sm.Scalar(data=1), angle)
        return sm.if_eq(
            angle,
            0,
            self,
            self * (sm.sin((1 - t) * angle2) / sm.sin(angle2))
            + other * (sm.sin(t * angle2) / sm.sin(angle2)),
        )

    def to_point3(self) -> Point3:
        result = Point3.from_casadi_sx(self.casadi_sx)
        result.reference_frame = self.reference_frame
        return result


@dataclass(eq=False, init=False)
class Quaternion(sm.SymbolicMathType, SpatialType, SubclassJSONSerializer):
    """
    Represents a quaternion, which is a mathematical entity used to encode
    rotations in three-dimensional space.

    The Quaternion class provides methods for creating quaternion objects
    from various representations, such as axis-angle, roll-pitch-yaw,
    and rotation matrices. It supports operations to define and manipulate
    rotations in 3D space efficiently. Quaternions are used extensively
    in physics, computer graphics, robotics, and aerospace engineering
    to represent orientations and rotations.
    """

    def __init__(
        self,
        x: sm.ScalarData = 0,
        y: sm.ScalarData = 0,
        z: sm.ScalarData = 0,
        w: sm.ScalarData = 1,
        reference_frame: Optional[KinematicStructureEntity] = None,
    ):
        """

        :param x: X-coordinate of the point. Defaults to 0.
        :param y: Y-coordinate of the point. Defaults to 0.
        :param z: Z-coordinate of the point. Defaults to 0.
        :param w: Z-coordinate of the point. Defaults to 1.
        :param reference_frame:
        It will be visualized at the origin of the vis_frame
        """
        self.casadi_sx = sm.to_sx([x, y, z, w])
        self.reference_frame = reference_frame

    def _verify_type(self):
        if self.shape != (4, 1):
            raise WrongDimensionsError(
                expected_dimensions=(4, 1), actual_dimensions=self.shape
            )

    def __neg__(self) -> Quaternion:
        return Quaternion.from_iterable(self.casadi_sx.__neg__())

    @classmethod
    def _from_json(cls, data: Dict[str, Any], **kwargs) -> Self:
        reference_frame = cls._parse_optional_frame_from_json(
            data, key="reference_frame_id", **kwargs
        )
        return cls.from_iterable(
            data["data"],
            reference_frame=reference_frame,
        )

    def to_json(self) -> Dict[str, Any]:
        if not self.is_constant():
            raise SpatialTypeNotJsonSerializable(self)
        result = super().to_json()
        if self.reference_frame is not None:
            result["reference_frame_id"] = to_json(self.reference_frame.id)
        result["data"] = self.to_np().tolist()
        return result

    @classmethod
    def from_iterable(
        cls,
        data: sm.VectorData,
        reference_frame: Optional[KinematicStructureEntity] = None,
    ) -> Quaternion:
        """
        Creates an instance of Quaternion from provided iterable data.

        This class method is used to construct a Quaternion object by processing the given
        data and optionally assigning a reference frame. The data can represent
        different array-like objects compatible with the desired format for a Quaternion
        instance. The provided iterable or array should follow a 1D structure to avoid
        raised errors.

        :param data: The array-like data or object such as a list, tuple, or numpy array
            used to initialize the Quaternion instance.
        :param reference_frame: A reference to a `KinematicStructureEntity` object,
            representing the frame of reference for the Quaternion instance. If the data
            has a `reference_frame` attribute, and this parameter is not specified,
            it will be taken from the data.

        :return: Returns an instance of Quaternion initialized with the processed data
            and an optional reference frame.
        """
        if hasattr(data, "shape") and len(data.shape) > 1 and data.shape[1] != 1:
            raise ValueError("The iterable must be a 1d list, tuple or array")
        return cls(
            x=data[0],
            y=data[1],
            z=data[2],
            w=data[3],
            reference_frame=reference_frame,
        )

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
    def z(self) -> sm.Scalar:
        return self[2]

    @z.setter
    def z(self, value: sm.ScalarData):
        self[2] = value

    @property
    def w(self) -> sm.Scalar:
        return self[3]

    @w.setter
    def w(self, value: sm.ScalarData):
        self[3] = value

    @classmethod
    def from_axis_angle(
        cls,
        axis: Vector3,
        angle: sm.ScalarData,
        reference_frame: Optional[KinematicStructureEntity] = None,
    ) -> Quaternion:
        """
        Creates a quaternion from an axis-angle representation.

        This method uses the axis of rotation and the rotation angle (in radians)
        to construct a quaternion representation of the rotation. Optionally,
        a reference frame can be specified to which the resulting quaternion is
        associated.

        :param axis: A 3D vector representing the axis of rotation.
        :param angle: The rotation angle in radians.
        :param reference_frame: An optional reference frame entity associated
            with the quaternion, if applicable.
        :return: A quaternion representing the rotation defined by
            the given axis and angle.
        """
        half_angle = angle / 2
        return cls(
            x=axis[0] * sm.sin(half_angle),
            y=axis[1] * sm.sin(half_angle),
            z=axis[2] * sm.sin(half_angle),
            w=sm.cos(half_angle),
            reference_frame=reference_frame,
        )

    @classmethod
    def from_rpy(
        cls,
        roll: sm.ScalarData,
        pitch: sm.ScalarData,
        yaw: sm.ScalarData,
        reference_frame: Optional[KinematicStructureEntity] = None,
    ) -> Quaternion:
        """
        Creates a Quaternion instance from specified roll, pitch, and yaw angles.

        The method computes the quaternion representation of the given roll, pitch,
        and yaw angles using trigonometric transformations based on their
        half-angle values for efficient calculations.

        :param roll: The roll angle in radians.
        :param pitch: The pitch angle in radians.
        :param yaw: The yaw angle in radians.
        :param reference_frame: Optional reference frame entity associated with
            the quaternion.
        :return: A Quaternion instance representing the rotation defined by the
            specified roll, pitch, and yaw angles.
        """
        roll = sm.to_sx(roll)
        pitch = sm.to_sx(pitch)
        yaw = sm.to_sx(yaw)
        roll_half = sm.to_sx(roll / 2.0)
        pitch_half = sm.to_sx(pitch / 2.0)
        yaw_half = sm.to_sx(yaw / 2.0)

        c_roll = ca.cos(roll_half)
        s_roll = ca.sin(roll_half)
        c_pitch = ca.cos(pitch_half)
        s_pitch = ca.sin(pitch_half)
        c_yaw = ca.cos(yaw_half)
        s_yaw = ca.sin(yaw_half)

        cc = c_roll * c_yaw
        cs = c_roll * s_yaw
        sc = s_roll * c_yaw
        ss = s_roll * s_yaw

        x = c_pitch * sc - s_pitch * cs
        y = c_pitch * ss + s_pitch * cc
        z = c_pitch * cs - s_pitch * sc
        w = c_pitch * cc + s_pitch * ss

        return cls(x=x, y=y, z=z, w=w, reference_frame=reference_frame)

    @classmethod
    def from_rotation_matrix(
        cls, r: Union[RotationMatrix, HomogeneousTransformationMatrix]
    ) -> Quaternion:
        """
        Creates a Quaternion object initialized from a given rotation matrix.

        This method constructs a quaternion representation of the provided rotation matrix. It is designed to handle
        different cases of rotation matrix configurations to ensure numerical stability during computation. The resultant
        quaternion adheres to the expected mathematical relationship with the given rotation matrix.

        :param r: The input matrix representing a rotation. It can be either a `RotationMatrix` or `TransformationMatrix`.
                  This matrix is expected to have a valid mathematical structure typical for rotation matrices.

        :return: A new instance of `Quaternion` corresponding to the given rotation matrix `r`.
        """
        q = rotation_matrix_to_quaternion(r.to_generic_matrix())
        return cls.from_iterable(q, reference_frame=r.reference_frame)

    def conjugate(self) -> Quaternion:
        return Quaternion(
            x=-self[0],
            y=-self[1],
            z=-self[2],
            w=self[3],
            reference_frame=self.reference_frame,
        )

    def multiply(self, q: Quaternion) -> Quaternion:
        return Quaternion(
            x=self.x * q.w + self.y * q.z - self.z * q.y + self.w * q.x,
            y=-self.x * q.z + self.y * q.w + self.z * q.x + self.w * q.y,
            z=self.x * q.y - self.y * q.x + self.z * q.w + self.w * q.z,
            w=-self.x * q.x - self.y * q.y - self.z * q.z + self.w * q.w,
            reference_frame=self.reference_frame,
        )

    def diff(self, q: Quaternion) -> Quaternion:
        """
        :return: quaternion p, such that self*p=q
        """
        return self.conjugate().multiply(q)

    def normalize(self) -> None:
        norm_ = self.to_generic_vector().norm()
        self.x /= norm_
        self.y /= norm_
        self.z /= norm_
        self.w /= norm_

    def to_axis_angle(self) -> Tuple[Vector3, sm.Scalar]:
        self.normalize()
        w2 = sm.sqrt(1 - self.w**2)
        m = sm.if_eq_zero(w2, sm.Scalar(1), w2)  # avoid /0
        angle = sm.if_eq_zero(
            w2, sm.Scalar(0), sm.Scalar(2 * sm.acos(sm.limit(self.w, -1, 1)))
        )
        x = sm.if_eq_zero(w2, sm.Scalar(0), self.x / m)
        y = sm.if_eq_zero(w2, sm.Scalar(0), self.y / m)
        z = sm.if_eq_zero(w2, sm.Scalar(1), self.z / m)
        return (
            Vector3(x=x, y=y, z=z, reference_frame=self.reference_frame),
            angle,
        )

    def to_generic_vector(self) -> sm.Vector:
        return sm.Vector.from_casadi_sx(self.casadi_sx)

    def to_rotation_matrix(self) -> RotationMatrix:
        return RotationMatrix.from_quaternion(self)

    def to_rpy(self) -> Tuple[sm.Scalar, sm.Scalar, sm.Scalar]:
        return self.to_rotation_matrix().to_rpy()

    def dot(self, other: Quaternion) -> sm.Scalar:
        if isinstance(other, Quaternion):
            return sm.Scalar(ca.mtimes(self.casadi_sx.T, other.casadi_sx))
        return NotImplemented

    def slerp(self, other: Quaternion, t: sm.ScalarData) -> Quaternion:
        """
        Spherical linear interpolation that takes into account that q == -q
        t=0 will return self and t=1 will return other.
        :param other: the other quaternion
        :param t: float, 0-1
        :return: 4x1 Matrix; Return spherical linear interpolation between two quaternions.
        """
        cos_half_theta = self.dot(other)

        if0 = -cos_half_theta
        other = sm.if_greater_zero(if0, -other, other)
        cos_half_theta = sm.if_greater_zero(if0, -cos_half_theta, cos_half_theta)

        if1 = abs(cos_half_theta) - 1.0

        # enforce acos(x) with -1 < x < 1
        cos_half_theta = min(sm.Scalar(1), cos_half_theta)
        cos_half_theta = max(sm.Scalar(-1), cos_half_theta)

        half_theta = sm.acos(cos_half_theta)

        sin_half_theta = sm.sqrt(1.0 - cos_half_theta * cos_half_theta)
        if2 = 0.001 - abs(sin_half_theta)

        ratio_a = (sm.sin((1.0 - t) * half_theta)).safe_division(sin_half_theta)
        ratio_b = sm.sin(t * half_theta).safe_division(sin_half_theta)

        mid_quaternion = Quaternion.from_iterable(
            self.to_generic_vector() * 0.5 + other.to_generic_vector() * 0.5
        )
        slerped_quaternion = Quaternion.from_iterable(
            self.to_generic_vector() * ratio_a + other.to_generic_vector() * ratio_b
        )

        return sm.if_greater_eq_zero(
            if1, self, sm.if_greater_zero(if2, mid_quaternion, slerped_quaternion)
        )


@dataclass(eq=False, init=False)
class Pose(sm.SymbolicMathType, SpatialType, SubclassJSONSerializer):

    def __init__(
        self,
        position: Optional[Point3] = None,
        orientation: Optional[Quaternion] = None,
        reference_frame: Optional[KinematicStructureEntity] = None,
    ):
        """
        Initialize a 3D point with orientation and a reference frame in a kinematic structure.

        :param position: The 3D position of the point, represented as a Point3 object.
                        If None, a default position is assumed.
        :param orientation: The orientation of the point in 3D space represented as a Quaternion.
                            If None, default orientation is assumed.
        :param reference_frame: The reference frame (kinematic structure entity) relative to which
                                this point is defined. This may be None if the point is not tied
                                to any specific reference frame.
        """
        if position is None:
            position = Point3()
        if orientation is None:
            orientation = Quaternion()
        transformation_matrix = (
            HomogeneousTransformationMatrix.from_point_rotation_matrix(
                point=position, rotation_matrix=orientation.to_rotation_matrix()
            )
        )
        self._casadi_sx = transformation_matrix._casadi_sx
        self.reference_frame = reference_frame

    def _verify_type(self):
        if self.shape != (4, 4):
            raise WrongDimensionsError(
                expected_dimensions=(4, 4), actual_dimensions=self.shape
            )
        self[3, 0] = 0.0
        self[3, 1] = 0.0
        self[3, 2] = 0.0
        self[3, 3] = 1.0

    @classmethod
    def _from_json(cls, data: Dict[str, Any], **kwargs) -> Self:
        reference_frame = cls._parse_optional_frame_from_json(
            data, key="reference_frame_id", **kwargs
        )
        return cls.from_xyz_quaternion(
            *data["position"][:3],
            *data["rotation"],
            reference_frame=reference_frame,
        )

    def to_json(self) -> Dict[str, Any]:
        if not self.is_constant():
            raise SpatialTypeNotJsonSerializable(self)
        result = super().to_json()
        if self.reference_frame is not None:
            result["reference_frame_id"] = to_json(self.reference_frame.id)
        result["position"] = self.to_position().to_np().tolist()
        result["rotation"] = self.to_quaternion().to_np().tolist()
        return result

    @classmethod
    def from_xyz_rpy(
        cls,
        x: sm.ScalarData = 0,
        y: sm.ScalarData = 0,
        z: sm.ScalarData = 0,
        roll: sm.ScalarData = 0,
        pitch: sm.ScalarData = 0,
        yaw: sm.ScalarData = 0,
        reference_frame: Optional[KinematicStructureEntity] = None,
    ) -> Self:
        """
        Creates a Pose object from position (x, y, z) and Euler angles
        (roll, pitch, yaw) values. The function also accepts optional reference and
        child frame parameters.

        :param x: The x-coordinate of the position
        :param y: The y-coordinate of the position
        :param z: The z-coordinate of the position
        :param roll: The rotation around the x-axis
        :param pitch: The rotation around the y-axis
        :param yaw: The rotation around the z-axis
        :param reference_frame: The reference frame for the transformation
        :return: A Pose object created using the provided
            position and orientation values
        """
        p = Point3(x=x, y=y, z=z)
        r = Quaternion.from_rpy(roll, pitch, yaw)
        return cls(p, r, reference_frame=reference_frame)

    @classmethod
    def from_xyz_quaternion(
        cls,
        pos_x: sm.ScalarData = 0,
        pos_y: sm.ScalarData = 0,
        pos_z: sm.ScalarData = 0,
        quat_x: sm.ScalarData = 0,
        quat_y: sm.ScalarData = 0,
        quat_z: sm.ScalarData = 0,
        quat_w: sm.ScalarData = 1,
        reference_frame: Optional[KinematicStructureEntity] = None,
    ) -> Self:
        """
        Creates a `Pose` instance from the provided position coordinates and quaternion
        values representing rotation. This method constructs a 3D point for the position and a rotation
        matrix derived from the quaternion, and initializes the transformation matrix with these along
        with optional reference and child frame entities.

        :param pos_x: X coordinate of the position in space.
        :param pos_y: Y coordinate of the position in space.
        :param pos_z: Z coordinate of the position in space.
        :param quat_w: W component of the quaternion representing rotation.
        :param quat_x: X component of the quaternion representing rotation.
        :param quat_y: Y component of the quaternion representing rotation.
        :param quat_z: Z component of the quaternion representing rotation.
        :param reference_frame: Optional reference frame for the transformation matrix.
        :return: A `Pose` object constructed from the given parameters.
        """
        p = Point3(x=pos_x, y=pos_y, z=pos_z)
        r = Quaternion(w=quat_w, x=quat_x, y=quat_y, z=quat_z)
        return cls(p, r, reference_frame=reference_frame)

    @classmethod
    def from_xyz_axis_angle(
        cls,
        x: sm.ScalarData = 0,
        y: sm.ScalarData = 0,
        z: sm.ScalarData = 0,
        axis: Vector3 | sm.NumericalVector = None,
        angle: sm.ScalarData = 0,
        reference_frame: Optional[KinematicStructureEntity] = None,
    ) -> Self:
        """
        Creates an instance of the class from x, y, z coordinates, axis and angle.

        This class method generates an object using provided spatial coordinates and a
        rotation defined by an axis and angle. The resulting object is defined with
        a specified reference frame and child frame.

        :param x: Initial x-coordinate.
        :param y: Initial y-coordinate.
        :param z: Initial z-coordinate.
        :param axis: Vector defining the axis of rotation. Defaults to Vector3(0, 0, 1) if not specified.
        :param angle: Angle of rotation around the specified axis, in radians.
        :param reference_frame: Reference frame entity to be associated with the object.
        :return: An instance of the class with the specified transformations applied.
        """
        if axis is None:
            axis = Vector3(0, 0, 1)
        rotation_matrix = Quaternion.from_axis_angle(axis=axis, angle=angle)
        point = Point3(x=x, y=y, z=z)
        return cls(
            position=point,
            orientation=rotation_matrix,
            reference_frame=reference_frame,
        )

    @property
    def x(self) -> sm.Scalar:
        return self[0, 3]

    @x.setter
    def x(self, value: sm.ScalarData):
        self[0, 3] = value

    @property
    def y(self) -> sm.Scalar:
        return self[1, 3]

    @y.setter
    def y(self, value: sm.ScalarData):
        self[1, 3] = value

    @property
    def z(self) -> sm.Scalar:
        return self[2, 3]

    @z.setter
    def z(self, value: sm.ScalarData):
        self[2, 3] = value

    def to_position(self) -> Point3:
        result = Point3.from_iterable(
            self[:4, 3:], reference_frame=self.reference_frame
        )
        return result

    def to_rotation_matrix(self) -> RotationMatrix:
        return RotationMatrix(data=self, reference_frame=self.reference_frame)

    def to_quaternion(self) -> Quaternion:
        return self.to_rotation_matrix().to_quaternion()

    def to_homogeneous_matrix(self) -> Self:
        return HomogeneousTransformationMatrix(
            data=self, reference_frame=self.reference_frame
        )


@sm.substitution_cache
def rotation_matrix_to_quaternion(r: Matrix):
    """
    This method constructs a quaternion representation of the provided rotation matrix. It is designed to handle
    different cases of rotation matrix configurations to ensure numerical stability during computation. The resultant
    quaternion adheres to the expected mathematical relationship with the given rotation matrix.

    .. note:: this method uses basic symbolic types and substitution caching, because it is building a large computational graph.

    :param r: The input 3x3 matrix representing a rotation.
    :return: A new instance of `Vector` corresponding to the quaternion for the given rotation matrix `r`.
    """
    q = sm.Vector(data=(0, 0, 0, 0))
    t = r.trace()

    if0 = t - r[3, 3]

    if1 = r[1, 1] - r[0, 0]

    m_i_i = sm.if_greater_zero(if1, r[1, 1], r[0, 0])
    m_i_j = sm.if_greater_zero(if1, r[1, 2], r[0, 1])
    m_i_k = sm.if_greater_zero(if1, r[1, 0], r[0, 2])

    m_j_i = sm.if_greater_zero(if1, r[2, 1], r[1, 0])
    m_j_j = sm.if_greater_zero(if1, r[2, 2], r[1, 1])
    m_j_k = sm.if_greater_zero(if1, r[2, 0], r[1, 2])

    m_k_i = sm.if_greater_zero(if1, r[0, 1], r[2, 0])
    m_k_j = sm.if_greater_zero(if1, r[0, 2], r[2, 1])
    m_k_k = sm.if_greater_zero(if1, r[0, 0], r[2, 2])

    if2 = r[2, 2] - m_i_i

    m_i_i = sm.if_greater_zero(if2, r[2, 2], m_i_i)
    m_i_j = sm.if_greater_zero(if2, r[2, 0], m_i_j)
    m_i_k = sm.if_greater_zero(if2, r[2, 1], m_i_k)

    m_j_i = sm.if_greater_zero(if2, r[0, 2], m_j_i)
    m_j_j = sm.if_greater_zero(if2, r[0, 0], m_j_j)
    m_j_k = sm.if_greater_zero(if2, r[0, 1], m_j_k)

    m_k_i = sm.if_greater_zero(if2, r[1, 2], m_k_i)
    m_k_j = sm.if_greater_zero(if2, r[1, 0], m_k_j)
    m_k_k = sm.if_greater_zero(if2, r[1, 1], m_k_k)

    t = sm.if_greater_zero(if0, t, m_i_i - (m_j_j + m_k_k) + r[3, 3])
    q[0] = sm.if_greater_zero(
        if0,
        r[2, 1] - r[1, 2],
        sm.if_greater_zero(
            if2, m_i_j + m_j_i, sm.if_greater_zero(if1, m_k_i + m_i_k, t)
        ),
    )
    q[1] = sm.if_greater_zero(
        if0,
        r[0, 2] - r[2, 0],
        sm.if_greater_zero(
            if2, m_k_i + m_i_k, sm.if_greater_zero(if1, t, m_i_j + m_j_i)
        ),
    )
    q[2] = sm.if_greater_zero(
        if0,
        r[1, 0] - r[0, 1],
        sm.if_greater_zero(
            if2, t, sm.if_greater_zero(if1, m_i_j + m_j_i, m_k_i + m_i_k)
        ),
    )
    q[3] = sm.if_greater_zero(if0, t, m_k_j - m_j_k)

    q *= 0.5 / sm.sqrt(t * r[3, 3])
    return q


# %% type hints

GenericSpatialType = TypeVar(
    "GenericSpatialType",
    bound=SpatialType,
)

GenericHomogeneousSpatialType = TypeVar(
    "GenericHomogeneousSpatialType",
    Point3,
    Vector3,
    HomogeneousTransformationMatrix,
    RotationMatrix,
)

GenericRotatableSpatialType = TypeVar(
    "GenericRotatableSpatialType",
    Vector3,
    HomogeneousTransformationMatrix,
    RotationMatrix,
)
