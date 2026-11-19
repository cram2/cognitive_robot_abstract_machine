"""
The symbolic casadi spatial types of three-dimensional space, such as points, vectors,
rotations, poses, displacements, twists and screw axes given in a frame, the changes
between frames, and how each change acts on a value. Their counterparts on a plane are
in :mod:`krrood_spatial_math.spatial_types_2_dimensional`.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from copy import deepcopy, copy
from dataclasses import dataclass, field
from enum import Enum, StrEnum

import casadi as ca
import numpy as np
from typing_extensions import Any, overload, Self, Callable, TypeVar

import krrood.symbolic_math.symbolic_math as sm
from krrood.adapters.json_serializer import SubclassJSONSerializer
from krrood.patterns.interface import Interface
from krrood.patterns.mixin import Mixin
from krrood.patterns.classproperty import classproperty
from krrood.symbolic_math.exceptions import (
    WrongDimensionsError,
    UnsupportedOperationError,
)
from krrood.symbolic_math.symbolic_math import Matrix, to_sx
from krrood_spatial_math.cartesian_frame_3_dimensional import (
    CartesianFrame3Dimensional,
    CartesianFrame3DimensionalReference,
    EmbeddedPlaneRelation,
)
from krrood_spatial_math.coordinate_system import (
    GenericCartesianFrame,
    CoordinateRepresentation,
    RigidCoordinateChange,
)
from krrood_spatial_math.line import Line, LineSegment
from krrood_spatial_math.point import CartesianPoint
from krrood_spatial_math.vector import CartesianVector
from krrood_spatial_math.exceptions import (
    InsufficientVectorsError,
    ReferenceFrameMismatchError,
)


class _ConstantMatrixParts(ca.SX, Enum):
    """
    The entries a homogeneous matrix always holds, whatever it represents.

    Assigned as whole rows and columns rather than element by element, because
    :meth:`SymbolicSpatialType._verify_type` runs on every intermediate product and each casadi
    element write costs about as much as the whole slice.
    """

    HOMOGENEOUS_BOTTOM_ROW = [[0.0, 0.0, 0.0, 1.0]]
    """
    The last row every 4x4 homogeneous matrix ends in.
    """

    ZERO_TRANSLATION = [[0.0], [0.0], [0.0]]
    """
    The translation column of a matrix that carries rotation only.
    """


class SpatialFrameKey(StrEnum):
    """
    The entities a spatial type is expressed against.

    Both halves of the serialization name the entity the same way, so the name lives
    here rather than at every place that reads or writes one.
    """

    REFERENCE = "reference_frame"
    """The entity a value is given relative to."""

    CHILD = "child_frame"
    """The entity a transformation points at."""


@dataclass(eq=False, repr=False)
class SymbolicSpatialType(ABC):
    """
    A symbolic matrix of spatial mathematics, given relative to a
    :class:`~krrood_spatial_math.cartesian_frame_3_dimensional.CartesianFrame3Dimensional`: a value given in
    that frame, or a change into it.
    """

    casadi_sx: ca.SX
    """
    Implement Symbolic Math protocol.
    """

    reference_frame: CartesianFrame3Dimensional | None = field(
        kw_only=True, default=None
    )
    """
    The reference frame associated with the object.

    Can be None if no reference frame is required or applicable.
    """

    def transform(
        self,
        target_frame_T_reference_frame: HomogeneousTransformationMatrix3Dimensional,
    ) -> Self:
        """
        This object re-expressed in ``target_frame_T_reference_frame``'s reference
        frame.

        :param target_frame_T_reference_frame: The transformation from this object's
            reference frame to the frame it should be expressed in.
        :return: The object in the transformation's reference frame.
        """
        return target_frame_T_reference_frame @ self

    def ensure_same_reference_frame(
        self, other: SymbolicSpatialType, context: str
    ) -> None:
        """
        :param other: A value combined with this one, whose coordinates must be given in
            the same frame.
        :param context: What ``other`` is to the operation, for the error message.
        :raises ReferenceFrameMismatchError: If both values have a reference frame and
            the frames differ.
        """
        if (
            self.reference_frame is not None
            and other.reference_frame is not None
            and other.reference_frame is not self.reference_frame
        ):
            raise ReferenceFrameMismatchError(
                expected_frame=self.reference_frame,
                actual_frame=other.reference_frame,
                context=context,
            )

    def to_json(self, **kwargs) -> dict[str, Any]:
        """
        The json of a spatial type: the frames it is expressed in, and its value, as
        numbers if it is constant and as the expression computing it otherwise.
        """
        result = super().to_json(**kwargs)
        self._frames_to_json(result)
        if self.is_constant():
            result.update(self._constant_to_json())
        else:
            result[sm.SymbolicMathJSONKey.EXPRESSION] = (
                sm.ExpressionGraph.from_expression(expression=self).to_json(**kwargs)
            )
        return result

    def _frames_to_json(self, data: dict[str, Any]) -> None:
        """
        Put references to the frames this value is expressed in into its json.

        :param data: The json of this value.
        """
        if self.reference_frame is not None:
            CartesianFrame3DimensionalReference(
                subject=SpatialFrameKey.REFERENCE
            ).write(data=data, identified_object=self.reference_frame)

    @abstractmethod
    def _constant_to_json(self) -> dict[str, Any]:
        """
        :return: The json entries holding the numbers of this constant value.
        """

    @classmethod
    def _from_json(cls, *, data: dict[str, Any], **kwargs: Any) -> Self:
        """
        Extends :meth:`SubclassJSONSerializer._from_json` by reading a constant value
        from its numbers and a symbolic one from its expression graph.
        """
        if sm.SymbolicMathJSONKey.EXPRESSION not in data:
            return cls._from_constant_json(data=data, **kwargs)
        result = sm.ExpressionGraph.from_json(
            data=data[sm.SymbolicMathJSONKey.EXPRESSION], **kwargs
        ).to_expression(expression_type=cls)
        result._frames_from_json(data, **kwargs)
        return result

    def _frames_from_json(self, data: dict[str, Any], **kwargs) -> None:
        """
        Set the frames this value is expressed in from references in its json.

        :param data: The json of this value.
        :param kwargs: The kwargs of the ``_from_json`` that is reading it.
        """
        self.reference_frame = self._parse_optional_frame_from_json(
            data, frame=SpatialFrameKey.REFERENCE, **kwargs
        )

    @classmethod
    @abstractmethod
    def _from_constant_json(cls, *, data: dict[str, Any], **kwargs: Any) -> Self:
        """
        :param data: The json of a constant value, as :meth:`_constant_to_json` wrote it.
        :param kwargs: The kwargs of the ``_from_json`` that is reading it.
        :return: The constant value.
        """

    @classmethod
    def _parse_optional_frame_from_json(
        cls, data: dict[str, Any], frame: SpatialFrameKey, **kwargs
    ) -> CartesianFrame3Dimensional | None:
        """
        Resolve a kinematic structure entity a serialized spatial type refers to.

        :param data: parsed JSON data
        :param frame: which of its frames the spatial type is asked for
        :param kwargs: addition kwargs of _from_json
        :return: None if the spatial type refers to no such entity.
        """
        reference = CartesianFrame3DimensionalReference(subject=frame)
        if not data.get(reference.id_key):
            return None
        return reference.resolve(data=data, **kwargs)

    @staticmethod
    def _ensure_consistent_frame(
        spatial_objects: list[SymbolicSpatialType | None],
    ) -> CartesianFrame3Dimensional | None:
        """
        Ensures that all provided spatial objects have a consistent reference frame.

        If a mismatch in the reference frames is detected among the non-null spatial
        objects, an exception is raised. If the list contains only null objects, None is
        returned.

        This method is primarily used to validate the reference frames of spatial
        objects before proceeding with further operations.

        :param spatial_objects: A list containing zero or more spatial objects, which
            can either be instances of ReferenceFrameMixin or None.
        :return: The common reference frame of the spatial objects if consistent, or
            None if no valid reference frame exists.
        :raises SpatialTypesError: Raised when the reference frames of provided input
            spatial objects are inconsistent.
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
                    raise ReferenceFrameMismatchError(
                        expected_frame=reference_frame,
                        actual_frame=spatial_object.reference_frame,
                        context=f"input parameter of type {type(spatial_object).__name__}",
                    )
        return reference_frame

    def _copy_with_data(self, data: ca.SX) -> Self:
        """
        Build a copy of this spatial object around already-copied matrix data.

        Bypasses the constructor: `data` comes from an object of this very type, so it
        already satisfies :meth:`_verify_type`, and it contains the same free variables,
        which are carried over rather than re-derived from the expression graph.

        :param data: The copied matrix data the result takes ownership of.
        :return: A copy of this object holding `data`.
        """
        result = type(self).__new__(type(self))
        result._casadi_sx = data
        result.reference_frame = self.reference_frame
        result.pinned_free_variables = self.pinned_free_variables
        return result

    def __deepcopy__(self, memo) -> Self:
        """
        Copy the matrix, but share the frames.

        Frames are references to kinematic structure entities, which belong to the world
        rather than to this object, so a deep copy must not duplicate them.
        """
        if id(self) in memo:
            return memo[id(self)]
        with sm.CasadiLock():
            result = self._copy_with_data(deepcopy(self.casadi_sx))
        memo[id(self)] = result
        return result


GenericSpatialType = TypeVar("GenericSpatialType", bound=SymbolicSpatialType)
"""
The kind of symbolic spatial value or change a method returns unchanged in type, such as
the value a frame expresses in its own coordinates.
"""


@dataclass(eq=False, repr=False)
class SymbolicSpatialChange(
    sm.SymbolicMathType,
    SymbolicSpatialType,
    RigidCoordinateChange[GenericCartesianFrame],
    SubclassJSONSerializer,
    ABC,
):
    """
    A rigid change, as a symbolic matrix, of coordinates given in a frame derived from
    its :attr:`child_frame` into the same kind of frame derived from its
    :attr:`~SymbolicSpatialType.reference_frame`.
    """

    child_frame: CartesianFrame3Dimensional | None = field(kw_only=True, default=None)
    """
    The frame the coordinates this change acts on are given in.
    """

    def _frames_to_json(self, data: dict[str, Any]) -> None:
        super()._frames_to_json(data)
        if self.child_frame is not None:
            CartesianFrame3DimensionalReference(subject=SpatialFrameKey.CHILD).write(
                data=data, identified_object=self.child_frame
            )

    def _frames_from_json(self, data: dict[str, Any], **kwargs) -> None:
        super()._frames_from_json(data, **kwargs)
        self.child_frame = self._parse_optional_frame_from_json(
            data, frame=SpatialFrameKey.CHILD, **kwargs
        )

    def _copy_with_data(self, data: ca.SX) -> Self:
        result = super()._copy_with_data(data)
        result.child_frame = self.child_frame
        return result

    def ensure_operand_in_child_frame(self, operand: SymbolicSpatialType) -> None:
        """
        :param operand: A value or change this change is applied to.
        :raises ReferenceFrameMismatchError: If both this change's child frame and the
            operand's reference frame are set and differ.
        """
        if (
            self.child_frame is not None
            and operand.reference_frame is not None
            and operand.reference_frame is not self.child_frame
        ):
            raise ReferenceFrameMismatchError(
                expected_frame=self.child_frame,
                actual_frame=operand.reference_frame,
                context=f"the {type(operand).__name__} this change is applied to",
            )


@dataclass(eq=False, repr=False, kw_only=True)
class SymbolicSpatialValue3Dimensional(
    sm.SymbolicMathType,
    SymbolicSpatialType,
    CoordinateRepresentation[CartesianFrame3Dimensional],
    SubclassJSONSerializer,
    ABC,
):
    """
    A value, as a symbolic matrix, such as a point or a pose, given in its
    :attr:`~SymbolicSpatialType.reference_frame`, its coordinate system.
    """

    @property
    def coordinate_system(self) -> CartesianFrame3Dimensional | None:
        return self.reference_frame


# %% how a change or a rotation acts on a value


class SpecialOrthogonalGroupAction(Interface):
    """
    A value or change the rotation group SO(3) acts on: a :class:`RotationMatrix` turns
    it about the origin of its reference frame, giving one of the same type.
    """

    @abstractmethod
    def apply_rotation(self, *, rotation: RotationMatrix) -> Self:
        """
        :param rotation: The rotation from this value's reference frame to the rotation's
            reference frame.
        :return: This value turned by ``rotation``, in the reference frame of
            ``rotation``.
        """


GenericSpecialOrthogonalGroupAction = TypeVar(
    "GenericSpecialOrthogonalGroupAction", bound=SpecialOrthogonalGroupAction
)
"""
The kind of value or change a rotation is applied to.
"""


class SpecialEuclideanGroupAction(SpecialOrthogonalGroupAction, Interface):
    """
    A value or change the rigid motion group SE(3) acts on: a
    :class:`HomogeneousTransformationMatrix3Dimensional` moves it, giving one of the same type.

    SO(3) is the subgroup of SE(3) that keeps the origin, so it acts on it too.
    """

    @abstractmethod
    def apply_transformation(
        self, *, transformation: HomogeneousTransformationMatrix3Dimensional
    ) -> Self:
        """
        :param transformation: A transformation whose child frame is this value's
            reference frame.
        :return: This value, given in the reference frame of ``transformation``.
        """


GenericSpecialEuclideanGroupAction = TypeVar(
    "GenericSpecialEuclideanGroupAction", bound=SpecialEuclideanGroupAction
)
"""
The kind of value or change a transformation is applied to.
"""


class HomogeneousCoordinates(SpecialEuclideanGroupAction, Mixin):
    """
    A value stored as homogeneous coordinates, which a transformation or a rotation acts
    on by multiplying them from the left.
    """

    def apply_transformation(
        self, *, transformation: HomogeneousTransformationMatrix3Dimensional
    ) -> Self:
        return self._left_product(matrix=transformation)

    def apply_rotation(self, *, rotation: RotationMatrix) -> Self:
        return self._left_product(matrix=rotation)

    def _left_product(
        self, *, matrix: HomogeneousTransformationMatrix3Dimensional | RotationMatrix
    ) -> Self:
        """
        :param matrix: The transformation or rotation applied to this value.
        :return: The product of ``matrix`` and this value, of this value's type, in the
            reference frame of ``matrix``.
        """
        result = type(self).from_casadi_sx(
            casadi_sx=ca.mtimes(matrix.casadi_sx, self.casadi_sx)
        )
        result.reference_frame = matrix.reference_frame
        return result


class Orientation(SpecialEuclideanGroupAction, Mixin):
    """
    An orientation, which a transformation turns by its rotation and never shifts, since
    an orientation has no position.
    """

    def apply_transformation(
        self, *, transformation: HomogeneousTransformationMatrix3Dimensional
    ) -> Self:
        return self.apply_rotation(rotation=transformation.rotation_matrix)


@dataclass(eq=False, init=False, repr=False)
class HomogeneousTransformationMatrix3Dimensional(
    SymbolicSpatialChange[CartesianFrame3Dimensional],
    SpecialEuclideanGroupAction,
):
    """
    Represents a 4x4 transformation matrix used in kinematics and transformations.

    A `TransformationMatrix` encapsulates relationships between a parent coordinate
    system (reference frame) and a child coordinate system through rotation and
    translation. It provides utilities to derive transformations, compute dot products,
    and create transformations from various inputs such as Euler angles or quaternions.

    It is only a change of coordinates from its :attr:`child_frame` into its
    :attr:`reference_frame`, never a value given in a frame: the value it corresponds to
    is a :class:`Pose3Dimensional`.

    .. warning::
        The matrix must be rigid: its upper left 3x3 block a rotation and its last row
        ``[0, 0, 0, 1]``. :meth:`inverse` and the rotation it is composed of are wrong
        for a matrix that scales or shears.
    """

    def __init__(
        self,
        data: sm.MatrixData | None = None,
        reference_frame: CartesianFrame3Dimensional | None = None,
        child_frame: CartesianFrame3Dimensional | None = None,
    ):
        """
        :param data: A 4x4 matrix of some form that represents the rotation matrix.
        """
        self.reference_frame = reference_frame
        self.child_frame = child_frame
        if data is None:
            self._casadi_sx = ca.SX.eye(4)
        elif isinstance(data, SymbolicSpatialType):
            # create a copy if data is a spatial type, because they are often still being used
            self.casadi_sx = copy(data.casadi_sx)
        else:
            self.casadi_sx = sm.to_sx(data)
        super().__post_init__()

    @property
    def source_coordinate_system(self) -> CartesianFrame3Dimensional | None:
        return self.child_frame

    @property
    def target_coordinate_system(self) -> CartesianFrame3Dimensional | None:
        return self.reference_frame

    def _verify_type(self):
        if self.shape != (4, 4):
            raise WrongDimensionsError(
                expected_dimensions=(4, 4), actual_dimensions=self.shape
            )
        self[3, :] = _ConstantMatrixParts.HOMOGENEOUS_BOTTOM_ROW

    @classmethod
    def _from_constant_json(cls, data: dict[str, Any], **kwargs) -> Self:
        reference_frame = cls._parse_optional_frame_from_json(
            data, frame=SpatialFrameKey.REFERENCE, **kwargs
        )
        child_frame = cls._parse_optional_frame_from_json(
            data, frame=SpatialFrameKey.CHILD, **kwargs
        )
        return cls.from_xyz_quaternion(
            *data["position"][:3],
            *data["rotation"],
            reference_frame=reference_frame,
            child_frame=child_frame,
        )

    @classmethod
    def create_with_variables(
        cls, name: str, resolver: Callable[[], np.ndarray] | None = None
    ) -> Self:
        """
        Creates a TransformationMatrix object with float variables variables in all
        relevant entries.

        :param name: Name for the variables.
        :param resolver: Callable that returns the actual transformation matrix when
            called.
        :return: TransformationMatrix object with float variables.
        """
        transformation_matrix = []
        for row in range(3):
            column_variables = []
            for column in range(4):
                variable = sm.FloatVariable(
                    name=f"{cls.__name__}_{name}[{row},{column}]",
                )
                column_variables.append(variable)
                if resolver is not None:
                    variable.resolve = lambda: resolver()[row, column]
            transformation_matrix.append(column_variables)
        transformation_matrix.append([0, 0, 0, 1])
        return cls(transformation_matrix)

    def _constant_to_json(self) -> dict[str, Any]:
        return {
            "position": self.position.to_np().tolist(),
            "rotation": self.quaternion.to_np().tolist(),
        }

    @classmethod
    def from_point_rotation_matrix(
        cls,
        point: CartesianPoint3Dimensional | None = None,
        rotation_matrix: RotationMatrix | None = None,
        reference_frame: CartesianFrame3Dimensional | None = None,
        child_frame: CartesianFrame3Dimensional | None = None,
    ) -> HomogeneousTransformationMatrix3Dimensional:
        """
        Constructs a TransformationMatrix object from a given point, a rotation matrix,
        a reference frame, and a child frame.

        :param point: The 3D point used to set the translation part of the
            transformation matrix. If None, no translation is applied.
        :param rotation_matrix: The rotation matrix defines the rotational component of
            the transformation. If None, the identity matrix is assumed.
        :param reference_frame: The reference frame for the transformation matrix. It
            specifies the parent coordinate system.
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
        reference_frame: CartesianFrame3Dimensional | None = None,
        child_frame: CartesianFrame3Dimensional | None = None,
    ) -> HomogeneousTransformationMatrix3Dimensional:
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
        :return: A TransformationMatrix object created using the provided position and
            orientation values
        """
        p = CartesianPoint3Dimensional(x=x, y=y, z=z)
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
        reference_frame: CartesianFrame3Dimensional | None = None,
        child_frame: CartesianFrame3Dimensional | None = None,
    ) -> HomogeneousTransformationMatrix3Dimensional:
        """
        Creates a `TransformationMatrix` instance from the provided position coordinates
        and quaternion values representing rotation. This method constructs a 3D point
        for the position and a rotation matrix derived from the quaternion, and
        initializes the transformation matrix with these along with optional reference
        and child frame entities.

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
        p = CartesianPoint3Dimensional(x=pos_x, y=pos_y, z=pos_z)
        r = RotationMatrix.from_quaternion(
            quaternion=Quaternion(w=quat_w, x=quat_x, y=quat_y, z=quat_z)
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
        axis: Vector3Dimensional | sm.NumericalVector = None,
        angle: sm.ScalarData = 0,
        reference_frame: CartesianFrame3Dimensional | None = None,
        child_frame: CartesianFrame3Dimensional | None = None,
    ) -> Self:
        """
        Creates an instance of the class from x, y, z coordinates, axis and angle.

        This class method generates an object using provided spatial coordinates and a
        rotation defined by an axis and angle. The resulting object is defined with a
        specified reference frame and child frame.

        :param x: Initial x-coordinate.
        :param y: Initial y-coordinate.
        :param z: Initial z-coordinate.
        :param axis: Vector defining the axis of rotation. Defaults to Vector3Dimensional(0, 0, 1)
            if not specified.
        :param angle: Angle of rotation around the specified axis, in radians.
        :param reference_frame: Reference frame entity to be associated with the object.
        :param child_frame: Child frame entity associated with the object.
        :return: An instance of the class with the specified transformations applied.
        """
        if axis is None:
            axis = Vector3Dimensional(0, 0, 1)
        elif not isinstance(axis, Vector3Dimensional):
            axis = Vector3Dimensional.from_iterable(axis)
        rotation_matrix = RotationMatrix.from_axis_angle(
            AxisAngle(axis=axis, angle=angle)
        )
        point = CartesianPoint3Dimensional(x=x, y=y, z=z)
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
        self, other: GenericSpecialEuclideanGroupAction
    ) -> GenericSpecialEuclideanGroupAction:
        """
        This transformation applied to ``other``, a value given in its child frame or a
        change into it.

        :param other: The value or change this transformation is applied to.
        :return: ``other``, given in this transformation's reference frame.
        :raises UnsupportedOperationError: If this transformation does not act on
            ``other``.
        :raises ReferenceFrameMismatchError: If ``other`` is given in a frame other than
            this transformation's child frame.
        """
        if not isinstance(other, SpecialEuclideanGroupAction):
            raise UnsupportedOperationError(operation="dot", left=self, right=other)
        self.ensure_operand_in_child_frame(other)
        return other.apply_transformation(transformation=self)

    def apply_transformation(
        self, *, transformation: HomogeneousTransformationMatrix3Dimensional
    ) -> HomogeneousTransformationMatrix3Dimensional:
        return HomogeneousTransformationMatrix3Dimensional(
            ca.mtimes(transformation.casadi_sx, self.casadi_sx),
            reference_frame=transformation.reference_frame,
            child_frame=self.child_frame,
        )

    def apply_rotation(
        self, *, rotation: RotationMatrix
    ) -> HomogeneousTransformationMatrix3Dimensional:
        return HomogeneousTransformationMatrix3Dimensional(
            ca.mtimes(rotation.casadi_sx, self.casadi_sx),
            reference_frame=rotation.reference_frame,
            child_frame=self.child_frame,
        )

    def __matmul__(
        self, other: GenericSpecialEuclideanGroupAction
    ) -> GenericSpecialEuclideanGroupAction:
        return self.dot(other)

    def inverse(self) -> HomogeneousTransformationMatrix3Dimensional:
        inv = HomogeneousTransformationMatrix3Dimensional(
            child_frame=self.reference_frame, reference_frame=self.child_frame
        )
        inv[:3, :3] = self[:3, :3].T
        inv[:3, 3] = (-inv[:3, :3]).dot(other=self[:3, 3])
        return inv

    @staticmethod
    def invert_numeric_matrix(f1_T_f2: np.ndarray) -> np.ndarray:
        """
        Invert a numeric rigid homogeneous transformation without building a symbolic
        expression.

        :param f1_T_f2: A 4x4 rigid homogeneous transformation from frame f2 to f1.
        :return: f2_T_f1, a new 4x4 array of the same dtype.
        """
        R = f1_T_f2[:3, :3]
        t = f1_T_f2[:3, 3]
        Rt = R.T
        f2_T_f1 = np.empty((4, 4), dtype=f1_T_f2.dtype)
        f2_T_f1[:3, :3] = Rt
        f2_T_f1[:3, 3] = -Rt @ t
        f2_T_f1[3] = [0, 0, 0, 1]
        return f2_T_f1

    @property
    def position(self) -> CartesianPoint3Dimensional:
        """
        :return: The translation this transformation is composed of.
        """
        result = CartesianPoint3Dimensional.from_casadi_sx(
            casadi_sx=self.casadi_sx[:4, 3]
        )
        result.reference_frame = self.reference_frame
        return result

    @property
    def quaternion(self) -> Quaternion:
        """
        :return: The rotation this transformation is composed of.
        """
        return self.rotation_matrix.quaternion

    @property
    def axis_angle(self) -> AxisAngle:
        """
        :return: The rotation this transformation is composed of.
        """
        return self.rotation_matrix.axis_angle

    @property
    def rotation_matrix(self) -> RotationMatrix:
        """
        :return: The rotation this transformation is composed of.
        """
        return RotationMatrix(data=self, reference_frame=self.reference_frame)

    @property
    def embedded_plane_relation(self) -> EmbeddedPlaneRelation:
        """
        How the x-y plane of the child frame lies relative to that of the reference
        frame; :attr:`EmbeddedPlaneRelation.UNKNOWN` for a transformation that is not constant.
        """
        if not self.is_constant():
            return EmbeddedPlaneRelation.UNKNOWN
        child_z_axis = self.rotation_matrix.to_np()[:3, 2]
        if not (
            np.isclose(self.z.to_np(), 0).all() and np.allclose(child_z_axis[:2], 0)
        ):
            return EmbeddedPlaneRelation.DIFFERENT_PLANES
        if child_z_axis[2] < 0:
            return EmbeddedPlaneRelation.SAME_PLANE_OPPOSITE_NORMAL
        return EmbeddedPlaneRelation.SAME_PLANE_SAME_NORMAL

    @property
    def pose(self) -> Pose3Dimensional:
        """
        :return: The pose this transformation moves its reference frame to.
        """
        result = Pose3Dimensional.from_casadi_sx(casadi_sx=copy(self.casadi_sx))
        result.reference_frame = self.reference_frame
        result.pinned_free_variables = list(self.pinned_free_variables)
        return result

    def copy_with_new_reference_frames(
        self,
        new_reference_frame: CartesianFrame3Dimensional,
        new_child_frame: CartesianFrame3Dimensional | None,
    ) -> HomogeneousTransformationMatrix3Dimensional:
        """
        Copies the transformation matrix and changes the reference and child frames.
        """
        return HomogeneousTransformationMatrix3Dimensional(
            data=deepcopy(self.casadi_sx),
            reference_frame=new_reference_frame,
            child_frame=new_child_frame,
        )

    def __hash__(self):
        if self.is_constant():
            return hash(
                (
                    *self.position.to_np().tolist(),
                    *self.quaternion.to_np().tolist(),
                    self.reference_frame,
                )
            )
        return super().__hash__()


@dataclass(eq=False, init=False, repr=False)
class RotationMatrix(
    SymbolicSpatialValue3Dimensional,
    HomogeneousCoordinates,
):
    """
    Class to represent a 4x4 symbolic rotation matrix tied to kinematic references.

    This class provides methods for creating and manipulating rotation matrices within
    the context of kinematic structures. It supports initialization using data such as
    quaternions, axis-angle, other matrices, or directly through vector definitions. The
    primary purpose is to facilitate rotational transformations and computations in a
    symbolic context, particularly for applications like robotic kinematics or
    mechanical engineering.
    """

    def __init__(
        self,
        data: sm.MatrixData | None = None,
        reference_frame: CartesianFrame3Dimensional | None = None,
    ):
        """
        :param data: A 4x4 matrix of some form that represents the rotation matrix.
        :param reference_frame:
        """
        self.reference_frame = reference_frame
        if data is None:
            self._casadi_sx = ca.SX.eye(4)
        else:
            empty_data = to_sx(Matrix.eye(size=4))
            empty_data[:3, :3] = sm.to_sx(data)[:3, :3]
            self._casadi_sx = empty_data
        super().__post_init__()

    def _verify_type(self):
        if self.shape != (4, 4):
            raise WrongDimensionsError(
                expected_dimensions=(4, 4), actual_dimensions=self.shape
            )
        self[:3, 3] = _ConstantMatrixParts.ZERO_TRANSLATION
        self[3, :] = _ConstantMatrixParts.HOMOGENEOUS_BOTTOM_ROW

    @classmethod
    def _from_constant_json(cls, data: dict[str, Any], **kwargs) -> Self:
        reference_frame = cls._parse_optional_frame_from_json(
            data, frame=SpatialFrameKey.REFERENCE, **kwargs
        )
        return Quaternion.from_iterable(
            data["quaternion"],
            reference_frame=reference_frame,
        ).rotation_matrix

    def _constant_to_json(self) -> dict[str, Any]:
        return {
            "quaternion": self.quaternion.to_np().tolist(),
        }

    @classmethod
    def from_axis_angle(cls, axis_angle: AxisAngle) -> RotationMatrix:
        """
        Conversion of unit axis and angle to 4x4 rotation matrix according to:
        https://www.euclideanspace.com/maths/geometry/rotations/conversions/angleToMatrix/index.htm

        :param axis_angle: The rotation to convert.
        :return: The rotation matrix, expressed in the frame of the axis angle.
        """
        # use casadi to prevent a bunch of sm.SymbolicMathType.__init__.py calls
        axis = axis_angle.casadi_sx[:3]
        angle = axis_angle.casadi_sx[3]
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
        return cls(s, reference_frame=axis_angle.reference_frame)

    @classmethod
    def _from_constant_quaternion(cls, quaternion: Quaternion) -> RotationMatrix:
        """
        Build a rotation matrix from a constant (fully numeric) quaternion.

        Uses numpy arithmetic instead of the symbolic CasADi path.  The symbolic path in
        :meth:`from_quaternion` constructs ~36 CasADi sub-expressions even for constant
        inputs; evaluating them in numpy is an order of magnitude faster and covers the
        common case for values loaded from a database.

        :param quaternion: Quaternion whose underlying CasADi expression satisfies
            ``quaternion.is_constant()``.
        :return: The corresponding rotation matrix.
        """
        x, y, z, w = quaternion.to_np().ravel()
        x2, y2, z2, w2 = x * x, y * y, z * z, w * w
        data = np.array(
            [
                [w2 + x2 - y2 - z2, 2 * x * y - 2 * w * z, 2 * x * z + 2 * w * y, 0],
                [2 * x * y + 2 * w * z, w2 - x2 + y2 - z2, 2 * y * z - 2 * w * x, 0],
                [2 * x * z - 2 * w * y, 2 * y * z + 2 * w * x, w2 - x2 - y2 + z2, 0],
                [0, 0, 0, 1],
            ]
        )
        return cls(data=data, reference_frame=quaternion.reference_frame)

    @classmethod
    def from_quaternion(cls, quaternion: Quaternion) -> RotationMatrix:
        """
        Unit quaternion to 4x4 rotation matrix according to:
        https://github.com/orocos/orocos_kinematics_dynamics/blob/master/orocos_kdl/src/frames.cpp#L167
        """
        if quaternion.is_constant():
            return cls._from_constant_quaternion(quaternion)
        x = quaternion[0]
        y = quaternion[1]
        z = quaternion[2]
        w = quaternion[3]
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
            reference_frame=quaternion.reference_frame,
        )

    def x_vector(self) -> Vector3Dimensional:
        return Vector3Dimensional(
            x=self[0, 0],
            y=self[1, 0],
            z=self[2, 0],
            reference_frame=self.reference_frame,
        )

    def y_vector(self) -> Vector3Dimensional:
        return Vector3Dimensional(
            x=self[0, 1],
            y=self[1, 1],
            z=self[2, 1],
            reference_frame=self.reference_frame,
        )

    def z_vector(self) -> Vector3Dimensional:
        return Vector3Dimensional(
            x=self[0, 2],
            y=self[1, 2],
            z=self[2, 2],
            reference_frame=self.reference_frame,
        )

    def dot(
        self, other: GenericSpecialOrthogonalGroupAction
    ) -> GenericSpecialOrthogonalGroupAction:
        """
        This rotation applied to ``other``, turning it about the origin of its reference
        frame.

        :param other: The value or change this rotation is applied to.
        :return: ``other`` turned, in this rotation's reference frame.
        :raises UnsupportedOperationError: If this rotation does not act on ``other``.
        """
        if not isinstance(other, SpecialOrthogonalGroupAction):
            raise UnsupportedOperationError(operation="dot", left=self, right=other)
        return other.apply_rotation(rotation=self)

    def __matmul__(
        self, other: GenericSpecialOrthogonalGroupAction
    ) -> GenericSpecialOrthogonalGroupAction:
        """
        :meth:`dot`, as the ``@`` operator.
        """
        return self.dot(other)

    @property
    def axis_angle(self) -> AxisAngle:
        """
        :return: The rotation this matrix describes.
        """
        return AxisAngle.from_rotation_matrix(self)

    @property
    def angle(self) -> sm.Scalar:
        """
        :return: The angle of the rotation around its axis in radians, in [0, pi].
        """
        return self.axis_angle.angle

    @property
    def generic_matrix(self) -> sm.Matrix:
        """
        :return: This rotation matrix as a matrix without spatial semantics.
        """
        return sm.Matrix.from_casadi_sx(casadi_sx=copy(self.casadi_sx))

    @classmethod
    def from_vectors(
        cls,
        x: Vector3Dimensional | None = None,
        y: Vector3Dimensional | None = None,
        z: Vector3Dimensional | None = None,
        reference_frame: CartesianFrame3Dimensional | None = None,
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
            raise InsufficientVectorsError(x=x, y=y, z=z)
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
    def from_x_axis(
        cls,
        x_vector: Vector3Dimensional,
        reference_frame: CartesianFrame3Dimensional | None = None,
    ) -> RotationMatrix:
        """
        Create a rotation matrix from the direction its x-axis points along.

        The y- and z-axis only complete the frame and are otherwise arbitrary.

        :param x_vector: the direction the x-axis should point along
        :param reference_frame: the frame the resulting rotation matrix is expressed in
        :return: a rotation matrix whose x-axis points along ``x``
        """
        # Crossing a direction with a world axis degenerates when the two are parallel, so
        # pick whichever of the X/Y axes yields the better-conditioned (longer) cross
        # product.
        frame_V_cross_x = x_vector.cross(Vector3Dimensional.X())
        frame_V_cross_y = x_vector.cross(Vector3Dimensional.Y())
        y = Vector3Dimensional.from_iterable(
            [
                sm.if_greater(
                    frame_V_cross_x.norm(),
                    frame_V_cross_y.norm(),
                    frame_V_cross_x[i],
                    frame_V_cross_y[i],
                )
                for i in range(3)
            ]
        )
        return cls.from_vectors(x=x_vector, y=y, reference_frame=reference_frame)

    @classmethod
    def from_rpy(
        cls,
        roll: sm.ScalarData | None = None,
        pitch: sm.ScalarData | None = None,
        yaw: sm.ScalarData | None = None,
        reference_frame: CartesianFrame3Dimensional | None = None,
    ) -> RotationMatrix:
        """
        Conversion of roll, pitch, yaw to 4x4 rotation matrix according to:

        https://github.com/orocos/orocos_kinematics_dynamics/blob/master/orocos_kdl/src/frames.cpp#L167
        """
        roll = 0 if roll is None else roll
        pitch = 0 if pitch is None else pitch
        yaw = 0 if yaw is None else yaw
        # Every sine and cosine is built once, and through ca.SX.unary, because casadi's
        # overloaded functions spend most of their time choosing the overload.
        sin_roll, cos_roll, sin_pitch, cos_pitch, sin_yaw, cos_yaw = (
            sm.Scalar.from_casadi_sx(casadi_sx=ca.SX.unary(operation, sm.to_sx(angle)))
            for angle in (roll, pitch, yaw)
            for operation in (ca.OP_SIN, ca.OP_COS)
        )
        rotation = [
            [
                cos_yaw * cos_pitch,
                (cos_yaw * sin_pitch * sin_roll) - (sin_yaw * cos_roll),
                (sin_yaw * sin_roll) + (cos_yaw * sin_pitch * cos_roll),
            ],
            [
                sin_yaw * cos_pitch,
                (cos_yaw * cos_roll) + (sin_yaw * sin_pitch * sin_roll),
                (sin_yaw * sin_pitch * cos_roll) - (cos_yaw * sin_roll),
            ],
            [-sin_pitch, cos_pitch * sin_roll, cos_pitch * cos_roll],
        ]
        s = ca.SX.eye(4)
        for row_index, row in enumerate(rotation):
            for column_index, entry in enumerate(row):
                s[row_index, column_index] = entry.casadi_sx
        result = cls(reference_frame=reference_frame)
        result.casadi_sx = s
        return result

    def inverse(self) -> RotationMatrix:
        return self.T

    @property
    def rpy(self) -> tuple[sm.Scalar, sm.Scalar, sm.Scalar]:
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

    @property
    def quaternion(self) -> Quaternion:
        """
        :return: The rotation this matrix describes.
        """
        return Quaternion.from_rotation_matrix(self)

    def normalize(self) -> None:
        """
        Scales each of the axes to the length of one.
        """
        scale_v = 1.0
        self[:3, 0] = self[:3, 0].scale(a=scale_v)
        self[:3, 1] = self[:3, 1].scale(a=scale_v)
        self[:3, 2] = self[:3, 2].scale(a=scale_v)

    @property
    def T(self) -> RotationMatrix:
        return RotationMatrix(self.casadi_sx.T, reference_frame=self.reference_frame)

    def rotational_distance(self, other: RotationMatrix) -> sm.Scalar:
        """
        Calculate the rotational error between two rotation matrices.

        This function computes the angular difference between two rotation matrices by
        computing the dot product of the first matrix and the inverse of the second.
        Subsequently, it generates the angle of the resulting rotation matrix.

        :param other: The second rotation matrix.
        :return: The angular error between the two rotation matrices as an expression.
        """
        r_distance = self.dot(other.inverse())
        return r_distance.angle


@dataclass(eq=False, init=False, repr=False)
class Vector3Dimensional(
    SymbolicSpatialValue3Dimensional,
    HomogeneousCoordinates,
    CartesianVector[CartesianFrame3Dimensional, sm.Scalar],
):
    """
    Representation of a 3D vector with reference frame support for homogenous
    transformations.

    This class provides a structured representation of 3D vectors. It includes
    support for operations such as addition, subtraction, scaling, dot product,
    cross product, and more. It is compatible with symbolic computations and
    provides methods to define standard basis vectors, normalize a vector, and
    compute geometric properties such as the angle between vectors. The class
    also includes support for working in different reference frames.

    .. note:: this is represented as a 4d vector, where the last entry is always a 0.
    """

    visualisation_frame: CartesianFrame3Dimensional = field(default=None)

    def __init__(
        self,
        x: sm.ScalarData = 0,
        y: sm.ScalarData = 0,
        z: sm.ScalarData = 0,
        reference_frame: CartesianFrame3Dimensional | None = None,
        visualisation_frame: CartesianFrame3Dimensional | None = None,
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
        super().__post_init__()

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
    def _from_constant_json(cls, data: dict[str, Any], **kwargs) -> Self:
        reference_frame = cls._parse_optional_frame_from_json(
            data, frame=SpatialFrameKey.REFERENCE, **kwargs
        )
        return cls.from_iterable(
            data["data"][:3],
            reference_frame=reference_frame,
        )

    def _constant_to_json(self) -> dict[str, Any]:
        return {
            "data": self.to_np().tolist(),
        }

    @classmethod
    def from_iterable(
        cls,
        data: sm.VectorData,
        reference_frame: CartesianFrame3Dimensional | None = None,
    ) -> Vector3Dimensional:
        """
        Creates an instance of Vector3Dimensional from provided iterable data.

        This class method is used to construct a Vector3Dimensional object by processing the given
        data and optionally assigning a reference frame. The data can represent
        different array-like objects compatible with the desired format for a Vector3Dimensional
        instance. The provided iterable or array should follow a 1D structure to avoid
        raised errors.

        :param data: The array-like data or object such as a list, tuple, or numpy array
            used to initialize the Vector3Dimensional instance.
        :param reference_frame: A reference to a `CartesianFrame3Dimensional` object,
            representing the frame of reference for the Vector3Dimensional instance. If the data
            has a `reference_frame` attribute, and this parameter is not specified, it
            will be taken from the data.
        :return: Returns an instance of Vector3Dimensional initialized with the processed data and
            an optional reference frame.
        """
        if isinstance(data, SymbolicSpatialType) and reference_frame is None:
            reference_frame = data.reference_frame
        if isinstance(data, Vector3Dimensional) and reference_frame is None:
            visualisation_frame = data.visualisation_frame
        else:
            visualisation_frame = None
        result = cls.from_casadi_sx(casadi_sx=copy(sm.to_sx(data)))
        result.reference_frame = reference_frame
        result.visualisation_frame = visualisation_frame
        return result

    @classmethod
    def X(
        cls, reference_frame: CartesianFrame3Dimensional | None = None
    ) -> Vector3Dimensional:
        return cls(x=1, y=0, z=0, reference_frame=reference_frame)

    @classmethod
    def Y(
        cls, reference_frame: CartesianFrame3Dimensional | None = None
    ) -> Vector3Dimensional:
        return cls(x=0, y=1, z=0, reference_frame=reference_frame)

    @classmethod
    def Z(
        cls, reference_frame: CartesianFrame3Dimensional | None = None
    ) -> Vector3Dimensional:
        return cls(x=0, y=0, z=1, reference_frame=reference_frame)

    @classmethod
    def NEGATIVE_X(
        cls, reference_frame: CartesianFrame3Dimensional | None = None
    ) -> Vector3Dimensional:
        return cls(x=-1, y=0, z=0, reference_frame=reference_frame)

    @classmethod
    def NEGATIVE_Y(
        cls, reference_frame: CartesianFrame3Dimensional | None = None
    ) -> Vector3Dimensional:
        return cls(x=0, y=-1, z=0, reference_frame=reference_frame)

    @classmethod
    def NEGATIVE_Z(
        cls, reference_frame: CartesianFrame3Dimensional | None = None
    ) -> Vector3Dimensional:
        return cls(x=0, y=0, z=-1, reference_frame=reference_frame)

    @classmethod
    def unit_vector(
        cls,
        x: sm.ScalarData = 0,
        y: sm.ScalarData = 0,
        z: sm.ScalarData = 0,
        reference_frame: CartesianFrame3Dimensional | None = None,
    ) -> Vector3Dimensional:
        v = cls(x=x, y=y, z=z, reference_frame=reference_frame)
        v.scale(1, unsafe=True)
        return v

    @classmethod
    def create_with_variables(
        cls, name: str, resolver: Callable[[], list[float] | np.ndarray] | None = None
    ) -> Self:
        """
        Creates a Vector3Dimensional object with float variables in all relevant entries.

        :param name: Name for the variables.
        :param resolver: Callable that returns the actual vector when called.
        :return: Vector3Dimensional object with float variables.
        """
        x = sm.FloatVariable(name=f"{name}.x")
        y = sm.FloatVariable(name=f"{name}.y")
        z = sm.FloatVariable(name=f"{name}.z")
        result = cls(
            x=x,
            y=y,
            z=z,
            reference_frame=None,
        )
        if resolver is not None:
            x.resolve = lambda: resolver()[0]
            y.resolve = lambda: resolver()[1]
            z.resolve = lambda: resolver()[2]
        return result

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
    def coordinates(self) -> tuple[sm.Scalar, sm.Scalar, sm.Scalar]:
        return self.x, self.y, self.z

    def __add__(self, other: Vector3Dimensional) -> Vector3Dimensional:
        """
        :raises ReferenceFrameMismatchError: If ``other`` is given in another frame.
        """
        if isinstance(other, Vector3Dimensional):
            self.ensure_same_reference_frame(other, "the added Vector3Dimensional")
            result = Vector3Dimensional.from_casadi_sx(
                casadi_sx=ca.SX.binary(ca.OP_ADD, self.casadi_sx, other.casadi_sx)
            )
        else:
            return NotImplemented
        result.reference_frame = self.reference_frame
        return result

    def __sub__(self, other: Vector3Dimensional) -> Vector3Dimensional:
        """
        :raises ReferenceFrameMismatchError: If ``other`` is given in another frame.
        """
        if isinstance(other, Vector3Dimensional):
            self.ensure_same_reference_frame(other, "the subtracted Vector3Dimensional")
            result = Vector3Dimensional.from_casadi_sx(
                casadi_sx=ca.SX.binary(ca.OP_SUB, self.casadi_sx, other.casadi_sx)
            )
        else:
            return NotImplemented
        result.reference_frame = self.reference_frame
        return result

    def __mul__(self, other: sm.ScalarData) -> Vector3Dimensional:
        if isinstance(other, sm.ScalarData):
            result = Vector3Dimensional.from_casadi_sx(
                casadi_sx=ca.SX.binary(ca.OP_MUL, self.casadi_sx, sm.to_sx(other))
            )
        else:
            return NotImplemented
        result.reference_frame = self.reference_frame
        return result

    def __rmul__(self, other: float) -> Vector3Dimensional:
        if isinstance(other, (int, float)):
            result = Vector3Dimensional.from_casadi_sx(
                casadi_sx=ca.SX.binary(ca.OP_MUL, self.casadi_sx, sm.to_sx(other))
            )
        else:
            return NotImplemented
        result.reference_frame = self.reference_frame
        return result

    def __truediv__(self, other: sm.ScalarData) -> Vector3Dimensional:
        if isinstance(other, sm.ScalarData):
            result = Vector3Dimensional.from_casadi_sx(
                casadi_sx=ca.SX.binary(ca.OP_DIV, self.casadi_sx, sm.to_sx(other))
            )
        else:
            return NotImplemented
        result.reference_frame = self.reference_frame
        return result

    def safe_division(
        self,
        *,
        other: sm.ScalarData,
        if_nan: Vector3Dimensional | None = None,
    ) -> sm.GenericSymbolicType:
        """
        A version of division where no sub-expression is ever NaN.

        The expression would evaluate to 'if_nan', but you should probably never work
        with the 'if_nan' result. However, if one sub-expressions is NaN, the whole
        expression evaluates to NaN, even if it is only in a branch of an if-else, that
        is not returned. This method is a workaround for such cases.
        """
        if if_nan is None:
            if_nan = Vector3Dimensional()
        save_denominator = sm.if_eq_zero(
            condition=other, if_result=sm.Scalar(1), else_result=other
        )
        return sm.if_eq_zero(
            other, if_result=if_nan, else_result=self / save_denominator
        )

    def __neg__(self) -> Vector3Dimensional:
        result = Vector3Dimensional.from_casadi_sx(
            casadi_sx=ca.SX.unary(ca.OP_NEG, self.casadi_sx)
        )
        result.reference_frame = self.reference_frame
        return result

    def __matmul__(self, other: Vector3Dimensional) -> sm.Scalar:
        return self.dot(other)

    def cross(self, other: Vector3Dimensional) -> Vector3Dimensional:
        result = ca.cross(self.casadi_sx[:3], other.casadi_sx[:3])
        result = self.__class__.from_iterable(result)
        result.reference_frame = self.reference_frame
        return result

    def scale(self, a: sm.ScalarData, unsafe: bool = False):
        if unsafe:
            self.casadi_sx = ((self / self.norm()) * a).casadi_sx
        else:
            self.casadi_sx = (self.safe_division(other=self.norm()) * a).casadi_sx

    def project_to_cone(
        self,
        frame_V_cone_axis: Vector3Dimensional,
        cone_theta: sm.ScalarData,
    ) -> Vector3Dimensional:
        """
        Projects a given vector onto the boundary of a cone defined by its axis and
        angle.

        This function computes the projection of a vector onto the boundary of a cone
        specified by its axis and half-angle. It handles special cases where the input
        vector is collinear with the cone's axis. The projection ensures the resulting
        vector lies within the cone's boundary.

        :param frame_V_cone_axis: The axis of the cone.
        :param cone_theta: The half-angle of the cone in radians. Can be a symbolic
            value or a float.
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

    def angle_between(self, other: Vector3Dimensional) -> sm.Scalar:
        return sm.acos(
            sm.limit(
                self @ other / (self.norm() * other.norm()),
                lower_limit=-1,
                upper_limit=1,
            )
        )

    def slerp(self, other: Vector3Dimensional, t: sm.ScalarData) -> Vector3Dimensional:
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

    @property
    def cartesian_point_3_dimensional(self) -> CartesianPoint3Dimensional:
        """
        :return: The point this vector reaches from the origin of the reference frame.
        """
        result = CartesianPoint3Dimensional.from_casadi_sx(
            casadi_sx=copy(self.casadi_sx)
        )
        result.reference_frame = self.reference_frame
        return result


@dataclass(eq=False, init=False, repr=False)
class CartesianPoint3Dimensional(
    SymbolicSpatialValue3Dimensional,
    HomogeneousCoordinates,
    CartesianPoint[CartesianFrame3Dimensional, Vector3Dimensional, sm.Scalar],
):
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
        reference_frame: CartesianFrame3Dimensional | None = None,
    ):
        """

        :param x: X-coordinate of the point. Defaults to 0.
        :param y: Y-coordinate of the point. Defaults to 0.
        :param z: Z-coordinate of the point. Defaults to 0.
        :param reference_frame:
        """
        self.casadi_sx = sm.to_sx([x, y, z, 1])
        self.reference_frame = reference_frame
        super().__post_init__()

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
        reference_frame: CartesianFrame3Dimensional | None = None,
    ) -> CartesianPoint3Dimensional:
        """
        Creates an instance of CartesianPoint3Dimensional from provided iterable data.

        This class method is used to construct a CartesianPoint3Dimensional object by processing the given
        data and optionally assigning a reference frame. The data can represent
        different array-like objects compatible with the desired format for a CartesianPoint3Dimensional
        instance. The provided iterable or array should follow a 1D structure to avoid
        raised errors.

        :param data: The array-like data or object such as a list, tuple, or numpy array
            used to initialize the CartesianPoint3Dimensional instance.
        :param reference_frame: A reference to a `CartesianFrame3Dimensional` object,
            representing the frame of reference for the CartesianPoint3Dimensional instance. If the data has
            a `reference_frame` attribute, and this parameter is not specified, it will
            be taken from the data.
        :return: Returns an instance of CartesianPoint3Dimensional initialized with the processed data and
            an optional reference frame.
        """
        if isinstance(data, SymbolicSpatialType) and reference_frame is None:
            reference_frame = data.reference_frame
        result = cls.from_casadi_sx(casadi_sx=copy(sm.to_sx(data)))
        result.reference_frame = reference_frame
        return result

    @classmethod
    def _from_constant_json(cls, data: dict[str, Any], **kwargs) -> Self:
        reference_frame = cls._parse_optional_frame_from_json(
            data, frame=SpatialFrameKey.REFERENCE, **kwargs
        )
        return cls.from_iterable(
            data["data"][:3],
            reference_frame=reference_frame,
        )

    @classmethod
    def create_with_variables(
        cls, name: str, resolver: Callable[[], list[float] | np.ndarray] | None = None
    ) -> Self:
        """
        Creates a Vector3Dimensional object with float variables in all relevant entries.

        :param name: Name for the variables.
        :param resolver: Callable that returns the actual vector when called.
        :return: Vector3Dimensional object with float variables.
        """
        x = sm.FloatVariable(name=f"{name}.x")
        y = sm.FloatVariable(name=f"{name}.y")
        z = sm.FloatVariable(name=f"{name}.z")
        result = cls(
            x=x,
            y=y,
            z=z,
            reference_frame=None,
        )
        if resolver is not None:
            x.resolve = lambda: resolver()[0]
            y.resolve = lambda: resolver()[1]
            z.resolve = lambda: resolver()[2]
        return result

    def _constant_to_json(self) -> dict[str, Any]:
        return {
            "data": self.to_np().tolist(),
        }

    def norm(self) -> sm.Scalar:
        return sm.Scalar.from_casadi_sx(casadi_sx=ca.norm_2(self[:3].casadi_sx))

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
    def coordinates(self) -> tuple[sm.Scalar, sm.Scalar, sm.Scalar]:
        return self.x, self.y, self.z

    def __add__(self, other: Vector3Dimensional) -> CartesianPoint3Dimensional:
        """
        :raises ReferenceFrameMismatchError: If ``other`` is given in another frame.
        """
        if isinstance(other, Vector3Dimensional):
            self.ensure_same_reference_frame(other, "the added Vector3Dimensional")
            result = CartesianPoint3Dimensional.from_casadi_sx(
                casadi_sx=ca.SX.binary(ca.OP_ADD, self.casadi_sx, other.casadi_sx)
            )
        else:
            return NotImplemented
        result.reference_frame = self.reference_frame
        return result

    @overload
    def __sub__(self, other: CartesianPoint3Dimensional) -> Vector3Dimensional: ...

    @overload
    def __sub__(self, other: Vector3Dimensional) -> CartesianPoint3Dimensional: ...

    def __sub__(self, other):
        """
        :raises ReferenceFrameMismatchError: If ``other`` is given in another frame.
        """
        if isinstance(other, CartesianPoint3Dimensional):
            self.ensure_same_reference_frame(
                other, "the subtracted CartesianPoint3Dimensional"
            )
            result = Vector3Dimensional.from_casadi_sx(
                casadi_sx=ca.SX.binary(ca.OP_SUB, self.casadi_sx, other.casadi_sx)
            )
        elif isinstance(other, Vector3Dimensional):
            self.ensure_same_reference_frame(other, "the subtracted Vector3Dimensional")
            result = CartesianPoint3Dimensional.from_casadi_sx(
                casadi_sx=ca.SX.binary(ca.OP_SUB, self.casadi_sx, other.casadi_sx)
            )
        else:
            return NotImplemented
        result.reference_frame = self.reference_frame
        return result

    def __neg__(self) -> CartesianPoint3Dimensional:
        result = CartesianPoint3Dimensional.from_casadi_sx(
            casadi_sx=ca.SX.unary(ca.OP_NEG, self.casadi_sx)
        )
        result.reference_frame = self.reference_frame
        return result

    def project_to_plane(
        self,
        frame_V_plane_vector1: Vector3Dimensional,
        frame_V_plane_vector2: Vector3Dimensional,
    ) -> tuple[CartesianPoint3Dimensional, sm.Scalar]:
        """
        Projects a point onto a plane defined by two vectors.

        This function assumes that all parameters are defined with respect to the same
        reference frame.

        :param frame_V_plane_vector1: First vector defining the plane
        :param frame_V_plane_vector2: Second vector defining the plane
        :return: Tuple of (projected point on the plane, signed distance from point to
            plane)
        """
        normal = frame_V_plane_vector1.cross(frame_V_plane_vector2)
        normal.scale(1)
        frame_V_current = self.vector3
        d = normal @ frame_V_current
        v: Vector3Dimensional = normal * d
        projection = self - v
        return projection, d

    def project_to_line(
        self, line_point: CartesianPoint3Dimensional, line_direction: Vector3Dimensional
    ) -> tuple[CartesianPoint3Dimensional, sm.Scalar]:
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
        self,
        line_start: CartesianPoint3Dimensional,
        line_end: CartesianPoint3Dimensional,
    ) -> tuple[sm.Scalar, CartesianPoint3Dimensional]:
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

    def midpoint(self, other: CartesianPoint3Dimensional) -> CartesianPoint3Dimensional:
        """
        :param other: The point to meet halfway.
        :return: The point midway between this one and ``other``.
        """
        return self + (other - self) / 2

    @property
    def vector3(self) -> Vector3Dimensional:
        """
        :return: The vector from the origin of the reference frame to this point.
        """
        result = Vector3Dimensional.from_casadi_sx(casadi_sx=copy(self.casadi_sx))
        result.reference_frame = self.reference_frame
        return result

    @property
    def generic_vector(self) -> sm.Vector:
        """
        :return: The coordinates of this point as a vector without spatial semantics.
        """
        return sm.Vector.from_casadi_sx(casadi_sx=copy(self.casadi_sx[:3]))


@dataclass(eq=False)
class Line3Dimensional(Line[CartesianPoint3Dimensional, Vector3Dimensional]):
    """
    An infinite straight line in 3D space.
    """

    point: CartesianPoint3Dimensional
    """
    A point the line passes through.
    """

    direction: Vector3Dimensional
    """
    The direction the line runs in, in the same reference frame as :attr:`point`.
    """


@dataclass(eq=False, kw_only=True)
class LineSegment3Dimensional(
    LineSegment[CartesianPoint3Dimensional, Line3Dimensional]
):
    """
    The part of a straight line in 3D space between two points.
    """

    start: CartesianPoint3Dimensional
    """
    The point the segment runs from.
    """

    end: CartesianPoint3Dimensional
    """
    The point the segment runs to, in the same reference frame as :attr:`start`.
    """

    @property
    def direction(self) -> Vector3Dimensional:
        """
        :return: The unit vector from :attr:`start` towards :attr:`end`.
        """
        direction = self.end - self.start
        direction.scale(1)
        return direction


@dataclass(eq=False, init=False, repr=False)
class Quaternion(
    SymbolicSpatialValue3Dimensional,
    Orientation,
):
    """
    Represents a quaternion, which is a mathematical entity used to encode rotations in
    three-dimensional space.

    The Quaternion class provides methods for creating quaternion objects from various
    representations, such as axis-angle, roll-pitch-yaw, and rotation matrices. It
    supports operations to define and manipulate rotations in 3D space efficiently.
    Quaternions are used extensively in physics, computer graphics, robotics, and
    aerospace engineering to represent orientations and rotations.
    """

    def __init__(
        self,
        x: sm.ScalarData = 0,
        y: sm.ScalarData = 0,
        z: sm.ScalarData = 0,
        w: sm.ScalarData = 1,
        reference_frame: CartesianFrame3Dimensional | None = None,
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
        super().__post_init__()
        if self.is_constant():
            self.normalize()

    def _verify_type(self):
        if self.shape != (4, 1):
            raise WrongDimensionsError(
                expected_dimensions=(4, 1), actual_dimensions=self.shape
            )

    def __neg__(self) -> Quaternion:
        return Quaternion.from_iterable(self.casadi_sx.__neg__())

    @classmethod
    def _from_constant_json(cls, data: dict[str, Any], **kwargs) -> Self:
        reference_frame = cls._parse_optional_frame_from_json(
            data, frame=SpatialFrameKey.REFERENCE, **kwargs
        )
        return cls.from_iterable(
            data["data"],
            reference_frame=reference_frame,
        )

    def _constant_to_json(self) -> dict[str, Any]:
        return {
            "data": self.to_np().tolist(),
        }

    @classmethod
    def from_iterable(
        cls,
        data: sm.VectorData,
        reference_frame: CartesianFrame3Dimensional | None = None,
    ) -> Quaternion:
        """
        Creates an instance of Quaternion from provided iterable data.

        This class method is used to construct a Quaternion object by processing the
        given data and optionally assigning a reference frame. The data can represent
        different array-like objects compatible with the desired format for a Quaternion
        instance. The provided iterable or array should follow a 1D structure to avoid
        raised errors.

        :param data: The array-like data or object such as a list, tuple, or numpy array
            used to initialize the Quaternion instance.
        :param reference_frame: A reference to a `CartesianFrame3Dimensional` object,
            representing the frame of reference for the Quaternion instance. If the data
            has a `reference_frame` attribute, and this parameter is not specified, it
            will be taken from the data.
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
    def from_axis_angle(cls, axis_angle: AxisAngle) -> Quaternion:
        """
        Creates a quaternion from an axis-angle representation.

        :param axis_angle: The rotation to convert.
        :return: A quaternion representing the rotation defined by the given axis and
            angle, expressed in the frame of the axis angle.
        """
        axis = axis_angle.axis
        half_angle = axis_angle.angle / 2
        return cls(
            x=axis[0] * sm.sin(half_angle),
            y=axis[1] * sm.sin(half_angle),
            z=axis[2] * sm.sin(half_angle),
            w=sm.cos(half_angle),
            reference_frame=axis_angle.reference_frame,
        )

    @classmethod
    def from_rpy(
        cls,
        roll: sm.ScalarData,
        pitch: sm.ScalarData,
        yaw: sm.ScalarData,
        reference_frame: CartesianFrame3Dimensional | None = None,
    ) -> Quaternion:
        """
        Creates a Quaternion instance from specified roll, pitch, and yaw angles.

        The method computes the quaternion representation of the given roll, pitch, and
        yaw angles using trigonometric transformations based on their half-angle values
        for efficient calculations.

        :param roll: The roll angle in radians.
        :param pitch: The pitch angle in radians.
        :param yaw: The yaw angle in radians.
        :param reference_frame: Optional reference frame entity associated with the
            quaternion.
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
        cls, r: RotationMatrix | HomogeneousTransformationMatrix3Dimensional
    ) -> Quaternion:
        """
        Creates a Quaternion object initialized from a given rotation matrix.

        This method constructs a quaternion representation of the provided rotation
        matrix. It is designed to handle different cases of rotation matrix
        configurations to ensure numerical stability during computation. The resultant
        quaternion adheres to the expected mathematical relationship with the given
        rotation matrix.

        :param r: The input matrix representing a rotation. It can be either a
            `RotationMatrix` or `TransformationMatrix`. This matrix is expected to have
            a valid mathematical structure typical for rotation matrices.
        :return: A new instance of `Quaternion` corresponding to the given rotation
            matrix `r`.
        """
        q = rotation_matrix_to_quaternion(r.generic_matrix)
        return cls.from_iterable(q, reference_frame=r.reference_frame)

    def conjugate(self) -> Quaternion:
        return Quaternion(
            x=-self[0],
            y=-self[1],
            z=-self[2],
            w=self[3],
            reference_frame=self.reference_frame,
        )

    def apply_rotation(self, *, rotation: RotationMatrix) -> Quaternion:
        result = rotation.quaternion.multiply(self)
        result.reference_frame = rotation.reference_frame
        return result

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

    def _normalize_numerically(self) -> None:
        """
        Normalize this quaternion in-place using numpy arithmetic.

        Called by :meth:`normalize` when the quaternion is constant.  Using numpy avoids
        constructing symbolic CasADi expressions, which is substantially faster for
        values loaded from a database.
        """
        values = self.to_np().ravel()
        self._casadi_sx = ca.SX(ca.DM(values / np.linalg.norm(values)))

    def normalize(self) -> None:
        if self._casadi_sx.is_constant():
            self._normalize_numerically()
            return
        norm_ = self.generic_vector.norm()
        self.x /= norm_
        self.y /= norm_
        self.z /= norm_
        self.w /= norm_

    @property
    def axis_angle(self) -> AxisAngle:
        """
        :return: The rotation this quaternion describes.
        """
        return AxisAngle.from_quaternion(self)

    @property
    def generic_vector(self) -> sm.Vector:
        """
        :return: This quaternion as a vector without spatial semantics.
        """
        return sm.Vector.from_casadi_sx(casadi_sx=copy(self.casadi_sx))

    @property
    def rotation_matrix(self) -> RotationMatrix:
        """
        :return: The rotation this quaternion describes.
        """
        return RotationMatrix.from_quaternion(self)

    @property
    def rpy(self) -> tuple[sm.Scalar, sm.Scalar, sm.Scalar]:
        """
        :return: roll, pitch, yaw
        """
        return self.rotation_matrix.rpy

    def dot(self, other: Quaternion) -> sm.Scalar:
        if isinstance(other, Quaternion):
            return sm.Scalar(ca.mtimes(self.casadi_sx.T, other.casadi_sx))
        return NotImplemented

    def rotational_distance(self, other: Quaternion) -> sm.Scalar:
        """
        Calculate the rotational error between two orientations.

        The dot product is taken as an absolute value, so the two quaternions that
        describe every rotation count as the same orientation.

        :param other: The second orientation.
        :return: The angle between the two orientations, 0 when they describe the same
            rotation.
        """
        return 2 * sm.safe_acos(abs(self.dot(other)))

    def slerp(self, other: Quaternion, t: sm.ScalarData) -> Quaternion:
        """
        Spherical linear interpolation that takes into account that q == -q t=0 will
        return self and t=1 will return other.

        :param other: the other quaternion
        :param t: float, 0-1
        :return: 4x1 Matrix; Return spherical linear interpolation between two
            quaternions.
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

        ratio_a = (sm.sin((1.0 - t) * half_theta)).safe_division(other=sin_half_theta)
        ratio_b = sm.sin(t * half_theta).safe_division(other=sin_half_theta)

        mid_quaternion = Quaternion.from_iterable(
            self.generic_vector * 0.5 + other.generic_vector * 0.5
        )
        slerped_quaternion = Quaternion.from_iterable(
            self.generic_vector * ratio_a + other.generic_vector * ratio_b
        )

        return sm.if_greater_eq_zero(
            if1, self, sm.if_greater_zero(if2, mid_quaternion, slerped_quaternion)
        )


@dataclass(eq=False, init=False, repr=False)
class AxisAngle(
    SymbolicSpatialValue3Dimensional,
    Orientation,
):
    """
    A rotation by an angle around a unit axis.
    """

    def __init__(
        self,
        axis: Vector3Dimensional | None = None,
        angle: sm.ScalarData = 0,
        reference_frame: CartesianFrame3Dimensional | None = None,
    ):
        """
        :param axis: The unit axis to rotate around. Defaults to the z-axis.
        :param angle: The angle to rotate by, in radians.
        :param reference_frame: The frame the axis is expressed in.
        """
        if axis is None:
            axis = Vector3Dimensional.Z()
        self.casadi_sx = sm.to_sx([axis.x, axis.y, axis.z, angle])
        self.reference_frame = reference_frame
        super().__post_init__()

    def _verify_type(self):
        if self.shape != (4, 1):
            raise WrongDimensionsError(
                expected_dimensions=(4, 1), actual_dimensions=self.shape
            )

    @classmethod
    def _from_constant_json(cls, data: dict[str, Any], **kwargs) -> Self:
        reference_frame = cls._parse_optional_frame_from_json(
            data, frame=SpatialFrameKey.REFERENCE, **kwargs
        )
        return cls(
            axis=Vector3Dimensional(*data["axis"]),
            angle=data["angle"],
            reference_frame=reference_frame,
        )

    def _constant_to_json(self) -> dict[str, Any]:
        return {
            "axis": self.axis.to_np()[:3].tolist(),
            "angle": float(self.angle),
        }

    @classmethod
    def from_quaternion(cls, quaternion: Quaternion) -> AxisAngle:
        """
        The identity rotation has no axis of its own and is read as a rotation by zero
        around the z-axis.

        :param quaternion: The rotation to describe, normalized in place.
        :return: The rotation the quaternion describes, with an angle in [0, 2 pi].
        """
        quaternion.normalize()
        sine_of_half_angle = sm.sqrt(1 - quaternion.w**2)
        divisor = sm.if_eq_zero(sine_of_half_angle, sm.Scalar(1), sine_of_half_angle)
        angle = sm.if_eq_zero(
            sine_of_half_angle,
            sm.Scalar(0),
            sm.Scalar(2 * sm.acos(sm.limit(quaternion.w, -1, 1))),
        )
        x = sm.if_eq_zero(sine_of_half_angle, sm.Scalar(0), quaternion.x / divisor)
        y = sm.if_eq_zero(sine_of_half_angle, sm.Scalar(0), quaternion.y / divisor)
        z = sm.if_eq_zero(sine_of_half_angle, sm.Scalar(1), quaternion.z / divisor)
        return cls(
            axis=Vector3Dimensional(x=x, y=y, z=z),
            angle=angle,
            reference_frame=quaternion.reference_frame,
        )

    @classmethod
    def from_rotation_matrix(cls, rotation_matrix: RotationMatrix) -> AxisAngle:
        """
        The identity rotation has no axis of its own and is read as a rotation by zero
        around the z-axis.

        :param rotation_matrix: The rotation to describe.
        :return: The rotation the matrix describes, with an angle in [0, pi].
        """
        rotation = rotation_matrix.generic_matrix[:3, :3]
        cosine = (rotation.trace() - 1) / 2
        # the guards keep the derivative finite where acos has none
        angle = sm.if_greater_eq(
            cosine,
            1,
            sm.Scalar(0),
            sm.if_less_eq(cosine, -1, sm.Scalar(np.pi), sm.acos(cosine)),
        )

        # 2 sin(angle) times the axis, which vanishes at half a turn
        scaled_axis = Vector3Dimensional(
            x=rotation[2, 1] - rotation[1, 2],
            y=rotation[0, 2] - rotation[2, 0],
            z=rotation[1, 0] - rotation[0, 1],
        )
        scaled_axis_length = scaled_axis.norm()
        axis_within_quarter_turn = sm.if_eq_zero(
            scaled_axis_length,
            Vector3Dimensional.Z(),
            scaled_axis / sm.if_eq_zero(scaled_axis_length, 1, scaled_axis_length),
        )

        # (1 - cos(angle)) times the axis times its transpose, which vanishes at no
        # turn; its column with the largest diagonal entry is the axis up to its sign
        axis_outer_product = (rotation + rotation.T) / 2 - sm.Matrix.eye(
            size=3
        ) * cosine
        columns = [
            Vector3Dimensional(*(axis_outer_product[row, column] for row in range(3)))
            for column in range(3)
        ]
        largest_column = sm.if_greater(
            rotation[2, 2],
            sm.max(rotation[0, 0], rotation[1, 1]),
            columns[2],
            sm.if_greater(rotation[1, 1], rotation[0, 0], columns[1], columns[0]),
        )
        axis_up_to_sign = largest_column / largest_column.norm()
        axis_beyond_quarter_turn = sm.if_less(
            axis_up_to_sign @ scaled_axis, 0, -axis_up_to_sign, axis_up_to_sign
        )

        axis = sm.if_greater_zero(
            cosine, axis_within_quarter_turn, axis_beyond_quarter_turn
        )
        return cls(
            axis=axis, angle=angle, reference_frame=rotation_matrix.reference_frame
        )

    @property
    def axis(self) -> Vector3Dimensional:
        """
        :return: The unit axis this rotation turns around.
        """
        return Vector3Dimensional(
            x=self[0], y=self[1], z=self[2], reference_frame=self.reference_frame
        )

    @property
    def angle(self) -> sm.Scalar:
        """
        :return: The angle this rotation turns by, in radians.
        """
        return self[3]

    @property
    def quaternion(self) -> Quaternion:
        """
        :return: The rotation this axis angle describes.
        """
        return Quaternion.from_axis_angle(self)

    @property
    def rotation_matrix(self) -> RotationMatrix:
        """
        :return: The rotation this axis angle describes.
        """
        return RotationMatrix.from_axis_angle(self)

    def apply_rotation(self, *, rotation: RotationMatrix) -> AxisAngle:
        result = AxisAngle.from_quaternion(
            rotation.quaternion.multiply(self.quaternion)
        )
        result.reference_frame = rotation.reference_frame
        return result


@dataclass(eq=False, init=False, repr=False, kw_only=True)
class VectorPair(
    SymbolicSpatialValue3Dimensional,
    ABC,
):
    """
    Six numbers made of two vectors given along the axes of the same reference frame:
    a rotational part in the upper three entries and a translational part in the lower
    three, such as the angular and the linear velocity of a :class:`Twist`.
    """

    def __init__(
        self,
        *,
        rotational_part: Vector3Dimensional | None,
        translational_part: Vector3Dimensional | None,
        reference_frame: CartesianFrame3Dimensional | None,
    ) -> None:
        """
        :param rotational_part: How the body turns, in the upper three entries; zero if
            None.
        :param translational_part: How a point of the body moves, in the lower three
            entries; zero if None.
        :param reference_frame: The frame along whose axes both vectors are given.
        """
        if rotational_part is None:
            rotational_part = Vector3Dimensional()
        if translational_part is None:
            translational_part = Vector3Dimensional()
        self.casadi_sx = sm.to_sx(
            [
                rotational_part.x,
                rotational_part.y,
                rotational_part.z,
                translational_part.x,
                translational_part.y,
                translational_part.z,
            ]
        )
        self.reference_frame = reference_frame
        super().__post_init__()

    @classproperty
    @abstractmethod
    def _rotational_part_name(cls) -> str:
        """
        :return: The name of the rotational part, which is both its keyword in the
            constructor and its key in the json.
        """

    @classproperty
    @abstractmethod
    def _translational_part_name(cls) -> str:
        """
        :return: The name of the translational part, which is both its keyword in the
            constructor and its key in the json.
        """

    def _verify_type(self) -> None:
        if self.shape != (6, 1):
            raise WrongDimensionsError(
                expected_dimensions=(6, 1), actual_dimensions=self.shape
            )

    @classmethod
    def _from_constant_json(cls, *, data: dict[str, Any], **kwargs: Any) -> Self:
        reference_frame = cls._parse_optional_frame_from_json(
            data, frame=SpatialFrameKey.REFERENCE, **kwargs
        )
        return cls(
            **{
                cls._rotational_part_name: Vector3Dimensional(
                    *data[cls._rotational_part_name]
                ),
                cls._translational_part_name: Vector3Dimensional(
                    *data[cls._translational_part_name]
                ),
            },
            reference_frame=reference_frame,
        )

    def _constant_to_json(self) -> dict[str, Any]:
        return {
            self._rotational_part_name: self._rotational_part.to_np()[:3].tolist(),
            self._translational_part_name: self._translational_part.to_np()[
                :3
            ].tolist(),
        }

    @property
    def _rotational_part(self) -> Vector3Dimensional:
        """
        :return: How the body turns, in the upper three entries.
        """
        return Vector3Dimensional(
            x=self[0], y=self[1], z=self[2], reference_frame=self.reference_frame
        )

    @property
    def _translational_part(self) -> Vector3Dimensional:
        """
        :return: How a point of the body moves, in the lower three entries.
        """
        return Vector3Dimensional(
            x=self[3], y=self[4], z=self[5], reference_frame=self.reference_frame
        )


@dataclass(eq=False, init=False, repr=False, kw_only=True)
class Displacement(VectorPair):
    """
    A rigid-body displacement: the transformation that rotates a body by
    :attr:`rotation_vector` about its origin and moves that origin by
    :attr:`translation`, both along the axes of the reference frame. Unlike a
    :class:`Twist`, it is not a velocity.
    """

    def __init__(
        self,
        *,
        rotation_vector: Vector3Dimensional | None = None,
        translation: Vector3Dimensional | None = None,
        reference_frame: CartesianFrame3Dimensional | None = None,
    ) -> None:
        """
        :param rotation_vector: The axis the body turned around, scaled by the angle it
            turned by in radians. Defaults to no rotation.
        :param translation: Where the body's origin ended up, relative to where it
            started, in meters. Defaults to no translation.
        :param reference_frame: The frame along whose axes the body moved.
        """
        super().__init__(
            rotational_part=rotation_vector,
            translational_part=translation,
            reference_frame=reference_frame,
        )

    @classproperty
    def _rotational_part_name(cls) -> str:
        return "rotation_vector"

    @classproperty
    def _translational_part_name(cls) -> str:
        return "translation"

    @property
    def rotation_vector(self) -> Vector3Dimensional:
        """
        :return: The axis the body turned around, scaled by the angle it turned by in
            radians.
        """
        return self._rotational_part

    @property
    def translation(self) -> Vector3Dimensional:
        """
        :return: Where the body's origin ended up, relative to where it started, in
            meters.
        """
        return self._translational_part


@dataclass(eq=False, init=False, repr=False, kw_only=True)
class Twist(VectorPair):
    """
    A twist ``(ω, v)``: how fast a rigid body turns, and how fast the point of the body
    at the origin of the reference frame moves, both along that frame's axes and per
    unit of the parameter they are taken with respect to, usually time. Unlike a
    :class:`Displacement`, it is a velocity (Lynch and Park, *Modern Robotics*, §3.3.2).
    """

    def __init__(
        self,
        *,
        angular_velocity: Vector3Dimensional | None = None,
        linear_velocity: Vector3Dimensional | None = None,
        reference_frame: CartesianFrame3Dimensional | None = None,
    ) -> None:
        """
        :param angular_velocity: The axis the body turns around, scaled by how fast it
            turns, in radians per unit of the parameter. Defaults to no turning.
        :param linear_velocity: How fast the point of the body at the origin of the
            reference frame moves, in meters per unit of the parameter. Defaults to
            standing still.
        :param reference_frame: The frame along whose axes both velocities are given.
        """
        super().__init__(
            rotational_part=angular_velocity,
            translational_part=linear_velocity,
            reference_frame=reference_frame,
        )

    @classproperty
    def _rotational_part_name(cls) -> str:
        return "angular_velocity"

    @classproperty
    def _translational_part_name(cls) -> str:
        return "linear_velocity"

    @property
    def angular_velocity(self) -> Vector3Dimensional:
        """
        :return: The axis the body turns around, scaled by how fast it turns, in radians
            per unit of the parameter.
        """
        return self._rotational_part

    @property
    def linear_velocity(self) -> Vector3Dimensional:
        """
        :return: How fast the point of the body at the origin of the reference frame
            moves, in meters per unit of the parameter.
        """
        return self._translational_part


@dataclass(eq=False, init=False, repr=False, kw_only=True)
class ScrewAxis(VectorPair):
    """
    How a rigid body's pose changes per unit of a position that moves it along a screw
    axis: the axis's line, the pitch along it and the unit of the position. Unlike a
    :class:`Twist`, it is not a velocity: it holds whatever the speed of the position.
    """

    def __init__(
        self,
        *,
        rotation_per_unit_position: Vector3Dimensional | None = None,
        translation_per_unit_position: Vector3Dimensional | None = None,
        reference_frame: CartesianFrame3Dimensional | None = None,
    ) -> None:
        """
        :param rotation_per_unit_position: The axis the body turns around, scaled by the
            angle it turns by per unit of position, in radians. Defaults to no turning.
        :param translation_per_unit_position: How far the point of the body at the
            origin of the reference frame moves per unit of position, in meters.
            Defaults to no moving.
        :param reference_frame: The frame along whose axes both parts are given.
        """
        super().__init__(
            rotational_part=rotation_per_unit_position,
            translational_part=translation_per_unit_position,
            reference_frame=reference_frame,
        )

    @classproperty
    def _rotational_part_name(cls) -> str:
        return "rotation_per_unit_position"

    @classproperty
    def _translational_part_name(cls) -> str:
        return "translation_per_unit_position"

    @property
    def rotation_per_unit_position(self) -> Vector3Dimensional:
        """
        :return: The axis the body turns around, scaled by the angle it turns by per
            unit of position, in radians.
        """
        return self._rotational_part

    @property
    def translation_per_unit_position(self) -> Vector3Dimensional:
        """
        :return: How far the point of the body at the origin of the reference frame
            moves per unit of position, in meters.
        """
        return self._translational_part

    @property
    def unit_direction(self) -> Vector3Dimensional:
        """
        :return: The unit vector along the axis: that of the rotation, or of the
            translation for an axis that does not turn the body (Lynch and Park,
            *Modern Robotics*, section 3.3.2.2). The zero vector for a zero screw axis.
        """
        rotation = self.rotation_per_unit_position
        translation = self.translation_per_unit_position
        angle_per_unit_position = rotation.norm()
        result = sm.if_greater_zero(
            angle_per_unit_position,
            rotation.safe_division(other=angle_per_unit_position),
            translation.safe_division(other=translation.norm()),
        )
        result.reference_frame = self.reference_frame
        return result

    def to_twist(self, *, position_velocity: sm.ScalarData) -> Twist:
        """
        :param position_velocity: How fast the position changes, per unit of the
            parameter the twist is taken with respect to, usually time.
        :return: The twist of the body while its position changes that fast.
        """
        return Twist(
            angular_velocity=self.rotation_per_unit_position * position_velocity,
            linear_velocity=self.translation_per_unit_position * position_velocity,
            reference_frame=self.reference_frame,
        )


@dataclass(eq=False, init=False, repr=False)
class Pose3Dimensional(
    SymbolicSpatialValue3Dimensional,
    HomogeneousCoordinates,
):
    """
    A position and an orientation, both given in one reference frame.
    """

    def __init__(
        self,
        position: CartesianPoint3Dimensional | None = None,
        orientation: Quaternion | None = None,
        reference_frame: CartesianFrame3Dimensional | None = None,
    ):
        """
        Initialize a 3D point with orientation and a reference frame in a kinematic
        structure.

        :param position: The 3D position of the point, represented as a CartesianPoint3Dimensional object.
            If None, a default position is assumed.
        :param orientation: The orientation of the point in 3D space represented as a
            Quaternion. If None, default orientation is assumed.
        :param reference_frame: The reference frame (kinematic structure entity)
            relative to which this point is defined. This may be None if the point is
            not tied to any specific reference frame.
        """
        if position is None:
            position = CartesianPoint3Dimensional()
        if orientation is None:
            orientation = Quaternion()
        transformation_matrix = (
            HomogeneousTransformationMatrix3Dimensional.from_point_rotation_matrix(
                point=position, rotation_matrix=orientation.rotation_matrix
            )
        )
        self._casadi_sx = transformation_matrix._casadi_sx
        self.reference_frame = reference_frame
        super().__post_init__()

    def _verify_type(self):
        if self.shape != (4, 4):
            raise WrongDimensionsError(
                expected_dimensions=(4, 4), actual_dimensions=self.shape
            )
        self[3, :] = _ConstantMatrixParts.HOMOGENEOUS_BOTTOM_ROW

    @classmethod
    def _from_constant_json(cls, data: dict[str, Any], **kwargs) -> Self:
        reference_frame = cls._parse_optional_frame_from_json(
            data, frame=SpatialFrameKey.REFERENCE, **kwargs
        )
        return cls.from_xyz_quaternion(
            *data["position"][:3],
            *data["rotation"],
            reference_frame=reference_frame,
        )

    def _constant_to_json(self) -> dict[str, Any]:
        return {
            "position": self.position.to_np().tolist(),
            "rotation": self.quaternion.to_np().tolist(),
        }

    @classmethod
    def from_xyz_rpy(
        cls,
        x: sm.ScalarData = 0,
        y: sm.ScalarData = 0,
        z: sm.ScalarData = 0,
        roll: sm.ScalarData = 0,
        pitch: sm.ScalarData = 0,
        yaw: sm.ScalarData = 0,
        reference_frame: CartesianFrame3Dimensional | None = None,
    ) -> Self:
        """
        Creates a Pose3Dimensional object from position (x, y, z) and Euler angles (roll, pitch,
        yaw) values. The function also accepts optional reference and child frame
        parameters.

        :param x: The x-coordinate of the position
        :param y: The y-coordinate of the position
        :param z: The z-coordinate of the position
        :param roll: The rotation around the x-axis
        :param pitch: The rotation around the y-axis
        :param yaw: The rotation around the z-axis
        :param reference_frame: The reference frame for the transformation
        :return: A Pose3Dimensional object created using the provided position and orientation
            values
        """
        p = CartesianPoint3Dimensional(x=x, y=y, z=z)
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
        reference_frame: CartesianFrame3Dimensional | None = None,
    ) -> Self:
        """
        Creates a `Pose3Dimensional` instance from the provided position coordinates and quaternion
        values representing rotation. This method constructs a 3D point for the position
        and a rotation matrix derived from the quaternion, and initializes the
        transformation matrix with these along with optional reference and child frame
        entities.

        :param pos_x: X coordinate of the position in space.
        :param pos_y: Y coordinate of the position in space.
        :param pos_z: Z coordinate of the position in space.
        :param quat_w: W component of the quaternion representing rotation.
        :param quat_x: X component of the quaternion representing rotation.
        :param quat_y: Y component of the quaternion representing rotation.
        :param quat_z: Z component of the quaternion representing rotation.
        :param reference_frame: Optional reference frame for the transformation matrix.
        :return: A `Pose3Dimensional` object constructed from the given parameters.
        """
        p = CartesianPoint3Dimensional(x=pos_x, y=pos_y, z=pos_z)
        r = Quaternion(w=quat_w, x=quat_x, y=quat_y, z=quat_z)
        return cls(p, r, reference_frame=reference_frame)

    @classmethod
    def from_xyz_axis_angle(
        cls,
        x: sm.ScalarData = 0,
        y: sm.ScalarData = 0,
        z: sm.ScalarData = 0,
        axis: Vector3Dimensional | sm.NumericalVector = None,
        angle: sm.ScalarData = 0,
        reference_frame: CartesianFrame3Dimensional | None = None,
    ) -> Self:
        """
        Creates an instance of the class from x, y, z coordinates, axis and angle.

        This class method generates an object using provided spatial coordinates and a
        rotation defined by an axis and angle. The resulting object is defined with a
        specified reference frame and child frame.

        :param x: Initial x-coordinate.
        :param y: Initial y-coordinate.
        :param z: Initial z-coordinate.
        :param axis: Vector defining the axis of rotation. Defaults to Vector3Dimensional(0, 0, 1)
            if not specified.
        :param angle: Angle of rotation around the specified axis, in radians.
        :param reference_frame: Reference frame entity to be associated with the object.
        :return: An instance of the class with the specified transformations applied.
        """
        if axis is None:
            axis = Vector3Dimensional(0, 0, 1)
        elif not isinstance(axis, Vector3Dimensional):
            axis = Vector3Dimensional.from_iterable(axis)
        rotation_matrix = Quaternion.from_axis_angle(AxisAngle(axis=axis, angle=angle))
        point = CartesianPoint3Dimensional(x=x, y=y, z=z)
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

    @property
    def roll(self) -> sm.Scalar:
        return self.rotation_matrix.rpy[0]

    @property
    def pitch(self) -> sm.Scalar:
        return self.rotation_matrix.rpy[1]

    @property
    def yaw(self) -> sm.Scalar:
        return self.rotation_matrix.rpy[2]

    @y.setter
    def y(self, value: sm.ScalarData):
        self[1, 3] = value

    @property
    def z(self) -> sm.Scalar:
        return self[2, 3]

    @z.setter
    def z(self, value: sm.ScalarData):
        self[2, 3] = value

    @property
    def position(self) -> CartesianPoint3Dimensional:
        result = CartesianPoint3Dimensional.from_casadi_sx(
            casadi_sx=self.casadi_sx[:4, 3]
        )
        result.reference_frame = self.reference_frame
        return result

    @property
    def orientation(self) -> Quaternion:
        return self.quaternion

    @property
    def quaternion(self) -> Quaternion:
        return self.rotation_matrix.quaternion

    @property
    def axis_angle(self) -> AxisAngle:
        """
        :return: The orientation of this pose.
        """
        return self.quaternion.axis_angle

    @property
    def rotation_matrix(self) -> RotationMatrix:
        """
        :return: The orientation of this pose.
        """
        return RotationMatrix(data=self, reference_frame=self.reference_frame)

    @property
    def homogeneous_matrix(self) -> HomogeneousTransformationMatrix3Dimensional:
        """
        :return: The transformation that moves the reference frame to this pose.
        """
        return HomogeneousTransformationMatrix3Dimensional(
            data=self, reference_frame=self.reference_frame
        )

    def __hash__(self):
        if self.is_constant():
            return hash(
                (
                    *self.position.to_np().tolist(),
                    *self.quaternion.to_np().tolist(),
                    self.reference_frame,
                )
            )
        return super().__hash__()


@sm.substitution_cache
def rotation_matrix_to_quaternion(r: Matrix):
    """
    This method constructs a quaternion representation of the provided rotation matrix.

    It is designed to handle different cases of rotation matrix configurations to ensure numerical stability during computation. The resultant
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
