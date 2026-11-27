"""
The alternative mappings and column types semantic_digital_twin stores its non-dataclass
types, such as the world and the spatial types, with.
"""

from __future__ import annotations

from abc import ABC
from dataclasses import dataclass, field
from io import BytesIO
from uuid import UUID

import numpy as np
import trimesh
import trimesh.exchange.stl
from sqlalchemy import TypeDecorator, types
from typing_extensions import Self

from krrood.ormatic.data_access_objects.alternative_mappings import AlternativeMapping
from krrood_spatial_math.point import Point
from semantic_digital_twin.mixin import HasSimulatorProperties
from krrood_spatial_math.spatial_types import (
    RotationMatrix,
    Vector3Dimensional,
    Point3Dimensional,
    HomogeneousTransformationMatrix3Dimensional,
)
from krrood_spatial_math.spatial_types import (
    Quaternion,
    Pose3Dimensional,
    AxisAngle,
    SymbolicSpatialType,
    Displacement,
    ScrewAxis,
    Twist,
)
from krrood_spatial_math.spatial_types_2_dimensional import (
    Point2Dimensional,
    Pose2Dimensional,
)
from semantic_digital_twin.world import World
from semantic_digital_twin.world_description.connections import Connection
from semantic_digital_twin.world_description.degree_of_freedom import DegreeOfFreedom
from semantic_digital_twin.world_description.world_entity import (
    SemanticAnnotation,
    KinematicStructureEntity,
    WorldEntity,
)
from semantic_digital_twin.world_description.world_modification import (
    WorldModelModificationBlock,
)
from semantic_digital_twin.world_description.world_state import WorldState


@dataclass(eq=False)
class WorldMapping(HasSimulatorProperties, AlternativeMapping[World]):
    """
    A world as ORMatic stores it: the modifications that build it again, its state and
    its entities.
    """

    kinematic_structure_entities: list[KinematicStructureEntity]
    connections: list[Connection]
    semantic_annotations: list[SemanticAnnotation]
    degrees_of_freedom: list[DegreeOfFreedom]
    state: WorldState
    name: str | None
    modification_history: list[WorldModelModificationBlock]

    @classmethod
    def from_domain_object(cls, *, obj: World):
        return cls(
            kinematic_structure_entities=obj.kinematic_structure_entities,
            connections=obj.connections,
            semantic_annotations=obj.semantic_annotations,
            degrees_of_freedom=list(obj.degrees_of_freedom),
            state=obj.state,
            name=obj.name,
            simulator_additional_properties=obj.simulator_additional_properties,
            modification_history=obj.modification_history,
        )

    def to_domain_object(self) -> World:
        result = World(name=self.name)

        with result.modify_world():
            for modification_block in self.modification_history:
                modification_block.apply(world=result)

            result.state = self.state
            result.state._world = result
        return result

    @classmethod
    def required_pre_build_classes(cls) -> list[type]:
        return [WorldState, SymbolicSpatialType, WorldEntity]

    __hash__ = AlternativeMapping.__hash__


@dataclass(eq=False)
class WorldStateMapping(AlternativeMapping[WorldState]):
    data: list[float]
    ids: list[UUID]

    @classmethod
    def from_domain_object(cls, *, obj: WorldState):
        return cls(
            data=obj.data,
            ids=obj.ids,
        )

    def to_domain_object(self) -> WorldState:
        return WorldState(
            _data=np.array(self.data, dtype=np.float64).reshape((4, len(self.ids))),
            _ids=self.ids,
            _index={name: idx for idx, name in enumerate(self.ids)},
        )


@dataclass(eq=False)
class Vector3DimensionalMapping(AlternativeMapping[Vector3Dimensional]):
    """
    The coordinates of a :class:`Vector3Dimensional`, as they evaluate in the world's
    state when it is stored, and the frame it is expressed in.
    """

    x: float
    y: float
    z: float

    reference_frame: KinematicStructureEntity | None

    @classmethod
    def from_domain_object(cls, *, obj: Vector3Dimensional) -> Self:
        x, y, z, _ = obj.evaluate().tolist()
        result = cls(x=x, y=y, z=z, reference_frame=obj.reference_frame)
        return result

    def to_domain_object(self) -> Vector3Dimensional:
        return Vector3Dimensional(
            x=self.x, y=self.y, z=self.z, reference_frame=self.reference_frame
        )


@dataclass(eq=False)
class PointMapping(AlternativeMapping[Point], ABC):
    """
    What every stored point given in a Cartesian frame keeps, whatever its dimension.
    """

    reference_frame: KinematicStructureEntity | None
    """
    The frame the point is given in, or *None* where it is given in no frame.
    """


@dataclass(eq=False, kw_only=True)
class Point3DimensionalMapping(PointMapping, AlternativeMapping[Point3Dimensional]):
    """
    A stored point given by three coordinates in its reference frame.
    """

    x: float
    """
    The amount along the frame's x-axis.
    """

    y: float
    """
    The amount along the frame's y-axis.
    """

    z: float
    """
    The amount along the frame's z-axis.
    """

    @classmethod
    def from_domain_object(cls, *, obj: Point3Dimensional) -> Self:
        x, y, z, _ = obj.evaluate().tolist()
        return cls(x=x, y=y, z=z, reference_frame=obj.reference_frame)

    def to_domain_object(self) -> Point3Dimensional:
        return Point3Dimensional(
            x=self.x, y=self.y, z=self.z, reference_frame=self.reference_frame
        )


@dataclass(eq=False)
class QuaternionMapping(AlternativeMapping[Quaternion]):
    """
    The components of a :class:`Quaternion`, as they evaluate in the world's state when
    it is stored, and the frame it is expressed in.
    """

    x: float
    y: float
    z: float
    w: float

    reference_frame: KinematicStructureEntity | None

    @classmethod
    def from_domain_object(cls, *, obj: Quaternion):
        x, y, z, w = obj.evaluate().tolist()
        result = cls(x=x, y=y, z=z, w=w, reference_frame=obj.reference_frame)
        return result

    def to_domain_object(self) -> Quaternion:
        return Quaternion(
            x=self.x,
            y=self.y,
            z=self.z,
            w=self.w,
            reference_frame=self.reference_frame,
        )


@dataclass(eq=False)
class AxisAngleMapping(AlternativeMapping[AxisAngle]):
    """
    The axis of an :class:`AxisAngle` and its angle, as the angle evaluates in the
    world's state when it is stored, and the frame it is expressed in.
    """

    axis: Vector3Dimensional
    angle: float
    reference_frame: KinematicStructureEntity | None

    @classmethod
    def from_domain_object(cls, *, obj: AxisAngle):
        return cls(
            axis=obj.axis,
            angle=float(obj.angle.evaluate()[0]),
            reference_frame=obj.reference_frame,
        )

    def to_domain_object(self) -> AxisAngle:
        return AxisAngle(
            axis=self.axis, angle=self.angle, reference_frame=self.reference_frame
        )

    @classmethod
    def required_pre_build_classes(cls) -> list[type]:
        return [Vector3Dimensional]


@dataclass(eq=False)
class DisplacementMapping(AlternativeMapping[Displacement]):
    translation: Vector3Dimensional
    rotation_vector: Vector3Dimensional
    reference_frame: KinematicStructureEntity | None

    @classmethod
    def from_domain_object(cls, *, obj: Displacement):
        return cls(
            translation=obj.translation,
            rotation_vector=obj.rotation_vector,
            reference_frame=obj.reference_frame,
        )

    def to_domain_object(self) -> Displacement:
        return Displacement(
            translation=self.translation,
            rotation_vector=self.rotation_vector,
            reference_frame=self.reference_frame,
        )

    @classmethod
    def required_pre_build_classes(cls) -> list[type]:
        return [Vector3Dimensional]


@dataclass(eq=False, kw_only=True)
class TwistMapping(AlternativeMapping[Twist]):
    """
    A :class:`Twist` stored as its two velocities and its reference frame.
    """

    angular_velocity: Vector3Dimensional
    """
    The axis the body turns around, scaled by how fast it turns.
    """

    linear_velocity: Vector3Dimensional
    """
    How fast the point of the body at the origin of the reference frame moves.
    """

    reference_frame: KinematicStructureEntity | None
    """
    The frame along whose axes both velocities are given.
    """

    @classmethod
    def from_domain_object(cls, *, obj: Twist) -> Self:
        return cls(
            angular_velocity=obj.angular_velocity,
            linear_velocity=obj.linear_velocity,
            reference_frame=obj.reference_frame,
        )

    def to_domain_object(self) -> Twist:
        return Twist(
            angular_velocity=self.angular_velocity,
            linear_velocity=self.linear_velocity,
            reference_frame=self.reference_frame,
        )

    @classmethod
    def required_pre_build_classes(cls) -> list[type]:
        """
        Extends :meth:`AlternativeMapping.required_pre_build_classes` by the
        :class:`Vector3Dimensional` both vectors are stored as.
        """
        return [Vector3Dimensional]


@dataclass(eq=False, kw_only=True)
class ScrewAxisMapping(AlternativeMapping[ScrewAxis]):
    """
    A :class:`ScrewAxis` stored as its rotation and translation per unit of position and
    its reference frame.
    """

    rotation_per_unit_position: Vector3Dimensional
    """
    The axis the body turns around, scaled by the angle it turns by per unit of
    position.
    """

    translation_per_unit_position: Vector3Dimensional
    """
    How far the point of the body at the origin of the reference frame moves per unit of
    position.
    """

    reference_frame: KinematicStructureEntity | None
    """
    The frame along whose axes both parts are given.
    """

    @classmethod
    def from_domain_object(cls, *, obj: ScrewAxis) -> Self:
        return cls(
            rotation_per_unit_position=obj.rotation_per_unit_position,
            translation_per_unit_position=obj.translation_per_unit_position,
            reference_frame=obj.reference_frame,
        )

    def to_domain_object(self) -> ScrewAxis:
        return ScrewAxis(
            rotation_per_unit_position=self.rotation_per_unit_position,
            translation_per_unit_position=self.translation_per_unit_position,
            reference_frame=self.reference_frame,
        )

    @classmethod
    def required_pre_build_classes(cls) -> list[type]:
        """
        Extends :meth:`AlternativeMapping.required_pre_build_classes` by the
        :class:`Vector3Dimensional` both vectors are stored as.
        """
        return [Vector3Dimensional]


@dataclass(eq=False)
class HomogeneousTransformationMatrix3DimensionalMapping(
    AlternativeMapping[HomogeneousTransformationMatrix3Dimensional]
):
    position: Point3Dimensional
    axis_angle: AxisAngle

    reference_frame: KinematicStructureEntity | None
    child_frame: KinematicStructureEntity | None

    @classmethod
    def from_domain_object(
        cls, *, obj: HomogeneousTransformationMatrix3Dimensional
    ) -> Self:
        return cls(
            position=obj.position,
            axis_angle=obj.axis_angle,
            reference_frame=obj.reference_frame,
            child_frame=obj.child_frame,
        )

    def to_domain_object(self) -> HomogeneousTransformationMatrix3Dimensional:
        return HomogeneousTransformationMatrix3Dimensional.from_point_rotation_matrix(
            point=self.position,
            rotation_matrix=self.axis_angle.to_coordinate_change(),
            reference_frame=self.reference_frame,
            child_frame=self.child_frame,
        )

    @classmethod
    def required_pre_build_classes(cls) -> list[type]:
        return [Point3Dimensional, AxisAngle]


@dataclass(eq=False)
class RotationMatrixMapping(
    HomogeneousTransformationMatrix3DimensionalMapping,
    AlternativeMapping[RotationMatrix],
):
    """
    A stored rotation: a stored transformation whose position is the origin.
    """

    def to_domain_object(self) -> RotationMatrix:
        rotation_matrix = self.axis_angle.to_coordinate_change()
        rotation_matrix.reference_frame = self.reference_frame
        rotation_matrix.child_frame = self.child_frame
        return rotation_matrix


@dataclass(eq=False, kw_only=True)
class Pose3DimensionalMapping(AlternativeMapping[Pose3Dimensional]):
    position: Point3Dimensional
    axis_angle: AxisAngle
    reference_frame: KinematicStructureEntity | None = field(default=None, kw_only=True)

    @classmethod
    def from_domain_object(cls, *, obj: Pose3Dimensional) -> Self:
        result = cls(position=obj.position, axis_angle=obj.axis_angle)
        result.reference_frame = obj.reference_frame
        return result

    def to_domain_object(self) -> Pose3Dimensional:
        return Pose3Dimensional(
            position=self.position,
            orientation=self.axis_angle.quaternion,
            reference_frame=self.reference_frame,
        )

    @classmethod
    def from_point_mapping_quaternion_mapping(
        cls,
        position: Point3DimensionalMapping,
        orientation: QuaternionMapping,
        reference_frame: KinematicStructureEntity,
    ) -> Pose3Dimensional:
        """
        Creates a :class:`Pose3Dimensional` from the mappings of its position and
        orientation.

        :param position: The mapping of the position.
        :param orientation: The mapping of the orientation.
        :param reference_frame: The frame the pose is given in.
        :return: The pose with that position and orientation.
        """
        return Pose3Dimensional(
            position=position.to_domain_object(),
            orientation=orientation.to_domain_object(),
            reference_frame=reference_frame,
        )

    @classmethod
    def required_pre_build_classes(cls) -> list[type]:
        return [Point3Dimensional, AxisAngle]


@dataclass(eq=False)
class Point2DimensionalMapping(PointMapping, AlternativeMapping[Point2Dimensional]):
    """
    A stored point given by two coordinates on the x-y plane of its reference frame, as
    they evaluate in the world's state when it is stored.
    """

    x: float
    """
    The amount along the frame's x-axis.
    """

    y: float
    """
    The amount along the frame's y-axis.
    """

    @classmethod
    def from_domain_object(cls, *, obj: Point2Dimensional) -> Self:
        x, y = obj.evaluate().tolist()[:2]
        return cls(x=x, y=y, reference_frame=obj.reference_frame)

    def to_domain_object(self) -> Point2Dimensional:
        return Point2Dimensional(
            x=self.x, y=self.y, reference_frame=self.reference_frame
        )


@dataclass(eq=False)
class Pose2DimensionalMapping(AlternativeMapping[Pose2Dimensional]):
    """
    The position of a :class:`Pose2Dimensional` and its yaw, as the yaw evaluates in the
    world's state when it is stored, and the frame it is expressed in.
    """

    position: Point2Dimensional
    yaw: float
    reference_frame: KinematicStructureEntity | None = field(default=None, kw_only=True)

    @classmethod
    def from_domain_object(cls, *, obj: Pose2Dimensional) -> Self:
        result = cls(position=obj.position, yaw=float(obj.yaw.evaluate()[0]))
        result.reference_frame = obj.reference_frame
        return result

    def to_domain_object(self) -> Pose2Dimensional:
        return Pose2Dimensional(
            position=self.position,
            yaw=self.yaw,
            reference_frame=self.reference_frame,
        )

    @classmethod
    def required_pre_build_classes(cls) -> list[type]:
        return [Point2Dimensional]


class TrimeshType(TypeDecorator):
    """
    Type that casts fields that are of type `type` to their class name on serialization
    and converts the name to the class itself through the globals on load.
    """

    impl = types.LargeBinary(4 * 1024 * 1024 * 1024 - 1)  # 4 GB max
    cache_ok = True

    def process_bind_param(self, value: trimesh.Trimesh, dialect):
        # return binary version of trimesh
        return trimesh.exchange.stl.export_stl(value)

    def process_result_value(self, value: impl, dialect) -> trimesh.Trimesh | None:
        if value is None:
            return None
        mesh = trimesh.Trimesh(**trimesh.exchange.stl.load_stl_binary(BytesIO(value)))
        return mesh
