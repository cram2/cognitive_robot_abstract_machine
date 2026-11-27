"""
The alternative mappings and column types semantic_digital_twin stores its
non-dataclass types, such as the world and the spatial types, with.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from io import BytesIO
from uuid import UUID

import numpy as np
import trimesh
import trimesh.exchange.stl
from sqlalchemy import TypeDecorator, types
from typing_extensions import Self

from krrood.ormatic.data_access_objects.alternative_mappings import AlternativeMapping
from semantic_digital_twin.mixin import HasSimulatorProperties
from krrood_spatial_math.spatial_types import (
    RotationMatrix,
    Vector3,
    Point3,
    HomogeneousTransformationMatrix,
)
from krrood_spatial_math.spatial_types import (
    Quaternion,
    Pose,
    Point2,
    Pose2D,
    AxisAngle,
    SpatialType,
    Displacement,
    ScrewAxis,
    Twist,
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


@dataclass(eq=False, kw_only=True)
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
        return [WorldState, SpatialType, WorldEntity]

    __hash__ = AlternativeMapping.__hash__


@dataclass(eq=False, kw_only=True)
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


@dataclass(eq=False, kw_only=True)
class Vector3Mapping(AlternativeMapping[Vector3]):
    """
    The coordinates of a :class:`Vector3`, as they evaluate in the world's state when
    it is stored, and the frame it is expressed in.
    """

    x: float
    y: float
    z: float

    reference_frame: KinematicStructureEntity | None

    @classmethod
    def from_domain_object(cls, *, obj: Vector3):
        x, y, z, _ = obj.evaluate().tolist()
        result = cls(x=x, y=y, z=z, reference_frame=obj.reference_frame)
        return result

    def to_domain_object(self) -> Vector3:
        return Vector3(
            x=self.x, y=self.y, z=self.z, reference_frame=self.reference_frame
        )


@dataclass(eq=False, kw_only=True)
class Point3Mapping(AlternativeMapping[Point3]):
    """
    The coordinates of a :class:`Point3`, as they evaluate in the world's state when it
    is stored, and the frame it is expressed in.
    """

    x: float
    y: float
    z: float

    reference_frame: KinematicStructureEntity | None

    @classmethod
    def from_domain_object(cls, *, obj: Point3):
        x, y, z, _ = obj.evaluate().tolist()
        result = cls(x=x, y=y, z=z, reference_frame=obj.reference_frame)
        return result

    def to_domain_object(self) -> Point3:
        return Point3(
            x=self.x, y=self.y, z=self.z, reference_frame=self.reference_frame
        )


@dataclass(eq=False, kw_only=True)
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


@dataclass(eq=False, kw_only=True)
class AxisAngleMapping(AlternativeMapping[AxisAngle]):
    """
    The axis of an :class:`AxisAngle` and its angle, as the angle evaluates in the
    world's state when it is stored, and the frame it is expressed in.
    """

    axis: Vector3
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
        return [Vector3]


@dataclass(eq=False, kw_only=True)
class DisplacementMapping(AlternativeMapping[Displacement]):
    translation: Vector3
    rotation_vector: Vector3
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
        return [Vector3]


@dataclass(eq=False, kw_only=True)
class TwistMapping(AlternativeMapping[Twist]):
    """
    A :class:`Twist` stored as its two velocities and its reference frame.
    """

    angular_velocity: Vector3
    """
    The axis the body turns around, scaled by how fast it turns.
    """

    linear_velocity: Vector3
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
        :class:`Vector3` both vectors are stored as.
        """
        return [Vector3]


@dataclass(eq=False, kw_only=True)
class ScrewAxisMapping(AlternativeMapping[ScrewAxis]):
    """
    A :class:`ScrewAxis` stored as its rotation and translation per unit of position and
    its reference frame.
    """

    rotation_per_unit_position: Vector3
    """
    The axis the body turns around, scaled by the angle it turns by per unit of position.
    """

    translation_per_unit_position: Vector3
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
        :class:`Vector3` both vectors are stored as.
        """
        return [Vector3]


@dataclass(eq=False, kw_only=True)
class RotationMatrixMapping(AlternativeMapping[RotationMatrix]):
    axis_angle: AxisAngle
    reference_frame: KinematicStructureEntity | None

    @classmethod
    def from_domain_object(cls, *, obj: RotationMatrix):
        return cls(axis_angle=obj.axis_angle, reference_frame=obj.reference_frame)

    def to_domain_object(self) -> RotationMatrix:
        rotation_matrix = self.axis_angle.rotation_matrix
        rotation_matrix.reference_frame = self.reference_frame
        return rotation_matrix

    @classmethod
    def required_pre_build_classes(cls) -> list[type]:
        return [AxisAngle]


@dataclass(eq=False, kw_only=True)
class HomogeneousTransformationMatrixMapping(
    AlternativeMapping[HomogeneousTransformationMatrix]
):
    position: Point3
    axis_angle: AxisAngle

    reference_frame: KinematicStructureEntity | None
    child_frame: KinematicStructureEntity | None

    @classmethod
    def from_domain_object(cls, *, obj: HomogeneousTransformationMatrix):
        return cls(
            position=obj.position,
            axis_angle=obj.axis_angle,
            reference_frame=obj.reference_frame,
            child_frame=obj.child_frame,
        )

    def to_domain_object(self) -> HomogeneousTransformationMatrix:
        return HomogeneousTransformationMatrix.from_point_rotation_matrix(
            point=self.position,
            rotation_matrix=self.axis_angle.rotation_matrix,
            reference_frame=self.reference_frame,
            child_frame=self.child_frame,
        )

    @classmethod
    def required_pre_build_classes(cls) -> list[type]:
        return [Point3, AxisAngle]


@dataclass(eq=False, kw_only=True)
class PoseMapping(AlternativeMapping[Pose]):
    position: Point3
    axis_angle: AxisAngle
    reference_frame: KinematicStructureEntity | None = field(default=None, kw_only=True)

    @classmethod
    def from_domain_object(cls, *, obj: Pose):
        result = cls(position=obj.position, axis_angle=obj.axis_angle)
        result.reference_frame = obj.reference_frame
        return result

    def to_domain_object(self) -> Pose:
        return Pose(
            position=self.position,
            orientation=self.axis_angle.quaternion,
            reference_frame=self.reference_frame,
        )

    @classmethod
    def from_point_mapping_quaternion_mapping(
        cls,
        *,
        position: Point3Mapping,
        orientation: QuaternionMapping,
        reference_frame: KinematicStructureEntity,
    ) -> Pose:
        """
        Creates a Pose instance from a Point3Mapping and a QuaternionMapping.

        This method constructs a Pose object by utilizing the provided Point3Mapping for
        the position and the QuaternionMapping for the orientation. The resulting Pose
        is associated with the specified reference frame.

        :param position: A Point3Mapping object that provides the position data for the
            Pose.
        :param orientation: A QuaternionMapping object that provides the orientation
            data for the Pose.
        :param reference_frame: The reference frame to which the Pose will be
            associated.
        :return: A Pose instance created from the given Point3Mapping and
            QuaternionMapping.
        """
        return Pose(
            position=position.to_domain_object(),
            orientation=orientation.to_domain_object(),
            reference_frame=reference_frame,
        )

    @classmethod
    def required_pre_build_classes(cls) -> list[type]:
        return [Point3, AxisAngle]


@dataclass(eq=False, kw_only=True)
class Point2Mapping(AlternativeMapping[Point2]):
    """
    The coordinates of a :class:`Point2`, as they evaluate in the world's state when it
    is stored, and the frame it is expressed in.
    """

    x: float
    y: float

    reference_frame: KinematicStructureEntity | None

    @classmethod
    def from_domain_object(cls, *, obj: Point2):
        x, y = obj.evaluate().tolist()[:2]
        result = cls(x=x, y=y, reference_frame=obj.reference_frame)
        return result

    def to_domain_object(self) -> Point2:
        return Point2(x=self.x, y=self.y, reference_frame=self.reference_frame)


@dataclass(eq=False, kw_only=True)
class Pose2DMapping(AlternativeMapping[Pose2D]):
    """
    The position of a :class:`Pose2D` and its yaw, as the yaw evaluates in the world's
    state when it is stored, and the frame it is expressed in.
    """

    position: Point2
    yaw: float
    reference_frame: KinematicStructureEntity | None = field(default=None, kw_only=True)

    @classmethod
    def from_domain_object(cls, *, obj: Pose2D):
        result = cls(position=obj.position, yaw=float(obj.yaw.evaluate()[0]))
        result.reference_frame = obj.reference_frame
        return result

    def to_domain_object(self) -> Pose2D:
        return Pose2D(
            position=self.position,
            yaw=self.yaw,
            reference_frame=self.reference_frame,
        )

    @classmethod
    def required_pre_build_classes(cls) -> list[type]:
        return [Point2]


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
