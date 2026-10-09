from dataclasses import dataclass, field
from io import BytesIO
from uuid import UUID

import numpy as np
import trimesh
import trimesh.exchange.stl
from sqlalchemy import TypeDecorator, types
from typing_extensions import List, Optional, Type

from krrood.ormatic.data_access_objects.alternative_mappings import AlternativeMapping
from semantic_digital_twin.mixin import HasSimulatorProperties
from semantic_digital_twin.spatial_types import (
    RotationMatrix,
    Vector3,
    Point3,
    HomogeneousTransformationMatrix,
)
from semantic_digital_twin.spatial_types.spatial_types import (
    Quaternion,
    Pose,
    Point2,
    Pose2D,
    AxisAngle,
    SpatialType,
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
    kinematic_structure_entities: List[KinematicStructureEntity]
    connections: List[Connection]
    semantic_annotations: List[SemanticAnnotation]
    degrees_of_freedom: List[DegreeOfFreedom]
    state: WorldState
    name: Optional[str]
    modification_history: List[WorldModelModificationBlock]

    @classmethod
    def from_domain_object(cls, obj: World):
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
                modification_block.apply(result)

            result.state = self.state
            result.state._world = result
        return result

    @classmethod
    def required_pre_build_classes(cls) -> List[Type]:
        return [WorldState, SpatialType, WorldEntity]

    __hash__ = AlternativeMapping.__hash__


@dataclass(eq=False)
class WorldStateMapping(AlternativeMapping[WorldState]):
    data: List[float]
    ids: List[UUID]

    @classmethod
    def from_domain_object(cls, obj: WorldState):
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
class Vector3Mapping(AlternativeMapping[Vector3]):
    x: float
    y: float
    z: float

    reference_frame: Optional[KinematicStructureEntity]

    @classmethod
    def from_domain_object(cls, obj: Vector3):
        x, y, z, _ = obj.to_np().tolist()
        result = cls(x=x, y=y, z=z, reference_frame=obj.reference_frame)
        return result

    def to_domain_object(self) -> Vector3:
        return Vector3(
            x=self.x, y=self.y, z=self.z, reference_frame=self.reference_frame
        )


@dataclass(eq=False)
class Point3Mapping(AlternativeMapping[Point3]):
    x: float
    y: float
    z: float

    reference_frame: Optional[KinematicStructureEntity]

    @classmethod
    def from_domain_object(cls, obj: Point3):
        x, y, z, _ = float(obj.x), float(obj.y), float(obj.z), obj.reference_frame
        result = cls(x=x, y=y, z=z, reference_frame=obj.reference_frame)
        return result

    def to_domain_object(self) -> Point3:
        return Point3(
            x=self.x, y=self.y, z=self.z, reference_frame=self.reference_frame
        )


@dataclass(eq=False)
class QuaternionMapping(AlternativeMapping[Quaternion]):
    x: float
    y: float
    z: float
    w: float

    reference_frame: Optional[KinematicStructureEntity]

    @classmethod
    def from_domain_object(cls, obj: Quaternion):
        x, y, z, w = float(obj.x), float(obj.y), float(obj.z), float(obj.w)
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
    axis: Vector3
    angle: float
    reference_frame: Optional[KinematicStructureEntity]

    @classmethod
    def from_domain_object(cls, obj: AxisAngle):
        return cls(
            axis=obj.axis, angle=float(obj.angle), reference_frame=obj.reference_frame
        )

    def to_domain_object(self) -> AxisAngle:
        return AxisAngle(
            axis=self.axis, angle=self.angle, reference_frame=self.reference_frame
        )

    @classmethod
    def required_pre_build_classes(cls) -> List[Type]:
        return [Vector3]


@dataclass(eq=False)
class RotationMatrixMapping(AlternativeMapping[RotationMatrix]):
    axis_angle: AxisAngle
    reference_frame: Optional[KinematicStructureEntity]

    @classmethod
    def from_domain_object(cls, obj: RotationMatrix):
        return cls(axis_angle=obj.axis_angle, reference_frame=obj.reference_frame)

    def to_domain_object(self) -> RotationMatrix:
        rotation_matrix = self.axis_angle.rotation_matrix
        rotation_matrix.reference_frame = self.reference_frame
        return rotation_matrix

    @classmethod
    def required_pre_build_classes(cls) -> List[Type]:
        return [AxisAngle]


@dataclass(eq=False)
class HomogeneousTransformationMatrixMapping(
    AlternativeMapping[HomogeneousTransformationMatrix]
):
    position: Point3
    axis_angle: AxisAngle

    reference_frame: Optional[KinematicStructureEntity]
    child_frame: Optional[KinematicStructureEntity]

    @classmethod
    def from_domain_object(cls, obj: HomogeneousTransformationMatrix):
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
    def required_pre_build_classes(cls) -> List[Type]:
        return [Point3, AxisAngle]


@dataclass(eq=False)
class PoseMapping(AlternativeMapping[Pose]):
    position: Point3
    axis_angle: AxisAngle
    reference_frame: Optional[KinematicStructureEntity] = field(
        default=None, kw_only=True
    )

    @classmethod
    def from_domain_object(cls, obj: Pose):
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
    def required_pre_build_classes(cls) -> List[Type]:
        return [Point3, AxisAngle]


@dataclass(eq=False)
class Point2Mapping(AlternativeMapping[Point2]):
    x: float
    y: float

    reference_frame: Optional[KinematicStructureEntity]

    @classmethod
    def from_domain_object(cls, obj: Point2):
        result = cls(
            x=float(obj.x), y=float(obj.y), reference_frame=obj.reference_frame
        )
        return result

    def to_domain_object(self) -> Point2:
        return Point2(x=self.x, y=self.y, reference_frame=self.reference_frame)


@dataclass(eq=False)
class Pose2DMapping(AlternativeMapping[Pose2D]):
    position: Point2
    yaw: float
    reference_frame: Optional[KinematicStructureEntity] = field(
        default=None, kw_only=True
    )

    @classmethod
    def from_domain_object(cls, obj: Pose2D):
        result = cls(position=obj.position, yaw=float(obj.yaw))
        result.reference_frame = obj.reference_frame
        return result

    def to_domain_object(self) -> Pose2D:
        return Pose2D.from_position_and_yaw(
            self.position,
            yaw=self.yaw,
            reference_frame=self.reference_frame,
        )

    @classmethod
    def required_pre_build_classes(cls) -> List[Type]:
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

    def process_result_value(self, value: impl, dialect) -> Optional[trimesh.Trimesh]:
        if value is None:
            return None
        mesh = trimesh.Trimesh(**trimesh.exchange.stl.load_stl_binary(BytesIO(value)))
        return mesh
