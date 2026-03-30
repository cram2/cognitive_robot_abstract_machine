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
from semantic_digital_twin.world_description.world_state import WorldState


@dataclass(eq=False)
class WorldMapping(HasSimulatorProperties, AlternativeMapping[World]):
    kinematic_structure_entities: List[KinematicStructureEntity]
    connections: List[Connection]
    semantic_annotations: List[SemanticAnnotation]
    degrees_of_freedom: List[DegreeOfFreedom]
    state: WorldState
    name: Optional[str] = field(default=None)

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
        )

    def to_domain_object(self) -> World:
        result = World(name=self.name)

        with result.modify_world():
            for entity in self.kinematic_structure_entities:
                result.add_kinematic_structure_entity(entity)

            for dof in self.degrees_of_freedom:
                result.add_degree_of_freedom(dof)

            for connection in self.connections:
                result.add_connection(connection)

            for semantic_annotation in self.semantic_annotations:
                result.add_semantic_annotation(semantic_annotation)

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
            data=obj._data.ravel().tolist(),
            ids=obj._ids,
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
        x, y, z, _ = obj.to_np().tolist()
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
        x, y, z, w = obj.to_np().tolist()
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
class RotationMatrixMapping(AlternativeMapping[RotationMatrix]):
    rotation: Quaternion
    reference_frame: Optional[KinematicStructureEntity]

    @classmethod
    def from_domain_object(cls, obj: RotationMatrix):
        result = cls(rotation=obj.to_quaternion(), reference_frame=obj.reference_frame)
        return result

    def to_domain_object(self) -> RotationMatrix:
        result = RotationMatrix.from_quaternion(self.rotation)
        return result

    @classmethod
    def required_pre_build_classes(cls) -> List[Type]:
        return [Quaternion]


@dataclass(eq=False)
class HomogeneousTransformationMatrixMapping(
    AlternativeMapping[HomogeneousTransformationMatrix]
):
    position: Point3
    rotation: Quaternion

    reference_frame: Optional[KinematicStructureEntity]
    child_frame: Optional[KinematicStructureEntity]

    @classmethod
    def from_domain_object(cls, obj: HomogeneousTransformationMatrix):
        position = obj.to_position()
        rotation = obj.to_quaternion()
        result = cls(
            position=position,
            rotation=rotation,
            reference_frame=obj.reference_frame,
            child_frame=obj.child_frame,
        )

        return result

    def to_domain_object(self) -> HomogeneousTransformationMatrix:
        return HomogeneousTransformationMatrix.from_point_rotation_matrix(
            point=self.position,
            rotation_matrix=self.rotation.to_rotation_matrix(),
            reference_frame=self.reference_frame,
            child_frame=self.child_frame,
        )

    @classmethod
    def required_pre_build_classes(cls) -> List[Type]:
        return [Quaternion, Point3]


@dataclass(eq=False)
class PoseMapping(AlternativeMapping[Pose]):
    position: Point3
    orientation: Quaternion
    reference_frame: Optional[KinematicStructureEntity] = field(
        default=None, kw_only=True
    )

    @classmethod
    def from_domain_object(cls, obj: Pose):
        position = obj.to_position()
        orientation = obj.to_quaternion()
        result = cls(position=position, orientation=orientation)
        result.reference_frame = obj.reference_frame
        return result

    def to_domain_object(self) -> Pose:
        return Pose(
            position=self.position,
            orientation=self.orientation,
            reference_frame=self.reference_frame,
        )

    @classmethod
    def from_point_mapping_quaternion_mapping(
        cls,
        point_mapping: Point3Mapping,
        quaternion_mapping: QuaternionMapping,
        reference_frame: KinematicStructureEntity,
    ) -> Pose:
        """
        Creates a Pose instance from a Point3Mapping and a QuaternionMapping.

        This method constructs a Pose object by utilizing the provided Point3Mapping for the position and the
        QuaternionMapping for the orientation. The resulting Pose is associated with the specified reference frame.

        :param point_mapping: A Point3Mapping object that provides the position data for the Pose.
        :param quaternion_mapping: A QuaternionMapping object that provides the orientation data for the Pose.
        :param reference_frame: The reference frame to which the Pose will be associated.
        :return: A Pose instance created from the given Point3Mapping and QuaternionMapping.
        """
        return Pose(
            position=point_mapping.to_domain_object(),
            orientation=quaternion_mapping.to_domain_object(),
            reference_frame=reference_frame,
        )

    @classmethod
    def required_pre_build_classes(cls) -> List[Type]:
        return [Point3, Quaternion]


class TrimeshType(TypeDecorator):
    """
    Type that casts fields that are of type `type` to their class name on serialization and converts the name
    to the class itself through the globals on load.
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
