from dataclasses import dataclass, field
from io import BytesIO
from uuid import UUID

import numpy as np
import trimesh
import trimesh.exchange.stl
from sqlalchemy import TypeDecorator, types
from typing_extensions import List
from typing_extensions import Optional

from krrood.ormatic.dao import AlternativeMapping
from ..datastructures.prefixed_name import PrefixedName
from ..spatial_types import (
    RotationMatrix,
    Vector3,
    Point3,
    HomogeneousTransformationMatrix,
)
from ..spatial_types.derivatives import DerivativeMap
from ..spatial_types.spatial_types import Quaternion, Pose
from ..world import World
from ..world_description.connections import Connection
from ..world_description.degree_of_freedom import DegreeOfFreedom, DegreeOfFreedomLimits
from ..world_description.world_entity import (
    SemanticAnnotation,
    KinematicStructureEntity,
)
from ..world_description.world_state import WorldState


@dataclass
class WorldMapping(AlternativeMapping[World]):
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
            result.delete_orphaned_dofs()
            result.state = self.state
            result.state._world = result

        return result


@dataclass
class WorldStateMapping(AlternativeMapping[WorldState]):
    data: List[float]
    ids: List[UUID]

    @classmethod
    def from_domain_object(cls, obj: WorldState):
        return cls(
            data=obj.data.ravel().tolist(),
            ids=obj._ids,
        )

    def to_domain_object(self) -> WorldState:
        return WorldState(
            data=np.array(self.data, dtype=np.float64).reshape((4, len(self.ids))),
            _ids=self.ids,
            _index={name: idx for idx, name in enumerate(self.ids)},
        )


@dataclass
class Vector3Mapping(AlternativeMapping[Vector3]):
    x: float
    y: float
    z: float

    reference_frame: Optional[KinematicStructureEntity] = field(
        init=False, default=None
    )

    @classmethod
    def from_domain_object(cls, obj: Vector3):
        x, y, z, _ = obj.to_np().tolist()
        result = cls(x=x, y=y, z=z)
        result.reference_frame = obj.reference_frame
        return result

    def to_domain_object(self) -> Vector3:
        return Vector3(x=self.x, y=self.y, z=self.z, reference_frame=None)


@dataclass
class Point3Mapping(AlternativeMapping[Point3]):
    x: float
    y: float
    z: float

    reference_frame: Optional[KinematicStructureEntity] = field(
        init=False, default=None
    )

    @classmethod
    def from_domain_object(cls, obj: Point3):
        x, y, z, _ = obj.to_np().tolist()
        result = cls(x=x, y=y, z=z)
        result.reference_frame = obj.reference_frame
        return result

    def to_domain_object(self) -> Point3:
        return Point3(x=self.x, y=self.y, z=self.z, reference_frame=None)


@dataclass
class QuaternionMapping(AlternativeMapping[Quaternion]):
    x: float
    y: float
    z: float
    w: float

    reference_frame: Optional[KinematicStructureEntity] = field(
        init=False, default=None
    )

    @classmethod
    def from_domain_object(cls, obj: Quaternion):
        x, y, z, w = obj.to_np().tolist()
        result = cls(x=x, y=y, z=z, w=w)
        result.reference_frame = obj.reference_frame
        return result

    def to_domain_object(self) -> Quaternion:
        return Quaternion(
            x=self.x,
            y=self.y,
            z=self.z,
            w=self.w,
            reference_frame=None,
        )


@dataclass
class RotationMatrixMapping(AlternativeMapping[RotationMatrix]):
    rotation: Quaternion
    reference_frame: Optional[KinematicStructureEntity] = field(
        init=False, default=None
    )

    @classmethod
    def from_domain_object(cls, obj: RotationMatrix):
        result = cls(rotation=obj.to_quaternion())
        result.reference_frame = obj.reference_frame
        return result

    def to_domain_object(self) -> RotationMatrix:
        result = RotationMatrix.from_quaternion(self.rotation)
        result.reference_frame = None
        return result


@dataclass
class HomogeneousTransformationMatrixMapping(
    AlternativeMapping[HomogeneousTransformationMatrix]
):
    position: Point3
    rotation: Quaternion
    reference_frame: Optional[KinematicStructureEntity] = field(
        init=False, default=None
    )
    child_frame: Optional[KinematicStructureEntity] = field(init=False, default=None)

    @classmethod
    def from_domain_object(cls, obj: HomogeneousTransformationMatrix):
        position = obj.to_position()
        rotation = obj.to_quaternion()
        result = cls(position=position, rotation=rotation)
        result.reference_frame = obj.reference_frame
        result.child_frame = obj.child_frame

        return result

    def to_domain_object(self) -> HomogeneousTransformationMatrix:
        return HomogeneousTransformationMatrix.from_point_rotation_matrix(
            point=self.position,
            rotation_matrix=RotationMatrix.from_quaternion(self.rotation),
            reference_frame=None,
            child_frame=self.child_frame,
        )


@dataclass
class PoseMapping(AlternativeMapping[Pose]):
    position: Point3
    rotation: Quaternion
    reference_frame: Optional[KinematicStructureEntity] = field(
        init=False, default=None
    )

    @classmethod
    def from_domain_object(cls, obj: Pose):
        position = obj.to_position()
        rotation = obj.to_quaternion()
        result = cls(position=position, rotation=rotation)
        result.reference_frame = obj.reference_frame
        return result

    def to_domain_object(self) -> Pose:
        return Pose(
            position=self.position,
            orientation=self.rotation,
            reference_frame=None,
        )


@dataclass
class DegreeOfFreedomLimitsMapping(AlternativeMapping[DegreeOfFreedomLimits]):
    lower: List[float]
    upper: List[float]

    @classmethod
    def from_domain_object(cls, obj: DegreeOfFreedomLimits):
        return cls(
            lower=obj.lower.data,
            upper=obj.upper.data,
        )

    def to_domain_object(self) -> DegreeOfFreedomLimits:
        return DegreeOfFreedomLimits(
            lower=DerivativeMap(data=self.lower), upper=DerivativeMap(data=self.upper)
        )


@dataclass
class DegreeOfFreedomMapping(AlternativeMapping[DegreeOfFreedom]):
    name: PrefixedName
    limits: DegreeOfFreedomLimits
    id: UUID

    @classmethod
    def from_domain_object(cls, obj: DegreeOfFreedom):
        return cls(
            name=obj.name,
            limits=obj.limits,
            id=obj.id,
        )

    def to_domain_object(self) -> DegreeOfFreedom:
        return DegreeOfFreedom(
            name=self.name,
            limits=DegreeOfFreedomLimits(
                lower=self.limits.lower, upper=self.limits.upper
            ),
            id=self.id,
        )


class TrimeshType(TypeDecorator):
    """
    Type that casts fields that are of type `type` to their class name on serialization and converts the name
    to the class itself through the globals on load.
    """

    impl = types.LargeBinary(4 * 1024 * 1024 * 1024 - 1)  # 4 GB max

    def process_bind_param(self, value: trimesh.Trimesh, dialect):
        # return binary version of trimesh
        return trimesh.exchange.stl.export_stl(value)

    def process_result_value(self, value: impl, dialect) -> Optional[trimesh.Trimesh]:
        if value is None:
            return None
        mesh = trimesh.Trimesh(**trimesh.exchange.stl.load_stl_binary(BytesIO(value)))
        return mesh
