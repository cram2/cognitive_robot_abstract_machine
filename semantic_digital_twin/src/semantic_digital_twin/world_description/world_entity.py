from __future__ import annotations

import inspect
import itertools
from abc import ABC, abstractmethod
from collections import deque
from collections.abc import Iterable, Mapping
from copy import deepcopy
from dataclasses import dataclass, field
from dataclasses import fields
from functools import lru_cache, cached_property
from uuid import UUID, uuid4

import numpy as np
import trimesh
import trimesh.boolean
from scipy.stats import geom
from trimesh.proximity import closest_point, nearby_faces
from trimesh.sample import sample_surface
from typing_extensions import (
    Deque,
    Type,
    TypeVar,
    Dict,
    Any,
    Self,
    ClassVar,
)
from typing_extensions import List, Optional, TYPE_CHECKING, Tuple
from typing_extensions import Set

from krrood.adapters.exceptions import JSON_TYPE_NAME
from krrood.adapters.json_serializer import (
    SubclassJSONSerializer,
    to_json,
    from_json,
)
from krrood.entity_query_language.predicate import Symbol
from krrood.symbolic_math.symbolic_math import Matrix
from .geometry import TriangleMesh
from .inertial_properties import Inertial
from .shape_collection import ShapeCollection, BoundingBoxCollection
from ..adapters.world_entity_kwargs_tracker import (
    KinematicStructureEntityKwargsTracker,
)
from ..datastructures.prefixed_name import PrefixedName
from ..exceptions import ReferenceFrameMismatchError
from ..spatial_types.spatial_types import (
    HomogeneousTransformationMatrix,
    Point3,
)
from ..utils import IDGenerator, type_string_to_type, camel_case_split

if TYPE_CHECKING:
    from ..world_description.degree_of_freedom import DegreeOfFreedom
    from ..world import World, GenericSemanticAnnotation

id_generator = IDGenerator()


@dataclass(eq=False)
class WorldEntity(Symbol):
    """
    A class representing an entity in the world.

    .. warning::
        The WorldEntity class is not meant to be instantiated directly.
    """

    _world: Optional[World] = field(default=None, repr=False, kw_only=True, hash=False)
    """
    The backreference to the world this entity belongs to.
    """

    _semantic_annotations: Set[SemanticAnnotation] = field(
        default_factory=set, init=False, repr=False, hash=False
    )
    """
    The semantic annotations this entity is part of.
    """

    name: PrefixedName = field(default=None, kw_only=True, hash=False)
    """
    The identifier for this world entity.
    """

    def __post_init__(self):
        if self.name is None:
            self.name = PrefixedName(f"{self.__class__.__name__}_{hash(self)}")

    def __eq__(self, other):
        if not isinstance(other, type(self)):
            return False
        return hash(self) == hash(other)

    def add_to_world(self, world: World):
        self._world = world
        world._world_entity_hash_table[hash(self)] = self

    def remove_from_world(self):
        self._world._world_entity_hash_table.pop(hash(self), None)
        self._world = None


@dataclass(eq=False)
class WorldEntityWithID(WorldEntity, SubclassJSONSerializer):
    """
    A WorldEntity that has a unique identifier.

    .. warning::
        The WorldEntity class is not meant to be instantiated directly.
    """

    id: UUID = field(default_factory=uuid4)
    """
    A unique identifier for this world entity.
    """

    @cached_property
    def _hash(self):
        return hash(self.id)

    def __hash__(self):
        return self._hash

    def add_to_world(self, world: World):
        super().add_to_world(world)

    def to_json(self) -> Dict[str, Any]:
        result = super().to_json()
        result["id"] = to_json(self.id)
        return result


@dataclass
class CollisionCheckingConfig(SubclassJSONSerializer):
    buffer_zone_distance: Optional[float] = None
    """
    Distance defining a buffer zone around the entity. The buffer zone represents a soft boundary where
    proximity should be monitored but minor violations are acceptable.
    """

    violated_distance: float = 0.0
    """
    Critical distance threshold that must not be violated. Any proximity below this threshold represents
    a severe collision risk requiring immediate attention.
    """

    disabled: Optional[bool] = None
    """
    Flag to enable/disable collision checking for this entity. When True, all collision checks are ignored.
    """

    max_avoided_bodies: int = 1
    """
    Maximum number of other bodies this body should avoid simultaneously.
    If more bodies than this are in the buffer zone, only the closest ones are avoided.
    """

    def to_json(self) -> Dict[str, Any]:
        json_data = super().to_json()
        for field_ in fields(self):
            value = getattr(self, field_.name)
            json_data[field_.name] = to_json(value)
        return json_data

    @classmethod
    def _from_json(cls, data: Dict[str, Any], **kwargs) -> Self:
        cls_kwargs = {}
        for field_name, json_field_data in data.items():
            if field_name == JSON_TYPE_NAME:
                continue
            cls_kwargs[field_name] = from_json(json_field_data)
        return cls(**cls_kwargs)


@dataclass(eq=False)
class KinematicStructureEntity(WorldEntityWithID, SubclassJSONSerializer, ABC):
    """
    An entity that is part of the kinematic structure of the world.
    """

    _world: Optional[World] = field(default=None, repr=False, hash=False, init=False)
    """
    Setting init=False because it should only be set by the World, not during initialization.
    """

    index: Optional[int] = field(default=None, init=False)
    """
    The index of the entity in `_world.kinematic_structure`.
    """

    def remove_from_world(self):
        super().remove_from_world()
        self.index = None

    @property
    @abstractmethod
    def combined_mesh(self) -> Optional[trimesh.Trimesh]:
        """
        Computes the combined mesh of this KinematicStructureEntity.
        """

    @property
    def center_of_mass(self) -> Point3:
        """
        Computes the center of mass of this KinematicStructureEntity.
        """
        # Center of mass in the body's local frame (collision geometry)
        com_local: np.ndarray[np.float64] = self.combined_mesh.center_mass  # (3,)
        # Transform to world frame using the body's global pose
        com = Point3(
            x=com_local[0],
            y=com_local[1],
            z=com_local[2],
            reference_frame=self,
        )
        world = self._world
        return world.transform(com, world.root)

    @property
    def global_pose(self) -> HomogeneousTransformationMatrix:
        """
        Computes the pose of the KinematicStructureEntity in the world frame.
        :return: TransformationMatrix representing the global pose.
        """
        return self._world.compute_forward_kinematics(self._world.root, self)

    @property
    def parent_connection(self) -> Connection:
        """
        Returns the parent connection of this KinematicStructureEntity.
        """
        return self._world.compute_parent_connection(self)

    @property
    def child_kinematic_structure_entities(self) -> List[KinematicStructureEntity]:
        """
        Returns the direct child KinematicStructureEntity of this entity.
        """
        return self._world.compute_child_kinematic_structure_entities(self)

    @property
    def parent_kinematic_structure_entity(self) -> KinematicStructureEntity:
        """
        Returns the parent KinematicStructureEntity of this entity.
        """
        return self._world.compute_parent_kinematic_structure_entity(self)

    def get_first_parent_connection_of_type(
        self, connection_type: Type[GenericConnection]
    ) -> GenericConnection:
        """
        Traverse the chain up until an active connection is found.
        """
        if self == self._world.root:
            raise ValueError(
                f"Cannot get controlled parent connection for root body {self._world.root.name}."
            )
        if isinstance(self.parent_connection, connection_type):
            return self.parent_connection
        return self.parent_connection.parent.get_first_parent_connection_of_type(
            connection_type
        )


@dataclass(eq=False)
class Body(KinematicStructureEntity, SubclassJSONSerializer):
    """
    Represents a body in the world.
    A body is a semantic atom, meaning that it cannot be decomposed into meaningful smaller parts.
    """

    visual: ShapeCollection = field(default_factory=ShapeCollection, repr=False)
    """
    List of shapes that represent the visual appearance of the link.
    The poses of the shapes are relative to the link.
    """

    collision: ShapeCollection = field(default_factory=ShapeCollection, repr=False)
    """
    List of shapes that represent the collision geometry of the link.
    The poses of the shapes are relative to the link.
    """

    collision_config: Optional[CollisionCheckingConfig] = field(
        default_factory=CollisionCheckingConfig
    )
    """
    Configuration for collision checking.
    """

    temp_collision_config: Optional[CollisionCheckingConfig] = None
    """
    Temporary configuration for collision checking, takes priority over `collision_config`.
    """

    index: Optional[int] = field(default=None, init=False)
    """
    The index of the entity in `_world.kinematic_structure`.
    """

    inertial: Optional[Inertial] = field(default_factory=Inertial)
    """
    Inertia properties of the body.
    """

    def __post_init__(self):
        if not self.name:
            self.name = PrefixedName(f"body_{id_generator(self)}")

        self.visual.reference_frame = self
        self.collision.reference_frame = self
        self.collision.transform_all_shapes_to_own_frame()
        self.visual.transform_all_shapes_to_own_frame()

    @property
    def combined_mesh(self) -> Optional[trimesh.Trimesh]:
        """
        Computes the combined mesh of this KinematicStructureEntity.
        """
        if not self.collision:
            return None
        return self.collision.combined_mesh

    def get_collision_config(self) -> CollisionCheckingConfig:
        if self.temp_collision_config is not None:
            return self.temp_collision_config
        return self.collision_config

    def set_static_collision_config(self, collision_config: CollisionCheckingConfig):
        merged_config = CollisionCheckingConfig(
            buffer_zone_distance=(
                collision_config.buffer_zone_distance
                if collision_config.buffer_zone_distance is not None
                else self.collision_config.buffer_zone_distance
            ),
            violated_distance=collision_config.violated_distance,
            disabled=(
                collision_config.disabled
                if collision_config.disabled is not None
                else self.collision_config.disabled
            ),
            max_avoided_bodies=collision_config.max_avoided_bodies,
        )
        self.collision_config = merged_config

    def set_static_collision_distances(
        self, buffer_zone_distance: float, violated_distance: float
    ):
        self.collision_config.buffer_zone_distance = buffer_zone_distance
        self.collision_config.violated_distance = violated_distance

    def set_temporary_collision_config(self, collision_config: CollisionCheckingConfig):
        self.temp_collision_config = collision_config

    def reset_temporary_collision_config(self):
        self.temp_collision_config = None

    def has_collision(
        self, volume_threshold: float = 1.001e-6, surface_threshold: float = 0.00061
    ) -> bool:
        """
        Check if collision geometry is mesh or simple shape with volume/surface bigger than thresholds.

        :param volume_threshold: Ignore simple geometry shapes with a volume less than this (in m^3)
        :param surface_threshold: Ignore simple geometry shapes with a surface area less than this (in m^2)
        :return: True if collision geometry is mesh or simple shape exceeding thresholds
        """
        return len(self.collision) > 0

    def compute_closest_points_multi(
        self, others: list[Body], sample_size=25
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Computes the closest points to each given body respectively.

        :param others: The list of bodies to compute the closest points to.
        :param sample_size: The number of samples to take from the surface of the other bodies.
        :return: A tuple containing: The points on the self body, the points on the other bodies, and the distances. All points are in the of this body.
        """

        @lru_cache(maxsize=None)
        def evaluated_geometric_distribution(n: int) -> np.ndarray:
            """
            Evaluates the geometric distribution for a given number of samples.
            :param n: The number of samples to evaluate.
            :return: An array of probabilities for each sample.
            """
            return geom.pmf(np.arange(1, n + 1), 0.5)

        query_points = []
        for other in others:
            # Calculate the closest vertex on this body to the other body
            closest_vert_id = self.collision[0].mesh.kdtree.query(
                (
                    self._world.compute_forward_kinematics_np(self, other)
                    @ other.collision[0].origin.to_np()
                )[:3, 3],
                k=1,
            )[1]
            closest_vert = self.collision[0].mesh.vertices[closest_vert_id]

            # Compute the closest faces on the other body to the closes vertex
            faces = nearby_faces(
                other.collision[0].mesh,
                [
                    (
                        self._world.compute_forward_kinematics_np(other, self)
                        @ self.collision[0].origin.to_np()
                    )[:3, 3]
                    + closest_vert
                ],
            )[0]
            face_weights = np.zeros(len(other.collision[0].mesh.faces))

            # Assign weights to the faces based on a geometric distribution
            face_weights[faces] = evaluated_geometric_distribution(len(faces))

            # Sample points on the surface of the other body
            q = sample_surface(
                other.collision[0].mesh, sample_size, face_weight=face_weights, seed=420
            )[0]
            # Make 4x4 transformation matrix from points
            points = np.tile(np.eye(4, dtype=np.float32), (len(q), 1, 1))
            points[:, :3, 3] = q

            # Transform from the mesh to the other mesh
            transform = (
                np.linalg.inv(self.collision[0].origin.to_np())
                @ self._world.compute_forward_kinematics_np(self, other)
                @ other.collision[0].origin.to_np()
            )
            points = points @ transform

            points = points[
                :, :3, 3
            ]  # Extract the points from the transformation matrix

            query_points.extend(points)

        # Actually compute the closest points
        points, dists = closest_point(self.collision[0].mesh, query_points)[:2]
        # Find the closest points for each body out of all the sampled points
        points = np.array(points).reshape(len(others), sample_size, 3)
        dists = np.array(dists).reshape(len(others), sample_size)
        dist_min = np.min(dists, axis=1)
        points_min_self = points[np.arange(len(others)), np.argmin(dists, axis=1), :]
        points_min_other = np.array(query_points).reshape(len(others), sample_size, 3)[
            np.arange(len(others)), np.argmin(dists, axis=1), :
        ]
        return points_min_self, points_min_other, dist_min

    def to_json(self) -> Dict[str, Any]:
        result = super().to_json()
        result["name"] = self.name.to_json()
        result["collision"] = self.collision.to_json()
        result["visual"] = self.visual.to_json()
        result["collision_config"] = to_json(self.collision_config)
        return result

    @classmethod
    def _from_json(cls, data: Dict[str, Any], **kwargs) -> Self:
        result = cls(
            name=PrefixedName.from_json(data["name"], **kwargs),
            id=from_json(data["id"]),
        )
        # add the new body so that the transformation matrices in the shapes can use it as reference frame.
        tracker = KinematicStructureEntityKwargsTracker.from_kwargs(kwargs)
        if not tracker.has_kinematic_structure_entity(result.id):
            tracker.add_kinematic_structure_entity(result)
        else:
            result = tracker.get_kinematic_structure_entity(result.id)

        collision = ShapeCollection.from_json(data["collision"], **kwargs)
        visual = ShapeCollection.from_json(data["visual"], **kwargs)

        for shape in itertools.chain(collision, visual):
            shape.origin.reference_frame = result

        result.collision = collision
        result.visual = visual
        result.collision_config = from_json(data["collision_config"])

        return result

    def get_semantic_annotations_by_type(
        self, type_: Type[GenericSemanticAnnotation]
    ) -> List[GenericSemanticAnnotation]:
        """
        Returns all semantic annotations of a given type which belong to this body.
        :param type_: The type of semantic annotations to return.
        :returns: A list of semantic annotations of the given type.
        """
        return list(
            filter(lambda sem: isinstance(sem, type_), self._semantic_annotations)
        )


@dataclass(eq=False)
class Region(KinematicStructureEntity):
    """
    Virtual KinematicStructureEntity representing a semantic region in the world.
    """

    area: ShapeCollection = field(default_factory=ShapeCollection, hash=False)
    """
    The shapes that represent the area of the region.
    """

    def __post_init__(self):
        self.area.reference_frame = self

    def __hash__(self):
        return id(self)

    @property
    def combined_mesh(self) -> Optional[trimesh.Trimesh]:
        if not self.area:
            return None
        return self.area.combined_mesh

    @classmethod
    def from_3d_points(
        cls,
        name: PrefixedName,
        points_3d: List[Point3],
        reference_frame: Optional[KinematicStructureEntity] = None,
        minimum_thickness: float = 0.005,
        sv_ratio_tol: float = 1e-7,
    ) -> Self:
        """
        Constructs a Region from a list of 3D points by creating a convex hull around them.
        The points are analyzed to determine if they are approximately planar. If they are,
        a minimum thickness is added to ensure the region has a non-zero volume.

        :param name: Prefixed name for the region.
        :param points_3d: List of 3D points.
        :param reference_frame: Optional reference frame.
        :param minimum_thickness: Minimum thickness to add if points are near-planar.
        :param sv_ratio_tol: Tolerance for determining planarity based on singular value ratio.

        :return: Region object.
        """
        points = np.asarray([point.to_np()[:3] for point in points_3d], dtype=float)
        points = np.unique(points, axis=0)
        assert (
            len(points) >= 3
        ), "At least 4 unique points are required to define a 3D region."

        centered_points = points - points.mean(axis=0, keepdims=True)
        assert np.any(centered_points), "Points must not be all identical."

        # We compute the principal axes of the point cloud using SVD.
        # This allows us to reason about the geometric thickness of our point cloud.
        # The axis with the smallest variance, located at the last index if our `principal_axis` is our `normal`
        # indicating the direction of the region's thickness.
        _, variance, principal_axis = np.linalg.svd(
            centered_points, full_matrices=False
        )
        smallest_variance_axis = principal_axis[-1]  # this is our normal
        unit_vector_normal = smallest_variance_axis / np.linalg.norm(
            smallest_variance_axis
        )

        # We compute the thickness, peak-to-peak (max - min), along the normal direction, to get the thickness of
        # the region.
        thickness_in_normal_direction = np.ptp(centered_points @ unit_vector_normal)
        is_near_planar = variance[0] > 0 and variance[-1] / variance[0] < sv_ratio_tol
        thickness_padding = (
            minimum_thickness / 2
            if thickness_in_normal_direction < minimum_thickness or is_near_planar
            else 0.0
        )

        # We do not provide any 2d shapes, since they would be very weird to handle with raytracing etc.
        # Thus we decided that in near-planar cases we add a minimum thickness to ensure we get a 3d shape.
        if thickness_padding > 0:
            P_aug = np.vstack(
                [
                    points + thickness_padding * unit_vector_normal,
                    points - thickness_padding * unit_vector_normal,
                ]
            )
        else:
            P_aug = points

        hull = trimesh.points.PointCloud(P_aug).convex_hull
        hull.remove_unreferenced_vertices()
        hull.update_faces(hull.nondegenerate_faces())
        hull.process()

        area_mesh = TriangleMesh(
            mesh=hull,
            origin=HomogeneousTransformationMatrix(reference_frame=reference_frame),
        )
        return cls(name=name, area=ShapeCollection([area_mesh]))

    def to_json(self) -> Dict[str, Any]:
        result = super().to_json()
        result["name"] = self.name.to_json()
        result["area"] = self.area.to_json()
        return result

    @classmethod
    def _from_json(cls, data: Dict[str, Any], **kwargs) -> Self:
        result = cls(
            name=PrefixedName.from_json(data["name"], id=from_json(data["id"]))
        )
        area = ShapeCollection.from_json(data["area"])
        for shape in area:
            shape.origin.reference_frame = result
        result.area = area
        return result


GenericKinematicStructureEntity = TypeVar(
    "GenericKinematicStructureEntity", bound=KinematicStructureEntity
)

GenericWorldEntity = TypeVar("GenericWorldEntity", bound=WorldEntity)


@dataclass
class SemanticAnnotation(WorldEntity, SubclassJSONSerializer):
    """
    Represents a semantic annotation on a set of bodies in the world.

    This class can hold references to certain bodies that gain meaning in this context.

    .. warning::

        The hash of a semantic annotation is based on the hash of its type and kinematic structure entities.
        Overwrite this with extreme care and only if you know what you are doing. Hashes are used inside rules to check if
        a new semantic annotation has been created. If you, for instance, just use the object identity, this will fail since python assigns
        new memory pointers always. The same holds for the equality operator.
        If you do not want to change the behavior, make sure to use @dataclass(eq=False) to decorate your class.
    """

    _synonyms: ClassVar[Set[str]] = set()
    """
    Additional names that can be used to match this object.
    """

    @classmethod
    @lru_cache(maxsize=None)
    def class_name_tokens(cls) -> Set[str]:
        """
        :return: Set of tokens from the class name.
        """
        return set(n.lower() for n in camel_case_split(cls.__name__))

    def __post_init__(self):
        if self.name is None:
            self.name = PrefixedName(
                name=f"{self.__class__.__name__}_{id_generator(self)}",
                prefix=self._world.name if self._world is not None else None,
            )
        for entity in self.kinematic_structure_entities:
            entity._semantic_annotations.add(self)

    def __hash__(self):
        return hash(
            tuple(
                [self.__class__]
                + sorted([kse.id for kse in self.kinematic_structure_entities])
            )
        )

    def __eq__(self, other):
        return hash(self) == hash(other)

    def to_json(self) -> Dict[str, Any]:
        result = {
            **super().to_json(),
        }

        for semantic_annotation_field in fields(self):
            value = getattr(self, semantic_annotation_field.name)
            if semantic_annotation_field.name.startswith(
                "_"
            ) or semantic_annotation_field.name.startswith("__"):
                continue
            if not issubclass(type(value), SubclassJSONSerializer):
                continue
            result[semantic_annotation_field.name] = value.to_json()
        return result

    @classmethod
    def _from_json(cls, data: Dict[str, Any], **kwargs) -> Self:
        semantic_annotation_fields = {f.name: f for f in fields(cls)}

        init_args = {}

        for k, v in semantic_annotation_fields.items():
            if k not in data.keys():
                continue
            field_type = type_string_to_type(data[k][JSON_TYPE_NAME])
            if issubclass(field_type, SubclassJSONSerializer):
                init_args[k] = field_type.from_json(data[k], **kwargs)

        return cls(**init_args)

    def _kinematic_structure_entities(
        self, visited: Set[int], aggregation_type: Type[GenericKinematicStructureEntity]
    ) -> Set[GenericKinematicStructureEntity]:
        """
        Recursively collects all entities that are part of this semantic annotation.
        """
        stack: Deque[object] = deque([self])
        entities: Set[aggregation_type] = set()

        while stack:
            obj = stack.pop()
            oid = id(obj)
            if oid in visited:
                continue
            visited.add(oid)

            match obj:
                case aggregation_type():
                    entities.add(obj)

                case SemanticAnnotation():
                    stack.extend(_attr_values(obj, aggregation_type))

                case Mapping():
                    stack.extend(
                        v
                        for v in obj.values()
                        if _is_entity_semantic_annotation_or_iterable(
                            v, aggregation_type
                        )
                    )

                case Iterable() if not isinstance(obj, (str, bytes, bytearray)):
                    stack.extend(
                        v
                        for v in obj
                        if _is_entity_semantic_annotation_or_iterable(
                            v, aggregation_type
                        )
                    )

        return entities

    @property
    def kinematic_structure_entities(self) -> Iterable[KinematicStructureEntity]:
        """
        Returns a Iterable of all relevant KinematicStructureEntity in this semantic annotation. The default behaviour is to aggregate all KinematicStructureEntity that are accessible
        through the properties and fields of this semantic annotation, recursively.
        If this behaviour is not desired for a specific semantic annotation, it can be overridden by implementing the `KinematicStructureEntity` property.
        """
        return self._kinematic_structure_entities(set(), KinematicStructureEntity)

    @property
    def bodies(self) -> Iterable[Body]:
        """
        Returns an Iterable of all relevant bodies in this semantic annotation. The default behaviour is to aggregate all bodies that are accessible
        through the properties and fields of this semantic annotation, recursively.
        If this behaviour is not desired for a specific semantic annotation, it can be overridden by implementing the `bodies` property.
        """
        return self._kinematic_structure_entities(set(), Body)

    @property
    def regions(self) -> Iterable[Region]:
        """
        Returns an Iterable of all relevant regions in this semantic annotation. The default behaviour is to aggregate all regions that are accessible
        through the properties and fields of this semantic annotation, recursively.
        If this behaviour is not desired for a specific semantic annotation, it can be overridden by implementing the `regions` property.
        """
        return self._kinematic_structure_entities(set(), Region)

    def as_bounding_box_collection_at_origin(
        self, origin: HomogeneousTransformationMatrix
    ) -> BoundingBoxCollection:
        """
        Returns a bounding box collection that contains the bounding boxes of all bodies in this semantic annotation.
        :param reference_frame: The reference frame to express the bounding boxes in.
        :returns: A collection of bounding boxes in world-space coordinates.
        """

        collections = iter(
            entity.collision.as_bounding_box_collection_at_origin(origin)
            for entity in self.kinematic_structure_entities
            if isinstance(entity, Body) and entity.has_collision()
        )
        bbs = BoundingBoxCollection([], origin.reference_frame)

        for bb_collection in collections:
            bbs = bbs.merge(bb_collection)

        return bbs

    def as_bounding_box_collection_in_frame(
        self, reference_frame: KinematicStructureEntity
    ) -> BoundingBoxCollection:
        """
        Provides the bounding box collection for this entity in the given reference frame.
        :param reference_frame: The reference frame to express the bounding boxes in.
        :returns: A collection of bounding boxes in world-space coordinates.
        """
        return self.as_bounding_box_collection_at_origin(
            HomogeneousTransformationMatrix(reference_frame=reference_frame)
        )


@dataclass(eq=False)
class RootedSemanticAnnotation(SemanticAnnotation):
    """
    Represents a semantic annotation that is rooted in a specific KinematicStructureEntity.
    """

    root: Body = field(default=None)

    @property
    def connections(self) -> List[Connection]:
        return self._world.get_connections_of_branch(self.root)

    @property
    def bodies(self) -> List[Body]:
        return self._world.get_kinematic_structure_entities_of_branch(self.root)

    @property
    def bodies_with_collisions(self) -> List[Body]:
        return [x for x in self.bodies if x.has_collision()]

    @property
    def bodies_with_enabled_collision(self) -> Set[Body]:
        return set(
            body
            for body in self.bodies
            if body.has_collision() and not body.get_collision_config().disabled
        )


@dataclass(eq=False)
class Agent(RootedSemanticAnnotation):
    """
    Represents an entity in the world that can act, move, or be controlled.

    Agents are dynamic bodies with semantic meaning — they may have intent,
    behavior, or be controlled by external or internal logic. Examples include
    robots, humans, or other autonomous actors.

    """

    ...


@dataclass(eq=False)
class Human(Agent):
    """
    Represents a human agent in the environment.

    A Person is an Agent that is not robotically actuated and does not provide
    kinematic chains, manipulators, or robot-specific components.

    This class exists primarily for semantic distinction, so that algorithms
    can treat human agents differently from robots if needed.
    """

    ...


@dataclass(eq=False)
class SemanticEnvironmentAnnotation(RootedSemanticAnnotation):
    """
    Represents a semantic annotation of the environment.
    """

    @property
    def kinematic_structure_entities(self) -> Set[KinematicStructureEntity]:
        """
        Returns a set of all KinematicStructureEntity in the environment semantic annotation.
        """
        return set(
            self._world.get_kinematic_structure_entities_of_branch(self.root)
        ) | {self.root}


@dataclass(eq=False)
class Connection(WorldEntity, SubclassJSONSerializer):
    """
    Represents a connection between two entities in the world.
    """

    _world: Optional[World] = field(default=None, repr=False, hash=False, init=False)
    """
    Setting init=False because it should only be set by the World, not during initialization.
    """

    parent: KinematicStructureEntity
    """
    The parent KinematicStructureEntity of the connection.
    """

    child: KinematicStructureEntity
    """
    The child KinematicStructureEntity of the connection.
    """

    parent_T_connection_expression: HomogeneousTransformationMatrix = field(
        default=None
    )
    _kinematics: HomogeneousTransformationMatrix = field(
        default_factory=HomogeneousTransformationMatrix, init=False
    )
    connection_T_child_expression: HomogeneousTransformationMatrix = field(default=None)
    """
    The origin expression of a connection is split into 2 transforms:
    1. parent_T_connection describes the pose of the connection relative to its parent and must be constant.
       It typically describes the fixed part of the origin expression, equivalent to the origin tag in urdf. 
    3. A transformation describing the connection's kinematics.
       For example, how a revolute joint rotate around its axis.
       This is always created by the connection and not should be copied.
    2. connection_T_child describes the pose of the child relative to the connection and must be constant.

    This split is necessary for copying Connections, because they need parent_T_connection as an input parameter and 
    connection_T_child is generated in the __post_init__ method.
    """

    def __post_init__(self):
        self.name = self.name or self._generate_default_name(
            parent=self.parent, child=self.child
        )

        # If I use default factories, I'd have to complicate the from_json, because I couldn't blindly pass these args
        if self.parent_T_connection_expression is None:
            self.parent_T_connection_expression = HomogeneousTransformationMatrix()
        if self.connection_T_child_expression is None:
            self.connection_T_child_expression = HomogeneousTransformationMatrix()

        if not self.parent_T_connection_expression.is_constant():
            raise RuntimeError(
                f"parent_T_connection must be constant for connection. This one contains free variables: {self.parent_T_connection_expression.free_variables()}"
            )
        if not self.connection_T_child_expression.is_constant():
            raise RuntimeError(
                f"connection_T_child must be constant for connection. This one contains free variables: {self.parent_T_connection_expression.free_variables()}"
            )

        if (
            self.parent_T_connection_expression.reference_frame is not None
            and self.parent_T_connection_expression.reference_frame != self.parent
        ):
            raise ReferenceFrameMismatchError(
                self.parent, self.parent_T_connection_expression.reference_frame
            )
        if (
            self.connection_T_child_expression.child_frame is not None
            and self.connection_T_child_expression.child_frame != self.child
        ):
            raise ReferenceFrameMismatchError(
                self.parent, self.connection_T_child_expression.child_frame
            )

        self.parent_T_connection_expression.reference_frame = self.parent
        self.connection_T_child_expression.child_frame = self.child

    def to_json(self) -> Dict[str, Any]:
        result = super().to_json()
        result["name"] = self.name.to_json()
        result["parent_id"] = to_json(self.parent.id)
        result["child_id"] = to_json(self.child.id)
        result["parent_T_connection_expression"] = (
            self.parent_T_connection_expression.to_json()
        )
        return result

    @classmethod
    def _from_json(cls, data: Dict[str, Any], **kwargs) -> Self:
        tracker = KinematicStructureEntityKwargsTracker.from_kwargs(kwargs)
        parent = tracker.get_kinematic_structure_entity(id=from_json(data["parent_id"]))
        child = tracker.get_kinematic_structure_entity(id=from_json(data["child_id"]))
        return cls(
            name=PrefixedName.from_json(data["name"]),
            parent=parent,
            child=child,
            parent_T_connection_expression=HomogeneousTransformationMatrix.from_json(
                data["parent_T_connection_expression"], **kwargs
            ),
        )

    @property
    def origin_expression(self) -> HomogeneousTransformationMatrix:
        return (
            self.parent_T_connection_expression
            @ self._kinematics
            @ self.connection_T_child_expression
        )

    @property
    def active_dofs(self) -> List[DegreeOfFreedom]:
        return []

    @property
    def passive_dofs(self) -> List[DegreeOfFreedom]:
        return []

    @property
    def is_controlled(self):
        return False

    @property
    def has_hardware_interface(self) -> bool:
        return False

    def add_to_world(self, world: World):
        self._world = world

    @classmethod
    def _generate_default_name(
        cls, parent: KinematicStructureEntity, child: KinematicStructureEntity
    ) -> PrefixedName:
        return PrefixedName(
            f"{parent.name.name}_T_{child.name.name}", prefix=child.name.prefix
        )

    def __hash__(self):
        return hash((self.parent, self.child))

    @property
    def origin(self) -> HomogeneousTransformationMatrix:
        """
        :return: The relative transform between the parent and child frame.
        """
        return self._world.compute_forward_kinematics(self.parent, self.child)

    @origin.setter
    def origin(self, value):
        raise NotImplementedError(
            f"Origin can not be set for Connection: {self.__class__.__name__}"
        )

    def origin_as_position_quaternion(self) -> Matrix:
        position = self.origin_expression.to_position()[:3]
        orientation = self.origin_expression.to_quaternion()
        return Matrix.vstack([position, orientation]).T

    @property
    def dofs(self) -> Set[DegreeOfFreedom]:
        """
        Returns the degrees of freedom associated with this connection.
        """
        dofs = set()

        if hasattr(self, "active_dofs"):
            dofs.update(set(self.active_dofs))
        if hasattr(self, "passive_dofs"):
            dofs.update(set(self.passive_dofs))

        return dofs

    @classmethod
    def create_with_dofs(
        cls,
        world: World,
        parent: KinematicStructureEntity,
        child: KinematicStructureEntity,
        name: Optional[PrefixedName] = None,
        *args,
        **kwargs,
    ) -> Self:
        """
        This method will automatically generate the degrees of freedom for this connection into the given world
        and initialize the connection with the generated dofs.
        :param world: Reference to the world where the dofs should be added.
        :param parent: parent of the connection.
        :param child: child of the connection.
        :param name: name of the connection.
        :param args: additional arguments the subclass might need
        :param kwargs: additional keyword arguments the subclass might need
        :return:
        """
        raise NotImplementedError(
            "ConnectionWithDofs.create_with_dofs is not implemented."
        )

    def _find_references_in_world(self, world: World) -> Tuple[
        KinematicStructureEntity,
        KinematicStructureEntity,
        HomogeneousTransformationMatrix,
        HomogeneousTransformationMatrix,
    ]:
        """
        Finds the reference frames to this connection in the given world and returns them as usable objects.
        :param world: Reference to the world where the reference frames are searched.
        :return: The other parent and child and new connection expressions with correct reference frames.
        """
        other_parent = world.get_kinematic_structure_entity_by_name(self.parent.name)
        other_child = world.get_kinematic_structure_entity_by_name(self.child.name)

        parent_T_connection = deepcopy(self.parent_T_connection_expression)
        parent_T_connection.reference_frame = (
            world.get_kinematic_structure_entity_by_name(
                parent_T_connection.reference_frame.name
            )
        )

        connection_T_child = deepcopy(self.connection_T_child_expression)
        connection_T_child.child_frame = world.get_kinematic_structure_entity_by_name(
            connection_T_child.child_frame.name
        )
        return (other_parent, other_child, parent_T_connection, connection_T_child)

    def copy_for_world(self, world: World) -> Self:
        """
        Copies this connection to the given world the parent and child references are updated to the new world as well
        as the references from the expression.
        :param world: World in which the connection should be copied.
        :return: The copied connection.
        """
        (
            other_parent,
            other_child,
            parent_T_connection_expression,
            connection_T_child_expression,
        ) = self._find_references_in_world(world)

        return self.__class__(
            other_parent,
            other_child,
            parent_T_connection_expression=parent_T_connection_expression,
            connection_T_child_expression=connection_T_child_expression,
            name=PrefixedName(self.name.name, prefix=self.name.prefix),
        )


GenericConnection = TypeVar("GenericConnection", bound=Connection)


def _is_entity_semantic_annotation_or_iterable(
    obj: object, aggregation_type: Type[KinematicStructureEntity]
) -> bool:
    """
    Determines if an object is a KinematicStructureEntity, a semantic annotation, or an Iterable (excluding strings and bytes).
    """
    return isinstance(obj, (aggregation_type, SemanticAnnotation)) or (
        isinstance(obj, Iterable) and not isinstance(obj, (str, bytes, bytearray))
    )


def _attr_values(
    semantic_annotation: SemanticAnnotation,
    aggregation_type: Type[GenericKinematicStructureEntity],
) -> Iterable[object]:
    """
    Yields all dataclass fields and set properties of this semantic annotation.
    Skips private fields (those starting with '_'), as well as the 'bodies' property.

    :param semantic_annotation: The semantic annotation to extract attributes from.
    """
    for f in fields(semantic_annotation):
        if f.name.startswith("_"):
            continue
        v = getattr(semantic_annotation, f.name, None)
        if _is_entity_semantic_annotation_or_iterable(v, aggregation_type):
            yield v

    for name, prop in inspect.getmembers(
        type(semantic_annotation), lambda o: isinstance(o, property)
    ):
        if name in {
            "kinematic_structure_entities",
            "bodies",
            "regions",
        } or name.startswith("_"):
            continue
        try:
            v = getattr(semantic_annotation, name)
        except Exception:
            continue
        if _is_entity_semantic_annotation_or_iterable(v, aggregation_type):
            yield v


@dataclass(eq=False)
class Actuator(WorldEntityWithID, SubclassJSONSerializer):
    """
    Represents an actuator in the world model.
    """

    _dofs: List[DegreeOfFreedom] = field(default_factory=list, init=False, repr=False)

    def to_json(self) -> Dict[str, Any]:
        result = super().to_json()
        result["name"] = self.name.to_json()
        result["dofs"] = to_json(self._dofs)
        return result

    @classmethod
    def _from_json(cls, data: Dict[str, Any], **kwargs) -> Actuator:
        actuator = cls(
            name=from_json(data["name"]),
            id=from_json(data["id"]),
        )
        dofs_data = data.get("dofs", [])
        assert (
            len(dofs_data) > 0
        ), "An actuator must have at least one degree of freedom."
        for dof_data in dofs_data:
            dof = from_json(dof_data)
            actuator.add_dof(dof)
        return actuator

    @property
    def dofs(self) -> List[DegreeOfFreedom]:
        """
        Returns the degrees of freedom associated with this actuator.
        """
        return self._dofs

    def add_dof(self, dof: DegreeOfFreedom) -> None:
        """
        Adds a degree of freedom to this actuator.

        :param dof: The degree of freedom to add.
        """
        self._dofs.append(dof)
