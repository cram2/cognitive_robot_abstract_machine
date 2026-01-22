from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field

import numpy as np
import trimesh
from probabilistic_model.probabilistic_circuit.rx.helper import (
    uniform_measure_of_event,
)
from random_events.product_algebra import Event
from typing_extensions import (
    TYPE_CHECKING,
    List,
    Optional,
    Self,
    Iterable,
    Type,
)

from krrood.ormatic.utils import classproperty
from ..datastructures.prefixed_name import PrefixedName
from ..datastructures.variables import SpatialVariables
from ..exceptions import (
    MismatchingWorld,
)
from ..spatial_types import Point3, HomogeneousTransformationMatrix, Vector3
from ..world import World
from ..world_description.connections import (
    FixedConnection,
)
from ..world_description.degree_of_freedom import DegreeOfFreedomLimits
from ..world_description.geometry import Scale
from ..world_description.shape_collection import BoundingBoxCollection
from ..world_description.world_entity import (
    SemanticAnnotation,
    Body,
    Region,
    KinematicStructureEntity,
    Connection,
)
from ..world_description.world_modification import synchronized_attribute_modification

if TYPE_CHECKING:
    from .semantic_annotations import (
        Drawer,
        Door,
        Handle,
        Hinge,
        Slider,
        Aperture,
    )


@dataclass(eq=False)
class IsPerceivable:
    """
    A mixin class for semantic annotations that can be perceived.
    """

    class_label: Optional[str] = field(default=None, kw_only=True)
    """
    The exact class label of the perceived object.
    """


@dataclass(eq=False)
class HasRootKinematicStructureEntity(SemanticAnnotation, ABC):
    """
    Base class for shared method for HasRootBody and HasRootRegion.
    """

    root: KinematicStructureEntity = field(kw_only=True)
    """
    The root kinematic structure entity of the semantic annotation.
    """

    @classproperty
    def _parent_connection_type(self) -> Type[Connection]:
        """
        The type of connection used to connect the root kinematic structure entity to the world.
        .. note:: Currently its always, except with sliders and hinges, but in the future this may change. So override if needed.
        """
        return FixedConnection

    @classmethod
    def _create_with_connection_in_world(
        cls,
        name: PrefixedName,
        world: World,
        kinematic_structure_entity: KinematicStructureEntity,
        world_root_T_self: Optional[HomogeneousTransformationMatrix] = None,
        connection_limits: Optional[DegreeOfFreedomLimits] = None,
        active_axis: Optional[Vector3] = None,
        connection_multiplier: float = 1.0,
        connection_offset: float = 0.0,
    ):
        """
        Create a new instance and connect its root entity to the world's root.

        :param name: The name of the semantic annotation.
        :param world: The world to add the annotation and entity to.
        :param kinematic_structure_entity: The root entity of the semantic annotation.
        :param world_root_T_self: The initial pose of the entity in the world root frame.
        :param connection_limits: The limits for the connection's degrees of freedom.
        :param active_axis: The active axis for the connection.
        :param connection_multiplier: The multiplier for the connection.
        :param connection_offset: The offset for the connection.
        :return: The created semantic annotation instance.
        """

        self_instance = cls(name=name, root=kinematic_structure_entity)
        world_root_T_self = world_root_T_self or HomogeneousTransformationMatrix()

        root = world.root
        world_root_T_self.reference_frame = root
        world_root_T_self.child_frame = kinematic_structure_entity

        if cls._parent_connection_type == FixedConnection:
            world_root_C_self = FixedConnection(
                parent=root,
                child=kinematic_structure_entity,
                parent_T_connection_expression=world_root_T_self,
            )
        else:
            world_root_C_self = cls._parent_connection_type.create_with_dofs(
                world=world,
                parent=root,
                child=kinematic_structure_entity,
                parent_T_connection_expression=world_root_T_self,
                multiplier=connection_multiplier,
                offset=connection_offset,
                axis=active_axis,
                dof_limits=connection_limits,
            )

        world.add_connection(world_root_C_self)
        world.add_semantic_annotation(self_instance)

        return self_instance

    def get_new_grandparent(
        self,
        parent_kinematic_structure_entity: KinematicStructureEntity,
    ):
        """
        Determine the new grandparent entity when changing the kinematic structure.

        :param parent_kinematic_structure_entity: The entity that will be the new parent.
        :return: The entity that will be the new grandparent.
        """
        grandparent_kinematic_structure_entity = (
            parent_kinematic_structure_entity.parent_connection.parent
        )
        new_hinge_parent = (
            grandparent_kinematic_structure_entity
            if grandparent_kinematic_structure_entity != self.root
            else self.root.parent_kinematic_structure_entity
        )
        return new_hinge_parent

    def _attach_parent_entity_in_kinematic_structure(
        self,
        new_parent_entity: KinematicStructureEntity,
    ):
        """
        Attach a new parent entity to this entity in the kinematic structure.

        :param new_parent_entity: The new parent entity to attach.
        """
        if new_parent_entity._world != self._world:
            raise MismatchingWorld(self._world, new_parent_entity._world)
        if new_parent_entity == self.root.parent_kinematic_structure_entity:
            return

        world = self._world

        root_T_self = self._offline_root_T_entity(self.root)
        root_T_new_parent = self._offline_root_T_entity(new_parent_entity)

        new_parent_T_self = root_T_new_parent.inverse() @ root_T_self

        parent_C_self = self.root.parent_connection
        world.remove_connection(parent_C_self)

        new_parent_C_self = FixedConnection(
            parent=new_parent_entity,
            child=self.root,
            parent_T_connection_expression=HomogeneousTransformationMatrix(
                new_parent_T_self.evaluate()
            ),
        )
        world.add_connection(new_parent_C_self)

    def _attach_child_entity_in_kinematic_structure(
        self,
        child_kinematic_structure_entity: KinematicStructureEntity,
    ):
        """
        Attach a new child entity to this entity in the kinematic structure.

        :param child_kinematic_structure_entity: The new child entity to attach.
        """
        if child_kinematic_structure_entity._world != self._world:
            raise MismatchingWorld(self._world, child_kinematic_structure_entity._world)

        if self == child_kinematic_structure_entity.parent_kinematic_structure_entity:
            return

        world = self._world
        root_T_self = self._offline_root_T_entity(self.root)
        root_T_new_child = self._offline_root_T_entity(child_kinematic_structure_entity)

        self_T_new_child = root_T_self.inverse() @ root_T_new_child

        parent_C_new_child = child_kinematic_structure_entity.parent_connection
        world.remove_connection(parent_C_new_child)

        self_C_new_child = FixedConnection(
            parent=self.root,
            child=child_kinematic_structure_entity,
            parent_T_connection_expression=HomogeneousTransformationMatrix(
                self_T_new_child.evaluate()
            ),
        )
        world.add_connection(self_C_new_child)

    def _offline_root_T_entity(
        self, entity: KinematicStructureEntity
    ) -> HomogeneousTransformationMatrix:
        """
        Computes root_T_entity without using the world's forward kinematics manager. This is done to avoid having to
        recompile and compute the forwardkinematics in this case.
        My reason of adding this is because otherwise, we would not be able to create for example a door and a handle
        in one modification block, and add the handle to the door in the same block. we would need to close the
        block, open a new one, and add the handle there. This is possible, but i think usage wise this is a lot nicer.

        :param entity: The entity to compute the root_T_entity for.

        :return: The root_T_entity of the entity.
        """
        world = entity._world
        future_root_T_self = entity.parent_connection.origin_expression
        parent_entity = entity.parent_kinematic_structure_entity
        while True:
            if parent_entity == world.root:
                break
            future_root_T_self = (
                parent_entity.parent_connection.origin_expression @ future_root_T_self
            )
            parent_entity = parent_entity.parent_kinematic_structure_entity
        return future_root_T_self


@dataclass(eq=False)
class HasRootBody(HasRootKinematicStructureEntity, ABC):
    """
    Abstract base class for all household objects. Each semantic annotation refers to a single Body.
    Each subclass automatically derives a MatchRule from its own class name and
    the names of its HouseholdObject ancestors. This makes specialized subclasses
    naturally more specific than their bases.
    """

    root: Body = field(kw_only=True)
    """
    The root body of the semantic annotation.
    """

    @property
    def bodies(self) -> Iterable[Body]:
        """
        The bodies that are part of the semantic annotation.
        """
        return [self.root]

    @classmethod
    def create_with_new_body_in_world(
        cls,
        name: PrefixedName,
        world: World,
        world_root_T_self: Optional[HomogeneousTransformationMatrix] = None,
        connection_limits: Optional[DegreeOfFreedomLimits] = None,
        active_axis: Optional[Vector3] = None,
        connection_multiplier: float = 1.0,
        connection_offset: float = 0.0,
        **kwargs,
    ) -> Self:
        """
        Create a new semantic annotation with a new body in the given world.

        :param name: The name of the semantic annotation.
        :param world: The world to add the annotation and body to.
        :param world_root_T_self: The initial pose of the body in the world root frame.
        :param connection_limits: The limits for the connection's degrees of freedom.
        :param active_axis: The active axis for the connection.
        :param connection_multiplier: The multiplier for the connection.
        :param connection_offset: The offset for the connection.
        :return: The created semantic annotation instance.
        """
        body = Body(name=name)

        return cls._create_with_connection_in_world(
            name=name,
            world=world,
            kinematic_structure_entity=body,
            world_root_T_self=world_root_T_self,
            connection_multiplier=connection_multiplier,
            connection_offset=connection_offset,
            active_axis=active_axis,
            connection_limits=connection_limits,
        )


@dataclass(eq=False)
class HasRootRegion(HasRootKinematicStructureEntity, ABC):
    """
    A mixin class for semantic annotations that have a region.
    """

    root: Region = field(kw_only=True)
    """
    The root region of the semantic annotation.
    """

    @property
    def regions(self) -> Iterable[Region]:
        """
        The regions that are part of the semantic annotation.
        """
        return [self.root]

    @classmethod
    def create_with_new_region_in_world(
        cls,
        name: PrefixedName,
        world: World,
        world_root_T_self: Optional[HomogeneousTransformationMatrix] = None,
        connection_limits: Optional[DegreeOfFreedomLimits] = None,
        active_axis: Optional[Vector3] = None,
        connection_multiplier: float = 1.0,
        connection_offset: float = 0.0,
        **kwargs,
    ) -> Self:
        """
        Create a new semantic annotation with a new region in the given world.

        :param name: The name of the semantic annotation.
        :param world: The world to add the annotation and region to.
        :param world_root_T_self: The initial pose of the region in the world root frame.
        :param connection_limits: The limits for the connection's degrees of freedom.
        :param active_axis: The active axis for the connection.
        :param connection_multiplier: The multiplier for the connection.
        :param connection_offset: The offset for the connection.
        :return: The created semantic annotation instance.
        """
        region = Region(name=name)

        return cls._create_with_connection_in_world(
            name=name,
            world=world,
            kinematic_structure_entity=region,
            world_root_T_self=world_root_T_self,
            connection_multiplier=connection_multiplier,
            connection_offset=connection_offset,
            active_axis=active_axis,
            connection_limits=connection_limits,
        )


@dataclass(eq=False)
class HasApertures(HasRootBody, ABC):
    """
    A mixin class for semantic annotations that have apertures.
    """

    apertures: List[Aperture] = field(default_factory=list, hash=False, kw_only=True)
    """
    The apertures of the semantic annotation.
    """

    @synchronized_attribute_modification
    def add_aperture(self, aperture: Aperture):
        """
        Cuts a hole in the semantic annotation's body for the given body annotation.

        :param aperture: The aperture to cut a hole for.
        """
        self._remove_aperture_geometry_from_parent(aperture)
        self._attach_child_entity_in_kinematic_structure(aperture.root)
        self.apertures.append(aperture)

    def _remove_aperture_geometry_from_parent(self, aperture: Aperture):
        """
        Remove the geometry of the aperture from the parent body's collision and visual geometry.

        :param aperture: The aperture whose geometry should be removed.
        """
        world = self._world
        world.update_forward_kinematics()
        hole_event = aperture.root.area.as_bounding_box_collection_in_frame(
            self.root
        ).event
        wall_event = self.root.collision.as_bounding_box_collection_in_frame(
            self.root
        ).event
        new_wall_event = wall_event - hole_event
        new_bounding_box_collection = BoundingBoxCollection.from_event(
            self.root, new_wall_event
        ).as_shapes()
        self.root.collision = new_bounding_box_collection
        self.root.visual = new_bounding_box_collection


@dataclass(eq=False)
class HasHinge(HasRootBody, ABC):
    """
    A mixin class for semantic annotations that have hinge joints.
    """

    hinge: Optional[Hinge] = field(default=None)
    """
    The hinge of the semantic annotation.
    """

    @synchronized_attribute_modification
    def add_hinge(
        self,
        hinge: Hinge,
    ):
        """
        Add a hinge to the semantic annotation.

        :param hinge: The hinge to add.
        """
        self._attach_parent_entity_in_kinematic_structure(
            hinge.root,
        )
        self.hinge = hinge


@dataclass(eq=False)
class HasSlider(HasRootKinematicStructureEntity, ABC):
    """
    A mixin class for semantic annotations that have slider joints.
    """

    slider: Optional[Slider] = field(default=None)
    """
    The slider of the semantic annotation.
    """

    @synchronized_attribute_modification
    def add_slider(
        self,
        slider: Slider,
    ):
        """
        Add a slider to the semantic annotation.

        :param slider: The slider to add.
        """
        self._attach_parent_entity_in_kinematic_structure(
            slider.root,
        )
        self.slider = slider


@dataclass(eq=False)
class HasDrawers(HasRootKinematicStructureEntity, ABC):
    """
    A mixin class for semantic annotations that have drawers.
    """

    drawers: List[Drawer] = field(default_factory=list, hash=False, kw_only=True)
    """
    The drawers of the semantic annotation.
    """

    @synchronized_attribute_modification
    def add_drawer(
        self,
        drawer: Drawer,
    ):
        """
        Add a drawer to the semantic annotation.

        :param drawer: The drawer to add.
        """

        self._attach_child_entity_in_kinematic_structure(drawer.root)
        self.drawers.append(drawer)


@dataclass(eq=False)
class HasDoors(HasRootKinematicStructureEntity, ABC):
    """
    A mixin class for semantic annotations that have doors.
    """

    doors: List[Door] = field(default_factory=list, hash=False, kw_only=True)
    """
    The doors of the semantic annotation.
    """

    @synchronized_attribute_modification
    def add_door(
        self,
        door: Door,
    ):
        """
        Add a door to the semantic annotation.

        :param door: The door to add.
        """

        self._attach_child_entity_in_kinematic_structure(door.root)
        self.doors.append(door)


@dataclass(eq=False)
class HasHandle(HasRootBody, ABC):
    """
    A mixin class for semantic annotations that have a handle.
    """

    handle: Optional[Handle] = None
    """
    The handle of the semantic annotation.
    """

    @synchronized_attribute_modification
    def add_handle(
        self,
        handle: Handle,
    ):
        """
        Adds a handle to the parent world with a fixed connection.

        :param handle: The handle to add.
        """
        self._attach_child_entity_in_kinematic_structure(
            handle.root,
        )
        self.handle = handle


@dataclass(eq=False)
class HasStorageSpace(HasRootBody, ABC):
    """
    A mixin class for semantic annotations that represent storage spaces. Used to afterthefact add object for example
    to a table, and have those objects move with the table when it is moved.
    """

    objects: List[HasRootBody] = field(default_factory=list, hash=False, kw_only=True)
    """
    The objects stored in the semantic annotation.
    """

    @synchronized_attribute_modification
    def add_object(self, object: HasRootBody):
        self._attach_child_entity_in_kinematic_structure(object.root)
        self.objects.append(object)


@dataclass(eq=False)
class HasSupportingSurface(HasStorageSpace, ABC):
    """
    A semantic annotation that represents a supporting surface.
    """

    supporting_surface: Region = field(default=None)
    """
    The supporting surface region of the semantic annotation.
    """

    @synchronized_attribute_modification
    def calculate_supporting_surface(
        self,
        upward_threshold: float = 0.95,
        clearance_threshold: float = 0.5,
        min_surface_area: float = 0.0225,  # 15cm x 15cm
    ):
        """
        Calculate and set the supporting surface region for the semantic annotation.

        :param upward_threshold: The threshold for the face normal to be considered upward-facing.
        :param clearance_threshold: The threshold for the vertical clearance above the surface.
        :param min_surface_area: The minimum area for a surface to be considered a supporting surface.
        """
        mesh = self.root.combined_mesh
        if mesh is None:
            return
        # --- Find upward-facing faces ---
        normals = mesh.face_normals
        upward_mask = normals[:, 2] > upward_threshold

        if not upward_mask.any():
            return

        # --- Find connected upward-facing regions ---
        upward_face_indices = np.nonzero(upward_mask)[0]
        submesh_up = mesh.submesh([upward_face_indices], append=True)
        face_groups = submesh_up.split(only_watertight=False)

        # Compute total area for each group
        large_groups = [g for g in face_groups if g.area >= min_surface_area]

        if not large_groups:
            return

        # --- Merge qualifying upward-facing submeshes ---
        candidates = trimesh.util.concatenate(large_groups)

        # --- Check vertical clearance using ray casting ---
        face_centers = candidates.triangles_center
        ray_origins = face_centers + np.array([0, 0, 0.01])  # small upward offset
        ray_dirs = np.tile([0, 0, 1], (len(ray_origins), 1))

        locations, index_ray, _ = mesh.ray.intersects_location(
            ray_origins=ray_origins, ray_directions=ray_dirs
        )

        # Compute distances to intersections (if any)
        distances = np.full(len(ray_origins), np.inf)
        distances[index_ray] = np.linalg.norm(
            locations - ray_origins[index_ray], axis=1
        )

        # Filter faces with enough space above
        clear_mask = (distances > clearance_threshold) | np.isinf(distances)

        if not clear_mask.any():
            raise ValueError(
                "No upward-facing surfaces with sufficient clearance found."
            )

        candidates_filtered = candidates.submesh([clear_mask], append=True)

        # --- Build the region ---
        points_3d = [
            Point3(
                x,
                y,
                z,
                reference_frame=self.root,
            )
            for x, y, z in candidates_filtered.vertices
        ]
        supporting_surface = Region.from_3d_points(
            name=PrefixedName(
                f"{self.root.name.name}_supporting_surface_region",
                self.root.name.prefix,
            ),
            points_3d=points_3d,
        )
        self_C_supporting_surface = FixedConnection(
            parent=self.root, child=supporting_surface
        )
        with self._world.modify_world():
            self._world.add_region(supporting_surface)
            self._world.add_connection(self_C_supporting_surface)
        self.supporting_surface = supporting_surface
        return supporting_surface

    def points_on_supporting_surface(self, amount: int = 100) -> List[Point3]:
        """
        Get points that are on the surface.

        :param amount: The number of points to return.
        :return: A list of points that are on the surface.
        """
        area_of_table = BoundingBoxCollection.from_shapes(self.root.collision)
        event = area_of_table.event
        p = uniform_measure_of_event(event)
        p = p.marginal(SpatialVariables.xy)
        samples = p.sample(amount)
        z_coordinate = np.full(
            (amount, 1), max([b.max_z for b in area_of_table]) + 0.01
        )
        samples = np.concatenate((samples, z_coordinate), axis=1)
        return [Point3(*s, reference_frame=self.root) for s in samples]


@dataclass(eq=False)
class HasCaseAsRootBody(HasSupportingSurface, ABC):
    """
    A mixin class for semantic annotations that have a case as root body.
    """

    @classproperty
    @abstractmethod
    def hole_direction(self) -> Vector3:
        """
        The direction of the physical hole of the geometry. For a drawer for example, this would always be Z.

        ..warning:: This does not describe the axis along, for example, a drawer opens. Its the physical opening where
        you can put something into the drawer.
        """
        ...

    @classmethod
    def create_with_new_body_in_world(
        cls,
        name: PrefixedName,
        world: World,
        world_root_T_self: Optional[HomogeneousTransformationMatrix] = None,
        connection_limits: Optional[DegreeOfFreedomLimits] = None,
        active_axis: Optional[Vector3] = None,
        connection_multiplier: float = 1.0,
        connection_offset: float = 0.0,
        *,
        scale: Scale = Scale(),
        wall_thickness: float = 0.01,
    ) -> Self:
        """
        Create a new semantic annotation with a new body in the given world.

        :param name: The name of the semantic annotation.
        :param world: The world to add the annotation and body to.
        :param world_root_T_self: The initial pose of the body in the world root frame.
        :param connection_limits: The limits for the connection's degrees of freedom.
        :param active_axis: The active axis for the connection.
        :param connection_multiplier: The multiplier for the connection.
        :param connection_offset: The offset for the connection.
        :param scale: The scale of the case.
        :param wall_thickness: The thickness of the case walls.
        :return: The created semantic annotation instance.
        """
        container_event = cls._create_container_event(scale, wall_thickness)

        body = Body(name=name)
        collision_shapes = BoundingBoxCollection.from_event(
            body, container_event
        ).as_shapes()
        body.collision = collision_shapes
        body.visual = collision_shapes
        return cls._create_with_connection_in_world(
            name=name,
            world=world,
            kinematic_structure_entity=body,
            world_root_T_self=world_root_T_self,
            connection_multiplier=connection_multiplier,
            connection_offset=connection_offset,
            active_axis=active_axis,
            connection_limits=connection_limits,
        )

    @classmethod
    def _create_container_event(cls, scale: Scale, wall_thickness: float) -> Event:
        """
        Return an event representing a container with walls of a specified thickness.

        :param scale: The scale of the container.
        :param wall_thickness: The thickness of the walls.
        :return: The event representing the container.
        """
        outer_box = scale.to_simple_event()
        inner_box = Scale(
            scale.x - wall_thickness,
            scale.y - wall_thickness,
            scale.z - wall_thickness,
        ).to_simple_event(cls.hole_direction, wall_thickness)

        container_event = outer_box.as_composite_set() - inner_box.as_composite_set()

        return container_event
