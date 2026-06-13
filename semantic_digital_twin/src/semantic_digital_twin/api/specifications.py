from __future__ import annotations
from abc import ABC, abstractmethod
from copy import deepcopy
from dataclasses import dataclass, field
from typing import Union, Optional, assert_never

from typing_extensions import Self, Type, Any

from random_events.product_algebra import Event
from semantic_digital_twin.datastructures.prefixed_name import PrefixedName
from semantic_digital_twin.robots.robot_parts import AbstractRobot
from semantic_digital_twin.semantic_annotations.mixins import (
    HasRootKinematicStructureEntity,
)
from semantic_digital_twin.adapters.urdf import URDFParser
from semantic_digital_twin.spatial_types import HomogeneousTransformationMatrix, Vector3
from semantic_digital_twin.world import World
from semantic_digital_twin.world_description.connections import (
    WheeledDrive,
    FixedConnection,
    ActiveConnection,
    ActiveConnection1DOF,
    Connection6DoF,
)
from semantic_digital_twin.world_description.geometry import (
    Scale,
    Color,
    Box,
    Mesh,
    Sphere,
    Cylinder,
)
from semantic_digital_twin.world_description.inertial_properties import Inertial
from semantic_digital_twin.world_description.shape_collection import (
    ShapeCollection,
    BoundingBoxCollection,
)
from semantic_digital_twin.world_description.world_entity import (
    Body,
    Region,
    KinematicStructureEntity,
    Connection,
)


def _build_connection(
    world: World,
    parent: KinematicStructureEntity,
    child: KinematicStructureEntity,
    connection_type: Type[Connection],
    parent_T_child: Optional[HomogeneousTransformationMatrix],
    axis: Optional[Vector3] = None,
    multiplier: float = 1.0,
    offset: float = 0.0,
) -> Connection:
    """
    Build a connection of ``connection_type`` between ``parent`` and ``child``.

    The pose offset is carried in the connection expression (so no two-phase
    ``.origin`` assignment is needed), mirroring
    :meth:`HasRootKinematicStructureEntity._create_with_connection_in_world`.

    :param parent_T_child: Pose of the child in the parent frame. Defaults to identity.
    :param axis: Movement axis, required for active (1-DoF) connections, ignored otherwise.
    :param multiplier: DoF multiplier for active connections.
    :param offset: DoF offset for active connections.
    """
    pose = parent_T_child or HomogeneousTransformationMatrix()
    pose.reference_frame = parent
    pose.child_frame = child

    if connection_type is FixedConnection:
        return FixedConnection(
            parent=parent, child=child, parent_T_connection_expression=pose
        )
    if issubclass(connection_type, ActiveConnection1DOF):
        if axis is None:
            raise ValueError(
                f"Active connection {connection_type.__name__} requires axis."
            )

        return connection_type.create_with_dofs(
            world=world,
            parent=parent,
            child=child,
            axis=axis,
            multiplier=multiplier,
            offset=offset,
            parent_T_connection_expression=pose,
        )
    # Connection6DoF and other passive multi-DoF connections.
    return connection_type.create_with_dofs(
        world=world,
        parent=parent,
        child=child,
        parent_T_connection_expression=pose,
    )


@dataclass
class WorldEntitySpawnSpecification(ABC):

    @abstractmethod
    def spawn(
        self,
        world: World,
        parent: KinematicStructureEntity | None = None,
        parent_T_self: HomogeneousTransformationMatrix | None = None,
        name: Union[str, PrefixedName, None] = None,
    ):
        """
        Instantiate the World Entity and add it to the given world.

        :param parent: The entity to attach to. If None, ``world.root`` is used.
        :param parent_T_self: Overrides the specification's stored default pose. If None, the stored default is used.
        :param name: Overrides the specification's own name. If None, the spec's name is used.
        """


@dataclass
class KinematicStructureEntitySpecification(WorldEntitySpawnSpecification):
    """
    Declarative, world-independent description of a kinematic structure entity.
    A specification is reusable: every materialization copies the prototype shapes and the
    pose, so the specification never becomes bound to an entity or world.
    """

    name: Union[str, PrefixedName]
    """
    The name of entities created from this specification. Can be overridden per spawn.
    """

    shapes: ShapeCollection = field(default_factory=ShapeCollection)
    """
    Prototype shapes with origins expressed in the entity frame.
    """

    child_specification: list[WorldEntitySpawnSpecification] = field(
        default_factory=list
    )
    """
    The child specifications of this specification. If set, the spawned entity will be a parent of the children.
    """

    def __post_init__(self):
        if isinstance(self.name, str):
            self.name = PrefixedName(self.name)

    @abstractmethod
    def spawn(
        self,
        world: World,
        parent: KinematicStructureEntity | None = None,
        parent_T_self: HomogeneousTransformationMatrix | None = None,
        name: Union[str, PrefixedName, None] = None,
    ):
        pass

    def _spawn_children(
        self, world: World, parent_entity: KinematicStructureEntity
    ) -> None:
        """Spawn every child specification as a child of ``parent_entity``."""
        for child in self.child_specification:
            child.spawn(world, parent=parent_entity)

    @classmethod
    def box(
        cls,
        name: Union[str, PrefixedName],
        scale: Scale,
        color: Optional[Color] = None,
        origin: Optional[HomogeneousTransformationMatrix] = None,
    ) -> Self:
        """
        Specification for a kinematic structure entity with a single box shape.
        :param name: The name of the body.
        :param scale: The extents of the box.
        :param color: The color of the box.
        :param origin: The origin of the box in the body frame. Defaults to identity.
        :return: The created specification.
        """
        return cls(
            name=name,
            shapes=Box(
                scale=scale,
                origin=origin or HomogeneousTransformationMatrix(),
                color=color or Color(),
            ).as_shape_collection(),
        )

    @classmethod
    def sphere(
        cls,
        name: Union[str, PrefixedName],
        radius: float,
        color: Optional[Color] = None,
        origin: Optional[HomogeneousTransformationMatrix] = None,
    ) -> Self:
        """
        Specification for a kinematic structure entity with a single sphere shape.
        :param name: The name of the kinematic structure entity.
        :param radius: The radius of the sphere.
        :param color: The color of the sphere.
        :param origin: The origin of the sphere in the kinematic structure entity frame. Defaults to identity.
        :return: The created specification.
        """
        return cls(
            name=name,
            shapes=Sphere(
                radius=radius,
                origin=origin or HomogeneousTransformationMatrix(),
                color=color or Color(),
            ).as_shape_collection(),
        )

    @classmethod
    def cylinder(
        cls,
        name: Union[str, PrefixedName],
        width: float,
        height: float,
        color: Optional[Color] = None,
        origin: Optional[HomogeneousTransformationMatrix] = None,
    ) -> Self:
        """
        Specification for a kinematic structure entity with a single cylinder shape.
        :param name: The name of the kinematic structure entity.
        :param width: The diameter of the cylinder.
        :param height: The height of the cylinder.
        :param color: The color of the cylinder.
        :param origin: The origin of the cylinder in the kinematic structure entity frame. Defaults to identity.
        :return: The created specification.
        """
        return cls(
            name=name,
            shapes=Cylinder(
                width=width,
                height=height,
                origin=origin or HomogeneousTransformationMatrix(),
                color=color or Color(),
            ).as_shape_collection(),
        )

    @classmethod
    def mesh(
        cls,
        name: Union[str, PrefixedName],
        filename: str,
        scale: Optional[Scale] = None,
        color: Optional[Color] = None,
        origin: Optional[HomogeneousTransformationMatrix] = None,
    ) -> Self:
        """
        Specification for a kinematic structure entity with a single mesh shape loaded from a file.
        :param name: The name of the kinematic structure entity.
        :param filename: The path of the mesh file.
        :param scale: The scale applied to the mesh.
        :param color: The color of the mesh.
        :param origin: The origin of the mesh in the kinematic structure entity frame. Defaults to identity.
        :return: The created specification.
        """
        return cls(
            name=name,
            shapes=Mesh(
                filename=filename,
                origin=origin or HomogeneousTransformationMatrix(),
                scale=scale or Scale(),
                color=color or Color(),
            ).as_shape_collection(),
        )

    @classmethod
    def from_event(cls, name: Union[str, PrefixedName], event: Event) -> Self:
        """
        Specification whose shapes are the bounding boxes of a random event.
        This is the construction used by semantic annotations with composite
        geometry (hollow handles, container cases, walls minus apertures, ...).
        :param name: The name of the entity.
        :param event: The event describing the geometry, in the entity frame.
        :return: The created specification.
        """
        # BoundingBoxCollection requires a reference frame, so the shapes are
        # built around a throwaway body and unbound again for the specification.
        anchor = Body(name=PrefixedName("spec_anchor"))
        return cls(
            name=name,
            shapes=BoundingBoxCollection.from_event(anchor, event)
            .as_shapes()
            .copy_without_reference_frame(),
        )


@dataclass
class BodySpecification(KinematicStructureEntitySpecification):

    inertial: Optional[Inertial] = None
    """
    Inertia properties of created bodies. None means the Body default.
    """

    visual_shapes: Optional[ShapeCollection] = None
    """
    Visual shapes when they differ from `shapes`. None shares `shapes` for both
    collision and visual (one collection); an empty list means no visual geometry.
    """

    def to_body(self, name: Optional[Union[str, PrefixedName]] = None) -> Body:
        """
        Create a new, world-independent body from this specification.
        :param name: Optional name override, e.g. for spawning multiple bodies
                     from the same specification.
        :return: The created body.
        """
        if isinstance(name, str):
            name = PrefixedName(name)

        shape_copy = self.shapes.copy_without_reference_frame()
        visual_shapes = (
            self.visual_shapes.copy_without_reference_frame()
            if self.visual_shapes
            else shape_copy
        )

        body = Body(
            name=name or self.name,
            collision=shape_copy,
            visual=visual_shapes,
        )
        if self.inertial is not None:
            body.inertial = deepcopy(self.inertial)
        return body

    def spawn(
        self,
        world: World,
        parent: KinematicStructureEntity | None = None,
        parent_T_self: HomogeneousTransformationMatrix | None = None,
        name: Union[str, PrefixedName, None] = None,
    ) -> Body:
        parent = parent or world.root
        body = self.to_body(name)
        with world.modify_world():
            connection = _build_connection(
                world, parent, body, FixedConnection, parent_T_self, None
            )
            world.add_connection(connection)
            self._spawn_children(world, body)
        return body


@dataclass
class RegionSpecification(KinematicStructureEntitySpecification):

    def to_region(self, name: Optional[Union[str, PrefixedName]] = None) -> Region:
        """
        Create a new, world-independent region from this specification.
        :param name: Optional name override.
        :return: The created region.
        """
        if isinstance(name, str):
            name = PrefixedName(name)
        return Region(
            name=name or self.name,
            area=self.shapes.copy_without_reference_frame(),
        )

    def spawn(
        self,
        world: World,
        parent: KinematicStructureEntity | None = None,
        parent_T_self: HomogeneousTransformationMatrix | None = None,
        name: Union[str, PrefixedName, None] = None,
    ) -> Region:
        parent = parent or world.root
        region = self.to_region(name)
        with world.modify_world():
            connection = _build_connection(
                world, parent, region, FixedConnection, parent_T_self, None
            )
            world.add_connection(connection)
            self._spawn_children(world, region)
        return region


@dataclass
class SemanticAnnotationWithRootSpecification(WorldEntitySpawnSpecification):
    """
    Declarative description of a semantic annotation rooted in a single kinematic
    structure entity. The annotation type owns the parent connection type (via its
    ``_parent_connection_type``); this specification only supplies the connection
    parameters for active connections.
    """

    name: Union[str, PrefixedName]
    """
    The name of the annotation.
    """

    semantic_annotation_type: Type[HasRootKinematicStructureEntity]
    """
    The type of the semantic annotation that is a subclass of HasRootKinematicStructureEntity.
    """

    root_specification: Optional[KinematicStructureEntitySpecification] = None
    """
    The specification of the root kinematic structure entity of the annotation.
    """

    axis: Optional[Vector3] = None
    """
    Movement axis for the parent connection. Required when the annotation's
    ``_parent_connection_type`` is an active connection (``ActiveConnection``); ignored otherwise.
    """

    multiplier: float = 1.0
    """
    DoF multiplier for the parent connection (active connections only).
    """

    offset: float = 0.0
    """
    DoF offset for the parent connection (active connections only).
    """

    annotation_kwargs: dict[
        str,
        Union[
            SemanticAnnotationWithRootSpecification,
            Any,
        ],
    ] = field(default_factory=dict)
    """
    The keyword arguments to pass to the annotation constructor.
    Spec-valued entries (nested annotations) are not yet supported and raise NotImplementedError.
    """

    def __post_init__(self):
        if isinstance(self.name, str):
            self.name = PrefixedName(self.name)
        if self.axis is None and issubclass(
            self.semantic_annotation_type._parent_connection_type, ActiveConnection
        ):
            raise ValueError(
                f"{self.semantic_annotation_type.__name__} attaches via "
                f"{self.semantic_annotation_type._parent_connection_type.__name__}, "
                f"which is an active connection, so axis is required."
            )

    def spawn(
        self,
        world: World,
        parent: KinematicStructureEntity | None = None,
        parent_T_self: HomogeneousTransformationMatrix | None = None,
        name: Union[str, PrefixedName, None] = None,
    ) -> HasRootKinematicStructureEntity:
        parent = parent or world.root
        name = name or self.name

        if self.root_specification is None:
            root_entity = Body(name=name)
        elif isinstance(self.root_specification, RegionSpecification):
            root_entity = self.root_specification.to_region(name)
        elif isinstance(self.root_specification, BodySpecification):
            root_entity = self.root_specification.to_body(name)
        else:
            assert_never(self.root_specification)

        for value in self.annotation_kwargs.values():
            if isinstance(value, WorldEntitySpawnSpecification):
                raise NotImplementedError(
                    "Spec-valued annotation_kwargs (nested annotations) are not yet "
                    "supported. Pass already-constructed values instead."
                )

        instance = self.semantic_annotation_type(
            name=name, root=root_entity, **self.annotation_kwargs
        )

        with world.modify_world():
            connection = _build_connection(
                world,
                parent,
                root_entity,
                self.semantic_annotation_type._parent_connection_type,
                parent_T_self,
                self.axis,
                self.multiplier,
                self.offset,
            )
            world.add_connection(connection)
            world.add_semantic_annotation(instance)
            if self.root_specification is not None:
                self.root_specification._spawn_children(world, root_entity)
        return instance


@dataclass
class BodyAndConnectionSpecification(WorldEntitySpawnSpecification):

    body_specification: BodySpecification

    connection_type: Type[Connection] = field(default=FixedConnection)

    parent_T_self: HomogeneousTransformationMatrix = field(
        default_factory=HomogeneousTransformationMatrix
    )

    axis: Optional[Vector3] = None
    """
    Movement axis for the parent connection. Required when ``connection_type`` is an active
    connection (``ActiveConnection``); ignored otherwise.
    """

    multiplier: float = 1.0
    """
    DoF multiplier for the parent connection (active connections only).
    """

    offset: float = 0.0
    """
    DoF offset for the parent connection (active connections only).
    """

    def __post_init__(self):
        if self.axis is None and issubclass(self.connection_type, ActiveConnection):
            raise ValueError(
                f"connection_type {self.connection_type.__name__} is an active connection, "
                f"so axis is required."
            )

    def spawn(
        self,
        world: World,
        parent: KinematicStructureEntity | None = None,
        parent_T_self: HomogeneousTransformationMatrix | None = None,
        name: Union[str, PrefixedName, None] = None,
    ) -> Body:
        parent = parent or world.root
        body = self.body_specification.to_body(name)
        pose = parent_T_self or self.parent_T_self
        with world.modify_world():
            connection = _build_connection(
                world,
                parent,
                body,
                self.connection_type,
                pose,
                self.axis,
                self.multiplier,
                self.offset,
            )
            world.add_connection(connection)
            self.body_specification._spawn_children(world, body)
        return body


@dataclass
class WorldSpecification:
    robot_semantic_annotation: Optional[Type[AbstractRobot]] = None
    drive_connection_type: Type[WheeledDrive] | None = None
    world_T_odom: HomogeneousTransformationMatrix | None = None
    odom_T_robot_start: HomogeneousTransformationMatrix | None = None
    starting_objects: list[WorldEntitySpawnSpecification] = field(default_factory=list)

    def to_world(self) -> World:
        """
        Materialize this specification into a new World.

        Without a robot, an empty world with a single root body is created. With a robot,
        the robot URDF is parsed and connected as ``map -> odom_combined -> drive -> robot``,
        with the localization and start poses applied. Finally all ``starting_objects`` are
        spawned relative to the world root.
        """
        if self.robot_semantic_annotation is None:
            world = World()
            with world.modify_world():
                world.add_body(Body(name=PrefixedName("root", "world")))
        else:
            world = URDFParser.from_file(
                self.robot_semantic_annotation.get_ros_file_path()
            ).parse()
            self.robot_semantic_annotation.from_world(world)

            with world.modify_world():
                robot_root = world.root
                map_body = Body(name=PrefixedName("map"))
                odom_body = Body(name=PrefixedName("odom_combined"))

                map_C_odom = Connection6DoF.create_with_dofs(
                    world=world, parent=map_body, child=odom_body
                )
                world.add_connection(map_C_odom)

                drive_connection_type = self.drive_connection_type or Connection6DoF
                odom_C_robot = drive_connection_type.create_with_dofs(
                    world=world, parent=odom_body, child=robot_root
                )
                world.add_connection(odom_C_robot)
                if issubclass(drive_connection_type, ActiveConnection):
                    odom_C_robot.has_hardware_interface = True

            # Poses touch DoF state, so they are set after the modification block.
            if self.world_T_odom is not None:
                map_C_odom.origin = self.world_T_odom
            if self.odom_T_robot_start is not None:
                odom_C_robot.origin = self.odom_T_robot_start

        for starting_object in self.starting_objects:
            starting_object.spawn(world)

        return world
