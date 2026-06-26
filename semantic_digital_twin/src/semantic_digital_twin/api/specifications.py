from __future__ import annotations

import difflib
from abc import ABC, abstractmethod
from copy import deepcopy
from dataclasses import dataclass, field, fields
from typing import ClassVar, Iterable, Union, Optional, TYPE_CHECKING

from typing_extensions import Self, Type, Any, Generic, TypeVar

from krrood.class_diagrams.attribute_introspector import DataclassOnlyIntrospector
from krrood.patterns.subclass_safe_generic import AbstractSubClassSafeGeneric
from krrood.utils import get_generic_type_params
from random_events.product_algebra import Event
from semantic_digital_twin.datastructures.prefixed_name import (
    PrefixedName,
    ensure_prefixed_name,
)
from semantic_digital_twin.adapters.urdf import URDFParser
from semantic_digital_twin.exceptions import MissingConnectionChildError
from semantic_digital_twin.spatial_types import HomogeneousTransformationMatrix, Vector3
from semantic_digital_twin.world import World
from semantic_digital_twin.world_description.connections import (
    WheeledDrive,
    FixedConnection,
    ActiveConnection,
    Connection6DoF,
    PrismaticConnection,
    RevoluteConnection,
)
from semantic_digital_twin.world_description.geometry import (
    Scale,
    Color,
    Box,
    Mesh,
    Sphere,
    Cylinder,
)
from semantic_digital_twin.world_description.degree_of_freedom import (
    DegreeOfFreedomLimits,
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

if TYPE_CHECKING:
    from semantic_digital_twin.semantic_annotations.mixins import (
        HasRootKinematicStructureEntity,
        PartWholeRelationship,
    )
    from semantic_digital_twin.robots.robot_parts import AbstractRobot


DomainObjectType = TypeVar("DomainObjectType", bound=KinematicStructureEntity)


@dataclass
class WorldEntitySpawnSpecification(ABC):

    name: Union[str, PrefixedName]
    """
    The name of entities created from this specification. Can be overridden per spawn.
    """

    def __post_init__(self):
        self.name = ensure_prefixed_name(self.name)

    @abstractmethod
    def spawn(
        self,
        world: World,
        name: Union[str, PrefixedName, None] = None,
        parent: KinematicStructureEntity | None = None,
        parent_T_self: HomogeneousTransformationMatrix | None = None,
    ):
        """
        Instantiate the World Entity and add it to the given world.

        :param parent: The entity to attach to. If None, ``world.root`` is used.
        :param parent_T_self: Overrides the specification's stored default pose. If None, the stored default is used.
        :param name: Overrides the specification's own name. If None, the spec's name is used.
        """

    def _spawn_children(
        self,
        world: World,
        parent: KinematicStructureEntity,
        children: Iterable[WorldEntitySpawnSpecification],
    ) -> None:
        """Spawn each child specification as a kinematic child of ``parent``."""
        for child in children:
            child.spawn(world, parent=parent)


@dataclass
class KinematicStructureEntitySpecification(
    Generic[DomainObjectType],
    AbstractSubClassSafeGeneric,
    WorldEntitySpawnSpecification,
):
    """
    Declarative, world-independent description of a kinematic structure entity.
    A specification is reusable: every materialization copies the prototype shapes and the
    pose, so the specification never becomes bound to an entity or world.

    The concrete domain-object type (e.g. ``Body``/``Region``) is bound as the generic
    parameter by each subclass and resolved at runtime in :meth:`to_domain_object`.
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

    parent_T_self: HomogeneousTransformationMatrix = field(
        default_factory=HomogeneousTransformationMatrix
    )
    """
    Default placement of the entity in its parent frame, used by :meth:`spawn` when the caller does not
    override it. Identity by default.
    """

    def to_domain_object(
        self, name: Union[str, PrefixedName, None] = None
    ) -> DomainObjectType:
        """Materialize a new, world-independent kinematic structure entity from this spec."""
        [domain_object_type] = get_generic_type_params(
            self, KinematicStructureEntitySpecification
        )
        return domain_object_type.from_shape_collection(
            ensure_prefixed_name(name) or self.name,
            self.shapes.copy_without_reference_frame(),
        )

    def spawn(
        self,
        world: World,
        name: Union[str, PrefixedName, None] = None,
        parent: KinematicStructureEntity | None = None,
        parent_T_self: HomogeneousTransformationMatrix | None = None,
    ) -> DomainObjectType:
        entity = self.to_domain_object(name)
        with world.modify_world():
            FixedConnectionSpecification().spawn(
                world,
                parent=parent,
                parent_T_self=parent_T_self or self.parent_T_self,
                child=entity,
            )
            self._spawn_children(world, entity, self.child_specification)
        return entity

    @classmethod
    def box(
        cls,
        name: Union[str, PrefixedName],
        scale: Scale,
        color: Optional[Color] = None,
        origin: Optional[HomogeneousTransformationMatrix] = None,
        child_specification: list[WorldEntitySpawnSpecification] | None = None,
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
            name,
            Box(
                scale=scale,
                origin=origin or HomogeneousTransformationMatrix(),
                color=color or Color(),
            ).as_shape_collection(),
            child_specification=child_specification or [],
        )

    @classmethod
    def sphere(
        cls,
        name: Union[str, PrefixedName],
        radius: float,
        color: Optional[Color] = None,
        origin: Optional[HomogeneousTransformationMatrix] = None,
        child_specification: list[WorldEntitySpawnSpecification] | None = None,
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
            name,
            Sphere(
                radius=radius,
                origin=origin or HomogeneousTransformationMatrix(),
                color=color or Color(),
            ).as_shape_collection(),
            child_specification=child_specification or [],
        )

    @classmethod
    def cylinder(
        cls,
        name: Union[str, PrefixedName],
        width: float,
        height: float,
        color: Optional[Color] = None,
        origin: Optional[HomogeneousTransformationMatrix] = None,
        child_specification: list[WorldEntitySpawnSpecification] | None = None,
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
            name,
            Cylinder(
                width=width,
                height=height,
                origin=origin or HomogeneousTransformationMatrix(),
                color=color or Color(),
            ).as_shape_collection(),
            child_specification=child_specification or [],
        )

    @classmethod
    def mesh(
        cls,
        name: Union[str, PrefixedName],
        filename: str,
        scale: Optional[Scale] = None,
        color: Optional[Color] = None,
        origin: Optional[HomogeneousTransformationMatrix] = None,
        child_specification: list[WorldEntitySpawnSpecification] | None = None,
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
            name,
            Mesh(
                filename=filename,
                origin=origin or HomogeneousTransformationMatrix(),
                scale=scale or Scale(),
                color=color or Color(),
            ).as_shape_collection(),
            child_specification=child_specification or [],
        )

    @classmethod
    def from_event(
        cls,
        name: Union[str, PrefixedName],
        event: Event,
        child_specification: list[WorldEntitySpawnSpecification] | None = None,
    ) -> Self:
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
            child_specification=child_specification or [],
        )


@dataclass
class BodySpecification(KinematicStructureEntitySpecification[Body]):

    inertial: Optional[Inertial] = None
    """
    Inertia properties of created bodies. None means the Body default.
    """

    visual_shapes: Optional[ShapeCollection] = None
    """
    Visual shapes when they differ from `shapes`. None shares `shapes` for both
    collision and visual (one collection); an empty list means no visual geometry.
    """

    def to_domain_object(self, name: Union[str, PrefixedName, None] = None) -> Body:
        """
        Create a new, world-independent body from this specification.
        :param name: Optional name override, e.g. for spawning multiple bodies
                     from the same specification.
        :return: The created body.
        """
        if self.visual_shapes is None:
            body = super().to_domain_object(name)
        else:
            body = Body(
                name=ensure_prefixed_name(name) or self.name,
                collision=self.shapes.copy_without_reference_frame(),
                visual=self.visual_shapes.copy_without_reference_frame(),
            )
        if self.inertial is not None:
            body.inertial = deepcopy(self.inertial)
        return body


@dataclass
class RegionSpecification(KinematicStructureEntitySpecification[Region]):
    pass


@dataclass
class SemanticAnnotationWithRootSpecification(WorldEntitySpawnSpecification):
    """
    Declarative description of a semantic annotation rooted in a single kinematic
    structure entity. The annotation type owns the parent connection specification type (via its
    ``_parent_connection_specification_type``); this specification only supplies the connection
    parameters for active connections.
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
    ``_parent_connection_specification_type`` is an active connection; ignored otherwise.
    """

    multiplier: float = 1.0
    """
    DoF multiplier for the parent connection (active connections only).
    """

    offset: float = 0.0
    """
    DoF offset for the parent connection (active connections only).
    """

    connection_limits: Optional[DegreeOfFreedomLimits] = None
    """
    Degree-of-freedom limits for the parent connection (active connections only).
    """

    annotation_kwargs: dict[
        str,
        Union[
            SemanticAnnotationWithRootSpecification,
            list[SemanticAnnotationWithRootSpecification],
            Any,
        ],
    ] = field(default_factory=dict)
    """
    Everything the annotation references, keyed by the constructor/part-whole field name. Inert values
    are passed to the annotation constructor. Spec-valued entries (``WorldEntitySpawnSpecification``,
    i.e. nested annotations) are spawned during :meth:`spawn` and mounted via the annotation's part-whole
    :meth:`~...mixins.PartWholeRelationship.add`, using the dict key as the target field name.
    """

    def spawn(
        self,
        world: World,
        name: Union[str, PrefixedName, None] = None,
        parent: KinematicStructureEntity | None = None,
        parent_T_self: HomogeneousTransformationMatrix | None = None,
    ) -> HasRootKinematicStructureEntity:
        from semantic_digital_twin.semantic_annotations.mixins import (
            _wrapped_part_whole_relationship_fields,
            PartWholeRelationship,
        )

        name = ensure_prefixed_name(name) or self.name

        part_whole_fields_by_name = {
            wrapped_field.name: wrapped_field
            for wrapped_field in _wrapped_part_whole_relationship_fields(
                self.semantic_annotation_type
            )
        }

        plain_kwargs = {}
        part_semantic_annotation_specs = {}
        unsupported_specs = {}

        for key, value in self.annotation_kwargs.items():
            wrapped_field = part_whole_fields_by_name.get(key)
            value_is_sequence = isinstance(value, (list, tuple))
            items = list(value) if value_is_sequence else [value]
            all_nested_annotation_specs = bool(items) and all(
                isinstance(item, SemanticAnnotationWithRootSpecification)
                for item in items
            )

            # A part-whole field takes one nested annotation spec, or a list of them for a
            # to-many relationship.
            if (
                wrapped_field is not None
                and all_nested_annotation_specs
                and (
                    wrapped_field.is_many_to_many_relationship or not value_is_sequence
                )
            ):
                part_semantic_annotation_specs[key] = items
            elif any(isinstance(item, WorldEntitySpawnSpecification) for item in items):
                unsupported_specs[key] = value
            else:
                plain_kwargs[key] = value

        if unsupported_specs:
            raise NotImplementedError(
                "Only nested part-whole annotation specs are supported in annotation_kwargs. These "
                "entries are not (e.g. raw entity specs, storage occupants, or non-part-whole fields): "
                f"{ {key: type(value).__name__ for key, value in unsupported_specs.items()} } "
                f"on {self.semantic_annotation_type.__name__}."
            )

        if self.root_specification is None:
            root_entity = Body(name=name)
        else:
            root_entity = self.root_specification.to_domain_object(name)

        instance = self.semantic_annotation_type(
            name=name, root=root_entity, **plain_kwargs
        )

        effective_pose = parent_T_self or (
            self.root_specification.parent_T_self
            if self.root_specification is not None
            else None
        )
        children = (
            self.root_specification.child_specification
            if self.root_specification is not None
            else ()
        )

        connection_specification = self.semantic_annotation_type._parent_connection_specification_type.from_kwargs(
            axis=self.axis,
            multiplier=self.multiplier,
            offset=self.offset,
            dof_limits=self.connection_limits,
        )

        with world.modify_world():
            connection_specification.spawn(
                world, parent=parent, parent_T_self=effective_pose, child=root_entity
            )
            world.add_semantic_annotation(instance)
            self._spawn_children(world, root_entity, children)

            if not isinstance(instance, PartWholeRelationship):
                return instance

            for field_name, part_specs in part_semantic_annotation_specs.items():
                for part_spec in part_specs:
                    part = part_spec.spawn(world, parent=root_entity)
                    instance.add(part, field_name=field_name)

        return instance


@dataclass
class ConnectionSpecification(WorldEntitySpawnSpecification, ABC):
    """
    Declarative, world- and kinematic-structure-entity-independent description of a connection.

    Each connection family is a concrete subclass that binds its :attr:`connection_type` and
    carries exactly the parameters that family uses. Materializing a specification forwards those
    parameters to the connection type's
    :meth:`~semantic_digital_twin.world_description.world_entity.Connection.create_with_dofs`.
    """

    name: Union[str, PrefixedName, None] = field(default=None, kw_only=True)
    """
    Optional connection name. If None, ``create_with_dofs`` auto-generates one from parent and child.
    """

    @property
    @abstractmethod
    def connection_type(self) -> Type[Connection]:
        """The connection type this specification materializes."""

    def _create_with_dofs_kwargs(self) -> dict[str, Any]:
        """Forward every public dataclass field except the connection ``name`` to ``create_with_dofs``."""
        discovered_attributes = DataclassOnlyIntrospector().discover(type(self))
        return {
            attribute.public_name: getattr(self, attribute.public_name)
            for attribute in discovered_attributes
            if attribute.public_name != "name"
        }

    @classmethod
    def from_kwargs(
        cls, *, name: PrefixedName | None = None, **connection_parameters
    ) -> Self:
        """
        Instantiate this specification, applying the parameters its connection family uses.

        Parameters that the family does not use are ignored, so a caller holding a bare
        specification type can parameterize any family uniformly.
        """
        return cls(name=name)

    def spawn(
        self,
        world: World,
        name: Union[str, PrefixedName, None] = None,
        parent: KinematicStructureEntity | None = None,
        parent_T_self: HomogeneousTransformationMatrix | None = None,
        *,
        child: KinematicStructureEntity | None = None,
    ) -> Connection:
        """
        Materialize the connection between ``parent`` and ``child`` and add it to the world.

        Unlike the entity specifications, a connection joins two pre-existing entities, so the
        child it connects must be supplied explicitly via ``child``.

        :param child: The kinematic structure entity that becomes the connection's child.
        :raises MissingConnectionChildError: If ``child`` is not provided.
        """
        if child is None:
            raise MissingConnectionChildError(connection_name=self.name)

        parent = parent or world.root
        connection_name = ensure_prefixed_name(name) or self.name

        parent_T_connection = parent_T_self or HomogeneousTransformationMatrix()
        parent_T_connection.reference_frame = parent
        parent_T_connection.child_frame = child

        with world.modify_world():
            connection = self.connection_type.create_with_dofs(
                world=world,
                parent=parent,
                child=child,
                name=connection_name,
                parent_T_connection_expression=parent_T_connection,
                **self._create_with_dofs_kwargs(),
            )
            world.add_connection(connection)
        return connection


@dataclass
class FixedConnectionSpecification(ConnectionSpecification):
    """Specification for a rigid :class:`~semantic_digital_twin.world_description.connections.FixedConnection`."""

    connection_type: ClassVar[Type[Connection]] = FixedConnection
    """The connection type this specification materializes."""


@dataclass
class Connection6DoFSpecification(ConnectionSpecification):
    """Specification for a free-floating :class:`~semantic_digital_twin.world_description.connections.Connection6DoF`."""

    connection_type: ClassVar[Type[Connection]] = Connection6DoF
    """The connection type this specification materializes."""


@dataclass
class ActiveConnection1DOFSpecification(ConnectionSpecification, ABC):
    """
    Specification for a single-DoF active connection. Concrete leaf subclasses bind the
    :attr:`connection_type` (e.g. prismatic or revolute).
    """

    axis: Optional[Vector3] = None
    """Movement axis of the connection. Required by ``create_with_dofs`` at spawn time."""

    multiplier: float = 1.0
    """Scaling factor applied to the degree of freedom's motion."""

    offset: float = 0.0
    """Constant offset applied to the degree of freedom's motion."""

    dof_limits: Optional[DegreeOfFreedomLimits] = None
    """Limits for the generated degree of freedom."""

    @classmethod
    def from_kwargs(
        cls,
        *,
        name: PrefixedName | None = None,
        axis: Vector3 | None = None,
        multiplier: float = 1.0,
        offset: float = 0.0,
        dof_limits: Optional[DegreeOfFreedomLimits] = None,
        **_,
    ) -> Self:
        return cls(
            name=name,
            axis=axis,
            multiplier=multiplier,
            offset=offset,
            dof_limits=dof_limits,
        )


@dataclass
class PrismaticConnectionSpecification(ActiveConnection1DOFSpecification):
    """Specification for a :class:`~semantic_digital_twin.world_description.connections.PrismaticConnection`."""

    connection_type: ClassVar[Type[Connection]] = PrismaticConnection
    """The connection type this specification materializes."""


@dataclass
class RevoluteConnectionSpecification(ActiveConnection1DOFSpecification):
    """Specification for a :class:`~semantic_digital_twin.world_description.connections.RevoluteConnection`."""

    connection_type: ClassVar[Type[Connection]] = RevoluteConnection
    """The connection type this specification materializes."""


@dataclass
class ConnectedBodySpecification(WorldEntitySpawnSpecification):
    """A body specification together with the connection that attaches it to its parent."""

    name: Union[str, PrefixedName, None] = field(default=None, kw_only=True)
    """
    Optional name override for the spawned body. Defaults to the wrapped body specification's name.
    """

    body_specification: BodySpecification
    """
    The geometry and pose specification of the body to spawn.
    """

    connection_specification: ConnectionSpecification = field(
        default_factory=FixedConnectionSpecification
    )
    """
    How the spawned body is connected to its parent. A fixed connection by default.
    """

    def __post_init__(self):
        if self.name is None:
            self.name = self.body_specification.name
        super().__post_init__()

    def spawn(
        self,
        world: World,
        name: Union[str, PrefixedName, None] = None,
        parent: KinematicStructureEntity | None = None,
        parent_T_self: HomogeneousTransformationMatrix | None = None,
    ) -> Body:
        body = self.body_specification.to_domain_object(name)
        pose = parent_T_self or self.body_specification.parent_T_self
        with world.modify_world():
            self.connection_specification.spawn(
                world, parent=parent, parent_T_self=pose, child=body
            )
            self._spawn_children(
                world, body, self.body_specification.child_specification
            )
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
