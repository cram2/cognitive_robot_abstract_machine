from __future__ import annotations
from abc import ABC, abstractmethod
from copy import deepcopy
from dataclasses import dataclass, field
from typing import Union, Optional, TYPE_CHECKING

from typing_extensions import Self, Type, Any, Generic, TypeVar

from krrood.class_diagrams.class_diagram import WrappedClass
from krrood.patterns.subclass_safe_generic import AbstractSubClassSafeGeneric
from krrood.utils import get_generic_type_params
from random_events.product_algebra import Event
from semantic_digital_twin.datastructures.prefixed_name import PrefixedName
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
        PartWholeRelationshipField,
    )
    from semantic_digital_twin.robots.robot_parts import AbstractRobot


DomainObjectType = TypeVar("DomainObjectType", bound=KinematicStructureEntity)


@dataclass
class WorldEntitySpawnSpecification(ABC):

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

    @staticmethod
    def _to_prefixed_name(
        name: Union[str, PrefixedName, None],
    ) -> Optional[PrefixedName]:
        """Normalize a name to a :class:`PrefixedName`, passing ``None`` through."""
        return PrefixedName(name) if isinstance(name, str) else name

    @staticmethod
    def _require_axis_for_active_connection(
        connection_type: Type[Connection], axis: Optional[Vector3]
    ) -> None:
        """Raise if an active connection is requested without a movement axis."""
        if axis is None and issubclass(connection_type, ActiveConnection):
            raise ValueError(
                f"{connection_type.__name__} is an active connection, so axis is required."
            )

    @staticmethod
    def _build_connection(
        world: World,
        parent: KinematicStructureEntity,
        child: KinematicStructureEntity,
        connection_type: Type[Connection],
        parent_T_child: Optional[HomogeneousTransformationMatrix],
        axis: Optional[Vector3] = None,
        multiplier: float = 1.0,
        offset: float = 0.0,
        dof_limits: Optional[DegreeOfFreedomLimits] = None,
    ) -> Connection:
        """
        Build a connection of ``connection_type`` between ``parent`` and ``child``.

        The parent_T_child offset is carried in the connection expression (so no two-phase
        ``.origin`` assignment is needed), mirroring
        :meth:`HasRootKinematicStructureEntity._create_with_connection_in_world`.

        :param parent_T_child: Pose of the child in the parent frame. Defaults to identity.
        :param axis: Movement axis, required for active (1-DoF) connections, ignored otherwise.
        :param multiplier: DoF multiplier for active connections.
        :param offset: DoF offset for active connections.
        :param dof_limits: Degree-of-freedom limits for active connections.
        """
        parent_T_child = parent_T_child or HomogeneousTransformationMatrix()
        parent_T_child.reference_frame = parent
        parent_T_child.child_frame = child

        if issubclass(connection_type, ActiveConnection1DOF) and axis is None:
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
            dof_limits=dof_limits,
            parent_T_connection_expression=parent_T_child,
        )

    def _attach(
        self,
        world: World,
        entity: KinematicStructureEntity,
        parent: Optional[KinematicStructureEntity],
        connection_type: Type[Connection],
        parent_T_self: Optional[HomogeneousTransformationMatrix],
        *,
        axis: Optional[Vector3] = None,
        multiplier: float = 1.0,
        offset: float = 0.0,
        dof_limits: Optional[DegreeOfFreedomLimits] = None,
        annotation: Optional[HasRootKinematicStructureEntity] = None,
        children: list[WorldEntitySpawnSpecification] = None,
    ) -> None:
        """
        Attach ``entity`` to ``parent`` (default ``world.root``) with a connection, optionally
        register ``annotation``, then spawn ``children`` beneath ``entity`` — all in one
        ``modify_world`` block.
        """
        parent = parent or world.root
        with world.modify_world():
            connection = self._build_connection(
                world,
                parent,
                entity,
                connection_type,
                parent_T_self,
                axis,
                multiplier,
                offset,
                dof_limits,
            )
            world.add_connection(connection)
            if annotation is not None:
                world.add_semantic_annotation(annotation)

            if children is None:
                return
            for child in children:
                child.spawn(world, parent=entity)


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

    parent_T_self: HomogeneousTransformationMatrix = field(
        default_factory=HomogeneousTransformationMatrix
    )
    """
    Default placement of the entity in its parent frame, used by :meth:`spawn` when the caller does not
    override it. Identity by default.
    """

    def __post_init__(self):
        self.name = self._to_prefixed_name(self.name)

    def to_domain_object(
        self, name: Union[str, PrefixedName, None] = None
    ) -> DomainObjectType:
        """Materialize a new, world-independent kinematic structure entity from this spec."""
        domain_object_type = get_generic_type_params(
            self, KinematicStructureEntitySpecification
        )[0]
        return domain_object_type.from_shape_collection(
            self._to_prefixed_name(name) or self.name,
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
        self._attach(
            world,
            entity,
            parent,
            FixedConnection,
            parent_T_self or self.parent_T_self,
            children=self.child_specification,
        )
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
                name=self._to_prefixed_name(name) or self.name,
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

    def __post_init__(self):
        self.name = self._to_prefixed_name(self.name)
        self._require_axis_for_active_connection(
            self.semantic_annotation_type._parent_connection_type, self.axis
        )

    def spawn(
        self,
        world: World,
        name: Union[str, PrefixedName, None] = None,
        parent: KinematicStructureEntity | None = None,
        parent_T_self: HomogeneousTransformationMatrix | None = None,
    ) -> HasRootKinematicStructureEntity:
        from semantic_digital_twin.semantic_annotations.mixins import (
            PartWholeRelationship,
        )

        name = self._to_prefixed_name(name) or self.name

        plain_kwargs = {}
        part_semantic_annotation_specs = {}
        other_semantic_annotation_specs = {}
        kinematic_structure_entity_specs = {}
        other_specs = {}

        for key, value in self.annotation_kwargs.items():

            if not isinstance(value, WorldEntitySpawnSpecification):
                plain_kwargs[key] = value
                continue

            if isinstance(value, KinematicStructureEntitySpecification):
                kinematic_structure_entity_specs[key] = value
                continue

            if isinstance(value, SemanticAnnotationWithRootSpecification):
                semantic_annotation_wrapped_class = WrappedClass(
                    value.semantic_annotation_type
                )
                field_of_interest = next(
                    field
                    for field in semantic_annotation_wrapped_class.fields
                    if field.name == key
                )
                if isinstance(field_of_interest, PartWholeRelationshipField):
                    part_semantic_annotation_specs[key] = value
                else:
                    other_semantic_annotation_specs[key] = value
                continue

            other_specs[key] = value

        if part_semantic_annotation_specs and not issubclass(
            self.semantic_annotation_type, PartWholeRelationship
        ):
            raise NotImplementedError(
                "Spec-valued annotation_kwargs (nested annotations) are only supported on part-whole "
                "annotations; pass already-constructed values for other annotation types."
            )

        if other_semantic_annotation_specs:
            raise NotImplementedError(
                "Non-PartWholeRelationshipFields are not supported yet"
            )

        if other_specs:
            raise NotImplementedError(
                f"Not sure how to handle these cases yet: { {k: type(v) for k, v in other_specs.items()} }"
            )

        if self.root_specification is None:
            root_entity = Body(name=name)
        else:
            root_entity = self.root_specification.to_domain_object(name)

        for key, value in kinematic_structure_entity_specs.items():
            resolved_kinematic_structure_entity = value.spawn(world, root_entity)
            plain_kwargs[key] = resolved_kinematic_structure_entity

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

        with world.modify_world():
            self._attach(
                world,
                root_entity,
                parent,
                self.semantic_annotation_type._parent_connection_type,
                effective_pose,
                axis=self.axis,
                multiplier=self.multiplier,
                offset=self.offset,
                dof_limits=self.connection_limits,
                annotation=instance,
                children=children,
            )
            for field_name, part_spec in part_semantic_annotation_specs.items():
                part = part_spec.spawn(world, parent=root_entity)
                instance.add(part, field_name=field_name)

        return instance


@dataclass
class BodyAndConnectionSpecification(WorldEntitySpawnSpecification):

    body_specification: BodySpecification

    connection_type: Type[Connection] = field(default=FixedConnection)

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

    connection_limits: Optional[DegreeOfFreedomLimits] = None
    """
    Degree-of-freedom limits for the parent connection (active connections only).
    """

    def __post_init__(self):
        self._require_axis_for_active_connection(self.connection_type, self.axis)

    def spawn(
        self,
        world: World,
        name: Union[str, PrefixedName, None] = None,
        parent: KinematicStructureEntity | None = None,
        parent_T_self: HomogeneousTransformationMatrix | None = None,
    ) -> Body:
        body = self.body_specification.to_domain_object(name)
        pose = parent_T_self or self.body_specification.parent_T_self
        self._attach(
            world,
            body,
            parent,
            self.connection_type,
            pose,
            axis=self.axis,
            multiplier=self.multiplier,
            offset=self.offset,
            dof_limits=self.connection_limits,
            children=self.body_specification.child_specification,
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
