from __future__ import annotations

from abc import ABC, abstractmethod
from copy import deepcopy
from dataclasses import dataclass, field
from typing import (
    Iterable,
    Union,
    Optional,
    TYPE_CHECKING,
    cast,
    Type,
    Any,
    Generic,
    List,
)

from typing_extensions import Self, TypeVar

from krrood.class_diagrams.attribute_introspector import DataclassOnlyIntrospector
from krrood.patterns.subclass_safe_generic import AbstractSubClassSafeGeneric
from krrood.utils import get_generic_type_params
from random_events.product_algebra import Event
from semantic_digital_twin.adapters.urdf import URDFParser
from semantic_digital_twin.datastructures.prefixed_name import (
    PrefixedName,
)
from semantic_digital_twin.exceptions import (
    MissingConnectionChildError,
    PartWholeCardinalityError,
    PartWholeFieldInAnnotationKwargs,
    UnknownPartWholeRelationshipField,
    MissingConnectionParentError,
)
from semantic_digital_twin.spatial_types import (
    HomogeneousTransformationMatrix,
    Point3,
    Vector3,
)
from semantic_digital_twin.world import World
from semantic_digital_twin.world_description.connections import (
    WheeledDrive,
    FixedConnection,
    ActiveConnection,
    Connection6DoF,
    PrismaticConnection,
    RevoluteConnection,
)
from semantic_digital_twin.world_description.degree_of_freedom import (
    DegreeOfFreedomLimits,
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
    WorldEntity,
)

if TYPE_CHECKING:
    from semantic_digital_twin.semantic_annotations.mixins import (
        HasRootKinematicStructureEntity,
        PartWholeRelationship,
    )
    from semantic_digital_twin.robots.robot_parts import AbstractRobot
    from semantic_digital_twin.adapters.package_resolver import PathResolver


TWorldEntity = TypeVar("TWorldEntity", bound=WorldEntity)
TKinematicStructureEntity = TypeVar(
    "TKinematicStructureEntity", bound=KinematicStructureEntity
)
TConnection = TypeVar("TConnection", bound=Connection)


@dataclass
class NamedSpecification(ABC):
    """
    Base for every specification: it carries a name and normalizes it. It deliberately declares no
    materialization contract, so entity-spawn specs and connection specs can derive their own
    (incompatible) verbs from it without one masquerading as the other.
    """

    name: Optional[str]
    """
    The name of entities created from this specification, as a plain string. ``None`` defers naming to 
    materialization.
    """

    def _resolved_name(self, name: Optional[str] = None) -> Optional[PrefixedName]:
        """
        Normalize the spawn-time name override, or the spec's own name, into a :class:`PrefixedName`.

        An already-prefixed name is returned unchanged; a bare string is wrapped. ``None`` is
        preserved so materialization can fall back to default name generation.
        """
        used_name = name if name is not None else self.name
        if used_name is None:
            return None
        return PrefixedName(name=used_name)


@dataclass
class SpawnSpecification(NamedSpecification, Generic[TWorldEntity], ABC):
    """
    Specification for a world entity that materializes itself together with the connection that
    attaches it to its parent. Materialized via :meth:`spawn`.
    """

    @abstractmethod
    def spawn(
        self,
        world: World,
        name: Optional[str] = None,
        parent: KinematicStructureEntity | None = None,
        parent_T_self: HomogeneousTransformationMatrix | None = None,
    ) -> TWorldEntity:
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
        children: Iterable[SpawnSpecification],
    ) -> None:
        """Spawn each child specification as a kinematic child of ``parent``."""
        for child in children:
            child.spawn(world, parent=parent)


@dataclass
class KinematicStructureEntitySpecification(
    SpawnSpecification[TKinematicStructureEntity],
    AbstractSubClassSafeGeneric,
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

    child_specification: list[KinematicStructureEntitySpecification] = field(
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

    def to_domain_object(self, name: Optional[str] = None) -> TKinematicStructureEntity:
        """
        Materialize a new, world-independent kinematic structure entity from this spec.

        The concrete domain-object type is resolved from this spec's bound generic parameter.

        :param name: Optional name override. If None, the spec's own name is used.
        """
        [domain_object_type] = get_generic_type_params(
            self, KinematicStructureEntitySpecification
        )
        return domain_object_type.from_shape_collection(
            self._resolved_name(name),
            self.shapes.copy_without_reference_frame(),
        )

    def _spawn_attached(
        self,
        world: World,
        connection_specification: ConnectionSpecification,
        name: Optional[str] = None,
        parent: KinematicStructureEntity | None = None,
        parent_T_self: HomogeneousTransformationMatrix | None = None,
    ) -> TKinematicStructureEntity:
        """
        Materialize this entity, attach it to ``parent`` via ``connection_specification``, and spawn
        its geometry children. This is the shared procedure behind both :meth:`spawn` and
        :meth:`ConnectedBodySpecification.spawn`; they differ only in which connection is used.
        """
        entity = self.to_domain_object(name)
        with world.modify_world():
            connection_specification.connect(
                world,
                parent=parent,
                child=entity,
                parent_T_connection=parent_T_self or self.parent_T_self,
            )
            for child in self.child_specification:
                child.spawn(world, parent=entity)
        return entity

    def spawn(
        self,
        world: World,
        name: Optional[str] = None,
        parent: KinematicStructureEntity | None = None,
        parent_T_self: HomogeneousTransformationMatrix | None = None,
    ) -> TKinematicStructureEntity:
        """
        Materialize the entity and attach it to ``parent`` via a fixed connection.

        :param parent: The entity to attach to. If None, ``world.root`` is used.
        :param parent_T_self: Overrides the specification's stored default pose. If None, the stored default is used.
        :param name: Overrides the specification's own name. If None, the spec's name is used.
        """
        return self._spawn_attached(
            world, FixedConnectionSpecification(), name, parent, parent_T_self
        )

    @classmethod
    def box(
        cls,
        name: str,
        scale: Scale,
        color: Optional[Color] = None,
        origin: Optional[HomogeneousTransformationMatrix] = None,
        child_specification: list[KinematicStructureEntitySpecification] | None = None,
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
        name: str,
        radius: float,
        color: Optional[Color] = None,
        origin: Optional[HomogeneousTransformationMatrix] = None,
        child_specification: list[KinematicStructureEntitySpecification] | None = None,
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
        name: str,
        width: float,
        height: float,
        color: Optional[Color] = None,
        origin: Optional[HomogeneousTransformationMatrix] = None,
        child_specification: list[KinematicStructureEntitySpecification] | None = None,
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
        name: str,
        filename: str,
        scale: Optional[Scale] = None,
        color: Optional[Color] = None,
        origin: Optional[HomogeneousTransformationMatrix] = None,
        child_specification: list[KinematicStructureEntitySpecification] | None = None,
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
        name: str,
        event: Event,
        child_specification: list[KinematicStructureEntitySpecification] | None = None,
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

    @classmethod
    def from_3d_points(
        cls,
        name: str,
        points_3d: List[Point3],
        minimum_thickness: float = 0.005,
        sv_ratio_tol: float = 1e-7,
        child_specification: list[KinematicStructureEntitySpecification] | None = None,
    ) -> Self:
        """
        Specification whose geometry is the convex hull of a point cloud.

        :param name: The name of the entity.
        :param points_3d: The points whose convex hull defines the geometry.
        :param minimum_thickness: Thickness added when the points are near-planar.
        :param sv_ratio_tol: Singular-value ratio tolerance for the planarity test.
        :return: The created specification.
        """
        return cls(
            name=name,
            shapes=ShapeCollection(
                [
                    Mesh.from_3d_points(
                        points_3d,
                        minimum_thickness=minimum_thickness,
                        sv_ratio_tol=sv_ratio_tol,
                    )
                ]
            ).copy_without_reference_frame(),
            child_specification=child_specification or [],
        )


@dataclass
class BodySpecification(KinematicStructureEntitySpecification[Body]):
    """
    Declarative description of a :class:`~semantic_digital_twin.world_description.world_entity.Body`.

    Extends the kinematic-structure-entity specification with body-only properties: inertial
    parameters and a separate visual shape collection.
    """

    inertial: Optional[Inertial] = None
    """
    Inertia properties of created bodies. None means the Body default.
    """

    visual_shapes: Optional[ShapeCollection] = None
    """
    Visual shapes when they differ from `shapes`. None shares `shapes` for both
    collision and visual (one collection); an empty list means no visual geometry.
    """

    def to_domain_object(self, name: Optional[str] = None) -> Body:
        """
        Create a new, world-independent body from this specification.
        :param name: Optional name override, e.g. for spawning multiple bodies
                     from the same specification.
        :return: The created body.
        """
        body = Body.from_shape_collection(
            self._resolved_name(name),
            self.shapes.copy_without_reference_frame(),
            visuals_shape_collection=self.visual_shapes,
        )
        if self.inertial is not None:
            body.inertial = deepcopy(self.inertial)
        return body


@dataclass
class RegionSpecification(KinematicStructureEntitySpecification[Region]):
    """
    Declarative description of a :class:`~semantic_digital_twin.world_description.world_entity.Region`.

    Carries no fields beyond the base kinematic-structure-entity specification; it only binds the
    materialized domain-object type to :class:`Region`.
    """


@dataclass
class SemanticAnnotationWithRootSpecification(
    SpawnSpecification["HasRootKinematicStructureEntity"]
):
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

    root_specification: KinematicStructureEntitySpecification
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

    annotation_kwargs: dict[str, Any] = field(default_factory=dict)
    """
    Inert keyword arguments passed straight to the annotation constructor, keyed by constructor field
    name. Nested annotation parts do not belong here; use :attr:`part_specifications`.
    """

    part_specifications: dict[
        str,
        Union[
            SemanticAnnotationWithRootSpecification,
            list[SemanticAnnotationWithRootSpecification],
        ],
    ] = field(default_factory=dict)
    """
    Nested annotation parts keyed by the target part-whole relationship field name. Each part is
    spawned during :meth:`spawn` and mounted via the annotation's
    :meth:`~...mixins.PartWholeRelationship.add`. A list value mounts several parts onto a to-many
    field; a single value mounts onto a singular field.
    """

    def __post_init__(self):
        """Validate the annotation kwargs and part specifications so misuse fails fast, before any world mutation."""
        self._validate_annotation_kwargs()
        self._validate_part_specifications(self.semantic_annotation_type)

    def spawn(
        self,
        world: World,
        name: Optional[str] = None,
        parent: KinematicStructureEntity | None = None,
        parent_T_self: HomogeneousTransformationMatrix | None = None,
    ) -> HasRootKinematicStructureEntity:
        """
        Materialize the annotation in ``world``: spawn its root entity, attach it to ``parent`` via the
        annotation's parent connection, register the annotation, and spawn its geometry children and
        mounted part specifications.

        :param parent: The entity to attach the root to. If None, ``world.root`` is used.
        :param parent_T_self: Overrides the root specification's stored default pose.
        :param name: Overrides the specification's own name. If None, the spec's name is used.
        """
        root_entity = self.root_specification.to_domain_object(name or self.name)

        instance = self.semantic_annotation_type(
            name=self._resolved_name(name), root=root_entity, **self.annotation_kwargs
        )

        effective_pose = parent_T_self or self.root_specification.parent_T_self
        children = self.root_specification.child_specification

        connection_specification = self.semantic_annotation_type._parent_connection_specification_type.from_kwargs(
            axis=self.axis,
            multiplier=self.multiplier,
            offset=self.offset,
            dof_limits=self.connection_limits,
        )

        with world.modify_world():
            connection_specification.connect(
                world,
                parent=parent,
                child=root_entity,
                parent_T_connection=effective_pose,
            )
            world.add_semantic_annotation(instance)
            for child in children:
                child.spawn(world, parent=root_entity)
            self._mount_part_specifications(world, instance, root_entity)

        return instance

    def _validate_annotation_kwargs(self) -> None:
        """
        Validate that :attr:`annotation_kwargs` carries no part-whole relationship field. Such fields
        must be supplied via :attr:`part_specifications` so they are spawned and mounted.

        :raises PartWholeFieldInAnnotationKwargs: If a key names a part-whole relationship field.
        """
        part_whole_field_names = self._part_whole_fields_by_name()
        misplaced_field_names = [
            field_name
            for field_name in self.annotation_kwargs
            if field_name in part_whole_field_names
        ]
        if misplaced_field_names:
            raise PartWholeFieldInAnnotationKwargs(
                annotation_type_name=self.semantic_annotation_type.__name__,
                field_names=misplaced_field_names,
            )

    def _validate_part_specifications(
        self, instance: type[HasRootKinematicStructureEntity]
    ) -> None:
        """
        Validate that every :attr:`part_specifications` key names a part-whole relationship field of
        the annotation and that list values target only to-many fields.

        :raises UnknownPartWholeRelationshipField: If a key is not a part-whole relationship field.
        :raises PartWholeCardinalityError: If a list is given for a singular field.
        """
        part_whole_fields_by_name = self._part_whole_fields_by_name()
        for field_name, value in self.part_specifications.items():
            wrapped_field = part_whole_fields_by_name.get(field_name)
            if wrapped_field is None:
                raise UnknownPartWholeRelationshipField(
                    annotation=instance,
                    field_name=field_name,
                    available_fields=list(part_whole_fields_by_name),
                )
            if (
                isinstance(value, list)
                and not wrapped_field.is_many_to_many_relationship
            ):
                raise PartWholeCardinalityError(
                    annotation_type_name=self.semantic_annotation_type.__name__,
                    field_name=field_name,
                )

    def _part_whole_fields_by_name(self) -> dict[str, Any]:
        """The annotation type's part-whole relationship fields, keyed by field name."""
        from semantic_digital_twin.semantic_annotations.mixins import (
            _wrapped_part_whole_relationship_fields,
        )

        return {
            wrapped_field.name: wrapped_field
            for wrapped_field in _wrapped_part_whole_relationship_fields(
                self.semantic_annotation_type
            )
        }

    def _mount_part_specifications(
        self,
        world: World,
        instance: PartWholeRelationship,
        root_entity: KinematicStructureEntity,
    ) -> None:
        """
        Spawn each nested part and mount it onto ``instance`` via the part-whole
        :meth:`~...mixins.PartWholeRelationship.add`, keyed by the target field name.

        .. note:: Assumes :meth:`_validate_part_specifications` has already run.
        """
        for field_name, value in self.part_specifications.items():
            part_specs = value if isinstance(value, list) else [value]
            for part_spec in part_specs:
                part = part_spec.spawn(world, parent=root_entity)
                instance.add(part, field_name=field_name)


@dataclass
class ConnectionSpecification(
    NamedSpecification, Generic[TConnection], AbstractSubClassSafeGeneric, ABC
):
    """
    Declarative, world- and kinematic-structure-entity-independent description of a connection.

    A connection joins two pre-existing entities, so it is *not* a
    :class:`SpawnSpecification` (which materializes an entity and its own parent connection). It is
    materialized via :meth:`connect`, which takes the ``child`` to attach.

    Each connection family is a concrete subclass that binds the connection type as its generic
    parameter (e.g. ``ConnectionSpecification[FixedConnection]``) and carries exactly the parameters
    that family uses. Materializing a specification forwards those parameters to the connection type's
    :meth:`~semantic_digital_twin.world_description.world_entity.Connection.create_with_dofs`.
    """

    name: Optional[str] = field(default=None, kw_only=True)
    """
    Optional connection name as a plain string. If None, ``create_with_dofs`` auto-generates one
    from parent and child. Wrapped into a :class:`PrefixedName` only at materialization time.
    """

    @property
    def connection_type(self) -> Type[TConnection]:
        """The connection type this specification materializes, from its bound generic parameter."""
        [connection_type] = get_generic_type_params(self, ConnectionSpecification)
        return connection_type

    def _create_with_dofs_kwargs(self) -> dict[str, Any]:
        """Forward every public dataclass field except the connection ``name`` to ``create_with_dofs``."""
        discovered_attributes = DataclassOnlyIntrospector().discover(type(self))
        instance_values = vars(self)
        result = {}
        for attribute in discovered_attributes:
            public_name = cast(str, attribute.public_name)
            if public_name != "name":
                result[public_name] = instance_values[public_name]

        return result

    @classmethod
    def from_kwargs(
        cls, *, name: Optional[str] = None, **connection_parameters
    ) -> Self:
        """
        Instantiate this specification, applying the parameters its connection family uses.

        Parameters that the family does not use are ignored, so a caller holding a bare
        specification type can parameterize any family uniformly.
        """
        return cls(name=name)

    def connect(
        self,
        world: World,
        parent: KinematicStructureEntity | None = None,
        child: KinematicStructureEntity | None = None,
        parent_T_connection: HomogeneousTransformationMatrix | None = None,
        name: Optional[str] = None,
    ) -> Connection:
        """
        Materialize the connection between ``parent`` and ``child`` and add it to the world.

        A connection joins two pre-existing entities, so the child it connects must be supplied
        explicitly via ``child``. If ``parent`` is omitted, ``world.root`` is used.

        :param child: The kinematic structure entity that becomes the connection's child.
        :param parent_T_connection: Placement of the connection in the parent frame. Identity if None.
        :raises MissingConnectionChildError: If ``child`` is not provided.
        """
        if child is None:
            raise MissingConnectionChildError(connection_name=self.name)

        parent = parent or world.root
        if parent is None:
            raise MissingConnectionParentError(connection_name=self.name)

        parent_T_connection = (
            deepcopy(parent_T_connection)
            if parent_T_connection is not None
            else HomogeneousTransformationMatrix()
        )
        parent_T_connection.reference_frame = parent
        parent_T_connection.child_frame = child

        with world.modify_world():
            connection = self.connection_type.create_with_dofs(
                world=world,
                parent=parent,
                child=child,
                name=self._resolved_name(name),
                parent_T_connection_expression=parent_T_connection,
                **self._create_with_dofs_kwargs(),
            )
            world.add_connection(connection)
        return connection


@dataclass
class FixedConnectionSpecification(ConnectionSpecification[FixedConnection]):
    """
    Declares a rigid :class:`~semantic_digital_twin.world_description.connections.FixedConnection`.

    Use this when two entities should keep a constant relative pose and never move with respect
    to each other.
    """


@dataclass
class Connection6DoFSpecification(ConnectionSpecification[Connection6DoF]):
    """
    Declares a free-floating :class:`~semantic_digital_twin.world_description.connections.Connection6DoF`.

    Use this when an entity may move and rotate freely relative to its parent, such as an object
    resting in the world that is not rigidly attached to anything.
    """


@dataclass
class ActiveConnection1DOFSpecification(ConnectionSpecification[TConnection], ABC):
    """
    Specification for a single-DoF active connection. Concrete leaf subclasses bind the connection
    type as their generic parameter (e.g. prismatic or revolute).
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
        name: Optional[str] = None,
        axis: Vector3 | None = None,
        multiplier: float = 1.0,
        offset: float = 0.0,
        dof_limits: Optional[DegreeOfFreedomLimits] = None,
        **_,
    ) -> Self:
        """
        Instantiate the specification from the single-DoF parameters (axis, multiplier, offset, limits).

        Parameters this connection family does not use are ignored, so a caller holding a bare
        specification type can parameterize any family uniformly.
        """
        return cls(
            name=name,
            axis=axis,
            multiplier=multiplier,
            offset=offset,
            dof_limits=dof_limits,
        )


@dataclass
class PrismaticConnectionSpecification(
    ActiveConnection1DOFSpecification[PrismaticConnection]
):
    """
    Declares a :class:`~semantic_digital_twin.world_description.connections.PrismaticConnection`.

    Use this for a single translational degree of freedom along the connection axis, such as a
    drawer sliding in or out.
    """


@dataclass
class RevoluteConnectionSpecification(
    ActiveConnection1DOFSpecification[RevoluteConnection]
):
    """
    Declares a :class:`~semantic_digital_twin.world_description.connections.RevoluteConnection`.

    Use this for a single rotational degree of freedom about the connection axis, such as a door
    swinging on its hinge.
    """


@dataclass
class ConnectedBodySpecification(SpawnSpecification):
    """A body specification together with the connection that attaches it to its parent."""

    name: Optional[str] = field(default=None, kw_only=True)
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

    def spawn(
        self,
        world: World,
        name: Optional[str] = None,
        parent: KinematicStructureEntity | None = None,
        parent_T_self: HomogeneousTransformationMatrix | None = None,
    ) -> Body:
        """
        Materialize the body and attach it to ``parent`` via the wrapped connection specification.

        :param parent: The entity to attach to. If None, ``world.root`` is used.
        :param parent_T_self: Overrides the body specification's stored default pose.
        :param name: Overrides the specification's own name. If None, the spec's name is used.
        """
        return self.body_specification._spawn_attached(
            world, self.connection_specification, name, parent, parent_T_self
        )


@dataclass
class WorldSpecification:
    """
    Declarative description of a world: an environment, an optional robot, and objects around them.

    The environment is supplied as a concrete :class:`World` (build one from a model file with
    :meth:`from_urdf` or :meth:`from_mjcf`). Applying it (:meth:`to_world`) optionally parses and
    merges a robot as ``world.root -> odom_combined -> drive -> robot``, then spawns all starting
    objects, and returns the augmented environment world.
    """

    world: World
    """
    The environment world that the robot and starting objects are added to. Its root is ``world.root``.
    """

    robot_semantic_annotation: Optional[Type[AbstractRobot]] = None
    """
    The robot to merge into the environment. If None, no robot is added.
    """

    drive_connection_type: Type[WheeledDrive] | None = None
    """
    The connection type attaching the robot to ``odom_combined``. Defaults to a free-floating connection.
    """

    world_T_odom: HomogeneousTransformationMatrix | None = None
    """
    The localization pose of ``odom_combined`` in the ``world.root`` frame. If None, identity is used.
    """

    odom_T_robot_start: HomogeneousTransformationMatrix | None = None
    """
    The start pose of the robot in the ``odom_combined`` frame. If None, identity is used.
    """

    starting_objects: list[SpawnSpecification] = field(default_factory=list)
    """
    Specifications spawned relative to the world root once the robot (if any) is in place.
    """

    @classmethod
    def from_urdf(
        cls,
        file_path: str,
        *,
        prefix: Optional[str] = None,
        path_resolver: Optional[PathResolver] = None,
        robot_semantic_annotation: Optional[Type[AbstractRobot]] = None,
        drive_connection_type: Optional[Type[WheeledDrive]] = None,
        world_T_odom: Optional[HomogeneousTransformationMatrix] = None,
        odom_T_robot_start: Optional[HomogeneousTransformationMatrix] = None,
        starting_objects: Optional[list[SpawnSpecification]] = None,
    ) -> Self:
        """
        Build a specification whose environment is parsed from a URDF file.

        :param file_path: Path to the environment URDF. This is never a robot description; the
            robot, if any, is supplied through ``robot_semantic_annotation``.
        :param prefix: Optional name prefix for the parsed environment.
        :param path_resolver: Resolver for mesh/package paths referenced by the URDF.
        """
        world = URDFParser.from_file(
            file_path, prefix=prefix, path_resolver=path_resolver
        ).parse()
        return cls(
            world=world,
            robot_semantic_annotation=robot_semantic_annotation,
            drive_connection_type=drive_connection_type,
            world_T_odom=world_T_odom,
            odom_T_robot_start=odom_T_robot_start,
            starting_objects=starting_objects or [],
        )

    @classmethod
    def from_mjcf(
        cls,
        file_path: str,
        *,
        prefix: Optional[str] = None,
        mimic_joints: Optional[dict[str, str]] = None,
        robot_semantic_annotation: Optional[Type[AbstractRobot]] = None,
        drive_connection_type: Optional[Type[WheeledDrive]] = None,
        world_T_odom: Optional[HomogeneousTransformationMatrix] = None,
        odom_T_robot_start: Optional[HomogeneousTransformationMatrix] = None,
        starting_objects: Optional[list[SpawnSpecification]] = None,
    ) -> Self:
        """
        Build a specification whose environment is parsed from an MJCF (MuJoCo XML) file.

        :param file_path: Path to the environment MJCF. This is never a robot description; the
            robot, if any, is supplied through ``robot_semantic_annotation``.
        :param prefix: Optional name prefix for the parsed environment.
        :param mimic_joints: Mapping of joint names to the joints they mimic.
        """
        from semantic_digital_twin.adapters.mjcf import MJCFParser

        world = MJCFParser(
            file_path=file_path, mimic_joints=mimic_joints or {}, prefix=prefix
        ).parse()
        return cls(
            world=world,
            robot_semantic_annotation=robot_semantic_annotation,
            drive_connection_type=drive_connection_type,
            world_T_odom=world_T_odom,
            odom_T_robot_start=odom_T_robot_start,
            starting_objects=starting_objects or [],
        )

    def to_domain_object(self) -> World:
        """
        Materialize a new World from this specification.

        A deep copy of :attr:`world` is augmented and returned, so the specification's stored world
        is never mutated and the method can be applied repeatedly. When ``robot_semantic_annotation``
        is set, the robot is parsed from its own description and merged as
        ``world.root -> odom_combined -> drive -> robot``, with the localization and start poses
        applied. Finally all ``starting_objects`` are spawned relative to the world root.
        """
        world = deepcopy(self.world)
        if self.robot_semantic_annotation is not None:
            self._setup_robot(world)

        for starting_object in self.starting_objects:
            starting_object.spawn(world)

        return world

    def _setup_robot(self, world: World) -> None:
        """
        Set up the robot in ``world``, inserting ``odom_combined`` between the world root and the robot root.
        """
        with world.modify_world():
            odom_body = Body(name=PrefixedName("odom_combined"))
            root_C_odom = Connection6DoF.create_with_dofs(
                world=world, parent=cast(Body, world.root), child=odom_body
            )
            world.add_connection(root_C_odom)

            robot_world = URDFParser.from_file(
                self.robot_semantic_annotation.get_ros_file_path()
            ).parse()
            robot_root_id = robot_world.root.id

            drive_connection_type = self.drive_connection_type or Connection6DoF
            odom_C_robot = drive_connection_type.create_with_dofs(
                world=world, parent=odom_body, child=cast(Body, robot_world.root)
            )
            world.merge_world(robot_world, root_connection=odom_C_robot)
            if issubclass(drive_connection_type, ActiveConnection):
                odom_C_robot.has_hardware_interface = True

        self.robot_semantic_annotation.from_branch_in_world(
            cast(Body, world.get_world_entity_with_id_by_id(robot_root_id))
        )

        # Poses touch DoF state, so they are set after the modification block.
        if self.world_T_odom is not None:
            root_C_odom.origin = self.world_T_odom
        if self.odom_T_robot_start is not None:
            odom_C_robot.origin = self.odom_T_robot_start
