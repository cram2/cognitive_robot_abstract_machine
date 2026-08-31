from __future__ import annotations

from abc import ABC, abstractmethod
from copy import deepcopy
from dataclasses import dataclass, field
from typing import (
    TYPE_CHECKING,
    cast,
    Type,
    Any,
    Generic,
)
from uuid import UUID, uuid4

from typing_extensions import Self, TypeVar

from krrood.class_diagrams.attribute_introspector import DataclassOnlyIntrospector
from krrood.class_diagrams.class_diagram import WrappedClass
from krrood.class_diagrams.wrapped_field import WrappedField
from krrood.patterns.subclass_safe_generic import SubClassSafeGeneric
from krrood.utils import get_generic_type_parameters
from random_events.product_algebra import Event
from semantic_digital_twin.adapters.urdf import URDFParser
from semantic_digital_twin.adapters.world_model_parser import WorldModelParser
from semantic_digital_twin.datastructures.prefixed_name import (
    PrefixedName,
)
from semantic_digital_twin.exceptions import (
    MergedRobotAnnotationNotFound,
    PartWholeCardinalityError,
    PartWholeFieldInAnnotationKwargs,
    UnknownPartWholeRelationshipField,
    MissingConnectionParentError,
)
from semantic_digital_twin.semantic_annotations.part_whole import (
    IsPartWholeRelationship,
)
from semantic_digital_twin.spatial_types import (
    HomogeneousTransformationMatrix,
    Point3,
    Vector3,
)
from semantic_digital_twin.world import World
from semantic_digital_twin.world_description.connections import (
    ActiveConnection,
    FixedConnection,
    Connection6DoF,
    PrismaticConnection,
    RevoluteConnection,
    ScrewConnection,
)
from semantic_digital_twin.world_description.degree_of_freedom import (
    DegreeOfFreedomLimits,
)
from semantic_digital_twin.world_description.geometry import (
    VolumetricBoundingBox,
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


# %% specification type parameters
TWorldEntity = TypeVar("TWorldEntity", bound=WorldEntity)
TKinematicStructureEntity = TypeVar(
    "TKinematicStructureEntity", bound=KinematicStructureEntity
)
TConnection = TypeVar("TConnection", bound=Connection)
TSemanticAnnotation = TypeVar(
    "TSemanticAnnotation", bound="HasRootKinematicStructureEntity"
)


# %% specification base classes


@dataclass
class NamedSpecification(ABC):
    """
    Base for every specification: it carries a name and normalizes it.

    It deliberately declares no materialization contract, so entity-spawn specs and
    connection specs can derive their own (incompatible) verbs from it without one
    masquerading as the other.
    """

    name: str | None
    """
    The name of entities created from this specification, as a plain string.

    ``None`` defers naming to materialization.
    """

    def _resolved_name(self, name: str | None = None) -> PrefixedName | None:
        """
        Normalize the spawn-time name override, or the spec's own name, into a
        :class:`PrefixedName`.

        A bare string is wrapped into a :class:`PrefixedName`. ``None`` is preserved so
        materialization can fall back to default name generation.

        :param name: Overrides the specification's own name. If None, the spec's name is
            used.
        :return: The normalized name, or None when neither name is set.
        """
        used_name = name or self.name
        if used_name is None:
            return None
        return PrefixedName(name=used_name)


@dataclass
class SpawnSpecification(NamedSpecification, Generic[TWorldEntity], ABC):
    """
    Specification for a world entity that materializes itself together with the
    connection that attaches it to its parent.

    Materialized via :meth:`spawn`.
    """

    @abstractmethod
    def spawn(
        self,
        world: World,
        name: str | None = None,
        parent: KinematicStructureEntity | None = None,
        parent_T_self: HomogeneousTransformationMatrix | None = None,
    ) -> TWorldEntity:
        """
        Instantiate the World Entity and add it to the given world.

        :param world: The world the entity and its parent connection are added to.
        :param name: Overrides the specification's own name. If None, the spec's name is
            used.
        :param parent: The entity to attach to. If None, ``world.root`` is used.
        :param parent_T_self: Overrides the specification's stored default pose. If
            None, the stored default is used.
        :return: The materialized world entity.
        """


# %% connection specifications


@dataclass
class ConnectionSpecification(
    NamedSpecification, Generic[TConnection], SubClassSafeGeneric, ABC
):
    """
    World- and kinematic-structure-entity-independent description of a connection.

    A connection joins two pre-existing entities, so it is *not* a
    :class:`SpawnSpecification` (which materializes an entity and its own parent connection). It is
    materialized via :meth:`connect`, which takes the ``child`` to attach.

    Each connection family is a concrete subclass that binds the connection type as its generic
    parameter (e.g. ``ConnectionSpecification[FixedConnection]``) and carries exactly the parameters
    that family uses. Materializing a specification forwards those parameters to the connection type's
    :meth:`~semantic_digital_twin.world_description.world_entity.Connection.create_with_dofs`.
    """

    name: str | None = field(default=None, kw_only=True)
    """
    Optional connection name as a plain string.

    If None, ``create_with_dofs`` auto-generates one from parent and child. Wrapped into
    a :class:`PrefixedName` only at materialization time.
    """

    @property
    def connection_type(self) -> Type[TConnection]:
        """
        The connection type this specification materializes, from its bound generic
        parameter.
        """
        [connection_type] = get_generic_type_parameters(self, ConnectionSpecification)
        return connection_type

    def _create_with_dofs_kwargs(self) -> dict[str, Any]:
        """
        Forward every public dataclass field except the connection ``name`` to
        ``create_with_dofs``.

        :return: The connection parameters, keyed by ``create_with_dofs`` parameter
            name.
        """
        discovered_attributes = DataclassOnlyIntrospector().discover(type(self))
        instance_values = vars(self)
        result = {}
        for attribute in discovered_attributes:
            public_name = cast(str, attribute.public_name)
            if public_name != "name":
                result[public_name] = instance_values[public_name]

        return result

    def connect(
        self,
        world: World,
        child: KinematicStructureEntity,
        parent: KinematicStructureEntity | None = None,
        parent_T_connection: HomogeneousTransformationMatrix | None = None,
        name: str | None = None,
    ) -> Connection:
        """
        Materialize the connection between ``parent`` and ``child`` and add it to the
        world.

        A connection joins two pre-existing entities, so the child it connects is
        mandatory. If ``parent`` is omitted, ``world.root`` is used.

        :param world: The world the connection is added to.
        :param child: The kinematic structure entity that becomes the connection's
            child.
        :param parent: The kinematic structure entity that becomes the connection's
            parent. If None, ``world.root`` is used.
        :param parent_T_connection: Placement of the connection in the parent frame.
            Identity if None.
        :param name: Overrides the specification's own name. If None, the spec's name is
            used.
        :return: The materialized connection.
        :raises MissingConnectionParentError: If no parent is given and the world has no
            root.
        """
        parent = parent or world.root
        if parent is None:
            raise MissingConnectionParentError(connection_name=self.name)

        parent_T_connection = (
            parent_T_connection.copy_with_new_reference_frames(
                new_reference_frame=parent, new_child_frame=child
            )
            if parent_T_connection is not None
            else HomogeneousTransformationMatrix(
                reference_frame=parent, child_frame=child
            )
        )

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
    Declares a rigid
    :class:`~semantic_digital_twin.world_description.connections.FixedConnection`.

    Use this when two entities should keep a constant relative pose and never move with
    respect to each other.
    """


@dataclass
class Connection6DoFSpecification(ConnectionSpecification[Connection6DoF]):
    """
    Declares a free-floating
    :class:`~semantic_digital_twin.world_description.connections.Connection6DoF`.

    Use this when an entity may move and rotate freely relative to its parent, such as
    an object resting in the world that is not rigidly attached to anything.
    """


@dataclass
class ActiveConnection1DOFSpecification(ConnectionSpecification[TConnection], ABC):
    """
    Specification for a single-DoF active connection.

    Concrete leaf subclasses bind the connection type as their generic parameter (e.g.
    prismatic or revolute).
    """

    axis: Vector3 = field(kw_only=True)
    """
    Movement axis of the connection.

    Mandatory: a single-DoF connection without an axis has no meaning, so it cannot be
    constructed.
    """

    multiplier: float = 1.0
    """
    Scaling factor applied to the degree of freedom's motion.
    """

    offset: float = 0.0
    """
    Constant offset applied to the degree of freedom's motion.
    """

    dof_limits: DegreeOfFreedomLimits | None = None
    """
    Limits for the generated degree of freedom.
    """


@dataclass
class PrismaticConnectionSpecification(
    ActiveConnection1DOFSpecification[PrismaticConnection]
):
    """
    Declares a
    :class:`~semantic_digital_twin.world_description.connections.PrismaticConnection`.

    Use this for a single translational degree of freedom along the connection axis,
    such as a drawer sliding in or out.
    """


@dataclass
class RevoluteConnectionSpecification(
    ActiveConnection1DOFSpecification[RevoluteConnection]
):
    """
    Declares a
    :class:`~semantic_digital_twin.world_description.connections.RevoluteConnection`.

    Use this for a single rotational degree of freedom about the connection axis, such
    as a door swinging on its hinge.
    """


@dataclass
class ScrewConnectionSpecification(ActiveConnection1DOFSpecification[ScrewConnection]):
    """
    Declares a
    :class:`~semantic_digital_twin.world_description.connections.ScrewConnection`.

    Use this where rotation about the connection axis and translation along it are
    coupled into a single degree of freedom, such as the thread between a bottle and its
    cap.
    """

    screw_pitch: float = field(kw_only=True)
    """
    The distance between adjacent threads along the connection axis in meters.

    Mandatory: a thread without a pitch couples no translation to its rotation, so it
    cannot be constructed.
    """


# %% kinematic structure entity specifications


@dataclass
class KinematicStructureEntitySpecification(
    SpawnSpecification[TKinematicStructureEntity],
    SubClassSafeGeneric,
):
    """
    World-independent, reusable description of a kinematic structure entity.

    A specification is reusable: every materialization copies the prototype shapes and
    the pose, so the specification never becomes bound to an entity or world.

    The concrete domain-object type (e.g. ``Body``/``Region``) is bound as the generic
    parameter by each subclass and resolved at runtime in :meth:`to_domain_object`.
    """

    shapes: ShapeCollection = field(default_factory=ShapeCollection)
    """
    Prototype shapes with origins expressed in the entity frame.
    """

    child_specifications: list[KinematicStructureEntitySpecification] = field(
        default_factory=list
    )
    """
    The child specifications of this specification.

    If set, the spawned entity will be a parent of the children.
    """

    parent_T_self: HomogeneousTransformationMatrix = field(
        default_factory=HomogeneousTransformationMatrix
    )
    """
    Default placement of the entity in its parent frame, used by :meth:`spawn` when the
    caller does not override it.

    Identity by default.
    """

    connection_specification: ConnectionSpecification | None = None
    """
    How the spawned entity attaches to its parent.

    ``None`` means :meth:`spawn` uses a fixed connection.
    """

    @property
    def scale(self) -> Scale:
        """
        The extents of this specification's geometry.

        This is the world-independent counterpart of
        :attr:`~semantic_digital_twin.semantic_annotations.mixins.HasRootKinematicStructureEntity.scale`,
        so a specification can be measured before anything is spawned.
        """
        bounds = self.shapes.combined_mesh.bounds
        return Scale(*(bounds[1] - bounds[0]))

    def to_domain_object(self, name: str | None = None) -> TKinematicStructureEntity:
        """
        Materialize a new, world-independent kinematic structure entity from this spec.

        The concrete domain-object type is resolved from this spec's bound generic
        parameter.

        :param name: Optional name override. If None, the spec's own name is used.
        :return: The created kinematic structure entity.
        """
        [domain_object_type] = get_generic_type_parameters(
            self, KinematicStructureEntitySpecification
        )
        return domain_object_type.from_shape_collection(
            self._resolved_name(name),
            self.shapes.copy_without_reference_frame(),
        )

    def attach_and_spawn_children(
        self,
        world: World,
        entity: KinematicStructureEntity,
        connection_specification: ConnectionSpecification,
        parent: KinematicStructureEntity | None = None,
        parent_T_self: HomogeneousTransformationMatrix | None = None,
    ) -> None:
        """
        Attach an already materialized ``entity`` to ``parent`` via
        ``connection_specification`` and spawn this specification's children below it.

        This is the shared tail of every spawn: entity specifications call it on
        themselves, and annotation specifications call it on their root specification, so
        the attach-and-descend sequence exists only once.

        :param world: The world the connection and the children are added to.
        :param entity: The materialized entity to attach.
        :param connection_specification: How the entity attaches to its parent.
        :param parent: The entity to attach to. If None, ``world.root`` is used.
        :param parent_T_self: Overrides the specification's stored default pose. If
            None, the stored default is used.
        """
        with world.modify_world():
            connection_specification.connect(
                world,
                child=entity,
                parent=parent,
                parent_T_connection=(parent_T_self or self.parent_T_self),
            )
            for child in self.child_specifications:
                child.spawn(world, parent=entity)

    def _spawn_attached(
        self,
        world: World,
        connection_specification: ConnectionSpecification,
        name: str | None = None,
        parent: KinematicStructureEntity | None = None,
        parent_T_self: HomogeneousTransformationMatrix | None = None,
    ) -> TKinematicStructureEntity:
        """
        Materialize this entity, attach it to ``parent`` via
        ``connection_specification``, and spawn its geometry children.

        :param world: The world the entity and its parent connection are added to.
        :param connection_specification: How the entity attaches to its parent.
        :param name: Overrides the specification's own name. If None, the spec's name is
            used.
        :param parent: The entity to attach to. If None, ``world.root`` is used.
        :param parent_T_self: Overrides the specification's stored default pose. If
            None, the stored default is used.
        :return: The materialized kinematic structure entity.
        """
        entity = self.to_domain_object(name)
        self.attach_and_spawn_children(
            world, entity, connection_specification, parent, parent_T_self
        )
        return entity

    def spawn(
        self,
        world: World,
        name: str | None = None,
        parent: KinematicStructureEntity | None = None,
        parent_T_self: HomogeneousTransformationMatrix | None = None,
    ) -> TKinematicStructureEntity:
        """
        Materialize the entity and attach it to ``parent`` via
        :attr:`connection_specification`, defaulting to a fixed connection when none is
        set.

        :param world: The world the entity and its parent connection are added to.
        :param name: Overrides the specification's own name. If None, the spec's name is
            used.
        :param parent: The entity to attach to. If None, ``world.root`` is used.
        :param parent_T_self: Overrides the specification's stored default pose. If
            None, the stored default is used.
        :return: The materialized kinematic structure entity.
        """
        connection_specification = (
            self.connection_specification or FixedConnectionSpecification()
        )
        return self._spawn_attached(
            world, connection_specification, name, parent, parent_T_self
        )

    @classmethod
    def box(
        cls,
        name: str,
        scale: Scale,
        color: Color | None = None,
        origin: HomogeneousTransformationMatrix | None = None,
        parent_T_self: HomogeneousTransformationMatrix | None = None,
        child_specifications: list[KinematicStructureEntitySpecification] | None = None,
        connection_specification: ConnectionSpecification | None = None,
    ) -> Self:
        """
        Specification for a kinematic structure entity with a single box shape.

        :param name: The name of the body.
        :param scale: The extents of the box.
        :param color: The color of the box.
        :param origin: The origin of the box in the body frame. Defaults to identity.
        :param parent_T_self: The default placement of the entity in its parent frame.
            Defaults to identity.
        :param child_specifications: Specifications spawned as kinematic children of the
            entity. Defaults to none.
        :param connection_specification: How the entity attaches to its parent. Defaults
            to a fixed connection.
        :return: The created specification.
        """
        return cls(
            name,
            Box(
                scale=scale,
                origin=(origin or HomogeneousTransformationMatrix()),
                color=color or Color(),
            ).as_shape_collection(),
            child_specifications=(child_specifications or []),
            parent_T_self=(parent_T_self or HomogeneousTransformationMatrix()),
            connection_specification=connection_specification,
        )

    @classmethod
    def sphere(
        cls,
        name: str,
        radius: float,
        color: Color | None = None,
        origin: HomogeneousTransformationMatrix | None = None,
        parent_T_self: HomogeneousTransformationMatrix | None = None,
        child_specifications: list[KinematicStructureEntitySpecification] | None = None,
        connection_specification: ConnectionSpecification | None = None,
    ) -> Self:
        """
        Specification for a kinematic structure entity with a single sphere shape.

        :param name: The name of the kinematic structure entity.
        :param radius: The radius of the sphere.
        :param color: The color of the sphere.
        :param origin: The origin of the sphere in the kinematic structure entity frame.
            Defaults to identity.
        :param parent_T_self: The default placement of the entity in its parent frame.
            Defaults to identity.
        :param child_specifications: Specifications spawned as kinematic children of the
            entity. Defaults to none.
        :param connection_specification: How the entity attaches to its parent. Defaults
            to a fixed connection.
        :return: The created specification.
        """
        return cls(
            name,
            Sphere(
                radius=radius,
                origin=(origin or HomogeneousTransformationMatrix()),
                color=color or Color(),
            ).as_shape_collection(),
            child_specifications=(child_specifications or []),
            parent_T_self=(parent_T_self or HomogeneousTransformationMatrix()),
            connection_specification=connection_specification,
        )

    @classmethod
    def cylinder(
        cls,
        name: str,
        width: float,
        height: float,
        color: Color | None = None,
        origin: HomogeneousTransformationMatrix | None = None,
        parent_T_self: HomogeneousTransformationMatrix | None = None,
        child_specifications: list[KinematicStructureEntitySpecification] | None = None,
        connection_specification: ConnectionSpecification | None = None,
    ) -> Self:
        """
        Specification for a kinematic structure entity with a single cylinder shape.

        :param name: The name of the kinematic structure entity.
        :param width: The diameter of the cylinder.
        :param height: The height of the cylinder.
        :param color: The color of the cylinder.
        :param origin: The origin of the cylinder in the kinematic structure entity
            frame. Defaults to identity.
        :param parent_T_self: The default placement of the entity in its parent frame.
            Defaults to identity.
        :param child_specifications: Specifications spawned as kinematic children of the
            entity. Defaults to none.
        :param connection_specification: How the entity attaches to its parent. Defaults
            to a fixed connection.
        :return: The created specification.
        """
        return cls(
            name,
            Cylinder(
                width=width,
                height=height,
                origin=(origin or HomogeneousTransformationMatrix()),
                color=color or Color(),
            ).as_shape_collection(),
            child_specifications=(child_specifications or []),
            parent_T_self=(parent_T_self or HomogeneousTransformationMatrix()),
            connection_specification=connection_specification,
        )

    @classmethod
    def mesh(
        cls,
        name: str,
        filename: str,
        scale: Scale | None = None,
        color: Color | None = None,
        origin: HomogeneousTransformationMatrix | None = None,
        parent_T_self: HomogeneousTransformationMatrix | None = None,
        child_specifications: list[KinematicStructureEntitySpecification] | None = None,
        connection_specification: ConnectionSpecification | None = None,
    ) -> Self:
        """
        Specification for a kinematic structure entity with a single mesh shape loaded
        from a file.

        :param name: The name of the kinematic structure entity.
        :param filename: The path of the mesh file.
        :param scale: The scale applied to the mesh.
        :param color: The color of the mesh.
        :param origin: The origin of the mesh in the kinematic structure entity frame.
            Defaults to identity.
        :param parent_T_self: The default placement of the entity in its parent frame.
            Defaults to identity.
        :param child_specifications: Specifications spawned as kinematic children of the
            entity. Defaults to none.
        :param connection_specification: How the entity attaches to its parent. Defaults
            to a fixed connection.
        :return: The created specification.
        """
        return cls(
            name,
            Mesh(
                filename=filename,
                origin=(origin or HomogeneousTransformationMatrix()),
                scale=scale or Scale(),
                color=color or Color(),
            ).as_shape_collection(),
            child_specifications=(child_specifications or []),
            parent_T_self=(parent_T_self or HomogeneousTransformationMatrix()),
            connection_specification=connection_specification,
        )

    @classmethod
    def from_event(
        cls,
        name: str,
        event: Event,
        parent_T_self: HomogeneousTransformationMatrix | None = None,
        child_specifications: list[KinematicStructureEntitySpecification] | None = None,
        connection_specification: ConnectionSpecification | None = None,
    ) -> Self:
        """
        Specification whose shapes are the bounding boxes of a random event.

        This is the construction used by semantic annotations with composite geometry
        (hollow handles, container cases, walls minus apertures, ...).

        :param name: The name of the entity.
        :param event: The event describing the geometry, in the entity frame.
        :param parent_T_self: The default placement of the entity in its parent frame.
            Defaults to identity.
        :param child_specifications: Specifications spawned as kinematic children of the
            entity. Defaults to none.
        :param connection_specification: How the entity attaches to its parent. Defaults
            to a fixed connection.
        :return: The created specification.
        """
        # BoundingBoxCollection requires a reference frame, so the shapes are
        # built around a throwaway body and unbound again for the specification.
        anchor = Body(name=PrefixedName("spec_anchor"))
        return cls(
            name=name,
            shapes=BoundingBoxCollection.from_event(
                VolumetricBoundingBox, anchor, event
            )
            .as_shapes()
            .copy_without_reference_frame(),
            child_specifications=(child_specifications or []),
            parent_T_self=(parent_T_self or HomogeneousTransformationMatrix()),
            connection_specification=connection_specification,
        )

    @classmethod
    def from_3d_points(
        cls,
        name: str,
        points_3d: list[Point3],
        minimum_thickness: float = 0.005,
        singular_value_ratio_tolerance: float = 1e-7,
        parent_T_self: HomogeneousTransformationMatrix | None = None,
        child_specifications: list[KinematicStructureEntitySpecification] | None = None,
        connection_specification: ConnectionSpecification | None = None,
    ) -> Self:
        """
        Specification whose geometry is the convex hull of a point cloud.

        :param name: The name of the entity.
        :param points_3d: The points whose convex hull defines the geometry.
        :param minimum_thickness: Thickness added when the points are near-planar.
        :param singular_value_ratio_tolerance: Singular-value ratio tolerance for the
            planarity test.
        :param parent_T_self: The default placement of the entity in its parent frame.
            Defaults to identity.
        :param child_specifications: Specifications spawned as kinematic children of the
            entity. Defaults to none.
        :param connection_specification: How the entity attaches to its parent. Defaults
            to a fixed connection.
        :return: The created specification.
        """
        return cls(
            name=name,
            shapes=ShapeCollection(
                [
                    Mesh.from_3d_points(
                        points_3d,
                        minimum_thickness=minimum_thickness,
                        singular_value_ratio_tolerance=singular_value_ratio_tolerance,
                    )
                ]
            ).copy_without_reference_frame(),
            child_specifications=(child_specifications or []),
            parent_T_self=(parent_T_self or HomogeneousTransformationMatrix()),
            connection_specification=connection_specification,
        )


@dataclass
class BodySpecification(KinematicStructureEntitySpecification[Body]):
    """
    World-independent description of a
    :class:`~semantic_digital_twin.world_description.world_entity.Body`.

    Extends the kinematic-structure-entity specification with body-only properties: inertial
    parameters and a separate visual shape collection.
    """

    inertial: Inertial | None = None
    """
    Inertia properties of created bodies.

    None means the Body default.
    """

    visual_shapes: ShapeCollection | None = None
    """
    Visual shapes when they differ from `shapes`.

    None shares `shapes` for both collision and visual (one collection); an empty list
    means no visual geometry.
    """

    def to_domain_object(self, name: str | None = None) -> Body:
        """
        Create a new, world-independent body from this specification.

        :param name: Optional name override, e.g. for spawning multiple bodies from the
            same specification.
        :return: The created body.
        """
        body = Body.from_shape_collection(
            self._resolved_name(name),
            self.shapes.copy_without_reference_frame(),
            visuals_shape_collection=(
                self.visual_shapes.copy_without_reference_frame()
                if self.visual_shapes is not None
                else None
            ),
        )
        if self.inertial is not None:
            body.inertial = deepcopy(self.inertial)
        return body


@dataclass
class RegionSpecification(KinematicStructureEntitySpecification[Region]):
    """
    World-independent description of a
    :class:`~semantic_digital_twin.world_description.world_entity.Region`.

    Carries no fields beyond the base kinematic-structure-entity specification; it only
    binds the materialized domain-object type to :class:`Region`.
    """


# %% semantic annotation specifications


@dataclass
class PartSpecificationBinding:
    """
    Nested annotation parts together with the part-whole relationship field they fill.

    Parts are held in a list of bindings rather than keyed by field name in a mapping,
    so that the nesting is part of the persisted specification instead of being dropped
    on the way to the database.
    """

    field_name: str
    """
    The name of the part-whole relationship field the parts are mounted onto.
    """

    specifications: list[SemanticAnnotationWithRootSpecification] = field(
        default_factory=list
    )
    """
    The part specifications spawned and mounted onto the field.
    """


@dataclass
class SemanticAnnotationWithRootSpecification(SpawnSpecification[TSemanticAnnotation]):
    """
    World-independent description of a semantic annotation rooted in a single kinematic
    structure entity.

    The annotation's root entity is what attaches to the parent, so its parent
    connection lives on ``root_specification.connection_specification`` and nowhere
    else. Leaving it unset falls back to the annotation type's own
    :meth:`~semantic_digital_twin.semantic_annotations.mixins.HasRootKinematicStructureEntity.parent_connection_specification`.

    ..note:: There is deliberately no way to pass loose connection parameters here. Each
        connection family carries exactly the parameters it uses, so an inapplicable
        parameter is a construction error rather than a silently ignored field.
    """

    semantic_annotation_type: Type[TSemanticAnnotation]
    """
    The type of the semantic annotation that is a subclass of
    HasRootKinematicStructureEntity.
    """

    root_specification: KinematicStructureEntitySpecification
    """
    The specification of the root kinematic structure entity of the annotation.

    Its :attr:`connection_specification` is the annotation's parent connection.
    """

    annotation_kwargs: dict[str, Any] = field(default_factory=dict)
    """
    Inert keyword arguments passed straight to the annotation constructor, keyed by
    constructor field name.

    Nested annotation parts do not belong here; use :attr:`part_bindings`.

    .. note:: These values are of arbitrary type and are therefore not persisted with the
        specification.
    """

    part_bindings: list[PartSpecificationBinding] = field(default_factory=list)
    """
    Nested annotation parts, each bound to the part-whole relationship field it fills.

    Each part is spawned during :meth:`spawn` and mounted via the annotation's
    :meth:`PartWholeRelationship.add`.
    """

    def __post_init__(self):
        """
        Validate the annotation kwargs and part bindings so misuse fails fast, before
        any world mutation.
        """
        self._validate_annotation_kwargs()
        self._validate_part_bindings(self.semantic_annotation_type)

    def spawn(
        self,
        world: World,
        name: str | None = None,
        parent: KinematicStructureEntity | None = None,
        parent_T_self: HomogeneousTransformationMatrix | None = None,
    ) -> TSemanticAnnotation:
        """
        Materialize the annotation in ``world``: spawn its root entity, attach it to
        ``parent``, register the annotation, and spawn its geometry children and mounted
        part specifications.

        The root's connection is the root specification's
        :attr:`connection_specification`, falling back to the annotation type's own
        :meth:`~semantic_digital_twin.semantic_annotations.mixins.HasRootKinematicStructureEntity.parent_connection_specification`
        when it is unset.

        :param world: The world the annotation, its root and its parts are added to.
        :param name: Overrides the specification's own name. If None, the spec's name is
            used.
        :param parent: The entity to attach the root to. If None, ``world.root`` is
            used.
        :param parent_T_self: Overrides the root specification's stored default pose.
        :return: The materialized semantic annotation.
        """
        root_entity = self.root_specification.to_domain_object(name or self.name)

        instance = self.semantic_annotation_type(
            name=self._resolved_name(name), root=root_entity, **self.annotation_kwargs
        )

        connection_specification = (
            self.root_specification.connection_specification
            or self.semantic_annotation_type.parent_connection_specification()
        )

        with world.modify_world():
            self.root_specification.attach_and_spawn_children(
                world, root_entity, connection_specification, parent, parent_T_self
            )
            world.add_semantic_annotation(instance)
            self._mount_part_specifications(world, instance, root_entity)

        return instance

    def _validate_annotation_kwargs(self) -> None:
        """
        Validate that :attr:`annotation_kwargs` carries no part-whole relationship
        field.

        Such fields must be supplied via :attr:`part_bindings` so they are spawned and
        mounted.

        :raises PartWholeFieldInAnnotationKwargs: If a key names a part-whole
            relationship field.
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

    def _validate_part_bindings(
        self, annotation_type: type[TSemanticAnnotation]
    ) -> None:
        """
        Validate that every binding targets a part-whole relationship field of the
        annotation and that only to-many fields are given more than one part.

        :param annotation_type: The annotation type whose part-whole fields are
            validated against.
        :raises UnknownPartWholeRelationshipField: If a binding does not name a part-
            whole relationship field.
        :raises PartWholeCardinalityError: If several parts target a singular field.
        """
        part_whole_fields_by_name = self._part_whole_fields_by_name()
        for binding in self.part_bindings:
            wrapped_field = part_whole_fields_by_name.get(binding.field_name)
            if wrapped_field is None:
                raise UnknownPartWholeRelationshipField(
                    annotation=annotation_type,
                    field_name=binding.field_name,
                    available_fields=list(part_whole_fields_by_name),
                )
            if (
                len(binding.specifications) > 1
                and not wrapped_field.is_many_to_many_relationship
            ):
                raise PartWholeCardinalityError(
                    annotation_type_name=self.semantic_annotation_type.__name__,
                    field_name=binding.field_name,
                )

    def _part_whole_fields_by_name(self) -> dict[str, WrappedField]:
        """
        The annotation type's part-whole relationship fields, keyed by field name.

        :return: The wrapped part-whole relationship fields, keyed by field name.
        """
        return {
            wrapped_field.name: wrapped_field
            for wrapped_field in WrappedClass(
                self.semantic_annotation_type
            ).fields_with_metadata(IsPartWholeRelationship)
        }

    def _mount_part_specifications(
        self,
        world: World,
        instance: PartWholeRelationship,
        root_entity: KinematicStructureEntity,
    ) -> None:
        """
        Spawn each nested part and mount it onto ``instance`` via the part-whole
        :meth:`PartWholeRelationship.add`, into the field its binding names.

        .. note:: Assumes :meth:`_validate_part_bindings` has already run.

        :param world: The world the parts are added to.
        :param instance: The annotation the spawned parts are mounted onto.
        :param root_entity: The annotation's root, which the parts are attached to.
        """
        for binding in self.part_bindings:
            for part_specification in binding.specifications:
                part = part_specification.spawn(world, parent=root_entity)
                instance.add(part, field_name=binding.field_name)


# %% robot specifications


@dataclass
class RobotSpecification:
    """
    World-independent description of a robot placed into a world: which robot, where its
    localization frame sits, and where the robot starts within it.

    Materialized via :meth:`spawn`, which merges the robot as ``world.root -> odom ->
    drive -> robot``.
    """

    semantic_annotation_type: Type[AbstractRobot]
    """
    The robot to merge into the world.
    """

    world_T_odom: HomogeneousTransformationMatrix | None = None
    """
    The localization pose of the robot's ``odom`` in the ``world.root`` frame.

    If None, identity is used.
    """

    odom_T_robot_start: HomogeneousTransformationMatrix | None = None
    """
    The start pose of the robot in its ``odom`` frame.

    If None, identity is used.
    """

    def spawn(self, world: World) -> AbstractRobot:
        """
        Parse the robot from its own description and merge it into ``world`` as
        ``world.root -> odom -> connection -> robot``.

        The connection attaching the robot to its ``odom`` is the drive declared by the
        robot's mobile base, or a fixed connection when the robot has no mobile base. An
        active drive is marked as controlled; the localization and start poses are
        applied afterwards.

        The robot is annotated while it still owns the world it was parsed into, so that
        the annotation's name-based lookups cannot be confused by an equally named joint
        of a robot already present in ``world``.

        :param world: The world the robot is merged into.
        :return: The semantic annotation of the merged robot.
        """
        connection_type = self.semantic_annotation_type.get_drive_connection_type()
        is_active = issubclass(connection_type, ActiveConnection)

        robot_world = URDFParser.from_file(
            self.semantic_annotation_type.get_ros_file_path()
        ).parse()
        robot_id = self.semantic_annotation_type.from_world(robot_world).id

        with world.modify_world():
            odom_body = self._create_odom_body()
            root_C_odom = Connection6DoF.create_with_dofs(
                world=world, parent=cast(Body, world.root), child=odom_body
            )
            world.add_connection(root_C_odom)

            # A fixed connection has no DoFs, so its start pose must be set at creation;
            # an active drive carries it as DoF state applied after the block.
            odom_C_robot = connection_type.create_with_dofs(
                world=world,
                parent=odom_body,
                child=cast(Body, robot_world.root),
                parent_T_connection_expression=(
                    None if is_active else self.odom_T_robot_start
                ),
            )
            world.merge_world(robot_world, root_connection=odom_C_robot)
            if is_active:
                odom_C_robot.has_hardware_interface = True

        # Poses touch DoF state, so they are set after the modification block.
        if self.world_T_odom is not None:
            root_C_odom.origin = self.world_T_odom.copy_with_new_reference_frames(
                new_reference_frame=world.root, new_child_frame=odom_body
            )
        if is_active and self.odom_T_robot_start is not None:
            odom_C_robot.origin = self.odom_T_robot_start

        return cast("AbstractRobot", world.get_semantic_annotation_by_id(robot_id))

    @staticmethod
    def _create_odom_body() -> Body:
        """
        Create the localization body of a single robot.

        Body names are not unique across a world, so the body's own identifier prefixes
        its name. Identifiers are unique even across processes, which keeps the odom
        bodies of several robots distinguishable.

        :return: The created odom body.
        """
        identifier = uuid4()
        return Body(
            name=PrefixedName(name="odom", prefix=str(identifier)), id=identifier
        )


# %% world specifications


@dataclass
class WorldSpecification:
    """
    World-independent description of a world: an environment, the robots in it, and
    objects around them.

    The environment is described by the parser that builds it (obtained from a model file
    with :meth:`from_urdf`, :meth:`from_mjcf` or :meth:`from_gazebo`). Applying it
    (:meth:`to_domain_object`) parses the environment anew, merges every robot into it,
    then spawns all starting objects, and returns the augmented environment world.
    """

    world_parser: WorldModelParser | None = None
    """
    The parser building the environment the robots and starting objects are added to.

    ``None`` describes an environment holding nothing but its root body.
    """

    robots: list[RobotSpecification] = field(default_factory=list)
    """
    The robots merged into the environment, each with its own localization and start
    pose.
    """

    objects: list[SpawnSpecification] = field(default_factory=list)
    """
    Specifications spawned relative to the world root once the robots are in place.
    """

    @classmethod
    def from_urdf(
        cls,
        file_path: str,
        *,
        prefix: str | None = None,
        path_resolver: PathResolver | None = None,
        robots: list[RobotSpecification] | None = None,
        objects: list[SpawnSpecification] | None = None,
    ) -> Self:
        """
        Build a specification whose environment is parsed from a URDF file.

        :param file_path: Path to the environment URDF. This is never a robot
            description; robots are supplied through ``robots``.
        :param prefix: Optional name prefix for the parsed environment.
        :param path_resolver: Resolver for mesh/package paths referenced by the URDF.
        :param robots: The robots merged into the environment.
        :param objects: Specifications spawned once the robots are in place.
        :return: The created specification.
        """
        world_parser = URDFParser.from_file(
            file_path, prefix=prefix, path_resolver=path_resolver
        )
        return cls(
            world_parser=world_parser,
            robots=robots or [],
            objects=objects or [],
        )

    @classmethod
    def from_mjcf(
        cls,
        file_path: str,
        *,
        prefix: str | None = None,
        mimic_joints: dict[str, str] | None = None,
        robots: list[RobotSpecification] | None = None,
        objects: list[SpawnSpecification] | None = None,
    ) -> Self:
        """
        Build a specification whose environment is parsed from an MJCF (MuJoCo XML)
        file.

        :param file_path: Path to the environment MJCF. This is never a robot
            description; robots are supplied through ``robots``.
        :param prefix: Optional name prefix for the parsed environment.
        :param mimic_joints: Mapping of joint names to the joints they mimic.
        :param robots: The robots merged into the environment.
        :param objects: Specifications spawned once the robots are in place.
        :return: The created specification.
        """
        from semantic_digital_twin.adapters.mjcf import MJCFParser

        world_parser = MJCFParser(
            file_path=file_path,
            mimic_joints=mimic_joints or {},
            prefix=prefix,
        )
        return cls(
            world_parser=world_parser,
            robots=robots or [],
            objects=objects or [],
        )

    @classmethod
    def from_gazebo(
        cls,
        file_path: str,
        *,
        prefix: str | None = None,
        path_resolver: PathResolver | None = None,
        robots: list[RobotSpecification] | None = None,
        objects: list[SpawnSpecification] | None = None,
    ) -> Self:
        """
        Build a specification whose environment is parsed from a Gazebo SDF world or
        model file.

        :param file_path: Path to the environment world or model file. This is never a
            robot description; robots are supplied through ``robots``.
        :param prefix: Optional name prefix for the parsed environment.
        :param path_resolver: Resolver for the ``model://`` and mesh URIs the file
            references. Defaults to one that searches next to the file.
        :param robots: The robots merged into the environment.
        :param objects: Specifications spawned once the robots are in place.
        :return: The created specification.
        """
        from semantic_digital_twin.adapters.gazebo import GazeboParser

        world_parser = GazeboParser.from_file(
            file_path, prefix=prefix, path_resolver=path_resolver
        )
        return cls(
            world_parser=world_parser,
            robots=robots or [],
            objects=objects or [],
        )

    def to_domain_object(self) -> World:
        """
        Materialize a new World from this specification.

        The environment is parsed anew, so the method can be applied repeatedly and no
        two results share an entity identifier. Every robot is merged into the world
        first, then all ``objects`` are spawned relative to the world root.

        :return: The augmented environment world.
        """
        if self.world_parser is not None:
            world = self.world_parser.parse()
        else:
            world = World.create_with_root_body()
        for robot_specification in self.robots:
            robot_specification.spawn(world)

        for object_specification in self.objects:
            object_specification.spawn(world)

        return world
