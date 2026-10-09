"""
Specifications of kinematic structure entities, such as bodies and regions.
"""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass, field

from typing_extensions import Self, TypeVar

from krrood.patterns.subclass_safe_generic import SubClassSafeGeneric
from krrood.utils import get_generic_type_parameters
from random_events.product_algebra import Event
from semantic_digital_twin.datastructures.prefixed_name import PrefixedName
from semantic_digital_twin.spatial_types import HomogeneousTransformationMatrix, Point3
from semantic_digital_twin.specifications.base import SpawnSpecification
from semantic_digital_twin.specifications.connections import (
    ConnectionSpecification,
    FixedConnectionSpecification,
)
from semantic_digital_twin.world import World
from semantic_digital_twin.world_description.geometry import (
    Box,
    Color,
    Cylinder,
    Mesh,
    Scale,
    Sphere,
    VolumetricBoundingBox,
)
from semantic_digital_twin.world_description.inertial_properties import Inertial
from semantic_digital_twin.world_description.shape_collection import (
    BoundingBoxCollection,
    ShapeCollection,
)
from semantic_digital_twin.world_description.world_entity import (
    Body,
    KinematicStructureEntity,
    Region,
)

# %% specification type parameters
TKinematicStructureEntity = TypeVar(
    "TKinematicStructureEntity", bound=KinematicStructureEntity
)


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

    It is where the entity sits while its connection is at its zero position, also when
    the connection carries a ``connection_T_child`` offset. Identity by default.
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
                parent_T_connection=connection_specification.parent_T_connection_for_child_at(
                    parent_T_self or self.parent_T_self
                ),
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
