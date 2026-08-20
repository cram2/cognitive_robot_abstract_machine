from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Tuple

import numpy as np
import trimesh
from krrood.class_diagrams.class_diagram import WrappedClass
from krrood.entity_query_language.factories import variable_from, entity, variable, an
from krrood.ormatic.utils import classproperty
from krrood.patterns.subclass_safe_generic import SubClassSafeGeneric
from probabilistic_model.distributions.gaussian import GaussianDistribution
from probabilistic_model.distributions.helper import make_dirac
from probabilistic_model.probabilistic_circuit.rx.helper import (
    uniform_measure_of_event,
)
from probabilistic_model.probabilistic_circuit.rx.probabilistic_circuit import (
    ProbabilisticCircuit,
    ProductUnit,
    SumUnit,
    leaf,
)
from random_events.product_algebra import Event
from random_events.set import Set as EventSet
from random_events.variable import Symbolic
from typing_extensions import (
    TYPE_CHECKING,
    Generic,
    List,
    Optional,
    Self,
    Set,
    Type,
    TypeVar,
)

from semantic_digital_twin.api import (
    BodySpecification,
    ConnectionSpecification,
    FixedConnectionSpecification,
    PartSpecificationBinding,
    RegionSpecification,
    SemanticAnnotationWithRootSpecification,
    KinematicStructureEntitySpecification,
)
from semantic_digital_twin.datastructures.prefixed_name import PrefixedName
from semantic_digital_twin.datastructures.variables import SpatialVariables
from semantic_digital_twin.exceptions import (
    AmbiguousPart,
    CannotBeAPartOf,
    UnknownPartWholeRelationshipField,
)
from semantic_digital_twin.reasoning.predicates import is_supported_by
from semantic_digital_twin.semantic_annotations.part_whole import (
    IsPartWholeRelationship,
)
from semantic_digital_twin.spatial_types import (
    Point3,
    HomogeneousTransformationMatrix,
    Vector3,
)
from semantic_digital_twin.world_description.connections import (
    FixedConnection,
)
from semantic_digital_twin.world_description.geometry import Scale
from semantic_digital_twin.world_description.shape_collection import (
    BoundingBoxCollection,
)
from semantic_digital_twin.world_description.world_entity import (
    SemanticAnnotation,
    Body,
    Region,
    KinematicStructureEntity,
    Connection,
)
from semantic_digital_twin.world_description.world_modification import (
    synchronized_attribute_modification,
)

if TYPE_CHECKING:
    from semantic_digital_twin.world import World
    from semantic_digital_twin.semantic_annotations.semantic_annotations import (
        Drawer,
        Door,
        Handle,
        Aperture,
        MechanicalJoint,
        Leg,
        Sink,
        ShelfLayer,
    )
    from semantic_digital_twin.world import World


@dataclass(eq=False)
class IsPerceivable:
    """
    A mixin class for semantic annotations that can be perceived.
    """

    class_label: Optional[str] = field(default=None, kw_only=True)
    """
    The exact class label of the perceived object.
    """


TKinematicStructureEntity = TypeVar(
    "TKinematicStructureEntity", bound=KinematicStructureEntity
)


@dataclass(eq=False)
class HasRootKinematicStructureEntity(
    SemanticAnnotation, Generic[TKinematicStructureEntity], SubClassSafeGeneric
):
    """
    Base class for shared method for HasRootBody and HasRootRegion.

    Building a specification takes two steps:
    :meth:`get_default_root_kinematic_structure_entity_specification` describes the root entity's geometry,
    and :meth:`get_annotation_specification` wraps that root into the spawnable
    annotation specification.

    .. code-block:: python

        Handle.get_annotation_specification(
            "handle",
            Handle.get_default_root_kinematic_structure_entity_specification(
                scale=Scale(0.1, 0.05, 0.05), thickness=0.01
            ),
        )

    Every geometry parameter belongs to the first step, so a type's geometry is
    described in exactly one place.
    """

    root: TKinematicStructureEntity = field(kw_only=True)
    """
    The root kinematic structure entity of the semantic annotation.
    """

    @property
    def scale(self) -> Scale:
        return Scale(
            *(self.root.combined_mesh.bounds[1] - self.root.combined_mesh.bounds[0])
        )

    @property
    def min_max_points(self) -> Tuple[Point3, Point3]:
        min = Point3.from_iterable(self.root.combined_mesh.bounds[0])
        max = Point3.from_iterable(self.root.combined_mesh.bounds[1])
        return min, max

    def __hash__(self):
        return hash((self.__class__, self.root))

    @classmethod
    def parent_connection_specification(cls) -> ConnectionSpecification:
        """
        Build the connection that attaches this annotation's root entity to its parent.

        Annotation types whose connection takes parameters override this with their own
        concrete signature, so callers get the parameters that family actually uses and
        nothing else (e.g. :meth:`Slider.parent_connection_specification` takes an
        ``axis``, this one takes none).

        :meth:`get_annotation_specification` calls this to fill in the connection
        when the caller supplies none. To parameterize it, call this method yourself and
        hand the result to that factory.

        .. warning:: Every override must stay callable with no arguments, since that is
            how the default is resolved. Add parameters only with defaults.

        .. note:: Currently always fixed, except for sliders and hinges. This may change
            in the future, so override where needed.

        :return: A freshly built connection specification.
        """
        return FixedConnectionSpecification()

    @classmethod
    @abstractmethod
    def get_default_root_kinematic_structure_entity_specification(
        cls,
        name: str | None = None,
        scale: Optional[Scale] = None,
        connection_specification: Optional[ConnectionSpecification] = None,
    ) -> KinematicStructureEntitySpecification:
        """
        Build this type's default root entity specification, the geometry only.

        Implemented once per root entity kind (:class:`HasRootBody` yields a body
        specification, :class:`HasRootRegion` a region one) and overridden by types
        whose geometry takes further parameters, such as a handle's ``thickness`` or a
        case's ``wall_thickness``. Those parameters live here and nowhere else, so a
        type's geometry is described in a single place.

        .. warning:: The result carries no annotation. Spawning it directly puts a bare
            body or region into the world; pass it to
            :meth:`get_annotation_specification` to obtain the annotation.

        :param name: The name of entities created from the specification. ``None``
            leaves naming to the spawning annotation, which overrides it with its own
            name.
        :param scale: The scale the geometry is generated from.
        :param connection_specification: Connection attaching the entity to its parent.
            ``None`` defers to the annotation type's default at spawn time.
        :return: The root entity specification.
        """

    @classmethod
    def get_annotation_specification(
        cls,
        name: str,
        root_specification: KinematicStructureEntitySpecification,
        *,
        parent_connection_specification: Optional[ConnectionSpecification] = None,
        annotation_kwargs: Optional[dict] = None,
        part_specifications: Optional[dict] = None,
    ) -> SemanticAnnotationWithRootSpecification[Self]:
        """
        Wrap a root entity specification, typically from
        :meth:`get_default_root_kinematic_structure_entity_specification`, into the spawnable annotation
        specification.

        The root geometry is always supplied by the caller. That builder owns every
        geometry parameter (a scale, a handle's ``thickness``, a case's
        ``wall_thickness``), so geometry is described in exactly one place.

        :param name: The name of the annotation and its root entity.
        :param root_specification: The specification of the annotation's root entity.
        :param parent_connection_specification: Connection attaching the root to its
            parent. When omitted, :meth:`parent_connection_specification` supplies this
            type's default.
        :param annotation_kwargs: Inert keyword arguments for the annotation
            constructor.
        :param part_specifications: Nested annotation parts keyed by part-whole
            relationship field name.
        :return: The annotation specification.
        """
        if parent_connection_specification is not None:
            root_specification.connection_specification = (
                parent_connection_specification
            )
        return SemanticAnnotationWithRootSpecification(
            name=name,
            semantic_annotation_type=cls,
            root_specification=root_specification,
            annotation_kwargs=(
                annotation_kwargs if annotation_kwargs is not None else {}
            ),
            part_bindings=cls._to_part_bindings(part_specifications),
        )

    @staticmethod
    def _to_part_bindings(
        part_specifications: Optional[dict],
    ) -> list[PartSpecificationBinding]:
        """
        Convert the keyword-friendly mapping of parts into the bindings the
        specification stores.

        :param part_specifications: Nested annotation parts keyed by part-whole
            relationship field name, each a single specification or a list of them.
        :return: One binding per named field.
        """
        if part_specifications is None:
            return []
        return [
            PartSpecificationBinding(
                field_name=field_name,
                specifications=(list(value) if isinstance(value, list) else [value]),
            )
            for field_name, value in part_specifications.items()
        ]

    def _mount_strategy(
        self,
        main_has_root_body_annotation: HasRootBody,
        relationship: IsPartWholeRelationship,
    ) -> None:
        """
        Realize the relationship between this annotation (as a part) and the
        ``main_has_root_body_annotation`` (the whole) in the kinematic structure. The
        default is to become a kinematic child of the whole; parts with a different
        strategy (e.g. mechanical joints that re-parent the whole, apertures that cut
        it) override this.

        :param main_has_root_body_annotation: The annotation (the whole) this one is
            being added to as a part.
        :param relationship: The metadata of the part-whole relationship field being
            mounted into, describing how the mount affects the whole.
        """
        main_has_root_body_annotation._world.move_branch(
            self.root, main_has_root_body_annotation.root, True
        )

    @property
    def global_transform(self) -> HomogeneousTransformationMatrix:
        return self.root.global_transform

    @property
    def connections(self) -> list[Connection]:
        return self._world.get_connections_of_branch(self.root)

    def _kinematic_structure_entities(
        self, visited: Set[int]
    ) -> list[KinematicStructureEntity]:
        if id(self) in visited:
            return []
        visited.add(id(self))
        return self._world.get_kinematic_structure_entities_of_branch(self.root)


TBody = TypeVar("TBody", bound=Body)


@dataclass(eq=False)
class HasRootBody(HasRootKinematicStructureEntity[TBody]):
    """
    Abstract base class for all objects which have a unambiguous root reference frame.

    Each semantic annotation refers to a single Body. Each subclass automatically
    derives a MatchRule from its own class name and the names of its ancestors.
     This makes specialized subclasses naturally more specific than their bases.
    """

    @classmethod
    def create_with_new_body_in_world(
        cls,
        name: str,
        world: World,
        world_root_T_self: Optional[HomogeneousTransformationMatrix] = None,
        parent_connection_specification: Optional[ConnectionSpecification] = None,
        scale: Optional[Scale] = None,
    ) -> Self:
        """
        Create a new semantic annotation with a new body in the given world.

        :param name: The name of the semantic annotation.
        :param world: The world to add the annotation and body to.
        :param world_root_T_self: The initial pose of the body in the world root frame.
        :param parent_connection_specification: Connection attaching the body to the
            world root. When omitted, this type's default parent connection applies.
        :param scale: The scale used to generate the geometry of the body. When omitted,
            the type's default geometry scale applies.
        :return: The created semantic annotation instance.
        """
        return cls.get_annotation_specification(
            name,
            cls.get_default_root_kinematic_structure_entity_specification(scale=scale),
            parent_connection_specification=parent_connection_specification,
        ).spawn(world, parent_T_self=world_root_T_self)

    @classmethod
    def get_default_root_kinematic_structure_entity_specification(
        cls,
        name: str | None = None,
        scale: Optional[Scale] = None,
        connection_specification: Optional[ConnectionSpecification] = None,
    ) -> BodySpecification:
        """
        Build the default body specification whose geometry matches what
        :meth:`create_with_new_body_in_world` generates from ``scale``.

        This is the geometry-extraction counterpart of the factory: instead of
        mutating a world, it returns a reusable, world-independent specification.

        .. warning:: The result carries no annotation. Spawning it directly puts a bare
            body into the world; pass it to :meth:`get_annotation_specification` to
            obtain the annotation.

        :param name: The name of bodies created from the specification. ``None`` leaves
            naming to the spawning annotation, which overrides it with its own name.
        :param scale: The scale used to generate the box geometry. ``None`` yields a
            geometry-less specification, matching a factory call without a scale.
        :param connection_specification: Connection attaching the entity to its parent.
            ``None`` defers to the annotation type's default at spawn time.
        :return: A body specification with a single solid box derived from ``scale``.
        """
        if scale is None:
            return BodySpecification(
                name=name, connection_specification=connection_specification
            )
        return BodySpecification.from_event(
            name,
            scale.to_simple_event().as_composite_set(),
            connection_specification=connection_specification,
        )


TRegion = TypeVar("TRegion", bound=Region)


@dataclass(eq=False)
class HasRootRegion(HasRootKinematicStructureEntity[TRegion]):
    """
    A mixin class for semantic annotations that have a region.
    """

    @classmethod
    def create_with_new_region_in_world(
        cls,
        name: str,
        world: World,
        world_root_T_self: Optional[HomogeneousTransformationMatrix] = None,
        parent_connection_specification: Optional[ConnectionSpecification] = None,
        scale: Optional[Scale] = None,
    ) -> Self:
        """
        Create a new semantic annotation with a new region in the given world.

        :param name: The name of the semantic annotation.
        :param world: The world to add the annotation and region to.
        :param world_root_T_self: The initial pose of the region in the world root
            frame.
        :param parent_connection_specification: Connection attaching the region to the
            world root. When omitted, this type's default parent connection applies.
        :param scale: The scale used to generate the region area geometry.
        :return: The created semantic annotation instance.
        """
        return cls.get_annotation_specification(
            name,
            cls.get_default_root_kinematic_structure_entity_specification(scale=scale),
            parent_connection_specification=parent_connection_specification,
        ).spawn(world, parent_T_self=world_root_T_self)

    @classmethod
    def get_default_root_kinematic_structure_entity_specification(
        cls,
        name: str | None = None,
        scale: Optional[Scale] = None,
        connection_specification: Optional[ConnectionSpecification] = None,
    ) -> RegionSpecification:
        """
        Build the default region specification whose geometry matches what
        :meth:`create_with_new_region_in_world` generates.

        .. warning:: The result carries no annotation. Spawning it directly puts a bare
            region into the world; pass it to :meth:`get_annotation_specification` to
            obtain the annotation.

        :param name: The name of regions created from the specification. ``None`` leaves
            naming to the spawning annotation, which overrides it with its own name.
        :param scale: The scale used to generate the region area geometry. ``None`` yields a
            geometry-less specification, matching the bare region of the base factory.
        :param connection_specification: Connection attaching the entity to its parent.
            ``None`` defers to the annotation type's default at spawn time.
        :return: A region specification.
        """
        if scale is None:
            return RegionSpecification(
                name=name, connection_specification=connection_specification
            )
        return RegionSpecification.from_event(
            name,
            scale.to_simple_event().as_composite_set(),
            connection_specification=connection_specification,
        )


@dataclass(eq=False)
class PartWholeRelationship(HasRootKinematicStructureEntity):
    """
    Base for annotations that have structural *parts* (the part-whole relation).

    Each part mixin (``HasHandle``, ``HasDoors``, ...) declares a typed part-whole relationship
    field. The unified :meth:`add` routes a part to the field whose element type matches it and lets
    the part mount itself (:meth:`HasRootKinematicStructureEntity._mount_strategy`).
    """

    @synchronized_attribute_modification
    def add(
        self, part: HasRootKinematicStructureEntity, *, field_name: str = ""
    ) -> None:
        """
        Add ``part`` as a structural part, routing it to the matching part-whole
        relationship field by type.

        :param part: The part to add.
        :param field_name: Optional name of the target part-whole relationship field.
            When given, only that field is considered (and ``part`` must still match its
            element type), which resolves the ambiguity when ``type(part)`` matches
            several fields. When empty (default), the field is resolved by type alone.
        :raises UnknownPartWholeRelationshipField: If ``field_name`` is given but is not
            a part-whole relationship field of this annotation.
        :raises CannotBeAPartOf: If no part-whole relationship field of this annotation
            accepts ``type(part)``.
        :raises AmbiguousPart: If ``type(part)`` matches more than one part-whole
            relationship field.
        """
        candidate_fields = WrappedClass(type(self)).fields_with_metadata(
            IsPartWholeRelationship
        )
        if field_name:
            named_fields = [
                wrapped_part_whole_relationship_field
                for wrapped_part_whole_relationship_field in candidate_fields
                if wrapped_part_whole_relationship_field.field.name == field_name
            ]
            if not named_fields:
                raise UnknownPartWholeRelationshipField(
                    type(self),
                    field_name,
                    [
                        wrapped_part_whole_relationship_field.field.name
                        for wrapped_part_whole_relationship_field in candidate_fields
                    ],
                )
            candidate_fields = named_fields
        matches = [
            wrapped_part_whole_relationship_field
            for wrapped_part_whole_relationship_field in candidate_fields
            if isinstance(part, wrapped_part_whole_relationship_field.type_endpoint)
        ]
        if not matches:
            raise CannotBeAPartOf(self, part)
        if len(matches) > 1:
            raise AmbiguousPart(self, part, [match.field for match in matches])

        [match] = matches
        part._mount_strategy(self, IsPartWholeRelationship.of_wrapped_field(match))
        if match.is_many_to_many_relationship:
            getattr(self, match.field.name).append(part)
        else:
            setattr(self, match.field.name, part)


@dataclass(eq=False)
class HasApertures(HasRootBody, PartWholeRelationship):
    """
    A mixin class for semantic annotations that have apertures.

    An aperture is an opening through this annotation's material, so mounting one
    removes its volume from the annotation's collision and visual geometry.
    """

    apertures: List[Aperture] = field(
        default_factory=list,
        hash=False,
        kw_only=True,
        metadata=IsPartWholeRelationship(
            removes_part_geometry_from_whole=True
        ).as_dict(),
    )
    """
    The apertures of the semantic annotation.
    """


@dataclass(eq=False)
class HasMechanicalJoint(HasRootBody, PartWholeRelationship):
    """
    A mixin class for semantic annotations that have mechanical joints.
    """

    mechanical_joint: Optional[MechanicalJoint] = field(
        default=None,
        metadata=IsPartWholeRelationship().as_dict(),
    )
    """
    The mechanical joint of the semantic annotation.
    """

    def _mount_strategy(
        self,
        main_has_root_body_annotation: HasRootBody,
        relationship: IsPartWholeRelationship,
    ) -> None:
        """
        Mount this annotation onto the whole through its mechanical joint, so the joint
        keeps carrying it.

        Moving this annotation on its own would pull it out from under its joint and
        leave a door or drawer rigidly attached to the whole, unable to move.

        :param main_has_root_body_annotation: The annotation (the whole) this one is
            being added to as a part.
        :param relationship: The metadata of the part-whole relationship field being
            mounted into, describing how the mount affects the whole.
        """
        if (
            self.mechanical_joint is None
            or self.root.parent_kinematic_structure_entity
            is not self.mechanical_joint.root
        ):
            super()._mount_strategy(main_has_root_body_annotation, relationship)
            return

        main_has_root_body_annotation._world.move_branch(
            self.mechanical_joint.root,
            main_has_root_body_annotation.root,
            enable_unsafe_inside_world_block=True,
        )

    def _kinematic_structure_entities(
        self, visited: Set[int]
    ) -> list[KinematicStructureEntity]:
        if id(self) in visited:
            return []
        visited.add(id(self))
        kinematic_structure_entities = (
            self._world.get_kinematic_structure_entities_of_branch(self.root)
        )
        if self.mechanical_joint is not None:
            kinematic_structure_entities.append(self.mechanical_joint.root)
        return kinematic_structure_entities

    def create_default_mechanical_joint(
        self, mechanical_joint_type: Type[MechanicalJoint]
    ) -> None:
        """
        Give this annotation a mechanical joint of ``mechanical_joint_type`` when its
        root is wired straight to its parent with a connection of the same kind that
        joint type provides, and no mechanical joint carries it yet.

        Formats like URDF often attach a door or drawer to its cabinet with a bare
        active connection (e.g. revolute for a door, prismatic for a drawer) and no
        dedicated joint body. This inserts one, carrying over the axis, multiplier,
        offset and limits of the existing connection, so :attr:`mechanical_joint`
        reflects the joint that already moves it.

        :param mechanical_joint_type: The mechanical joint type to create, e.g.
            ``Hinge`` for a door or ``Slider`` for a drawer.
        """
        if self.mechanical_joint is not None:
            return
        connection = self.root.parent_connection
        connection_type = (
            mechanical_joint_type.parent_connection_specification().connection_type
        )
        if not isinstance(connection, connection_type):
            return
        joint = mechanical_joint_type.create_with_new_body_in_world(
            name=f"{self.root.name.name}_{mechanical_joint_type.__name__.lower()}",
            world=self._world,
            world_root_T_self=self.root.global_transform,
            parent_connection_specification=mechanical_joint_type.parent_connection_specification(
                axis=connection.axis,
                multiplier=connection.multiplier,
                offset=connection.offset,
                dof_limits=connection.raw_dof.limits,
            ),
        )
        self.add(joint)


@dataclass(eq=False)
class HasDrawers(PartWholeRelationship):
    """
    A mixin class for semantic annotations that have drawers.
    """

    drawers: List[Drawer] = field(
        default_factory=list,
        hash=False,
        kw_only=True,
        metadata=IsPartWholeRelationship().as_dict(),
    )
    """
    The drawers of the semantic annotation.
    """


@dataclass(eq=False)
class HasShelfLayers(PartWholeRelationship):
    """
    A mixin class for semantic annotations that have shelf layers.
    """

    shelf_layers: List[ShelfLayer] = field(
        default_factory=list,
        hash=False,
        kw_only=True,
        metadata=IsPartWholeRelationship().as_dict(),
    )
    """
    The shelf layers of the semantic annotation.
    """


@dataclass(eq=False)
class HasDoors(PartWholeRelationship):
    """
    A mixin class for semantic annotations that have doors.
    """

    doors: List[Door] = field(
        default_factory=list,
        hash=False,
        kw_only=True,
        metadata=IsPartWholeRelationship().as_dict(),
    )
    """
    The doors of the semantic annotation.
    """


@dataclass(eq=False)
class HasHandle(HasRootBody, PartWholeRelationship):
    """
    A mixin class for semantic annotations that have a handle.
    """

    handle: Optional[Handle] = field(
        default=None,
        metadata=IsPartWholeRelationship().as_dict(),
    )
    """
    The handle of the semantic annotation.
    """


THasRootBody = TypeVar("THasRootBody", bound=HasRootBody)
"""
A type variable for HasRootBody.
"""


@dataclass(eq=False)
class HasLegs(PartWholeRelationship):
    """
    A mixin class for semantic annotations that have legs.
    """

    legs: List[Leg] = field(
        default_factory=list,
        hash=False,
        kw_only=True,
        metadata=IsPartWholeRelationship().as_dict(),
    )
    """
    The legs of the semantic annotation.
    """


@dataclass(eq=False)
class HasSink(PartWholeRelationship):
    """
    A mixin class for semantic annotations that have a sink.
    """

    sink: Optional[Sink] = field(
        default=None,
        metadata=IsPartWholeRelationship().as_dict(),
    )
    """
    The sink of the semantic annotation.
    """


@dataclass(eq=False)
class IsStorageSpace(HasRootBody, Generic[THasRootBody], SubClassSafeGeneric):
    """
    A mixin class for semantic annotations that represent storage spaces.

    Used to afterthefact add object for example to a table, and have those objects move
    with the table when it is moved.
    """

    objects: List[THasRootBody] = field(default_factory=list, hash=False, kw_only=True)
    """
    The occupants currently contained in/on this annotation.
    """

    @synchronized_attribute_modification
    def add_object(self, object: HasRootBody):
        self._world.move_branch(
            object.root, self.root, enable_unsafe_inside_world_block=True
        )
        self.objects.append(object)

    def get_objects_of_type(
        self, object_type: Type[SemanticAnnotation]
    ) -> List[HasRootBody]:
        """
        Returns all objects of a given type in the semantic annotation.

        ..warning:: object_type does not have to be a subclass of HasRootBody, as some semantic concepts, for example
        Food may not necessarily inherit from HasRootBody, but some objects stored in here may inherit from Food as well
        as HasRootBody.

        :param object_type: The type of the semantic annotations to return.

        :return: A list of HasRootBody objects of the given type.
        """
        return [obj for obj in self.objects if isinstance(obj, object_type)]


@dataclass(eq=False)
class HasSupportingSurface(IsStorageSpace):
    """
    A semantic annotation that represents a supporting surface.
    """

    supporting_surface: Region = field(default=None)
    """
    The supporting surface region of the semantic annotation.
    """

    def calculate_supporting_surface(
        self,
        upward_threshold: float = 0.95,
        clearance_threshold: float = 0.5,
        min_surface_area: float = 0.0225,  # 15cm x 15cm
    ) -> Optional[Region]:
        """
        Calculate the supporting surface region for the semantic annotation, add it to
        the world, and set it as the supporting surface of self.

        :param upward_threshold: The threshold for the face normal to be considered
            upward-facing.
        :param clearance_threshold: The threshold for the vertical clearance above the
            surface.
        :param min_surface_area: The minimum area for a surface to be considered a
            supporting surface.
        :return: The supporting surface region, or None if no suitable region could be
            found.
        """
        mesh = self.root.combined_mesh
        if mesh is None:
            return None
        # --- Find upward-facing faces ---
        normals = mesh.face_normals
        upward_mask = normals[:, 2] > upward_threshold

        if not upward_mask.any():
            return None

        # --- Find connected upward-facing regions ---
        upward_face_indices = np.nonzero(upward_mask)[0]
        submesh_up = mesh.submesh([upward_face_indices], append=True)
        face_groups = submesh_up.split(only_watertight=False)

        # Compute total area for each group
        large_groups = [g for g in face_groups if g.area >= min_surface_area]

        if not large_groups:
            return None

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
            return None

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

        supporting_surface_z_position = self.root.collision.scale.z / 2
        self_C_supporting_surface = FixedConnection(
            parent=self.root,
            child=supporting_surface,
            parent_T_connection_expression=HomogeneousTransformationMatrix.from_xyz_rpy(
                z=supporting_surface_z_position, reference_frame=self.root
            ),
        )
        self._world.add_region(supporting_surface)
        self._world.add_connection(self_C_supporting_surface)
        self.add_supporting_surface(supporting_surface)
        return supporting_surface

    def infer_objects_on_surface(self):
        """
        Infer and add objects that are supported by this surface to the storage space.

        This method queries the world for bodies that are supported by this annotation's
        root body, finds their corresponding semantic annotations, and adds them to the
        objects list if they are not already present.
        """
        bodies = variable_from(self._world.bodies_with_collision)
        body = entity(bodies).where(
            is_supported_by(
                supported_body=bodies,
                supporting_body=self.root,
            )
        )
        objects = an(
            entity(
                semantic_annotation := variable(
                    HasRootBody, domain=self._world.semantic_annotations
                )
            ).where(semantic_annotation.root == body)
        ).evaluate()
        for obj in objects:
            if obj in self.objects:
                continue
            self.add_object(obj)

    @synchronized_attribute_modification
    def add_supporting_surface(self, region: Region):
        self._world.move_branch(
            region, self.root, enable_unsafe_inside_world_block=True
        )
        self.supporting_surface = region

    def sample_points_from_surface(
        self,
        body_to_sample_for: Optional[HasRootBody] = None,
        category_of_interest: Optional[Type[SemanticAnnotation]] = None,
        amount: int = 100,
    ) -> List[Point3]:
        """
        Samples points from a surface around the semantic annotation.

        The surface is determined by the supporting surface of the semantic annotation and is truncated by the objects on the surface. The points are sampled
        using a Gaussian mixture model.

        ..warning:: Calling this method when the self.supporting_surface is None will cause the method to calculate the
            surface and add it to the world, resulting in model updates being published if the synchronizer is running.

        :param body_to_sample_for: The physical object to sample points for.
        :param category_of_interest: The type of object sample points around.
        :param amount: The number of points to sample.

        :return: A list of sampled points, sorted by distance to the around_object.
        """
        if self.supporting_surface is None:
            with self._world.modify_world():
                supporting_surface = self.calculate_supporting_surface()
            if supporting_surface is None:
                return []

        largest_xy_object_dimension = 0.1
        z_object_dimension = 0.0
        if body_to_sample_for:
            largest_xy_object_dimension = body_to_sample_for.root.combined_mesh.extents[
                :2
            ].max()
            z_object_dimension = body_to_sample_for.root.combined_mesh.extents[2]

        self_max_z = self.supporting_surface.area.max_point.z
        z_coordinate = np.full(
            (amount, 1),
            self_max_z + (z_object_dimension / 2),
        )

        surface_circuit = self._build_surface_sampler(
            category_of_interest=category_of_interest,
            object_bloat=largest_xy_object_dimension,
        )

        if surface_circuit is None:
            return []

        samples = surface_circuit.sample(amount)
        samples = samples[np.argsort(surface_circuit.log_likelihood(samples))[::-1]]
        samples = np.concatenate((samples, z_coordinate), axis=1)

        if category_of_interest:
            return [
                Point3(*s[1:], reference_frame=self.supporting_surface) for s in samples
            ]
        return [Point3(*s, reference_frame=self.supporting_surface) for s in samples]

    def _build_surface_sampler(
        self,
        category_of_interest: Optional[Type[SemanticAnnotation]] = None,
        object_bloat: float = 0.1,
    ):
        """
        Build a probabilistic circuit representing the supporting surface, truncated by
        the objects on the surface, and with Gaussian mixtures around the objects of
        interest.

        :param category_of_interest: The type of object sample points around.
        :param object_bloat: The amount of bloat to apply to the object event.
        """
        truncated_event_2d = self._2d_surface_sample_space_excluding_objects(
            object_bloat
        )

        objects_of_interest = (
            self.get_objects_of_type(category_of_interest)
            if category_of_interest
            else []
        )
        if objects_of_interest:
            return self._2d_gaussian_sampler_from_2d_sample_space(
                objects_of_interest=objects_of_interest,
                # using values too low makes sampling from truncated gaussians very unstable
                variance=1,
                sample_space=truncated_event_2d,
            )
        else:
            return uniform_measure_of_event(truncated_event_2d)

    def _2d_surface_sample_space_excluding_objects(self, object_bloat: float) -> Event:
        """
        Compute a 2D event representing the supporting surface, truncated by the objects
        on the surface.

        :param object_bloat: The amount of bloat to apply to the object events.
        """
        area_of_self = BoundingBoxCollection.from_shapes(self.supporting_surface.area)
        area_of_self.transform_all_shapes_to_own_frame()
        event = area_of_self.event

        event_2d = event.marginal(SpatialVariables.xy)
        for obj in self.objects:
            bounding_box = obj.root.collision.as_bounding_box_collection_in_frame(
                self.supporting_surface
            ).bounding_box()
            bounding_box.enlarge_all(object_bloat)
            object_event = bounding_box.simple_event.as_composite_set()
            object_event_2d = object_event.marginal(SpatialVariables.xy)
            event_2d = event_2d - object_event_2d
        return event_2d

    def _2d_gaussian_sampler_from_2d_sample_space(
        self,
        objects_of_interest: List[HasRootBody],
        variance: float,
        sample_space: Event,
    ) -> Optional[ProbabilisticCircuit]:
        """
        Create a Gaussian mixture model from a list of points, truncated by an event.

        :param objects_of_interest: Objects of interest to sample around. The Gaussian
            mixtures will be centered around the positions of these objects on the
            surface.
        :param variance: The standard deviation to use for the Gaussian mixtures.
        :param sample_space: The event to truncate the Gaussian mixture model with.
        :return: A probabilistic circuit representing the Gaussian mixture model
            truncated by the event, or None if the event has zero measure.
        """
        surface_circuit = self._untruncated_2d_gaussian_sampler(
            objects_of_interest=objects_of_interest,
            variance=variance,
        )
        sample_space.fill_missing_variables(surface_circuit.variables)
        surface_circuit.log_truncated_in_place(sample_space)

        return surface_circuit

    def _untruncated_2d_gaussian_sampler(
        self,
        objects_of_interest: List[HasRootBody],
        variance: float,
    ) -> ProbabilisticCircuit:
        """
        Create a Gaussian mixture model from a list of points, without truncation.

        This method is extracted from the `_2d_gaussian_sampler_from_2d_sample_space`
        method so that the generated distribution can be tested properly, which cannot
        be done after truncation.
        """
        surface_circuit = ProbabilisticCircuit()
        surface_circuit_root = SumUnit(probabilistic_circuit=surface_circuit)

        objects_of_interest_variable = Symbolic(
            name="objects_of_interest",
            domain=EventSet.from_iterable(objects_of_interest),
        )

        for object_of_interest in objects_of_interest:
            surface_P_obj = self._world.transform(
                object_of_interest.root.global_transform, self.supporting_surface
            )

            p_object_root = ProductUnit(probabilistic_circuit=surface_circuit)
            surface_circuit_root.add_subcircuit(p_object_root, 1.0)

            object_of_interest_p = make_dirac(
                objects_of_interest_variable, object_of_interest
            )

            x_p = GaussianDistribution(
                variable=SpatialVariables.x.value,
                location=float(surface_P_obj.x),
                scale=variance,
            )
            y_p = GaussianDistribution(
                variable=SpatialVariables.y.value,
                location=float(surface_P_obj.y),
                scale=variance,
            )

            p_object_root.add_subcircuit(leaf(object_of_interest_p, surface_circuit))
            p_object_root.add_subcircuit(leaf(x_p, surface_circuit))
            p_object_root.add_subcircuit(leaf(y_p, surface_circuit))

        return surface_circuit


@dataclass(eq=False)
class HasCaseAsRootBody(HasSupportingSurface):
    """
    A mixin class for semantic annotations that have a case as root body.
    """

    @classproperty
    @abstractmethod
    def hole_direction(self) -> Vector3:
        """
        The direction of the physical hole of the geometry.

        For a drawer for example, this would always be Z.
                ..warning:: This does not describe the axis along, for example, a drawer opens. Its the physical opening where
                you can put something into the drawer.
        """

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

    @classmethod
    def get_default_root_kinematic_structure_entity_specification(
        cls,
        name: str | None = None,
        scale: Optional[Scale] = None,
        connection_specification: Optional[ConnectionSpecification] = None,
        *,
        wall_thickness: float = 0.01,
    ) -> BodySpecification:
        """
        Build the default body specification for a container/case, matching the hollow
        geometry generated by :meth:`create_with_new_body_in_world`.

        :param name: The name of bodies created from the specification.
        :param scale: The outer scale of the case. Defaults to a unit :class:`Scale`
            when omitted.
        :param wall_thickness: The thickness of the case walls.
        :param connection_specification: Connection attaching the entity to its parent.
            ``None`` defers to the annotation type's default at spawn time.
        :return: A body specification with hollow container geometry.
        """
        scale = scale if scale is not None else Scale()
        return BodySpecification.from_event(
            name,
            cls._create_container_event(scale, wall_thickness),
            connection_specification=connection_specification,
        )
