from __future__ import annotations
from abc import ABC, abstractmethod
from copy import deepcopy
from dataclasses import dataclass, field
from typing import Union, Optional

from typing_extensions import Self, Type, Any

from random_events.product_algebra import Event
from semantic_digital_twin.datastructures.prefixed_name import PrefixedName
from semantic_digital_twin.robots.robot_parts import AbstractRobot
from semantic_digital_twin.semantic_annotations.mixins import (
    HasRootKinematicStructureEntity,
)
from semantic_digital_twin.spatial_types import HomogeneousTransformationMatrix, Vector3
from semantic_digital_twin.world import World
from semantic_digital_twin.world_description.connections import (
    WheeledDrive,
    FixedConnection,
    PrismaticConnection,
    RevoluteConnection,
    Connection6DoF,
    ActiveConnectionParameters,
)
from semantic_digital_twin.world_description.degree_of_freedom import (
    DegreeOfFreedomLimits,
)
from semantic_digital_twin.world_description.geometry import (
    Shape,
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


@dataclass
class WorldEntitySpawnSpecification(ABC):

    @abstractmethod
    def spawn(
        self,
        name: Union[str, PrefixedName],
        world: World,
        parent_T_self: HomogeneousTransformationMatrix | None = None,
        parent: KinematicStructureEntity | None = None,
    ):
        """
        If parent is None, world.root is used as a parent
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
        name: Union[str, PrefixedName],
        world: World,
        parent_T_self: HomogeneousTransformationMatrix | None = None,
        parent: KinematicStructureEntity | None = None,
    ):
        pass

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
        name: Union[str, PrefixedName],
        world: World,
        parent_T_self: HomogeneousTransformationMatrix | None = None,
        parent: KinematicStructureEntity | None = None,
    ):
        pass


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
        name: Union[str, PrefixedName],
        world: World,
        parent_T_self: HomogeneousTransformationMatrix | None = None,
        parent: KinematicStructureEntity | None = None,
    ):
        pass


@dataclass
class SemanticAnnotationWithRootSpecification(WorldEntitySpawnSpecification):
    """
    Represents a specification for semantic annotations that have a root structure.

    The classmethods in this class correctly expose the same signature as the KinematicStructureEntitySpecification, because
    the goal is that this signature is automatically passed on to create a root_specification without the user needing to first
    create one by hand
    """

    name: Union[str, PrefixedName]
    """
    The name of the annotation.
    """

    semantic_annotation_type: Type[HasRootKinematicStructureEntity]
    """
    The type of the semantic annotation that is a subclass of HasRootKinematicStructureEntity.
    """

    root_specification: Optional[KinematicStructureEntitySpecification]
    """
    The specification of the root kinematic structure entity of the annotation.
    """

    active_connection_parameters: ActiveConnectionParameters | None = field(
        default=None
    )

    annotation_kwargs: dict[
        str,
        Union[
            SemanticAnnotationWithRootSpecification,
            Any,
        ],
    ] = field(default_factory=dict)
    """
    The keyword arguments to pass to the annotation constructor. 
    If the values are WorldEntitySpecification, they will be spawned when this Specification is spawned
    """

    def spawn(
        self,
        name: Union[str, PrefixedName],
        world: World,
        parent_T_self: HomogeneousTransformationMatrix | None = None,
        parent: KinematicStructureEntity | None = None,
    ):
        pass


@dataclass
class BodyAndConnectionSpecification(WorldEntitySpawnSpecification):

    body_specification: BodySpecification

    connection_type: Type[Connection] = field(default=FixedConnection)

    parent_T_child: HomogeneousTransformationMatrix = field(
        default_factory=HomogeneousTransformationMatrix
    )

    active_connection_parameters: ActiveConnectionParameters | None = field(
        default=None
    )

    def spawn(
        self,
        name: Union[str, PrefixedName],
        world: World,
        parent_T_self: HomogeneousTransformationMatrix | None = None,
        parent: KinematicStructureEntity | None = None,
    ):
        pass


@dataclass
class WorldSpecification:
    robot_semantic_annotation: Type[AbstractRobot]
    drive_connection_type: Type[WheeledDrive] | None = None
    world_T_odom: HomogeneousTransformationMatrix | None = None
    odom_T_robot_start: HomogeneousTransformationMatrix | None = None
    starting_objects: list[WorldEntitySpawnSpecification] = field(default_factory=list)

    def create(self) -> World: ...
