"""
Semantic annotations of the furniture, appliances, doors, handles and other things a
world contains.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing_extensions import Iterable, Iterator, Self, TYPE_CHECKING


import numpy as np

from krrood.patterns.classproperty import classproperty
from krrood.symbolic_math import symbolic_math
from random_events.interval import closed
from random_events.product_algebra import SimpleEvent
from semantic_digital_twin.datastructures.alignment import AlignmentPair
from semantic_digital_twin.datastructures.prefixed_name import PrefixedName
from semantic_digital_twin.datastructures.variables import SpatialVariables
from semantic_digital_twin.exceptions import (
    NoGraspGeometry,
    HandleWiderThanLong,
    InvalidPlaneDimensions,
    InvalidHingeActiveAxis,
    MissingSemanticAnnotationError,
    MissingMovableJointError,
)
from semantic_digital_twin.reasoning.predicates import InsideOf
from semantic_digital_twin.robots.exceptions import MissingMobileBaseError
from semantic_digital_twin.robots.robot_part_mixins import HasMobileBase
from semantic_digital_twin.semantic_annotations.part_whole.relationship_marker import (
    IsPartWholeRelationship,
)
from semantic_digital_twin.semantic_annotations.mixins import (
    HasSupportingSurface,
    HasRootRegion,
    HasCaseAsRootBody,
    HasMovableJoint,
    IsPerceivable,
    HasRootBody,
    IsStorageSpace,
    IsGraspable,
    HasDrawnGraspsByDefault,
)
from semantic_digital_twin.semantic_annotations.part_whole.furniture_parts import (
    HasCounterTop,
    HasCabinets,
    HasDrawers,
    HasDoors,
    HasHandle,
    HasHandles,
    HasApertures,
    HasLegs,
    HasSink,
    HasShelfLayers,
    HasRim,
)
from semantic_digital_twin.semantic_annotations.part_whole.fruit_parts import (
    HasCore,
    HasFlesh,
    HasPeel,
    HasPit,
    HasPith,
    HasSeedChamber,
    HasSeeds,
    HasStem,
)
from semantic_digital_twin.semantic_annotations.part_whole.egg_parts import (
    HasAlbumen,
    HasShell,
    HasYolk,
)
from semantic_digital_twin.semantic_annotations.qualities.quality import (
    MassDensity,
    MaterialElasticity,
    Quality,
    SkinThickness,
)
from semantic_digital_twin.semantic_annotations.qualities.relationship_marker import (
    IsQualityRelationship,
)
from semantic_digital_twin.semantic_annotations.qualities.mixins import (
    HasRipeness,
    HasSkinThickness,
)
from semantic_digital_twin.grasping.grasp_candidates import GraspCandidate
from semantic_digital_twin.grasping.grasp_strategies import (
    GraspABar,
    GraspAcrossTheLength,
    GraspStrategy,
    PinchARim,
)
from krrood_spatial_math.spatial_types import (
    Point3,
    HomogeneousTransformationMatrix,
    Vector3,
)
from krrood_spatial_math.spatial_types import Pose, RotationMatrix
from semantic_digital_twin.world_description.connections import (
    FixedConnection,
)
from semantic_digital_twin.world_description.geometry import (
    VolumetricBoundingBox,
    Scale,
    Color,
)
from semantic_digital_twin.world_description.shape_collection import (
    BoundingBoxCollection,
)
from semantic_digital_twin.world_description.world_entity import (
    SemanticAnnotation,
    Body,
)
from semantic_digital_twin.specifications.connections import ConnectionSpecification
from semantic_digital_twin.specifications.kinematic_structure_entities import (
    BodySpecification,
    KinematicStructureEntitySpecification,
    RegionSpecification,
)
from semantic_digital_twin.specifications.semantic_annotations import (
    SemanticAnnotationWithRootSpecification,
)

if TYPE_CHECKING:
    from semantic_digital_twin.robots.robot_parts import AbstractRobot
    from semantic_digital_twin.world import World
    from semantic_digital_twin.world_description.graph_of_convex_sets.boxes import (
        PlanarGraphOfBoundingBoxes,
    )


@dataclass(eq=False)
class Furniture(SemanticAnnotation, ABC):
    """
    A semantic annotation that represents a piece of furniture.
    """


@dataclass(eq=False)
class Handle(IsGraspable):
    """
    A handle is a physical entity that can be grasped by a hand or a robotic gripper to
    open or close an object.
    """

    @classproperty
    def grasp_strategy(cls) -> GraspStrategy:
        return GraspABar()

    @classmethod
    def _create_handle_geometry(
        cls, scale: Scale, thickness: float = 0.0
    ) -> SimpleEvent:
        """
        Create a box event representing the handle.

        :param scale: The scale of the handle.
        :param thickness: The thickness of the handle walls.
        """
        x_interval = closed(-scale.x + thickness, 0)
        y_interval = closed(-scale.y / 2, scale.y / 2)
        z_interval = closed(
            -scale.z / 2 + thickness,
            scale.z / 2 - thickness,
        )

        return SimpleEvent.from_data(
            {
                SpatialVariables.x.value: x_interval,
                SpatialVariables.y.value: y_interval,
                SpatialVariables.z.value: z_interval,
            }
        )

    @classmethod
    def get_default_root_kinematic_structure_entity_specification(
        cls,
        name: str | None = None,
        scale: Scale | None = None,
        connection_specification: ConnectionSpecification | None = None,
        *,
        thickness: float = 0.005,
    ) -> BodySpecification:
        """
        Build the default handle body specification, matching the hollow handle geometry
        generated by :meth:`create_with_new_body_in_world`.

        :param name: The name of bodies created from the specification.
        :param scale: The scale of the handle: depth as x, width as y and length as z.
            Defaults to ``Scale(0.05, 0.02, 0.1)``.
        :param thickness: The thickness of the handle walls.
        :param connection_specification: Connection attaching the entity to its parent.
            ``None`` defers to the annotation type's default at spawn time.
        :return: A body specification with the handle geometry.
        :raises HandleWiderThanLong: If the width of ``scale`` exceeds its length.
        """
        scale = scale or Scale(0.05, 0.02, 0.1)
        if scale.y > scale.z:
            raise HandleWiderThanLong(scale=scale)
        handle_event = cls._create_handle_geometry(scale=scale).as_composite_set()
        inner_box = cls._create_handle_geometry(
            scale=scale, thickness=thickness
        ).as_composite_set()
        handle_event -= inner_box

        return BodySpecification.from_event(
            name, handle_event, connection_specification=connection_specification
        )

    def default_grasp_candidates(self) -> list[GraspCandidate]:
        """
        The grasps at the centre of the handle's bar that close around it, one
        approaching from each of the four sides around it.

        The bar runs along the longest extent of the root's collision geometry.

        :raises NoGraspGeometry: If the root body has no collision geometry to tell
            which way the bar runs.
        """
        if not self.root.has_collision():
            raise NoGraspGeometry(self)
        mesh = self.root.combined_mesh
        bar_axis = np.argmax(mesh.extents)
        bar_direction = Vector3.from_iterable(np.eye(3)[bar_axis])
        across_bar = np.delete(np.eye(3), bar_axis, axis=0)
        return [
            GraspCandidate(
                self,
                Pose(
                    position=Point3.from_iterable(mesh.bounds.mean(axis=0)),
                    orientation=RotationMatrix.from_vectors(
                        x=approach, y=bar_direction.cross(approach)
                    ).quaternion,
                    reference_frame=self.root,
                ),
            )
            for approach in map(Vector3.from_iterable, [*across_bar, *-across_bar])
        ]


@dataclass(eq=False)
class Dishwasher(HasCaseAsRootBody, HasDoors, HasDrawers):
    """
    A dishwasher is a kitchen appliance used for cleaning dishes, utensils, and
    cookware.

    It typically has a front door that opens to reveal racks for loading dirty items and
    a control panel for selecting wash cycles.
    """

    @classproperty
    def _hole_direction_axis(cls) -> Vector3:
        return Vector3.NEGATIVE_X()


@dataclass(eq=False)
class WashingMachine(HasCaseAsRootBody, HasDoors, HasDrawers):
    """
    A washing machine is a household appliance that washes laundry in a drum.

    It typically has a front door to load the drum and a pull-out detergent drawer.
    """

    @classproperty
    def _hole_direction_axis(cls) -> Vector3:
        return Vector3.NEGATIVE_X()


@dataclass(eq=False)
class Aperture(HasRootRegion):
    """
    An opening in a physical entity.

    An example is like a hole in a wall that can be used to enter a room.
    """

    @classmethod
    def create_with_new_region_in_world_from_body(
        cls,
        name: str,
        world: World,
        body: Body,
        parent_T_self: HomogeneousTransformationMatrix | None = None,
    ) -> Self:
        world.update_forward_kinematics()
        body_scale = (
            body.collision.as_bounding_box_collection_in_frame(body)
            .bounding_box()
            .scale
        )
        return cls.create_with_new_region_in_world(
            name, world, parent_T_self, scale=body_scale
        )

    def _mount_strategy(
        self,
        main_has_root_body_annotation: HasRootBody,
        relationship: IsPartWholeRelationship,
    ) -> None:
        """
        Cut this aperture out of the whole's geometry when the relationship calls for
        it, then mount as a kinematic child.

        An aperture mounted into a field that does not remove geometry marks a passage
        rather than a hole, so the whole keeps its material.
        """
        if relationship.removes_part_geometry_from_whole:
            self._remove_aperture_geometry_from_parent(main_has_root_body_annotation)
        super()._mount_strategy(main_has_root_body_annotation, relationship)

    def _remove_aperture_geometry_from_parent(self, parent: HasRootBody):
        """
        Remove the geometry of the aperture from the parent body's collision and visual
        geometry.

        :param parent: The parent from which the aperture geometry is removed.
        """
        world = parent._world
        world.update_forward_kinematics()
        hole_event = self.root.area.as_bounding_box_collection_in_frame(
            parent.root
        ).event
        wall_event = parent.root.collision.as_bounding_box_collection_in_frame(
            parent.root
        ).event
        new_wall_event = wall_event - hole_event
        new_bounding_box_collection = BoundingBoxCollection.from_event(
            VolumetricBoundingBox, parent.root, new_wall_event
        ).as_shapes()

        parent.root.collision = new_bounding_box_collection
        parent.root.visual = new_bounding_box_collection

    @classmethod
    def get_default_root_kinematic_structure_entity_specification(
        cls,
        name: str | None = None,
        scale: Scale | None = None,
        connection_specification: ConnectionSpecification | None = None,
    ) -> RegionSpecification:
        """
        Build the default region specification whose ``area`` geometry matches what
        :meth:`create_with_new_region_in_world` generates from ``scale``.

        :param name: The name of regions created from the specification.
        :param scale: The scale used to generate the region area geometry. Defaults to
            ``Scale()``.
        :param connection_specification: Connection attaching the entity to its parent.
            ``None`` defers to the annotation type's default at spawn time.
        :return: A region specification with a single box area derived from ``scale``.
        """
        scale = scale or Scale()
        return RegionSpecification.from_event(
            name,
            scale.to_simple_event().as_composite_set(),
            connection_specification=connection_specification,
        )


@dataclass(eq=False)
class EntryWay(Aperture):
    """
    The passage a door covers.

    It is coextensive with the door leaf, so mounting it on its door leaves the door's
    geometry intact; mounting it on a :class:`Wall` cuts the doorway out of the wall.
    """

    @classmethod
    def get_default_root_kinematic_structure_entity_specification(
        cls,
        name: str | None = None,
        scale: Scale | None = None,
        connection_specification: ConnectionSpecification | None = None,
        *,
        color: Color | None = None,
    ) -> RegionSpecification:
        """
        Build the default entry way region specification: a box coextensive with the
        door leaf, translucent so the passage is visible without occluding the door.

        :param name: The name of regions created from the specification.
        :param scale: The scale of the passage. Defaults to ``Scale()``.
        :param connection_specification: Connection attaching the entity to its parent.
            ``None`` defers to the annotation type's default at spawn time.
        :param color: The colour of the passage. Defaults to translucent white.
        :return: A region specification with the passage geometry.
        """
        return RegionSpecification.box(
            name,
            scale or Scale(),
            color=color or Color.from_rgba(red=1.0, green=1.0, blue=1.0, opacity=0.2),
            connection_specification=connection_specification,
        )


@dataclass(eq=False)
class Door(HasHandle, HasMovableJoint):
    """
    A door is a physical entity that has covers an opening, has a movable body and a
    handle.
    """

    entry_way: EntryWay | None = field(
        default=None, metadata=IsPartWholeRelationship().as_dict()
    )
    """
    The entry way of the door: the passage the door covers.

    It is coextensive with the door leaf, so mounting it leaves the door's geometry
    intact; mounting the same entry way on a :class:`Wall` cuts the doorway out of it.
    """

    @classproperty
    def default_scale(cls) -> Scale:
        """
        The scale the door's geometry and its entry way are built from when the caller
        supplies none.
        """
        return Scale(0.03, 1, 2)

    @classmethod
    def get_annotation_specification(
        cls,
        name: str,
        root_specification: KinematicStructureEntitySpecification,
        *,
        parent_connection_specification: ConnectionSpecification | None = None,
        annotation_kwargs: dict | None = None,
        part_specifications: dict | None = None,
    ) -> SemanticAnnotationWithRootSpecification[Self]:
        """
        Extend the door specification with its entry way, declared as a nested part so a
        door always carries the passage it covers.

        The entry way is coextensive with the door leaf, so it is sized from the root
        specification's own geometry rather than from a separately supplied scale. A
        caller-supplied ``entry_way`` entry replaces the default one.

        :param name: The name of the annotation and its root entity.
        :param root_specification: The specification of the door's root body.
        :param parent_connection_specification: Connection attaching the root to its
            parent.
        :param annotation_kwargs: Inert keyword arguments for the annotation
            constructor.
        :param part_specifications: Nested annotation parts keyed by part-whole
            relationship field name.
        :return: The annotation specification.
        """
        entry_way_specification = EntryWay.get_annotation_specification(
            f"{name}_entry_way",
            EntryWay.get_default_root_kinematic_structure_entity_specification(
                scale=root_specification.scale
            ),
        )
        return super().get_annotation_specification(
            name,
            root_specification,
            parent_connection_specification=parent_connection_specification,
            annotation_kwargs=annotation_kwargs,
            part_specifications={
                "entry_way": entry_way_specification,
                **(part_specifications if part_specifications is not None else {}),
            },
        )

    @classmethod
    def get_default_root_kinematic_structure_entity_specification(
        cls,
        name: str | None = None,
        scale: Scale | None = None,
        connection_specification: ConnectionSpecification | None = None,
    ) -> BodySpecification:
        """
        Build the default door body specification.

        The door geometry is the base solid box; the ``EntryWay`` is a nested part
        specification and is not part of the door's own geometry.

        :param name: The name of bodies created from the specification.
        :param scale: The scale of the door. ``scale.x`` must be the smallest dimension.
            Defaults to :attr:`default_scale`.
        :param connection_specification: Connection attaching the entity to its parent.
            ``None`` defers to the annotation type's default at spawn time.
        :return: A body specification with the door box geometry.
        """
        scale = scale or cls.default_scale
        if not (scale.x < scale.y and scale.x < scale.z):
            raise InvalidPlaneDimensions(scale, clazz=Door)
        return super().get_default_root_kinematic_structure_entity_specification(
            name, scale, connection_specification
        )

    def calculate_self_T_movable_joint(
        self, axis: Vector3
    ) -> HomogeneousTransformationMatrix:
        """
        Calculate where the door's hinge sits: on the edge opposite its handle.

        :param axis: The axis the door swings about, in the door frame, along one of the
            frame's axes in either direction.
        :return: The pose of the hinge in the door frame.
        """
        if self.handle is None:
            raise MissingSemanticAnnotationError(self.__class__, Handle)

        connection = self.handle.root.parent_connection
        door_P_handle = connection.origin_expression.position
        scale = self.root.collision.scale

        # The axis' sign only decides which way the door swings, not which edge it
        # swings about.
        match [abs(component) for component in axis.to_np().tolist()]:
            case [0, 1, 0, 0]:
                sign = (
                    symbolic_math.sign(-1 * door_P_handle.z)
                    if door_P_handle.z != 0
                    else 1
                )
                offset = sign * (scale.z / 2)
                door_T_hinge = HomogeneousTransformationMatrix.from_xyz_rpy(z=offset)

            case [0, 0, 1, 0]:
                sign = (
                    symbolic_math.sign(-1 * door_P_handle.y)
                    if door_P_handle.y != 0
                    else 1
                )
                offset = sign * (scale.y / 2)
                door_T_hinge = HomogeneousTransformationMatrix.from_xyz_rpy(y=offset)
            case [1, 0, 0, 0]:
                sign = (
                    symbolic_math.sign(-1 * door_P_handle.x)
                    if door_P_handle.x != 0
                    else 1
                )
                offset = sign * (scale.x / 2)
                door_T_hinge = HomogeneousTransformationMatrix.from_xyz_rpy(x=offset)

            case _:
                raise InvalidHingeActiveAxis(axis=axis)

        return door_T_hinge


@dataclass(eq=False)
class DoubleDoor(SemanticAnnotation):
    """
    A semantic annotation that represents a double door with left and right doors.
    """

    door_0: Door = field(kw_only=True)
    door_1: Door = field(kw_only=True)

    def calculate_left_right_door_from_view_point(
        self, world_T_view_point: HomogeneousTransformationMatrix
    ) -> tuple[Door, Door]:
        """
        Calculate which door is the left and which is the right door based on a given
        view point.

        :param world_T_view_point: The transformation matrix of the view point.
        :return: A tuple containing the left and right door. the first door is the left
            door, the second door is the right door.
        """
        world_T_door_0 = self.door_0.root.global_transform
        view_point_T_door_0 = world_T_view_point.inverse() @ world_T_door_0
        world_T_door_1 = self.door_1.root.global_transform
        view_point_T_door_1 = world_T_view_point.inverse() @ world_T_door_1
        if view_point_T_door_0.y > view_point_T_door_1.y:
            return self.door_0, self.door_1
        else:
            return self.door_1, self.door_0


@dataclass(eq=False)
class Drawer(Furniture, HasCaseAsRootBody, HasHandle, HasMovableJoint):
    @classproperty
    def _hole_direction_axis(cls) -> Vector3:
        return Vector3.Z()

    def calculate_self_T_movable_joint(
        self, axis: Vector3
    ) -> HomogeneousTransformationMatrix:
        """
        A drawer slides, which moves it the same wherever its rails sit, so they sit at
        its origin.

        :param axis: The axis the drawer slides along, in the drawer frame.
        :return: The identity: the rails' frame is the drawer's frame.
        """
        return HomogeneousTransformationMatrix()


@dataclass(eq=False)
class Elevator(HasCaseAsRootBody, HasDoors, HasMovableJoint):
    """
    An elevator in the world, consists of three walls a floor, double doors for entering and a prismatic drive that moves
    the elevator to other floors.
    """

    @classproperty
    def _hole_direction_axis(cls) -> Vector3:
        return Vector3.NEGATIVE_X()

    def calculate_self_T_movable_joint(
        self, axis: Vector3
    ) -> HomogeneousTransformationMatrix:
        """
        An elevator's drive slides it, which moves it the same wherever the drive sits,
        so it sits at the cabin's origin.

        :param axis: The axis the cabin travels along, in the cabin frame.
        :return: The identity: the drive's frame is the cabin's frame.
        """
        return HomogeneousTransformationMatrix()

    def open(self):
        """
        Opens the elevator doors
        """
        for door in self.doors:
            if door.movable_joint is None:
                raise MissingMovableJointError(door)
            door.movable_joint.position = door.movable_joint.dof.limits.upper.position

    def close(self):
        """
        Closes the elevator doors
        """
        for door in self.doors:
            if door.movable_joint is None:
                raise MissingMovableJointError(door)
            door.movable_joint.position = door.movable_joint.dof.limits.lower.position

    def drive_position_for_floor(self, floor: Level) -> float:
        """
        The drive position at which the elevator serves the given floor.

        The half height accounts for the case body's origin sitting at its ground rather
        than at its centre.

        :param floor: The floor the elevator should serve.
        :return: The position to drive the elevator's joint to.
        """
        return float(floor.floor_plane[0].z)

    def drive_to_floor(self, floor: Level):
        """
        Drives the elevator to the floor given
        """
        if self.movable_joint is None:
            raise MissingMovableJointError(self)
        self.movable_joint.position = self.drive_position_for_floor(floor)


############################### subclasses to Furniture


@dataclass(eq=False)
class ShelfLayer(HasSupportingSurface):
    """
    A horizontal surface used for storing objects, typically found inside cabinets or on
    walls.
    """

    _synonyms = {"level", "board"}


@dataclass(eq=False)
class Table(Furniture, HasSupportingSurface):
    """
    A semantic annotation that represents a table.
    """


@dataclass(eq=False)
class CounterTop(Furniture, HasSupportingSurface, HasSink):
    """
    A semantic annotation that represents a counter top.
    """

    _synonyms = {"countertop"}


@dataclass(eq=False)
class KitchenIsland(Furniture, HasRootBody, HasCabinets, HasCounterTop, HasDoors):
    """
    A run of kitchen cabinets standing free of the walls, under one counter top.
    """

    _synonyms = {"kitchen_island", "island"}


@dataclass(eq=False)
class Cabinet(Furniture, HasCaseAsRootBody, HasDoors, HasDrawers):
    @classproperty
    def _hole_direction_axis(cls) -> Vector3:
        return Vector3.NEGATIVE_X()


@dataclass(eq=False)
class Fridge(Cabinet): ...


@dataclass(eq=False)
class TumbleDryer(HasCaseAsRootBody, HasDoors, HasDrawers):
    """
    A tumble dryer is a household appliance that dries laundry in a rotating drum.

    It typically has a front door to load the drum and a pull-out water tank.
    """

    @classproperty
    def _hole_direction_axis(cls) -> Vector3:
        return Vector3.NEGATIVE_X()


@dataclass(eq=False)
class Oven(HasRootBody, HasDoors): ...


@dataclass(eq=False)
class Dresser(Cabinet): ...


@dataclass(eq=False)
class Cupboard(Cabinet): ...


@dataclass(eq=False)
class Wardrobe(Cabinet): ...


@dataclass(eq=False)
class Floor(HasSupportingSurface):
    """
    A floor that robots and furniture stand on.
    """

    def navigable_space(
        self, *, robot: AbstractRobot, margin: float
    ) -> PlanarGraphOfBoundingBoxes:
        """
        :param robot: A robot with a mobile base, standing on this floor.
        :param margin: How far, beyond its radius, the base is to keep from obstacles.
        :return: The free space above this floor that the robot's base can stand in,
            kept clear of every obstacle up to the robot's height by the base's radius
            and ``margin``.
        :raises MissingMobileBaseError: If the robot has no mobile base.
        """
        if not isinstance(robot, HasMobileBase):
            raise MissingMobileBaseError(robot_part=robot)
        return self.planar_free_space(
            max_height=robot.as_bounding_box_collection_in_frame(robot.root)
            .bounding_box()
            .scale.z,
            bloat_obstacles=robot.mobile_base.base_radius + margin,
        )

    @classmethod
    def create_with_new_body_from_polytope_in_world(
        cls,
        name: str,
        world: World,
        floor_polytope: list[Point3],
        world_root_T_self: HomogeneousTransformationMatrix | None = None,
    ) -> Self:
        """
        Create a Floor semantic annotation with a new body defined by the given list of
        Point3.

        :param name: The name of the floor body.
        :param floor_polytope: A list of 3D points defining the floor poly
        """
        return cls.get_annotation_specification(
            name,
            BodySpecification.from_3d_points(name, floor_polytope),
        ).spawn(world, parent_T_self=world_root_T_self)

    @classmethod
    def get_default_root_kinematic_structure_entity_specification(
        cls,
        name: str | None = None,
        scale: Scale | None = None,
        connection_specification: ConnectionSpecification | None = None,
    ) -> BodySpecification:
        """
        Build the default floor body specification, matching the convex-hull polytope
        geometry generated by :meth:`create_with_new_body_in_world` (which derives
        points from the scale bounding box).

        :param name: The name of bodies created from the specification.
        :param scale: The scale defining the floor polytope. Defaults to ``Scale()``.
        :param connection_specification: Connection attaching the entity to its parent.
            ``None`` defers to the annotation type's default at spawn time.
        :return: A body specification with the floor polytope mesh.
        """
        scale = scale or Scale()
        return BodySpecification.from_3d_points(
            name,
            scale.to_bounding_box().get_points(),
            connection_specification=connection_specification,
        )


@dataclass(eq=False)
class Room(SemanticAnnotation):
    """
    A closed area with a specific purpose.
    """

    floor: Floor = field(kw_only=True)
    """
    The room's floor.
    """


@dataclass(eq=False)
class Kitchen(Room): ...


@dataclass(eq=False)
class Bedroom(Room): ...


@dataclass(eq=False)
class Bathroom(Room): ...


@dataclass(eq=False)
class LivingRoom(Room): ...


@dataclass(eq=False)
class Level(HasRootRegion):
    """
    A level of a building
    """

    @property
    def floor_plane(self) -> tuple[Point3, Point3]:
        """
        The floor plane of a level, expressed as two points

        :return: The floor plane of the level
        """
        min_point_global = self._world.transform(
            self.root.area.min_point, self._world.root
        )
        max_point_global = self._world.transform(
            self.root.area.max_point, self._world.root
        )
        min_z = min_point_global.z
        return min_point_global, Point3(
            max_point_global.x,
            max_point_global.y,
            min_z,
            reference_frame=self._world.root,
        )


@dataclass(eq=False)
class GroundFloor(Level): ...


@dataclass(eq=False)
class FirstFloor(Level): ...


@dataclass(eq=False)
class SecondFloor(Level): ...


@dataclass(eq=False)
class Wall(HasApertures):
    """
    A wall is a physical entity that separates two spaces and can contain apertures.

    Doors are a computed property.
    """

    _synonyms = {"walls"}

    @property
    def doors(self) -> Iterable[Door]:
        return [
            door
            for door in self._world.get_semantic_annotations_by_type(Door)
            if door.entry_way
            and InsideOf(
                body=door.entry_way.root, other=self.root, minimum_containment_ratio=0.1
            )()
        ]

    @classmethod
    def _create_wall_event(cls, scale: Scale) -> SimpleEvent:
        """
        Return the collision shapes for the wall.

        A wall event is created based on the scale of the wall, and doors are removed
        from the wall event. The resulting bounding box collection is converted to
        shapes.
        """
        x_interval = closed(-scale.x / 2, scale.x / 2)
        y_interval = closed(-scale.y / 2, scale.y / 2)
        z_interval = closed(0, scale.z)

        return SimpleEvent.from_data(
            {
                SpatialVariables.x.value: x_interval,
                SpatialVariables.y.value: y_interval,
                SpatialVariables.z.value: z_interval,
            }
        )

    @classmethod
    def get_default_root_kinematic_structure_entity_specification(
        cls,
        name: str | None = None,
        scale: Scale | None = None,
        connection_specification: ConnectionSpecification | None = None,
    ) -> BodySpecification:
        """
        Build the default wall body specification, matching the geometry generated by
        :meth:`create_with_new_body_in_world` (a box whose z runs ``0..scale.z``).

        :param name: The name of bodies created from the specification.
        :param scale: The scale of the wall. ``scale.x`` must be the smallest dimension.
            Defaults to ``Scale()``.
        :param connection_specification: Connection attaching the entity to its parent.
            ``None`` defers to the annotation type's default at spawn time.
        :return: A body specification with the wall box geometry.
        """
        scale = scale or Scale()
        if not (scale.x < scale.y and scale.x < scale.z):
            raise InvalidPlaneDimensions(scale, clazz=Wall)
        return BodySpecification.from_event(
            name,
            cls._create_wall_event(scale).as_composite_set(),
            connection_specification=connection_specification,
        )

    def bloated_bounding_box_collection(
        self,
        origin: HomogeneousTransformationMatrix,
        bloat_amount: float,
        obstacle_height_clearance: float = 0.01,
    ) -> BoundingBoxCollection[VolumetricBoundingBox, Point3]:
        """
        Bloat this wall's bounding boxes along their thinner dimension only -- the
        side that faces the room -- rather than symmetrically in x and y.

        :param origin: The origin to express the bounding boxes relative to.
        :param bloat_amount: The amount to bloat by.
        :param obstacle_height_clearance: The amount to bloat by in z, regardless of
            ``bloat_amount``.
        :return: The bloated bounding boxes.
        """
        return BoundingBoxCollection(
            [
                (
                    bounding_box.bloat(bloat_amount, 0, obstacle_height_clearance)
                    if bounding_box.width > bounding_box.depth
                    else bounding_box.bloat(0, bloat_amount, obstacle_height_clearance)
                )
                for bounding_box in self.as_bounding_box_collection_at_origin(origin)
            ],
            origin.reference_frame,
        )


@dataclass(eq=False)
class Container(IsGraspable):
    """
    An object that holds contents and has an opening they go in and out through.
    """


@dataclass(eq=False)
class Cookware(IsGraspable):
    """
    An object used to cook with.
    """


@dataclass(eq=False)
class Tableware(IsGraspable):
    """
    An object a table is set with for a meal.
    """


@dataclass(eq=False)
class Bottle(Container):
    """
    Abstract class for bottles.
    """


@dataclass(eq=False)
class Statue(IsGraspable):
    """
    A sculpture small enough to be picked up.
    """


@dataclass(eq=False)
class SoapBottle(Bottle):
    """
    A soap bottle.
    """


@dataclass(eq=False)
class WineBottle(Bottle):
    """
    A wine bottle.
    """

    mass_density: MassDensity | None = field(
        default_factory=lambda: MassDensity.glass,
        kw_only=True,
        hash=False,
        metadata=IsQualityRelationship().as_dict(),
    )
    """
    The mass density of the glass the bottle is made of.
    """

    material_elasticity: MaterialElasticity | None = field(
        default_factory=lambda: MaterialElasticity.glass,
        kw_only=True,
        hash=False,
        metadata=IsQualityRelationship().as_dict(),
    )
    """
    The elasticity of the glass the bottle is made of.
    """


@dataclass(eq=False)
class MustardBottle(Bottle):
    """
    A mustard bottle.
    """


@dataclass(eq=False)
class DrinkingContainer(Container, Tableware): ...


@dataclass(eq=False)
class Cup(HasHandle, HasRim, DrinkingContainer, IsPerceivable):
    """
    A cup.
    """


@dataclass(eq=False)
class Mug(HasHandle, HasRim, DrinkingContainer):
    """
    A mug, held by its handle.
    """


@dataclass(eq=False)
class CookingContainer(Container, Cookware): ...


@dataclass(eq=False)
class Lid(Cookware): ...


@dataclass(eq=False)
class Pan(CookingContainer):
    """
    A pan.
    """


@dataclass(eq=False)
class PanLid(Lid):
    """
    A pan lid.
    """


@dataclass(eq=False)
class Pot(CookingContainer, HasHandles):
    """
    A pot.
    """


@dataclass(eq=False)
class PotLid(Lid):
    """
    A pot lid.
    """


@dataclass(eq=False)
class Plate(HasSupportingSurface, Tableware):
    """
    A plate.
    """


@dataclass(eq=False)
class Rim(HasDrawnGraspsByDefault):
    """
    The upper edge of an open container's wall, which a gripper pinches from above.
    """

    @classproperty
    def grasp_strategy(cls) -> GraspStrategy:
        return PinchARim()


@dataclass(eq=False)
class Bowl(HasRim, HasSupportingSurface, Container, Tableware, IsPerceivable):
    """
    A bowl, grasped by pinching the wall of its rim: it offers nothing to grip at its
    own origin, which is inside it.
    """


# Food Items
@dataclass(eq=False)
class Food(IsGraspable):
    """
    A Group class for Food.
    """


@dataclass(eq=False)
class TunaCan(Food):
    """
    A tuna can.
    """


@dataclass(eq=False)
class Bread(Food):
    """
    Bread.
    """

    mass_density: MassDensity | None = field(
        default_factory=lambda: MassDensity(
            value=Quality.closed_range(lower=130.0, upper=340.0)
        ),
        kw_only=True,
        hash=False,
        metadata=IsQualityRelationship().as_dict(),
    )
    """
    The apparent density of the crumb of five commercial breads
    :cite:`nussinovitch1990bread`.
    """

    _synonyms = {
        "bumpybread",
        "whitebread",
        "loafbread",
        "honeybread",
        "grainbread",
    }


@dataclass(eq=False)
class CheezeIt(Food, IsPerceivable):
    """
    Some type of cracker.
    """


@dataclass(eq=False)
class Pringles(Food):
    """
    Pringles chips.
    """


@dataclass(eq=False)
class GelatinBox(Food):
    """
    Gelatin box.
    """


@dataclass(eq=False)
class TomatoSoup(Food):
    """
    Tomato soup.
    """


@dataclass(eq=False)
class Candy(Food, IsPerceivable):
    """
    A candy.
    """

    ...


@dataclass(eq=False)
class Noodles(Food, IsPerceivable):
    """
    A container of noodles.
    """

    ...


@dataclass(eq=False)
class Cereal(Food, IsPerceivable):
    """
    A container of cereal.
    """

    ...


@dataclass(eq=False)
class Milk(Food, IsPerceivable):
    """
    A container of milk.
    """

    mass_density: MassDensity | None = field(
        default_factory=lambda: MassDensity(
            value=Quality.closed_range(lower=1027.0, upper=1033.0)
        ),
        kw_only=True,
        hash=False,
        metadata=IsQualityRelationship().as_dict(),
    )
    """
    Milk at 20 °C, which dominates a full container of it; being liquid, it has no
    elasticity :cite:`goff_milk_physical`.
    """


@dataclass(eq=False)
class SaltContainer(Container, IsPerceivable):
    """
    A container of salt.
    """


@dataclass(eq=False, kw_only=True)
class UnknownObject(HasRootBody, IsPerceivable):
    """
    An object perception found that no more specific semantic annotation describes.
    """


@dataclass(eq=False)
class Produce(Food):
    """
    In American English, produce generally refers to fresh fruits and vegetables
    intended to be eaten by humans.
    """

    pass


@dataclass(eq=False, kw_only=True)
class Fruit(Produce, HasRipeness):
    """
    Fruit.
    """


@dataclass(eq=False, kw_only=True)
class FruitPart(HasRootBody):
    """
    An anatomical part of a fruit, such as its peel or a seed.
    """


@dataclass(eq=False, kw_only=True)
class Peel(FruitPart, HasSkinThickness):
    """
    The outer layer of a fruit, its skin or rind (botanically the exocarp, Plant
    Ontology PO:0009085).
    """


@dataclass(eq=False, kw_only=True)
class Flesh(FruitPart):
    """
    The soft tissue between the peel and the inside of a fruit, which is usually eaten
    (botanically mostly the mesocarp, Plant Ontology PO:0009087).
    """


@dataclass(eq=False, kw_only=True)
class CitrusSegment(Flesh):
    """
    One of the wedge-shaped portions the flesh of a citrus fruit divides into, each
    filled with juice sacs inside a thin membrane (botanically a fruit locule, Plant
    Ontology PO:0025270).
    """


@dataclass(eq=False, kw_only=True)
class Core(FruitPart):
    """
    The tough centre of a fruit such as an apple that holds its seeds (botanically the
    endocarp, Plant Ontology PO:0009086).
    """


@dataclass(eq=False, kw_only=True)
class Seed(FruitPart):
    """
    A single seed that lies loose in a fruit's flesh or core (Plant Ontology
    PO:0009010).
    """


@dataclass(eq=False, kw_only=True)
class Stem(FruitPart):
    """
    The stalk a fruit hung from on its plant.
    """


@dataclass(eq=False, kw_only=True)
class Pit(FruitPart):
    """
    The single large, hard seed or stone at the centre of a fruit such as an avocado or
    a peach.
    """


@dataclass(eq=False, kw_only=True)
class SeedChamber(FruitPart):
    """
    The soft central tissue of a fruit such as a cucumber in which its small seeds are
    embedded.
    """


@dataclass(eq=False, kw_only=True)
class Pith(FruitPart):
    """
    The pale, firmer centre of a fruit such as a strawberry.
    """


@dataclass(eq=False)
class Vegetable(Produce):
    """
    Vegetable.
    """


@dataclass(eq=False, kw_only=True)
class TomatoPeel(Peel):
    """
    The peel of a tomato, its exocarp.
    """

    material_elasticity: MaterialElasticity | None = field(
        default_factory=lambda: MaterialElasticity(
            youngs_modulus=Quality.closed_range(lower=2.297e6, upper=9.59e6)
        ),
        kw_only=True,
        hash=False,
        metadata=IsQualityRelationship().as_dict(),
    )
    """
    Tensile strips of tomato skin :cite:`li2012tomato,kunabroniowska2011tomato`; its
    measured Poisson's ratio of 0.38 to 0.73 lies beyond the scale and is left out.
    """


@dataclass(eq=False, kw_only=True)
class TomatoFlesh(Flesh):
    """
    The flesh of a tomato, its mesocarp.
    """

    material_elasticity: MaterialElasticity | None = field(
        default_factory=lambda: MaterialElasticity(
            youngs_modulus=Quality.closed_range(lower=0.726e6, upper=0.846e6)
        ),
        kw_only=True,
        hash=False,
        metadata=IsQualityRelationship().as_dict(),
    )
    """
    Tomato mesocarp in compression :cite:`li2012tomato`.
    """


@dataclass(eq=False, kw_only=True)
class TomatoSeedChamber(SeedChamber):
    """
    The seed chamber of a tomato, its locular gel.
    """

    material_elasticity: MaterialElasticity | None = field(
        default_factory=lambda: MaterialElasticity(
            youngs_modulus=Quality.closed_range(lower=0.048e6, upper=0.124e6)
        ),
        kw_only=True,
        hash=False,
        metadata=IsQualityRelationship().as_dict(),
    )
    """
    The locular gel around tomato seeds in compression :cite:`li2012tomato`.
    """


@dataclass(eq=False, kw_only=True)
class Tomato(Fruit, HasPeel, HasFlesh, HasSeedChamber):
    """
    A tomato, whose seeds sit in a locular gel, its seed chamber.
    """


@dataclass(eq=False)
class Lettuce(Vegetable):
    """
    Lettuce.
    """

    material_elasticity: MaterialElasticity | None = field(
        default_factory=lambda: MaterialElasticity(
            youngs_modulus=Quality.closed_range(lower=2.23e6, upper=5.08e6)
        ),
        kw_only=True,
        hash=False,
        metadata=IsQualityRelationship().as_dict(),
    )
    """
    Tensile strips of leaves of var. Crispa lettuce :cite:`xavier2022lettuce`.
    """


@dataclass(eq=False)
class Carrot(Vegetable):
    """
    A carrot.
    """

    material_elasticity: MaterialElasticity | None = field(
        default_factory=lambda: MaterialElasticity(
            youngs_modulus=Quality.closed_range(lower=3.0e6, upper=8.31e6),
            poissons_ratio=Quality.closed_range(lower=0.35, upper=0.40),
        ),
        kw_only=True,
        hash=False,
        metadata=IsQualityRelationship().as_dict(),
    )
    """
    Raw carrot root in compression: a Japanese carrot before boiling
    :cite:`ogawa2015carrot`, and Chantenay and Nairobi along the root
    :cite:`morrell2019carrot`.
    """


@dataclass(eq=False, kw_only=True)
class ApplePeel(Peel):
    """
    The peel of an apple.
    """

    material_elasticity: MaterialElasticity | None = field(
        default_factory=lambda: MaterialElasticity(
            youngs_modulus=Quality.closed_range(lower=17.79e6, upper=24.00e6)
        ),
        kw_only=True,
        hash=False,
        metadata=IsQualityRelationship().as_dict(),
    )
    """
    Tensile strips of Fuji and Starkrimson apple peel in both directions
    :cite:`wang2015apple`.
    """

    skin_thickness: SkinThickness | None = field(
        default_factory=lambda: SkinThickness(
            value=Quality.closed_range(lower=33.3e-6, upper=73.1e-6)
        ),
        kw_only=True,
        hash=False,
        metadata=IsQualityRelationship().as_dict(),
    )
    """
    The histological skin thickness of apples across cultivars
    :cite:`homutova2006apple`.
    """


@dataclass(eq=False, kw_only=True)
class AppleFlesh(Flesh):
    """
    The flesh of an apple.
    """

    material_elasticity: MaterialElasticity | None = field(
        default_factory=lambda: MaterialElasticity(
            youngs_modulus=Quality.closed_range(lower=2.7e6, upper=4.6e6)
        ),
        kw_only=True,
        hash=False,
        metadata=IsQualityRelationship().as_dict(),
    )
    """
    Apple flesh in quasi-static compression; under impact it measures about 9 MPa
    :cite:`stropek2022apple`.
    """


@dataclass(eq=False, kw_only=True)
class Apple(Fruit, HasPeel, HasFlesh, HasCore, HasSeeds, HasStem):
    """
    An apple.
    """

    mass_density: MassDensity | None = field(
        default_factory=lambda: MassDensity(
            value=Quality.closed_range(lower=929.0, upper=946.0)
        ),
        kw_only=True,
        hash=False,
        metadata=IsQualityRelationship().as_dict(),
    )
    """
    The density of whole fruits of three early apple cultivars
    :cite:`ozturk2011apple`.
    """

    material_elasticity: MaterialElasticity | None = field(
        default_factory=lambda: MaterialElasticity(
            youngs_modulus=Quality.closed_range(lower=1.63e6, upper=3.94e6),
            poissons_ratio=Quality.closed_range(lower=0.173, upper=0.221),
        ),
        kw_only=True,
        hash=False,
        metadata=IsQualityRelationship().as_dict(),
    )
    """
    The elastic constants of whole Golden Delicious apples from acoustic and
    puncture tests :cite:`grotte2002golden`.
    """


@dataclass(eq=False, kw_only=True)
class AvocadoPeel(Peel):
    """
    The peel of an avocado.
    """

    skin_thickness: SkinThickness | None = field(
        default_factory=lambda: SkinThickness(
            value=Quality.closed_range(lower=1.6e-3, upper=1.6e-3)
        ),
        kw_only=True,
        hash=False,
        metadata=IsQualityRelationship().as_dict(),
    )
    """
    The skin of Hass avocados, from a plant-patent comparison of cultivars
    :cite:`uspp14239avocado`.
    """


@dataclass(eq=False, kw_only=True)
class AvocadoFlesh(Flesh):
    """
    The flesh of an avocado.
    """

    material_elasticity: MaterialElasticity | None = field(
        default_factory=lambda: MaterialElasticity(
            youngs_modulus=Quality.closed_range(lower=0.16e6, upper=2.29e6)
        ),
        kw_only=True,
        hash=False,
        metadata=IsQualityRelationship().as_dict(),
    )
    """
    Slices of Hass avocado flesh, from 2.29 MPa unripe down to 0.16 MPa after five
    days of ripening :cite:`farinas2021avocado`.
    """


@dataclass(eq=False, kw_only=True)
class Avocado(Fruit, HasPeel, HasFlesh, HasPit):
    """
    An avocado.
    """


@dataclass(eq=False, kw_only=True)
class BananaPeel(Peel):
    """
    The peel of a banana.
    """

    skin_thickness: SkinThickness | None = field(
        default_factory=lambda: SkinThickness(
            value=Quality.closed_range(lower=2.95e-3, upper=3.65e-3)
        ),
        kw_only=True,
        hash=False,
        metadata=IsQualityRelationship().as_dict(),
    )
    """
    The peel of green Nendran and Dwarf Cavendish bananas, measured with calipers
    :cite:`kachru1995banana`.
    """


@dataclass(eq=False, kw_only=True)
class BananaFlesh(Flesh):
    """
    The flesh of a banana.
    """

    material_elasticity: MaterialElasticity | None = field(
        default_factory=lambda: MaterialElasticity(
            youngs_modulus=Quality.closed_range(lower=1.09e6, upper=1.75e6)
        ),
        kw_only=True,
        hash=False,
        metadata=IsQualityRelationship().as_dict(),
    )
    """
    Cylinders of banana pulp in compression, from day 1 to day 12 after harvest
    :cite:`ciro2005banana`.
    """


@dataclass(eq=False, kw_only=True)
class Banana(Fruit, HasPeel, HasFlesh):
    """
    A banana.
    """


@dataclass(eq=False, kw_only=True)
class CucumberFlesh(Flesh):
    """
    The flesh of a cucumber.
    """

    material_elasticity: MaterialElasticity | None = field(
        default_factory=lambda: MaterialElasticity(
            youngs_modulus=Quality.closed_range(lower=2.844e6, upper=2.844e6)
        ),
        kw_only=True,
        hash=False,
        metadata=IsQualityRelationship().as_dict(),
    )
    """
    Cucumber flesh in compression with skin and seeds removed, 29 kgf/cm²
    :cite:`kojima1975cucumber`.
    """


@dataclass(eq=False, kw_only=True)
class Cucumber(Fruit, HasPeel, HasFlesh, HasSeedChamber):
    """
    A cucumber, a fruit in the botanical sense.
    """

    mass_density: MassDensity | None = field(
        default_factory=lambda: MassDensity(
            value=Quality.closed_range(lower=980.0, upper=980.0)
        ),
        kw_only=True,
        hash=False,
        metadata=IsQualityRelationship().as_dict(),
    )
    """
    The specific gravity of whole Kurume-H cucumbers :cite:`kojima1975cucumber`.
    """


@dataclass(eq=False, kw_only=True)
class CitrusFruit(Fruit, HasPeel):
    """
    A citrus fruit, such as an orange or a lemon, whose flesh is divided into segments
    (botanically a hesperidium, Plant Ontology PO:0030109).
    """

    citrus_segments: list[CitrusSegment] = field(
        default_factory=list,
        hash=False,
        metadata=IsPartWholeRelationship().as_dict(),
    )
    """
    The segments the flesh of the citrus fruit is divided into.
    """


@dataclass(eq=False, kw_only=True)
class OrangePeel(Peel):
    """
    The peel of an orange.
    """

    mass_density: MassDensity | None = field(
        default_factory=lambda: MassDensity(
            value=Quality.closed_range(lower=746.0, upper=746.0)
        ),
        kw_only=True,
        hash=False,
        metadata=IsQualityRelationship().as_dict(),
    )
    """
    Orange peel, from its quasi-static compression study :cite:`jentzsch2022citrus`.
    """

    material_elasticity: MaterialElasticity | None = field(
        default_factory=lambda: MaterialElasticity(
            youngs_modulus=Quality.closed_range(lower=1.985e4, upper=1.985e4),
            poissons_ratio=Quality.closed_range(lower=-0.09, upper=-0.03),
        ),
        kw_only=True,
        hash=False,
        metadata=IsQualityRelationship().as_dict(),
    )
    """
    Orange peel in quasi-static compression with digital image correlation; the
    Poisson's ratio is the median from 10 to 50 % axial strain
    :cite:`jentzsch2022citrus`.
    """

    skin_thickness: SkinThickness | None = field(
        default_factory=lambda: SkinThickness(
            value=Quality.closed_range(lower=5.4e-3, upper=7.0e-3)
        ),
        kw_only=True,
        hash=False,
        metadata=IsQualityRelationship().as_dict(),
    )
    """
    The mean thickness of orange peel, 6.2 mm, within one standard deviation
    :cite:`jentzsch2022citrus`.
    """


@dataclass(eq=False, kw_only=True)
class Orange(CitrusFruit):
    """
    An orange.
    """

    mass_density: MassDensity | None = field(
        default_factory=lambda: MassDensity(
            value=Quality.closed_range(lower=999.0, upper=1046.0)
        ),
        kw_only=True,
        hash=False,
        metadata=IsQualityRelationship().as_dict(),
    )
    """
    The density of whole oranges, from large to small grades
    :cite:`sharifi2007orange`.
    """

    material_elasticity: MaterialElasticity | None = field(
        default_factory=lambda: MaterialElasticity(
            youngs_modulus=Quality.closed_range(lower=0.584e6, upper=0.584e6),
            poissons_ratio=Quality.closed_range(lower=0.24, upper=0.24),
        ),
        kw_only=True,
        hash=False,
        metadata=IsQualityRelationship().as_dict(),
    )
    """
    The elastic constants of whole oranges in compression, for a drop simulation
    :cite:`namdari2018orange`.
    """


@dataclass(eq=False, kw_only=True)
class PeachPeel(Peel):
    """
    The peel of a peach.
    """

    material_elasticity: MaterialElasticity | None = field(
        default_factory=lambda: MaterialElasticity(
            youngs_modulus=Quality.closed_range(lower=1.08e6, upper=1.08e6)
        ),
        kw_only=True,
        hash=False,
        metadata=IsQualityRelationship().as_dict(),
    )
    """
    Tensile strips of Red Haven peach skin; the strips include flesh, so their
    thickness is no peel thickness :cite:`cermak2011peach`.
    """


@dataclass(eq=False, kw_only=True)
class Peach(Fruit, HasPeel, HasFlesh, HasPit, HasStem):
    """
    A peach.
    """

    material_elasticity: MaterialElasticity | None = field(
        default_factory=lambda: MaterialElasticity(
            youngs_modulus=Quality.closed_range(lower=0.89e6, upper=0.89e6),
            poissons_ratio=Quality.closed_range(lower=0.29, upper=0.29),
        ),
        kw_only=True,
        hash=False,
        metadata=IsQualityRelationship().as_dict(),
    )
    """
    The elastic constants of halved peaches in compression, for a drop simulation
    :cite:`kabas2015peach`.
    """


@dataclass(eq=False, kw_only=True)
class StrawberryFlesh(Flesh):
    """
    The flesh of a strawberry, its cortex.
    """

    material_elasticity: MaterialElasticity | None = field(
        default_factory=lambda: MaterialElasticity(
            youngs_modulus=Quality.closed_range(lower=0.274e6, upper=0.614e6)
        ),
        kw_only=True,
        hash=False,
        metadata=IsQualityRelationship().as_dict(),
    )
    """
    The cortex of strawberries in uniaxial compression from 35 °C down to 1 °C
    :cite:`an2022strawberry`.
    """


@dataclass(eq=False, kw_only=True)
class StrawberryPith(Pith):
    """
    The pith of a strawberry.
    """

    material_elasticity: MaterialElasticity | None = field(
        default_factory=lambda: MaterialElasticity(
            youngs_modulus=Quality.closed_range(lower=0.333e6, upper=0.695e6)
        ),
        kw_only=True,
        hash=False,
        metadata=IsQualityRelationship().as_dict(),
    )
    """
    The pith of strawberries in uniaxial compression from 35 °C down to 1 °C
    :cite:`an2022strawberry`.
    """


@dataclass(eq=False, kw_only=True)
class Strawberry(Fruit, HasPeel, HasFlesh, HasPith):
    """
    A strawberry.
    """


@dataclass(eq=False, kw_only=True)
class WatermelonPeel(Peel):
    """
    The rind of a watermelon.
    """

    material_elasticity: MaterialElasticity | None = field(
        default_factory=lambda: MaterialElasticity(
            youngs_modulus=Quality.closed_range(lower=4.937e6, upper=5.357e6)
        ),
        kw_only=True,
        hash=False,
        metadata=IsQualityRelationship().as_dict(),
    )
    """
    The green outer rind of Crimson Sweet and Charleston Gray watermelons
    :cite:`sadrnia2008watermelon`.
    """

    skin_thickness: SkinThickness | None = field(
        default_factory=lambda: SkinThickness(
            value=Quality.closed_range(lower=12.11e-3, upper=20.28e-3)
        ),
        kw_only=True,
        hash=False,
        metadata=IsQualityRelationship().as_dict(),
    )
    """
    The thickness of the whole watermelon rind :cite:`khater2016watermelon`.
    """


@dataclass(eq=False, kw_only=True)
class WatermelonFlesh(Flesh):
    """
    The flesh of a watermelon.
    """

    material_elasticity: MaterialElasticity | None = field(
        default_factory=lambda: MaterialElasticity(
            youngs_modulus=Quality.closed_range(lower=0.396e6, upper=0.536e6)
        ),
        kw_only=True,
        hash=False,
        metadata=IsQualityRelationship().as_dict(),
    )
    """
    Red watermelon flesh in compression :cite:`sadrnia2008watermelon`.
    """


@dataclass(eq=False, kw_only=True)
class Watermelon(Fruit, HasPeel, HasFlesh, HasSeeds):
    """
    A watermelon.
    """

    mass_density: MassDensity | None = field(
        default_factory=lambda: MassDensity(
            value=Quality.closed_range(lower=927.84, upper=973.57)
        ),
        kw_only=True,
        hash=False,
        metadata=IsQualityRelationship().as_dict(),
    )
    """
    The density of whole watermelons across storage conditions
    :cite:`khater2016watermelon`.
    """


@dataclass(eq=False, kw_only=True)
class EggPart(HasRootBody):
    """
    A part of an egg, such as its shell or its yolk.
    """


@dataclass(eq=False, kw_only=True)
class Shell(HasRootBody, HasSkinThickness):
    """
    The hard outer covering of an object such as an egg or a nut.
    """


@dataclass(eq=False, kw_only=True)
class Albumen(EggPart):
    """
    The white of an egg, the liquid around its yolk.
    """


@dataclass(eq=False, kw_only=True)
class Yolk(EggPart):
    """
    The yellow centre of an egg.
    """


@dataclass(eq=False, kw_only=True)
class EggShell(Shell):
    """
    The shell of a hen's egg.
    """

    material_elasticity: MaterialElasticity | None = field(
        default_factory=lambda: MaterialElasticity(
            youngs_modulus=Quality.closed_range(lower=19.0e9, upper=19.0e9)
        ),
        kw_only=True,
        hash=False,
        metadata=IsQualityRelationship().as_dict(),
    )
    """
    Full rings of eggshell in compression, around the egg :cite:`lin1995eggshell`.
    """

    skin_thickness: SkinThickness | None = field(
        default_factory=lambda: SkinThickness(
            value=Quality.closed_range(lower=0.403e-3, upper=0.444e-3)
        ),
        kw_only=True,
        hash=False,
        metadata=IsQualityRelationship().as_dict(),
    )
    """
    The shell thickness of eggs of four hen genotypes :cite:`hammershoj2021egg`.
    """


@dataclass(eq=False, kw_only=True)
class Egg(Food, HasShell, HasAlbumen, HasYolk):
    """
    A hen's egg.
    """

    mass_density: MassDensity | None = field(
        default_factory=lambda: MassDensity(
            value=Quality.closed_range(lower=1085.0, upper=1085.0)
        ),
        kw_only=True,
        hash=False,
        metadata=IsQualityRelationship().as_dict(),
    )
    """
    Whole fresh hen's eggs, weighed in air and in water; the density falls as an egg
    ages and loses mass :cite:`tsarenko2017egg`.
    """


@dataclass(eq=False)
class Salt(Food):
    """
    A pack or container of salt (e.g., salt shaker or salt can).
    """


@dataclass(eq=False)
class CoffeeTable(Table):
    """
    A coffee table.
    """


@dataclass(eq=False)
class DiningTable(Table, HasLegs):
    """
    A dining table.
    """


@dataclass(eq=False)
class SideTable(Table):
    """
    A side table.
    """

    _synonyms = {"bedside"}


@dataclass(eq=False)
class Desk(Table, HasLegs):
    """
    A desk.
    """


@dataclass(eq=False)
class Chair(Furniture, HasRootBody):
    """
    Abstract class for chairs.
    """


@dataclass(eq=False)
class OfficeChair(Chair):
    """
    An office chair.
    """


@dataclass(eq=False)
class Armchair(Chair):
    """
    An armchair.
    """


@dataclass(eq=False)
class TrashCan(HasCaseAsRootBody, Furniture):
    """
    Abstract class for Trash Can.
    """

    @classproperty
    def _hole_direction_axis(cls) -> Vector3:
        return Vector3.Z()


@dataclass(eq=False)
class Shelf(Cabinet, HasShelfLayers):
    """
    A shelving unit whose storage surfaces are modelled as explicit shelf layers.

    Each layer is a separate :class:`ShelfLayer` annotation, so objects can be placed on
    and reasoned about per layer.
    """


@dataclass(eq=False)
class Bed(Furniture):
    """
    A bed.
    """


@dataclass(eq=False)
class Sofa(Furniture, HasSupportingSurface):
    """
    A sofa.
    """


@dataclass(eq=False)
class Sink(HasRootBody):
    """
    A sink.
    """


@dataclass(eq=False)
class Kettle(CookingContainer): ...


@dataclass(eq=False)
class Decor(HasRootBody): ...


@dataclass(eq=False)
class WallDecor(Decor):
    """
    Wall decorations.
    """


@dataclass(eq=False)
class Cloth(IsGraspable):
    """
    A piece of fabric, such as a towel or a rag.
    """


@dataclass(eq=False)
class Poster(WallDecor):
    """
    A poster.
    """


@dataclass(eq=False)
class WallPanel(HasRootBody):
    """
    A wall panel.
    """


@dataclass(eq=False)
class Potato(Vegetable):
    """
    A potato.
    """

    mass_density: MassDensity | None = field(
        default_factory=lambda: MassDensity(
            value=Quality.closed_range(lower=1080.0, upper=1095.0)
        ),
        kw_only=True,
        hash=False,
        metadata=IsQualityRelationship().as_dict(),
    )
    """
    The specific gravity targeted for processing potato cultivars, so table potatoes
    can be lighter :cite:`wang2017potato`.
    """

    material_elasticity: MaterialElasticity | None = field(
        default_factory=lambda: MaterialElasticity(
            youngs_modulus=Quality.closed_range(lower=2.17e6, upper=5.18e6),
            poissons_ratio=Quality.closed_range(lower=0.406, upper=0.441),
        ),
        kw_only=True,
        hash=False,
        metadata=IsQualityRelationship().as_dict(),
    )
    """
    Cylinders of potato flesh in axial compression :cite:`solomon2006potato`, and a
    Poisson's ratio back-calculated with a finite element model
    :cite:`celik2026potato`.
    """


@dataclass(eq=False)
class GarbageBin(HasRootBody):
    """
    A garbage bin.
    """


@dataclass(eq=False)
class Drone(HasRootBody): ...


@dataclass(eq=False)
class ProcthorBox(HasRootBody): ...


@dataclass(eq=False)
class Houseplant(HasRootBody):
    """
    A houseplant.
    """


@dataclass(eq=False)
class SprayBottle(Bottle):
    """
    A spray bottle.
    """


@dataclass(eq=False)
class Vase(Container):
    """
    A vase.
    """


@dataclass(eq=False)
class Book(IsGraspable):
    """
    A book.
    """

    mass_density: MassDensity | None = field(
        default_factory=lambda: MassDensity.paper,
        kw_only=True,
        hash=False,
        metadata=IsQualityRelationship().as_dict(),
    )
    """
    The mass density of the paper the book is made of.
    """

    book_front: BookFront | None = None
    """
    The front cover of the book.
    """


@dataclass(eq=False)
class BookFront(HasRootBody): ...


@dataclass(eq=False)
class SaltPepperShaker(SaltContainer):
    """
    A salt and pepper shaker.
    """


@dataclass(eq=False)
class Cutlery(HasHandle, Tableware):
    """
    A piece of cutlery, lying flat, grasped from above and across its length.
    """

    mass_density: MassDensity | None = field(
        default_factory=lambda: MassDensity.steel,
        kw_only=True,
        hash=False,
        metadata=IsQualityRelationship().as_dict(),
    )
    """
    The mass density of the steel the cutlery is made of.
    """

    material_elasticity: MaterialElasticity | None = field(
        default_factory=lambda: MaterialElasticity.steel,
        kw_only=True,
        hash=False,
        metadata=IsQualityRelationship().as_dict(),
    )
    """
    The elasticity of the steel the cutlery is made of.
    """

    def default_grasp_candidates(self) -> list[GraspCandidate]:
        """
        :return: The grasp from above that closes across the piece, which lies flat.
        :raises NoGraspGeometry: If the root body has no collision geometry to tell
            which way the piece lies.
        """
        if not self.root.has_collision():
            raise NoGraspGeometry(self)
        bounding_box = self.root.collision.as_bounding_box_collection_in_frame(
            self.root
        ).bounding_box()
        along_x = bounding_box.x_interval.upper - bounding_box.x_interval.lower
        along_y = bounding_box.y_interval.upper - bounding_box.y_interval.lower
        finger_axis = Vector3.NEGATIVE_Y() if along_x >= along_y else Vector3.X()
        return [
            GraspCandidate(
                self,
                Pose(
                    orientation=RotationMatrix.from_vectors(
                        x=Vector3.NEGATIVE_Z(), y=finger_axis
                    ).quaternion,
                    reference_frame=self.root,
                ),
            )
        ]

    @classproperty
    def grasp_strategy(cls) -> GraspStrategy:
        return GraspAcrossTheLength()


@dataclass(eq=False)
class Fork(Cutlery):
    """
    A fork.
    """


@dataclass(eq=False)
class Knife(Cutlery):
    """
    A butter knife.
    """


@dataclass(eq=False)
class Spoon(Cutlery, IsPerceivable):
    """
    A spoon.
    """


@dataclass(eq=False)
class Pencil(IsGraspable):
    """
    A pencil.
    """


@dataclass(eq=False)
class Pen(IsGraspable):
    """
    A pen.
    """


@dataclass(eq=False)
class ShapeSortingPiece(IsGraspable):
    """
    A loose piece of a shape-sorting toy, which drops through the hole of its
    own outline.
    """

    grasp_height: float = field(kw_only=True)
    """
    How far above its bottom face the piece is gripped.

    Gripped low, the piece hangs as little as possible below the fingers when it is let
    go of over its hole; how low depends on how far the closing fingers reach past the
    point they are brought to.
    """

    def default_grasp_candidates(self) -> list[GraspCandidate]:
        """
        :return: The grasp from above, on the piece's vertical axis at
            :attr:`grasp_height`, closing along the piece's own x.
        :raises NoGraspGeometry: If the root body has no collision geometry to find the
            bottom face in.
        """
        if not self.root.has_collision():
            raise NoGraspGeometry(self)
        bottom_face = (
            self.root.collision.as_bounding_box_collection_in_frame(self.root)
            .bounding_box()
            .min_z
        )
        return [
            GraspCandidate(
                self,
                Pose(
                    position=Point3(
                        0.0,
                        0.0,
                        bottom_face + self.grasp_height,
                        reference_frame=self.root,
                    ),
                    orientation=RotationMatrix.from_vectors(
                        x=Vector3.NEGATIVE_Z(), y=Vector3.X()
                    ).quaternion,
                    reference_frame=self.root,
                ),
            )
        ]


@dataclass(eq=False)
class Baseball(IsGraspable):
    """
    A baseball.
    """


@dataclass(eq=False)
class BottleCap(HasMovableJoint):
    """
    A cap that closes a bottle, typically mounted on a screw joint.
    """

    def calculate_self_T_movable_joint(
        self, axis: Vector3
    ) -> HomogeneousTransformationMatrix:
        """
        A cap screws about its own centre line, so its thread sits at its origin.

        :param axis: The thread axis, in the cap frame.
        :return: The identity: the thread's frame is the cap's frame.
        """
        return HomogeneousTransformationMatrix()


@dataclass(eq=False)
class Agent(HasRootBody):
    """
    Represents an entity in the world that can act, move, or be controlled.

    Agents are dynamic bodies with semantic meaning — they may have intent, behavior, or
    be controlled by external or internal logic. Examples include robots, humans, or
    other autonomous actors.
    """


@dataclass(eq=False)
class Human(Agent, IsPerceivable):
    """
    Represents a human agent in the environment.

    A Person is an Agent that is not robotically actuated and does not provide kinematic
    chains, end_effectors, or robot-specific components.

    This class exists primarily for semantic distinction, so that algorithms can treat
    human agents differently from robots if needed.
    """


@dataclass(eq=False)
class Parcel(IsGraspable):
    """
    A parcel, as handled in a warehouse.
    """

    mass_density: MassDensity | None = field(
        default_factory=lambda: MassDensity.cardboard,
        kw_only=True,
        hash=False,
        metadata=IsQualityRelationship().as_dict(),
    )
    """
    The mass density of the cardboard the parcel is made of.
    """


@dataclass(eq=False)
class SemanticEnvironmentAnnotation(HasRootBody):
    """
    Represents a semantic annotation of the environment.
    """

    def obstacle_entities(
        self,
        *,
        search_space: BoundingBoxCollection[VolumetricBoundingBox, Point3],
        excluded_bodies: Iterable[Body] = (),
    ) -> list[Body]:
        """
        Collect the obstacle bodies to consider within ``search_space``.

        Filters out robot bodies so a robot does not treat itself as an obstacle, and
        bodies without meaningful collision geometry.

        :param search_space: The search space; its reference frame is used to look up
            the owning world.
        :param excluded_bodies: Bodies that are no obstacles either, such as an object
            to be reached into.
        :return: The obstacle bodies to consider.
        """
        world = search_space.reference_frame._world
        left_out = set(world.robot_bodies_with_collision) | set(excluded_bodies)
        return [body for body in self.bodies_with_collision if body not in left_out]

    def build_bloated_obstacle_collection(
        self,
        *,
        search_space: BoundingBoxCollection[VolumetricBoundingBox, Point3],
        semantic_wall_annotation: Wall | None = None,
        bloat_obstacles: float = 0.0,
        bloat_walls: float = 0.0,
        obstacle_height_clearance: float = 0.01,
        excluded_bodies: Iterable[Body] = (),
    ) -> BoundingBoxCollection[VolumetricBoundingBox, Point3]:
        """
        Collect and bloat this annotation's obstacle bounding boxes.

        Applies independent bloat amounts to obstacles and walls.

        :param search_space: The search space; its reference frame is used as the
            origin.
        :param semantic_wall_annotation: An optional wall annotation, bloated by its
            own rule (see :meth:`Wall.bloated_bounding_box_collection`).
        :param bloat_obstacles: Amount to expand each obstacle bounding box
            symmetrically in x and y.
        :param bloat_walls: Amount to expand wall bounding boxes in their thinner
            dimension.
        :param obstacle_height_clearance: Amount to expand every obstacle bounding box
            in z, regardless of ``bloat_obstacles``/``bloat_walls``.
        :param excluded_bodies: Bodies that are no obstacles, see
            :meth:`obstacle_entities`.
        :return: A collection of the bloated obstacle and wall bounding boxes.
        """
        world_root = search_space.reference_frame
        origin = HomogeneousTransformationMatrix(reference_frame=world_root)

        entities_to_consider = self.obstacle_entities(
            search_space=search_space, excluded_bodies=excluded_bodies
        )

        collections = (
            entity.collision.as_bounding_box_collection_at_origin(origin)
            for entity in entities_to_consider
        )
        obstacle_bounding_boxes = BoundingBoxCollection.merge_all(
            collections, world_root
        )

        bloated_obstacles = BoundingBoxCollection(
            [
                bounding_box.bloat(
                    bloat_obstacles, bloat_obstacles, obstacle_height_clearance
                )
                for bounding_box in obstacle_bounding_boxes
            ],
            world_root,
        )

        if semantic_wall_annotation is not None:
            bloated_obstacles = bloated_obstacles.merge(
                semantic_wall_annotation.bloated_bounding_box_collection(
                    origin, bloat_walls, obstacle_height_clearance
                )
            )

        return bloated_obstacles


@dataclass(eq=False)
class RoomWithWallsAndDoors(Room):
    """
    A room with a type description (e.g., Ktichen) and walls and doors.
    """

    room_type: str | None = field(kw_only=True, default=None)
    """
    Description of the type of the room in natural language.
    """

    walls: list[Wall] = field(kw_only=True, default_factory=list)
    """
    The walls enclosing this room.
    """

    doors: list[Door] = field(kw_only=True, default_factory=list)
    """
    The doors of the room.
    """


@dataclass(eq=False)
class DoorWithType(Door):
    """
    A Door that has a type description, e.g. "main entrance".
    """

    type_description: str | None = field(kw_only=True, default=None)


@dataclass(eq=False)
class Leg(HasRootBody):
    """
    A leg that supports a piece of furniture.
    """


@dataclass(eq=False)
class Cooktop(HasRootBody):
    """
    A cooktop surface for cooking.
    """


@dataclass(eq=False)
class Tool(IsGraspable, ABC):
    """
    A tool that is held by a robot's end effector to act on other bodies.
    """

    def _end_effector_name(self) -> PrefixedName:
        """
        :return: The name of the body that acts as the tool's tip, derived from the
            tool's root name.
        """
        root_name = self.root.name
        return PrefixedName(f"{root_name.name}_end_effector", root_name.prefix)

    def _find_end_effector_body(self) -> Body | None:
        if self.root._world is None:
            return None

        end_effector_name = self._end_effector_name()
        for body in self.root._world.bodies:
            if (
                body.name == end_effector_name
                or body.name.name == end_effector_name.name
            ):
                return body
        return None

    def get_tool_frame(self) -> Body:
        """
        :return: The body acting as the tool's tip, or the tool's root if no dedicated
            tip body exists.
        """
        end_effector = self._find_end_effector_body()
        if end_effector is None:
            return self.root
        return end_effector

    @abstractmethod
    def tool_alignment(self, target: Body | Pose) -> list[AlignmentPair]:
        """
        :param target: The body or pose the tool acts on.
        :return: The normal pairs that must stay aligned while the tool acts on the
            target.
        """


@dataclass(eq=False)
class ToolWithHandle(Tool, HasHandle, ABC):
    """
    A tool held by its handle, acting through a tip body located ahead of its root.

    If the tip body does not exist yet, it is created and rigidly connected to the
    tool's root.
    """

    def __post_init__(self):
        super().__post_init__()
        if self.root._world is None:
            return
        if self._find_end_effector_body() is not None:
            return
        self._create_end_effector_body()

    def _create_end_effector_body(self):
        world = self.root._world
        tip_body = Body(name=self._end_effector_name())
        with world.modify_world():
            world.add_kinematic_structure_entity(tip_body)
            world.add_connection(
                FixedConnection(
                    parent=self.root,
                    child=tip_body,
                    parent_T_connection_expression=HomogeneousTransformationMatrix.from_xyz_rpy(
                        x=0.1,
                        y=0.0,
                        z=0.0,
                        reference_frame=self.root,
                    ),
                )
            )


@dataclass(eq=False)
class Whisk(ToolWithHandle):
    """
    A whisk for mixing the contents of containers.
    """

    def tool_alignment(self, target: Body | Pose) -> list[AlignmentPair]:
        return [
            AlignmentPair(
                tip_normal=Vector3(-1, 0, 0, reference_frame=self.root),
                reference_normal=Vector3(0, 0, 1, reference_frame=target),
            )
        ]


@dataclass(eq=False)
class CuttingKnife(ToolWithHandle):
    """
    A knife for cutting food objects.
    """

    mass_density: MassDensity | None = field(
        default_factory=lambda: MassDensity.steel,
        kw_only=True,
        hash=False,
        metadata=IsQualityRelationship().as_dict(),
    )
    """
    The mass density of the steel the knife is made of.
    """

    material_elasticity: MaterialElasticity | None = field(
        default_factory=lambda: MaterialElasticity.steel,
        kw_only=True,
        hash=False,
        metadata=IsQualityRelationship().as_dict(),
    )
    """
    The elasticity of the steel the knife is made of.
    """

    def tool_alignment(self, target: Body | Pose) -> list[AlignmentPair]:
        return [
            AlignmentPair(
                tip_normal=Vector3(1, 0, 0, reference_frame=self.root),
                reference_normal=Vector3(0, 1, 0, reference_frame=target),
            ),
            AlignmentPair(
                tip_normal=Vector3(0, 0, 1, reference_frame=self.root),
                reference_normal=Vector3(0, 0, 1, reference_frame=target),
            ),
        ]


@dataclass(eq=False)
class PouringCup(Tool, Container):
    """
    A cup for pouring liquids into containers.
    """

    def tool_alignment(self, target: Body | Pose) -> list[AlignmentPair]:
        return [
            AlignmentPair(
                tip_normal=Vector3(0, 0, 1, reference_frame=self.root),
                reference_normal=Vector3(0, 0, 1, reference_frame=target),
            )
        ]


@dataclass(eq=False)
class Sponge(Tool):
    """
    A sponge for wiping surfaces.

    .. note:: The sponge is grasped so its local Z axis points away from the gripper,
        which is why its alignments use negative Z normals unlike the other tools.
    """

    mass_density: MassDensity | None = field(
        default_factory=lambda: MassDensity.polyurethane_foam,
        kw_only=True,
        hash=False,
        metadata=IsQualityRelationship().as_dict(),
    )
    """
    The mass density of the polyurethane foam the sponge is made of.
    """

    def tool_alignment(self, target: Body | Pose) -> list[AlignmentPair]:
        if isinstance(target, Body):
            return [
                AlignmentPair(
                    tip_normal=Vector3(0, 0, 1, reference_frame=self.root),
                    reference_normal=Vector3(0, 0, -1, reference_frame=target),
                )
            ]
        return [
            AlignmentPair(
                tip_normal=Vector3(0, 0, -1, reference_frame=self.root),
                reference_normal=self._pose_surface_normal(target),
            )
        ]

    def _pose_surface_normal(self, pose: Pose) -> Vector3:
        """
        :param pose: The pose whose surface normal is computed.
        :return: The pose's local Z axis expressed in the pose's reference frame.
        """
        reference_frame = (
            pose.reference_frame if pose.reference_frame is not None else self.root
        )
        rotation = pose.rotation_matrix.to_np()[:3, :3]
        return Vector3.from_iterable(
            rotation @ np.array([0.0, 0.0, 1.0]),
            reference_frame=reference_frame,
        )


@dataclass(eq=False)
class Microwave(IsStorageSpace, HasDoors):
    """
    A microwave oven, a kitchen appliance with a door that heats food placed inside it
    using microwave radiation.
    """


@dataclass(eq=False)
class Hood(HasRootBody):
    """
    A range hood mounted above a cooktop that vents cooking fumes.
    """


@dataclass(eq=False)
class Toaster(HasRootBody):
    """
    A countertop appliance for toasting slices of bread.
    """


@dataclass(eq=False)
class CoffeeMachine(HasRootBody):
    """
    A countertop appliance that brews coffee.
    """

    _synonyms = {"coffe"}
