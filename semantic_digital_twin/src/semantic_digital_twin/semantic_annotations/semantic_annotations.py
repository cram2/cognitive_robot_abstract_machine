from __future__ import annotations

import math
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import ClassVar, Iterable, Optional, Self, Tuple, TYPE_CHECKING, Union

import numpy as np
from typing_extensions import List, Type, Dict

from krrood.ormatic.utils import classproperty
from probabilistic_model.bayesian_network.bayesian_network import Node
from semantic_digital_twin.datastructures.alignment import AlignmentPair
from krrood.symbolic_math import symbolic_math
from random_events.interval import closed
from random_events.product_algebra import SimpleEvent
from semantic_digital_twin.datastructures.prefixed_name import PrefixedName
from semantic_digital_twin.datastructures.variables import SpatialVariables
from semantic_digital_twin.exceptions import (
    InvalidPlaneDimensions,
    InvalidHingeActiveAxis,
    MissingSemanticAnnotationError,
    MechanicalJointAlreadyMounted,
)
from semantic_digital_twin.reasoning.predicates import InsideOf
from semantic_digital_twin.semantic_annotations.mixins import (
    HasSupportingSurface,
    HasRootRegion,
    HasDrawers,
    HasDoors,
    HasHandle,
    HasCaseAsRootBody,
    HasMechanicalJoint,
    HasApertures,
    IsPerceivable,
    HasRootBody,
    IsStorageSpace,
    HasLegs,
    HasSink,
)
from semantic_digital_twin.spatial_types import (
    Point3,
    HomogeneousTransformationMatrix,
    Vector3,
    RotationMatrix,
)
from semantic_digital_twin.spatial_types.derivatives import DerivativeMap
from semantic_digital_twin.spatial_types.spatial_types import Pose
from semantic_digital_twin.world_description.connections import (
    RevoluteConnection,
    PrismaticConnection,
    FixedConnection,
)
from semantic_digital_twin.world_description.degree_of_freedom import (
    DegreeOfFreedomLimits,
)
from semantic_digital_twin.world_description.geometry import Scale, Mesh, Color
from semantic_digital_twin.world_description.shape_collection import (
    BoundingBoxCollection,
    ShapeCollection,
)
from semantic_digital_twin.world_description.world_entity import (
    SemanticAnnotation,
    Body,
    Region,
    Connection,
)

if TYPE_CHECKING:
    from semantic_digital_twin.world import World


@dataclass(eq=False)
class Furniture(SemanticAnnotation, ABC):
    """
    A semantic annotation that represents a piece of furniture.
    """


@dataclass(eq=False)
class Handle(HasRootBody):
    """
    A handle is a physical entity that can be grasped by a hand or a robotic gripper to open or close an object.
    """

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
        scale: Scale = Scale(0.1, 0.02, 0.02),
        thickness: float = 0.005,
    ) -> Self:
        handle_event = cls._create_handle_geometry(scale=scale).as_composite_set()

        inner_box = cls._create_handle_geometry(
            scale=scale, thickness=thickness
        ).as_composite_set()

        handle_event -= inner_box

        handle_body = Body(name=name)
        collision = BoundingBoxCollection.from_event(
            handle_body, handle_event
        ).as_shapes()
        handle_body.collision = collision
        handle_body.visual = collision
        return cls._create_with_connection_in_world(
            name, world, handle_body, world_root_T_self
        )

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


@dataclass(eq=False)
class Dishwasher(HasCaseAsRootBody, HasDoors, HasDrawers):
    """
    A dishwasher is a kitchen appliance used for cleaning dishes, utensils, and cookware. It typically has a front door that opens to reveal racks for loading dirty items and a control panel for selecting wash cycles.
    """

    @classproperty
    def hole_direction(self) -> Vector3:
        return Vector3.NEGATIVE_X()


@dataclass(eq=False)
class Aperture(HasRootRegion):
    """
    An opening in a physical entity.
    An example is like a hole in a wall that can be used to enter a room.
    """

    @classmethod
    def create_with_new_region_in_world(
        cls,
        name: PrefixedName,
        world: World,
        world_root_T_self: Optional[HomogeneousTransformationMatrix] = None,
        connection_limits: Optional[DegreeOfFreedomLimits] = None,
        active_axis: Vector3 = Vector3.Z(),
        connection_multiplier: float = 1.0,
        connection_offset: float = 0.0,
        *,
        scale: Scale = Scale(),
    ) -> Self:
        """
        Create a new semantic annotation with a new region in the given world.
        """
        aperture_region = Region(name=name)

        scale_event = scale.to_simple_event().as_composite_set()
        aperture_geometry = BoundingBoxCollection.from_event(
            aperture_region, scale_event
        ).as_shapes()
        aperture_region.area = aperture_geometry

        return cls._create_with_connection_in_world(
            name, world, aperture_region, world_root_T_self
        )

    @classmethod
    def create_with_new_region_in_world_from_body(
        cls,
        name: PrefixedName,
        world: World,
        body: Body,
        parent_T_self: Optional[HomogeneousTransformationMatrix] = None,
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

    def _mount_strategy(self, main_has_root_body_annotation: HasRootBody) -> None:
        # An aperture cuts its shape out of the whole's geometry, then mounts as a child.
        self._remove_aperture_geometry_from_parent(main_has_root_body_annotation)
        super()._mount_strategy(main_has_root_body_annotation)

    def _remove_aperture_geometry_from_parent(self, parent: HasRootBody):
        """
        Remove the geometry of the aperture from the parent body's collision and visual geometry.

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
            parent.root, new_wall_event
        ).as_shapes()

        parent.root.collision = new_bounding_box_collection
        parent.root.visual = new_bounding_box_collection


@dataclass(eq=False)
class MechanicalJoint(HasRootBody):
    """
    A mechanical joint is a physical entity that connects two bodies and allows one to move along or around a fixed axis
    """

    def _mount_strategy(self, main_has_root_body_annotation: HasRootBody) -> None:
        """
        Inserts the joint between the whole (``main_has_root_body_annotation``) and the whole's
        current parent, preserving the whole's ancestry.
        So
        whole_parent -(fixed)-> whole
        becomes
        whole_parent -(active)-> joint -(fixed)-> whole. The joint keeps its active connection
        (now anchored at the whole's parent); the whole hangs rigidly off the joint.
        """
        if (
            main_has_root_body_annotation.root.parent_kinematic_structure_entity
            == self.root
        ):
            return
        # used instead of World.compute_child_kinematic_structure_entities because its memoized
        if list(self._world.kinematic_structure.successors(self.root.index)):
            raise MechanicalJointAlreadyMounted(self, main_has_root_body_annotation)

        self._world.move_branch(
            self.root,
            main_has_root_body_annotation.root.parent_kinematic_structure_entity,
        )
        main_has_root_body_annotation._world.move_branch(
            main_has_root_body_annotation.root, self.root
        )

    @property
    def position(self):
        return self.root.parent_connection.position

    @position.setter
    def position(self, value):
        self.root.parent_connection.position = value


@dataclass(eq=False)
class Hinge(MechanicalJoint):
    """
    A hinge is a physical entity that connects two bodies and allows one to rotate around a fixed axis.
    """

    @classproperty
    def _parent_connection_type(self) -> Type[Connection]:
        return RevoluteConnection


@dataclass(eq=False)
class Slider(MechanicalJoint):
    """
    A Slider is a physical entity that connects two bodies and allows one to linearly translate along a fixed axis.
    """

    @classproperty
    def _parent_connection_type(self) -> Type[Connection]:
        return PrismaticConnection


@dataclass(eq=False)
class EntryWay(Aperture): ...


@dataclass(eq=False)
class Door(HasHandle, HasMechanicalJoint):
    """
    A door is a physical entity that has covers an opening, has a movable body and a handle.
    """

    entry_way: Optional[EntryWay] = field(default=None)
    """
    The entry way of the door.
    """

    @classmethod
    def create_with_new_body_in_world(
        cls,
        name: PrefixedName,
        world: World,
        world_root_T_self: Optional[HomogeneousTransformationMatrix] = None,
        connection_limits: Optional[DegreeOfFreedomLimits] = None,
        active_axis: Vector3 = Vector3.Z(),
        connection_multiplier: float = 1.0,
        connection_offset: float = 0.0,
        *,
        scale: Scale = Scale(0.03, 1, 2),
    ) -> Self:
        if not (scale.x < scale.y and scale.x < scale.z):
            raise InvalidPlaneDimensions(scale, clazz=Door)

        # This creates an event around the scale, so if the scale is 1 the event will be from -0.5 to 0.5.
        door_event = scale.to_simple_event().as_composite_set()
        door_body = Body(name=name)
        bounding_box_collection = BoundingBoxCollection.from_event(
            door_body, door_event
        )
        collision = bounding_box_collection.as_shapes()
        door_body.collision = collision
        door_body.visual = collision

        entry_way_name = PrefixedName(name.name + "entry_way", name.prefix)
        entry_way_region_name = PrefixedName(
            name.name + "entry_way_region", name.prefix
        )

        entry_way_region = Region(
            name=entry_way_region_name,
            area=ShapeCollection([Mesh.from_trimesh(mesh=door_body.combined_mesh)]),
        )
        entry_way_region.area.dye_shapes(Color(R=1.0, G=1.0, B=1.0, A=0.2))
        entry_way = EntryWay(name=entry_way_name, root=entry_way_region)
        world.add_region(entry_way.root)
        world.add_connection(FixedConnection(door_body, entry_way.root))
        world.add_semantic_annotation(entry_way)

        door = cls._create_with_connection_in_world(
            name, world, door_body, world_root_T_self
        )
        door.entry_way = entry_way
        return door

    def calculate_world_T_hinge_based_on_handle(
        self, opening_axis: Vector3
    ) -> HomogeneousTransformationMatrix:
        """
        Calculate the door pivot point based on the handle position and the door scale. The pivot point is on the opposite
        side of the handle.
        :return: The transformation matrix defining the door's pivot point.
        """
        if self.handle is None:
            raise MissingSemanticAnnotationError(self.__class__, Handle)

        connection = self.handle.root.parent_connection
        door_P_handle = connection.origin_expression.to_position()
        scale = self.root.collision.scale
        world_T_door = self.root.global_transform

        match opening_axis.to_np().tolist():
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
                raise InvalidHingeActiveAxis(axis=opening_axis)

        world_T_hinge = world_T_door @ door_T_hinge

        return world_T_hinge


@dataclass(eq=False)
class DoubleDoor(SemanticAnnotation):
    """
    A semantic annotation that represents a double door with left and right doors.
    """

    door_0: Door = field(kw_only=True)
    door_1: Door = field(kw_only=True)

    def calculate_left_right_door_from_view_point(
        self, world_T_view_point: HomogeneousTransformationMatrix
    ) -> Tuple[Door, Door]:
        """
        Calculate which door is the left and which is the right door based on a given view point.

        :param world_T_view_point: The transformation matrix of the view point.

        :return: A tuple containing the left and right door. the first door is the left door, the second door is the right door.
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
class Drawer(Furniture, HasCaseAsRootBody, HasHandle, HasMechanicalJoint):
    @classproperty
    def hole_direction(self) -> Vector3:
        return Vector3.Z()


@dataclass(eq=False)
class Elevator(HasRootBody):
    """
    An elevator in the world, consists of three walls a floor, double doors for entering and a prismatic drive that moves
    the elevator to other floors.

    ``scale`` (passed to :meth:`create_with_new_bodies_in_world`) defines the cabin as follows:
    ``scale.x`` is the cabin depth, along the axis the double doors face and open (the back wall
    sits opposite the doors); ``scale.y`` is the cabin width (the two side walls); ``scale.z`` is
    the cabin height, from floor level up to the ceiling.
    """

    WALL_THICKNESS: ClassVar[float] = 0.05
    """
    Thickness used for the three cabin walls and the two door leaves.
    """

    doors: DoubleDoor = field(kw_only=True, default=None)
    """
    Double doors through which one can access the elevator
    """

    drive: Slider = field(kw_only=True, default=None)
    """
    Prismatic drive that moves the elevator to other floors.
    """

    walls: List[Wall] = field(kw_only=True, default=None)
    """
    Other 3 walls of the elevator cabin, besides the double doors.
    """

    floor: Floor = field(kw_only=True, default=None)
    """
    Floor of the elevator cabin
    """

    anker_point: Body = field(kw_only=True, default=None)
    """
    root point below the elevator from which the elevator is driving up
    """

    floor_positions: Dict[str, float] = field(
        kw_only=True, default_factory=dict, init=False
    )
    """
    Positions of the drive which corresponds to the floor the elevator can reach.
    """

    @classmethod
    def create_with_new_bodies_in_world(
        cls, name: PrefixedName, world: World, scale: Scale
    ) -> Self:
        with world.modify_world():
            anker = Body(name=PrefixedName(f"{name.name}_anker", name.prefix))
            elevator = cls._create_with_connection_in_world(name, world, anker)

            floor = Floor.create_with_new_body_in_world(
                name=PrefixedName(f"{name.name}_floor", name.prefix),
                world=world,
                scale=scale.xy,
            )

            slider = Slider.create_with_new_body_in_world(
                name=PrefixedName(f"{name.name}_drive", name.prefix),
                world=world,
                active_axis=Vector3.Z(),
            )

            world.move_branch(slider.root, anker)
            world.move_branch(floor.root, slider.root)

            thickness = cls.WALL_THICKNESS
            wall_configs = (
                (0.0, Point3(-scale.x / 2, 0, 0), Scale(thickness, scale.y, scale.z)),
                (
                    math.pi / 2,
                    Point3(0, -scale.y / 2, 0),
                    Scale(thickness, scale.x, scale.z),
                ),
                (
                    math.pi / 2,
                    Point3(0, scale.y / 2, 0),
                    Scale(thickness, scale.x, scale.z),
                ),
            )
            walls = []
            for i, (yaw, translation, wall_scale) in enumerate(wall_configs):
                rotation = RotationMatrix.from_rpy(yaw=yaw)
                transform = HomogeneousTransformationMatrix.from_point_rotation_matrix(
                    translation, rotation, reference_frame=world.root
                )
                walls.append(
                    Wall.create_with_new_body_in_world(
                        name=PrefixedName(f"{name.name}_wall{i}", name.prefix),
                        world=world,
                        world_root_T_self=transform,
                        scale=wall_scale,
                    )
                )

            door_scale = Scale(thickness, scale.y / 2, scale.z)
            door1 = Door.create_with_new_body_in_world(
                name=PrefixedName(f"{name.name}_door0", name.prefix),
                world=world,
                world_root_T_self=HomogeneousTransformationMatrix.from_point_rotation_matrix(
                    Point3(scale.x / 2, -scale.y / 4, scale.z / 2),
                    reference_frame=world.root,
                ),
                scale=door_scale,
            )
            door2 = Door.create_with_new_body_in_world(
                name=PrefixedName(f"{name.name}_door1", name.prefix),
                world=world,
                world_root_T_self=HomogeneousTransformationMatrix.from_point_rotation_matrix(
                    Point3(scale.x / 2, scale.y / 4, scale.z / 2),
                    reference_frame=world.root,
                ),
                scale=door_scale,
            )
            double_door = DoubleDoor(
                name=PrefixedName(f"{name.name}_double_door", name.prefix),
                door_0=door1,
                door_1=door2,
            )
            world.add_semantic_annotation(double_door)

            for wall in walls:
                world.move_branch(wall.root, floor.root)
            for door in (door1, door2):
                world.move_branch(door.root, floor.root)

            door_travel = door_scale.y
            door_slider_configs = (
                (
                    door1,
                    DerivativeMap(position=-door_travel),
                    DerivativeMap(position=0.0),
                ),
                (
                    door2,
                    DerivativeMap(position=0.0),
                    DerivativeMap(position=door_travel),
                ),
            )
            for i, (door, lower, upper) in enumerate(door_slider_configs):
                door_slider = Slider.create_with_new_body_in_world(
                    name=PrefixedName(f"{name.name}_door{i}_drive", name.prefix),
                    world=world,
                    active_axis=(Vector3.Y() * ((-1) ** (i + 1))),
                    connection_limits=DegreeOfFreedomLimits(lower=lower, upper=upper),
                )
                door.add(door_slider)

            elevator.doors = double_door
            elevator.drive = slider
            elevator.walls = walls
            elevator.floor = floor
            elevator.anker_point = anker

        return elevator

    def open(self):
        """
        Opens the elevator doors
        """
        self.doors.door_0.mechanical_joint.position = self.doors.door_0.scale.y

        self.doors.door_1.mechanical_joint.position = self.doors.door_1.scale.y

    def close(self):
        """
        Closes the elevator doors
        """
        self.doors.door_0.mechanical_joint.position = 0
        self.doors.door_1.mechanical_joint.position = 0

    def add_floor(self, floor_name: str, floor_position: float):
        """
        Adds a floor to the possible targets for the elevator
        """
        self.floor_positions[floor_name] = floor_position

    def drive_to_floor(self, floor_name: str):
        """
        Drives the elevator to the floor given
        """
        self.drive.root.parent_connection.position = self.floor_positions[floor_name]


############################### subclasses to Furniture


@dataclass(eq=False)
class ShelfLayer(HasSupportingSurface):
    """
    A horizontal surface used for storing objects, typically found inside cabinets or on walls.
    """


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


@dataclass(eq=False)
class Cabinet(Furniture, HasCaseAsRootBody, HasDoors, HasDrawers):
    @classproperty
    def hole_direction(self) -> Vector3:
        return Vector3.NEGATIVE_X()


@dataclass(eq=False)
class Fridge(Cabinet): ...


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
    @classmethod
    def create_with_new_body_in_world(
        cls,
        name: PrefixedName,
        world: World,
        world_root_T_self: Optional[HomogeneousTransformationMatrix] = None,
        connection_limits: Optional[DegreeOfFreedomLimits] = None,
        active_axis: Vector3 = Vector3.Z(),
        connection_multiplier: float = 1.0,
        connection_offset: float = 0.0,
        *,
        scale: Scale = Scale(),
    ) -> Self:
        """
        Create a Floor semantic annotation with a new body defined by the given scale.

        :param name: The name of the floor body.
        :param scale: The scale defining the floor polytope.
        """
        polytope = scale.to_bounding_box().get_points()
        return cls.create_with_new_body_from_polytope_in_world(
            name=name,
            floor_polytope=polytope,
            world=world,
            world_root_T_self=world_root_T_self,
        )

    @classmethod
    def create_with_new_body_from_polytope_in_world(
        cls,
        name: PrefixedName,
        world: World,
        floor_polytope: List[Point3],
        world_root_T_self: Optional[HomogeneousTransformationMatrix] = None,
    ) -> Self:
        """
        Create a Floor semantic annotation with a new body defined by the given list of Point3.

        :param name: The name of the floor body.
        :param floor_polytope: A list of 3D points defining the floor poly
        """
        room_body = Body.from_3d_points(name=name, points_3d=floor_polytope)
        return cls._create_with_connection_in_world(
            name, world, room_body, world_root_T_self
        )


@dataclass(eq=False)
class Room(SemanticAnnotation):
    """
    A closed area with a specific purpose
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
class Wall(HasApertures):
    """
    A wall is a physical entity that separates two spaces and can contain apertures. Doors are a computed property.
    """

    @classmethod
    def create_with_new_body_in_world(
        cls,
        name: PrefixedName,
        world: World,
        world_root_T_self: Optional[HomogeneousTransformationMatrix] = None,
        connection_limits: Optional[DegreeOfFreedomLimits] = None,
        active_axis: Vector3 = Vector3.Z(),
        connection_multiplier: float = 1.0,
        connection_offset: float = 0.0,
        *,
        scale: Scale = Scale(),
    ) -> Self:
        if not (scale.x < scale.y and scale.x < scale.z):
            raise InvalidPlaneDimensions(scale, clazz=Wall)

        wall_body = Body(name=name)
        # This creates an event exactly as the scale, so if the scale is 1 the event will be from 0 to 1.
        wall_event = cls._create_wall_event(scale).as_composite_set()
        wall_collision = BoundingBoxCollection.from_event(
            wall_body, wall_event
        ).as_shapes()

        wall_body.collision = wall_collision
        wall_body.visual = wall_collision

        return cls._create_with_connection_in_world(
            name, world, wall_body, world_root_T_self
        )

    @property
    def doors(self) -> Iterable[Door]:
        return [
            door
            for door in self._world.get_semantic_annotations_by_type(Door)
            if door.entry_way and InsideOf(door.entry_way.root, self.root)() > 0.1
        ]

    @classmethod
    def _create_wall_event(cls, scale: Scale) -> SimpleEvent:
        """
        Return the collision shapes for the wall. A wall event is created based on the scale of the wall, and
        doors are removed from the wall event. The resulting bounding box collection is converted to shapes.
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


@dataclass(eq=False)
class Bottle(HasRootBody):
    """
    Abstract class for bottles.
    """


@dataclass(eq=False)
class Statue(HasRootBody): ...


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


@dataclass(eq=False)
class MustardBottle(Bottle):
    """
    A mustard bottle.
    """


@dataclass(eq=False)
class DrinkingContainer(HasRootBody): ...


@dataclass(eq=False)
class Cup(DrinkingContainer, IsPerceivable):
    """
    A cup.
    """


@dataclass(eq=False)
class Mug(DrinkingContainer):
    """
    A mug.
    """


@dataclass(eq=False)
class CookingContainer(HasRootBody): ...


@dataclass(eq=False)
class Lid(HasRootBody): ...


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
class Pot(CookingContainer):
    """
    A pot.
    """


@dataclass(eq=False)
class PotLid(Lid):
    """
    A pot lid.
    """


@dataclass(eq=False)
class Plate(HasSupportingSurface):
    """
    A plate.
    """


@dataclass(eq=False)
class Bowl(HasSupportingSurface, IsPerceivable):
    """
    A bowl.
    """


# Food Items
@dataclass(eq=False)
class Food(HasRootBody):
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

    _synonyms = {
        "bumpybread",
        "whitebread",
        "loafbread",
        "honeybread",
        "grainbread",
    }


@dataclass(eq=False)
class CheezeIt(Food):
    """
    Some type of cracker.
    """


@dataclass(eq=False)
class Pringles(Food):
    """
    Pringles chips
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


@dataclass(eq=False)
class SaltContainer(HasRootBody, IsPerceivable):
    """
    A container of salt.
    """


@dataclass(eq=False)
class Produce(Food):
    """
    In American English, produce generally refers to fresh fruits and vegetables intended to be eaten by humans.
    """

    pass


@dataclass(eq=False)
class Fruit(Produce):
    """
    Fruit.
    """


@dataclass(eq=False)
class Vegetable(Produce):
    """
    Vegetable.
    """


@dataclass(eq=False)
class Tomato(Fruit):
    """
    A tomato.
    """


@dataclass(eq=False)
class Lettuce(Vegetable):
    """
    Lettuce.
    """


@dataclass(eq=False)
class Carrot(Vegetable):
    """
    A carrot.
    """


@dataclass(eq=False)
class Apple(Fruit):
    """
    An apple.
    """


@dataclass(eq=False)
class Banana(Fruit):
    """
    A banana.
    """


@dataclass(eq=False)
class Orange(Fruit):
    """
    An orange.
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


@dataclass(eq=False)
class Desk(Table, HasLegs):
    """
    A desk.
    """


@dataclass(eq=False)
class Chair(Furniture):
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
class TrashCan(HasRootBody, Furniture):
    """
    Abstract class for Trash Can.
    """


@dataclass(eq=False)
class ShelvingUnit(Furniture):
    """
    A shelving unit.
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
class Cloth(HasRootBody): ...


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
class Potato(Vegetable): ...


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
class SprayBottle(HasRootBody):
    """
    A spray bottle.
    """


@dataclass(eq=False)
class Vase(HasRootBody):
    """
    A vase.
    """


@dataclass(eq=False)
class Book(HasRootBody):
    """
    A book.
    """

    book_front: Optional[BookFront] = None


@dataclass(eq=False)
class BookFront(HasRootBody): ...


@dataclass(eq=False)
class SaltPepperShaker(HasRootBody):
    """
    A salt and pepper shaker.
    """


@dataclass(eq=False)
class Cuttlery(HasRootBody): ...


@dataclass(eq=False)
class Fork(Cuttlery):
    """
    A fork.
    """


@dataclass(eq=False)
class Knife(Cuttlery):
    """
    A butter knife.
    """


@dataclass(eq=False)
class Spoon(Cuttlery, IsPerceivable): ...


@dataclass(eq=False)
class Pencil(HasRootBody):
    """
    A pencil.
    """


@dataclass(eq=False)
class Pen(HasRootBody):
    """
    A pen.
    """


@dataclass(eq=False)
class Baseball(HasRootBody):
    """
    A baseball.
    """


@dataclass(eq=False)
class LiquidCap(HasRootBody):
    """
    A liquid cap.
    """


@dataclass(eq=False)
class Agent(HasRootBody):
    """
    Represents an entity in the world that can act, move, or be controlled.

    Agents are dynamic bodies with semantic meaning — they may have intent,
    behavior, or be controlled by external or internal logic. Examples include
    robots, humans, or other autonomous actors.

    """


@dataclass(eq=False)
class Human(Agent):
    """
    Represents a human agent in the environment.

    A Person is an Agent that is not robotically actuated and does not provide
    kinematic chains, end_effectors, or robot-specific components.

    This class exists primarily for semantic distinction, so that algorithms
    can treat human agents differently from robots if needed.
    """


@dataclass(eq=False)
class SemanticEnvironmentAnnotation(HasRootBody):
    """
    Represents a semantic annotation of the environment.
    """


@dataclass(eq=False)
class RoomWithWallsAndDoors(Room):
    """
    A room with a type description (e.g., Ktichen) and walls and doors.
    """

    room_type: Optional[str] = field(kw_only=True, default=None)
    """
    Description of the type of the room in natural language.
    """

    walls: List[Wall] = field(kw_only=True, default_factory=list)
    """
    The walls enclosing this room.
    """

    doors: List[Door] = field(kw_only=True, default_factory=list)
    """
    The doors of the room.
    """


@dataclass(eq=False)
class DoorWithType(Door):
    """
    A Door that has a type description, e.g. "main entrance"
    """

    type_description: Optional[str] = field(kw_only=True, default=None)


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
class Tool(HasRootBody, ABC):
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

    def _find_end_effector_body(self) -> Optional[Body]:
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
    def tool_alignment(self, target: Union[Body, Pose]) -> List[AlignmentPair]:
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

    def tool_alignment(self, target: Union[Body, Pose]) -> List[AlignmentPair]:
        return [
            AlignmentPair(
                tip_normal=Vector3(-1, 0, 0, reference_frame=self.root),
                goal_normal=Vector3(0, 0, 1, reference_frame=target),
            )
        ]


@dataclass(eq=False)
class CuttingKnife(ToolWithHandle):
    """
    A knife for cutting food objects.
    """

    def tool_alignment(self, target: Union[Body, Pose]) -> List[AlignmentPair]:
        return [
            AlignmentPair(
                tip_normal=Vector3(1, 0, 0, reference_frame=self.root),
                goal_normal=Vector3(0, 1, 0, reference_frame=target),
            ),
            AlignmentPair(
                tip_normal=Vector3(0, 0, 1, reference_frame=self.root),
                goal_normal=Vector3(0, 0, 1, reference_frame=target),
            ),
        ]


@dataclass(eq=False)
class PouringCup(Tool):
    """
    A cup for pouring liquids into containers.
    """

    def tool_alignment(self, target: Union[Body, Pose]) -> List[AlignmentPair]:
        return [
            AlignmentPair(
                tip_normal=Vector3(0, 0, 1, reference_frame=self.root),
                goal_normal=Vector3(0, 0, 1, reference_frame=target),
            )
        ]


@dataclass(eq=False)
class Sponge(Tool):
    """
    A sponge for wiping surfaces.

    .. note:: The sponge is grasped so its local Z axis points away from the gripper,
        which is why its alignments use negative Z normals unlike the other tools.
    """

    def tool_alignment(self, target: Union[Body, Pose]) -> List[AlignmentPair]:
        if isinstance(target, Body):
            return [
                AlignmentPair(
                    tip_normal=Vector3(0, 0, 1, reference_frame=self.root),
                    goal_normal=Vector3(0, 0, -1, reference_frame=target),
                )
            ]
        return [
            AlignmentPair(
                tip_normal=Vector3(0, 0, -1, reference_frame=self.root),
                goal_normal=self._pose_surface_normal(target),
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
        rotation = pose.to_rotation_matrix().to_np()[:3, :3]
        return Vector3.from_iterable(
            rotation @ np.array([0.0, 0.0, 1.0]),
            reference_frame=reference_frame,
        )


@dataclass(eq=False)
class Microwave(IsStorageSpace, HasDoors):
    """
    A microwave oven, a kitchen appliance with a door that heats food placed inside it using
    microwave radiation.
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
