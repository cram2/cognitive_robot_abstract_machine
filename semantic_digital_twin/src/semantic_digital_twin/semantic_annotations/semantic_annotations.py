from __future__ import annotations

from abc import ABC
from dataclasses import dataclass, field
from typing import Iterable, Optional, Self, Tuple

from random_events.interval import closed
from random_events.product_algebra import SimpleEvent
from typing_extensions import List, Type

from krrood.ormatic.utils import classproperty
from krrood.symbolic_math import symbolic_math
from .mixins import (
    HasSupportingSurface,
    HasRootRegion,
    HasDrawers,
    HasDoors,
    HasHandle,
    HasCaseAsRootBody,
    HasHinge,
    HasSlider,
    HasApertures,
    IsPerceivable,
    HasRootBody,
    HasStorageSpace,
)
from ..datastructures.prefixed_name import PrefixedName
from ..datastructures.variables import SpatialVariables
from ..exceptions import (
    InvalidPlaneDimensions,
    InvalidHingeActiveAxis,
    MissingSemanticAnnotationError,
)
from ..reasoning.predicates import InsideOf
from ..spatial_types import Point3, HomogeneousTransformationMatrix, Vector3
from ..world import World
from ..world_description.connections import (
    RevoluteConnection,
    PrismaticConnection,
    FixedConnection,
)
from ..world_description.degree_of_freedom import DegreeOfFreedomLimits
from ..world_description.geometry import Scale, TriangleMesh
from ..world_description.shape_collection import BoundingBoxCollection, ShapeCollection
from ..world_description.world_entity import (
    SemanticAnnotation,
    Body,
    Region,
    Connection,
)


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

        x_interval = closed(0, scale.x - thickness)
        y_interval = closed(
            -scale.y / 2 + thickness,
            scale.y / 2 - thickness,
        )

        z_interval = closed(-scale.z / 2, scale.z / 2)

        return SimpleEvent(
            {
                SpatialVariables.x.value: x_interval,
                SpatialVariables.y.value: y_interval,
                SpatialVariables.z.value: z_interval,
            }
        )


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

        world._forward_kinematic_manager.recompile()
        world._forward_kinematic_manager.recompute()
        body_scale = (
            body.collision.as_bounding_box_collection_in_frame(body)
            .bounding_box()
            .scale
        )
        return cls.create_with_new_region_in_world(
            name, world, parent_T_self, scale=body_scale
        )


@dataclass(eq=False)
class Hinge(HasRootBody):
    """
    A hinge is a physical entity that connects two bodies and allows one to rotate around a fixed axis.
    """

    @classproperty
    def _parent_connection_type(self) -> Type[Connection]:
        return RevoluteConnection


@dataclass(eq=False)
class Slider(HasRootBody):
    """
    A Slider is a physical entity that connects two bodies and allows one to linearly translate along a fixed axis.
    """

    @classproperty
    def _parent_connection_type(self) -> Type[Connection]:
        return PrismaticConnection


@dataclass(eq=False)
class EntryWay(Aperture): ...


@dataclass(eq=False)
class Door(HasHandle, HasHinge):
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
            area=ShapeCollection([TriangleMesh(mesh=door_body.combined_mesh)]),
        )
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
        world_T_door = self.root.global_pose

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
        world_T_door_0 = self.door_0.root.global_pose
        view_point_T_door_0 = world_T_view_point.inverse() @ world_T_door_0
        world_T_door_1 = self.door_1.root.global_pose
        view_point_T_door_1 = world_T_view_point.inverse() @ world_T_door_1
        if view_point_T_door_0.y > view_point_T_door_1.y:
            return self.door_0, self.door_1
        else:
            return self.door_1, self.door_0


@dataclass(eq=False)
class Drawer(Furniture, HasCaseAsRootBody, HasHandle, HasSlider, HasStorageSpace):

    @classproperty
    def hole_direction(self) -> Vector3:
        return Vector3.Z()


############################### subclasses to Furniture


@dataclass(eq=False)
class Table(Furniture, HasSupportingSurface):
    """
    A semantic annotation that represents a table.
    """


@dataclass(eq=False)
class Cabinet(Furniture, HasCaseAsRootBody):
    @classproperty
    def hole_direction(self) -> Vector3:
        return Vector3.NEGATIVE_X()


@dataclass(eq=False)
class Fridge(Cabinet, HasDoors, HasDrawers): ...


@dataclass(eq=False)
class Dresser(Cabinet, HasDrawers, HasDoors): ...


@dataclass(eq=False)
class Cupboard(Cabinet, HasDoors): ...


@dataclass(eq=False)
class Wardrobe(Cabinet, HasDrawers, HasDoors): ...


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
        self = cls(root=room_body)
        self._create_with_connection_in_world(name, world, self.root, world_root_T_self)
        return self


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

        return SimpleEvent(
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
class Food(HasRootBody): ...


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

    ...


@dataclass(eq=False)
class SaltContainer(HasRootBody, IsPerceivable):
    """
    A container of salt.
    """

    ...


@dataclass(eq=False)
class Produce(Food):
    """
    In American English, produce generally refers to fresh fruits and vegetables intended to be eaten by humans.
    """

    pass


@dataclass(eq=False)
class Tomato(Produce):
    """
    A tomato.
    """


@dataclass(eq=False)
class Lettuce(Produce):
    """
    Lettuce.
    """


@dataclass(eq=False)
class Apple(Produce):
    """
    An apple.
    """


@dataclass(eq=False)
class Banana(Produce):
    """
    A banana.
    """


@dataclass(eq=False)
class Orange(Produce):
    """
    An orange.
    """


@dataclass(eq=False)
class CoffeeTable(Table):
    """
    A coffee table.
    """


@dataclass(eq=False)
class DiningTable(Table):
    """
    A dining table.
    """


@dataclass(eq=False)
class SideTable(Table):
    """
    A side table.
    """


@dataclass(eq=False)
class Desk(Table):
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
class Sofa(Furniture):
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
class Potato(Produce): ...


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
