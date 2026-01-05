from __future__ import annotations

from abc import ABC
from dataclasses import dataclass
from typing import Iterable, Optional, Self

from random_events.interval import closed
from random_events.product_algebra import SimpleEvent
from typing_extensions import List

from krrood.entity_query_language.entity import entity, variable
from krrood.entity_query_language.entity_result_processors import an
from krrood.ormatic.utils import classproperty
from .mixins import (
    HasRootBody,
    HasSupportingSurface,
    HasRootRegion,
    HasDrawers,
    HasDoors,
    HasHandle,
    HasCaseAsMainBody,
    HasHinge,
    HasLeftRightDoor,
    HasSlider,
    HasApertures,
    IsPerceivable,
)
from ..datastructures.prefixed_name import PrefixedName
from ..datastructures.variables import SpatialVariables
from ..exceptions import InvalidPlaneDimensions
from ..reasoning.predicates import InsideOf
from ..spatial_types import Point3, HomogeneousTransformationMatrix
from ..utils import Direction
from ..world import World
from ..world_description.geometry import Scale
from ..world_description.shape_collection import BoundingBoxCollection
from ..world_description.world_entity import (
    SemanticAnnotation,
    Body,
    KinematicStructureEntity,
    Region,
)


@dataclass(eq=False)
class Handle(HasRootBody):

    @classmethod
    def create_with_new_body_in_world(
        cls,
        name: PrefixedName,
        world: World,
        parent: KinematicStructureEntity,
        parent_T_self: Optional[HomogeneousTransformationMatrix] = None,
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
        return cls._create_with_fixed_connection_in_world(
            name, world, handle_body, parent, parent_T_self
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
class Fridge(
    HasCaseAsMainBody,
    HasDoors,
    HasDrawers,
):
    """
    A fridge that has a door and a body.
    """

    @classproperty
    def opening_direction(self) -> Direction:
        return Direction.NEGATIVE_X


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
        parent: KinematicStructureEntity,
        parent_T_self: Optional[HomogeneousTransformationMatrix] = None,
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

        return cls._create_with_fixed_connection_in_world(
            name, world, aperture_region, parent, parent_T_self
        )

    @classmethod
    def create_with_new_region_in_world_from_body(
        cls,
        name: PrefixedName,
        world: World,
        parent: KinematicStructureEntity,
        body: Body,
        parent_T_self: Optional[HomogeneousTransformationMatrix] = None,
    ) -> Self:
        body_scale = (
            body.collision.as_bounding_box_collection_in_frame(body)
            .bounding_box()
            .scale
        )
        return cls.create_with_new_region_in_world(
            name, world, parent, parent_T_self, scale=body_scale
        )


@dataclass(eq=False)
class Hinge(HasRootBody):
    """
    A hinge is a physical entity that connects two bodies and allows one to rotate around a fixed axis.
    """


@dataclass(eq=False)
class Slider(HasRootBody):
    """
    A Slider is a physical entity that connects two bodies and allows one to linearly translate along a fixed axis.
    """


@dataclass(eq=False)
class Door(HasRootBody, HasHandle, HasHinge):
    """
    A door is a physical entity that has covers an opening, has a movable body and a handle.
    """

    @classmethod
    def create_with_new_body_in_world(
        cls,
        name: PrefixedName,
        world: World,
        parent: KinematicStructureEntity,
        parent_T_self: Optional[HomogeneousTransformationMatrix] = None,
        *,
        scale: Scale = Scale(0.03, 1, 2),
    ) -> Self:
        if not (scale.x < scale.y and scale.x < scale.z):
            raise InvalidPlaneDimensions(scale)

        door_event = scale.to_simple_event().as_composite_set()
        door_body = Body(name=name)
        bounding_box_collection = BoundingBoxCollection.from_event(
            door_body, door_event
        )
        collision = bounding_box_collection.as_shapes()
        door_body.collision = collision
        door_body.visual = collision
        return cls._create_with_fixed_connection_in_world(
            name, world, door_body, parent, parent_T_self
        )


@dataclass(eq=False)
class DoubleDoor(SemanticAnnotation, HasLeftRightDoor):
    """
    A semantic annotation that represents a double door with left and right doors.
    """


@dataclass(eq=False)
class Drawer(HasCaseAsMainBody, HasHandle, HasSlider):

    @property
    def opening_direction(self) -> Direction:
        return Direction.Z


############################### subclasses to Furniture
@dataclass(eq=False)
class Furniture(SemanticAnnotation, ABC): ...


@dataclass(eq=False)
class Table(Furniture, HasSupportingSurface):
    """
    A semantic annotation that represents a table.
    """


@dataclass(eq=False)
class Cabinet(HasCaseAsMainBody, Furniture, HasDrawers, HasDoors):
    @property
    def opening_direction(self) -> Direction:
        return Direction.NEGATIVE_X


@dataclass(eq=False)
class Dresser(HasCaseAsMainBody, Furniture, HasDrawers, HasDoors):
    @property
    def opening_direction(self) -> Direction:
        return Direction.NEGATIVE_X


@dataclass(eq=False)
class Cupboard(HasCaseAsMainBody, Furniture, HasDoors):
    @property
    def opening_direction(self) -> Direction:
        return Direction.NEGATIVE_X


@dataclass(eq=False)
class Wardrobe(HasCaseAsMainBody, Furniture, HasDrawers, HasDoors):
    @property
    def opening_direction(self) -> Direction:
        return Direction.NEGATIVE_X


@dataclass(eq=False)
class Floor(HasSupportingSurface):

    @classmethod
    def create_with_new_body_in_world(
        cls,
        name: PrefixedName,
        world: World,
        parent: KinematicStructureEntity,
        parent_T_self: Optional[HomogeneousTransformationMatrix] = None,
        *,
        scale: Scale = Scale(),
    ) -> Self:
        """
        Create a Floor semantic annotation with a new body defined by the given scale.

        :param name: The name of the floor body.
        :param scale: The scale defining the floor polytope.
        """
        polytope = scale.to_bounding_box().get_points()
        self = cls.create_with_new_body_from_polytope(
            name=name, floor_polytope=polytope
        )
        self._create_with_fixed_connection_in_world(
            name, world, self.body, parent, parent_T_self
        )
        return self

    @classmethod
    def create_with_new_body_from_polytope(
        cls, name: PrefixedName, floor_polytope: List[Point3]
    ) -> Self:
        """
        Create a Floor semantic annotation with a new body defined by the given list of Point3.

        :param name: The name of the floor body.
        :param floor_polytope: A list of 3D points defining the floor poly
        """
        room_body = Body.from_3d_points(name=name, points_3d=floor_polytope)
        return cls(body=room_body)


@dataclass(eq=False)
class Room(SemanticAnnotation):
    """
    A closed area with a specific purpose
    """

    floor: Floor
    """
    The room's floor.
    """


@dataclass(eq=False)
class Wall(HasRootBody, HasApertures):

    @classmethod
    def create_with_new_body_in_world(
        cls,
        name: PrefixedName,
        world: World,
        parent: KinematicStructureEntity,
        parent_T_self: Optional[HomogeneousTransformationMatrix] = None,
        *,
        scale: Scale = Scale(),
    ) -> Self:
        if not (scale.x < scale.y and scale.x < scale.z):
            raise InvalidPlaneDimensions(scale)

        wall_body = Body(name=name)
        wall_event = cls._create_wall_event(scale).as_composite_set()
        wall_collision = BoundingBoxCollection.from_event(
            wall_body, wall_event
        ).as_shapes()

        wall_body.collision = wall_collision
        wall_body.visual = wall_collision

        return cls._create_with_fixed_connection_in_world(
            name, world, wall_body, parent, parent_T_self
        )

    @property
    def doors(self) -> Iterable[Door]:
        door = variable(Door, self._world.semantic_annotations)
        query = an(
            entity(door).where(InsideOf(self.body, door.entry_way.region)() > 0.1)
        )
        return query.evaluate()

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
class Chair(HasRootBody, Furniture):
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
class ShelvingUnit(HasRootBody, Furniture):
    """
    A shelving unit.
    """


@dataclass(eq=False)
class Bed(HasRootBody, Furniture):
    """
    A bed.
    """


@dataclass(eq=False)
class Sofa(HasRootBody, Furniture):
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
