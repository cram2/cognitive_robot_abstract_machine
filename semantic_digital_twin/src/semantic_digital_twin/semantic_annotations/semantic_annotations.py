from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable, Optional

import numpy as np
from probabilistic_model.probabilistic_circuit.rx.helper import uniform_measure_of_event
from typing_extensions import List

from krrood.entity_query_language.entity import entity, variable
from krrood.entity_query_language.entity_result_processors import an
from .mixins import (
    HasBody,
    HasSupportingSurface,
    Furniture,
    HasRegion,
    HasDrawers,
    HasDoors,
)
from ..datastructures.variables import SpatialVariables
from ..reasoning.predicates import InsideOf
from ..spatial_types import Point3
from ..world_description.shape_collection import BoundingBoxCollection
from ..world_description.world_entity import (
    SemanticAnnotation,
    Body,
)


@dataclass(eq=False)
class IsPerceivable:
    """
    A mixin class for semantic annotations that can be perceived.
    """

    class_label: Optional[str] = field(default=None, kw_only=True)
    """
    The exact class label of the perceived object.
    """


@dataclass(eq=False)
class Handle(HasBody): ...


@dataclass(eq=False)
class Container(HasBody): ...


@dataclass(eq=False)
class Fridge(SemanticAnnotation):
    """
    A fridge that has a door and a body.
    """

    container: Container = field(kw_only=True)
    door: Door = field(kw_only=True)


@dataclass(eq=False)
class Table(Furniture, HasBody):
    """
    A table.
    """

    def points_on_table(self, amount: int = 100) -> List[Point3]:
        """
        Get points that are on the table.

        :amount: The number of points to return.
        :returns: A list of points that are on the table.
        """
        area_of_table = BoundingBoxCollection.from_shapes(self.body.collision)
        event = area_of_table.event
        p = uniform_measure_of_event(event)
        p = p.marginal(SpatialVariables.xy)
        samples = p.sample(amount)
        z_coordinate = np.full(
            (amount, 1), max([b.max_z for b in area_of_table]) + 0.01
        )
        samples = np.concatenate((samples, z_coordinate), axis=1)
        return [Point3(*s, reference_frame=self.body) for s in samples]


@dataclass(eq=False)
class Aperture(HasRegion):
    """
    An opening in a physical entity.
    An example is like a hole in a wall that can be used to enter a room.
    """


@dataclass(eq=False)
class Door(HasBody):
    """
    A door is a physical entity that has covers an opening, has a movable body and a handle.
    """

    handle: Handle = field(kw_only=True)
    """
    The handle of the door.
    """


@dataclass(eq=False)
class DoubleDoor(SemanticAnnotation):
    left_door: Door = field(kw_only=True)
    right_door: Door = field(kw_only=True)


@dataclass(eq=False)
class Drawer(SemanticAnnotation):
    container: Container = field(kw_only=True)
    handle: Handle = field(kw_only=True)


############################### subclasses to Furniture
@dataclass(eq=False)
class Cabinet(Furniture, HasDrawers, HasDoors):
    container: Container = field(kw_only=True)


@dataclass(eq=False)
class Dresser(Furniture, HasDrawers, HasDoors):
    container: Container = field(kw_only=True)


@dataclass(eq=False)
class Cupboard(Furniture, HasDoors):
    container: Container = field(kw_only=True)


@dataclass(eq=False)
class Wardrobe(Furniture, HasDrawers, HasDoors):
    container: Container = field(kw_only=True)


class Floor(HasSupportingSurface): ...


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
class Wall(SemanticAnnotation):
    body: Body = field(kw_only=True)

    @property
    def doors(self) -> Iterable[Door]:
        door = variable(Door, self._world.semantic_annotations)
        query = an(
            entity(door).where(InsideOf(self.body, door.entry_way.region)() > 0.1)
        )
        return query.evaluate()


@dataclass(eq=False)
class Bottle(Container):
    """
    Abstract class for bottles.
    """


@dataclass(eq=False)
class Statue(HasBody): ...


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
class DrinkingContainer(Container, HasBody): ...


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
class CookingContainer(Container, HasBody): ...


@dataclass(eq=False)
class Lid(HasBody): ...


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
class Plate(HasBody, HasSupportingSurface):
    """
    A plate.
    """


@dataclass(eq=False)
class Bowl(HasBody, IsPerceivable, HasSupportingSurface):
    """
    A bowl.
    """


# Food Items


@dataclass(eq=False)
class Food(HasBody): ...


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
class Milk(Container, Food, IsPerceivable):
    """
    A container of milk.
    """

    ...


@dataclass(eq=False)
class SaltContainer(Container, IsPerceivable):
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
class Chair(HasBody, Furniture):
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
class ShelvingUnit(HasBody, Furniture):
    """
    A shelving unit.
    """


@dataclass(eq=False)
class Bed(HasBody, Furniture):
    """
    A bed.
    """


@dataclass(eq=False)
class Sofa(HasBody, Furniture):
    """
    A sofa.
    """


@dataclass(eq=False)
class Sink(HasBody):
    """
    A sink.
    """


@dataclass(eq=False)
class Kettle(CookingContainer): ...


@dataclass(eq=False)
class Decor(HasBody): ...


@dataclass(eq=False)
class WallDecor(Decor):
    """
    Wall decorations.
    """


@dataclass(eq=False)
class Cloth(HasBody): ...


@dataclass(eq=False)
class Poster(WallDecor):
    """
    A poster.
    """


@dataclass(eq=False)
class WallPanel(HasBody):
    """
    A wall panel.
    """


@dataclass(eq=False)
class Potato(Produce): ...


@dataclass(eq=False)
class GarbageBin(Container):
    """
    A garbage bin.
    """


@dataclass(eq=False)
class Drone(HasBody): ...


@dataclass(eq=False)
class ProcthorBox(Container): ...


@dataclass(eq=False)
class Houseplant(HasBody):
    """
    A houseplant.
    """


@dataclass(eq=False)
class SprayBottle(HasBody):
    """
    A spray bottle.
    """


@dataclass(eq=False)
class Vase(HasBody):
    """
    A vase.
    """


@dataclass(eq=False)
class Book(HasBody):
    """
    A book.
    """

    book_front: Optional[BookFront] = None


@dataclass(eq=False)
class BookFront(HasBody): ...


@dataclass(eq=False)
class SaltPepperShaker(HasBody):
    """
    A salt and pepper shaker.
    """


@dataclass(eq=False)
class Cuttlery(HasBody): ...


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
class Pencil(HasBody):
    """
    A pencil.
    """


@dataclass(eq=False)
class Pen(HasBody):
    """
    A pen.
    """


@dataclass(eq=False)
class Baseball(HasBody):
    """
    A baseball.
    """


@dataclass(eq=False)
class LiquidCap(HasBody):
    """
    A liquid cap.
    """

@dataclass(eq=False)
class Drink(SemanticAnnotation):
    """
    A Semantic annotation representing a drink item.
    """
    body: Body = field(kw_only=True)


@dataclass(eq=False)
class Cola(Drink):
    ...

@dataclass(eq=False)
class Fanta(Drink):
    ...

@dataclass(eq=False)
class Water(Drink):
    ...

@dataclass(eq=False)
class Beer(Drink):
    ...

# Food Items (with HasBody)
@dataclass(eq=False)
class Chips(Food, IsPerceivable):
    """
    A bag or can of chips.
    """


@dataclass(eq=False)
class Crispbread(Food, IsPerceivable):
    """
    A pack of crispbread.
    """


@dataclass(eq=False)
class Sauce(Food, IsPerceivable):
    """
    A bottle or container of sauce.
    """


@dataclass(eq=False)
class Soda(Food, IsPerceivable):
    """
    A can or bottle of soda.
    """


@dataclass(eq=False)
class SoyaDrink(Food, IsPerceivable):
    """
    A carton of soya drink.
    """


@dataclass(eq=False)
class Pudding(Food, IsPerceivable):
    """
    A box or container of pudding.
    """


@dataclass(eq=False)
class GelatinBox(Food, IsPerceivable):
    """
    A box of gelatin.
    """


@dataclass(eq=False)
class Sponge(Food, IsPerceivable):
    """
    A sponge (e.g., kitchen sponge).
    """


@dataclass(eq=False)
class Spatula(Food, IsPerceivable):
    """
    A spatula (e.g., kitchen spatula).
    """


@dataclass(eq=False)
class Rice(Food, IsPerceivable):
    """
    A pack of rice.
    """


@dataclass(eq=False)
class PancakeMix(Food, IsPerceivable):
    """
    A box of pancake mix.
    """


@dataclass(eq=False)
class Oregano(Food, IsPerceivable):
    """
    A shaker of oregano.
    """


@dataclass(eq=False)
class Liquorice(Food, IsPerceivable):
    """
    A bag of liquorice.
    """


@dataclass(eq=False)
class HoneyWafers(Food, IsPerceivable):
    """
    A pack of honey wafers.
    """


@dataclass(eq=False)
class Grapes(Food, IsPerceivable):
    """
    A bunch of grapes.
    """


@dataclass(eq=False)
class Peach(Food, IsPerceivable):
    """
    A peach.
    """


@dataclass(eq=False)
class Plum(Food, IsPerceivable):
    """
    A plum.
    """


@dataclass(eq=False)
class Strawberry(Food, IsPerceivable):
    """
    A strawberry.
    """


@dataclass(eq=False)
class Lemon(Food, IsPerceivable):
    """
    A lemon.
    """


@dataclass(eq=False)
class Orange(Food, IsPerceivable):
    """
    An orange.
    """


@dataclass(eq=False)
class Cucumber(Food, IsPerceivable):
    """
    A cucumber.
    """

@dataclass(eq=False)
class Zucchini(Food, IsPerceivable):
    """
    A zucchini.
    """

@dataclass(eq=False)
class Salt(Food, IsPerceivable):
    """
    A pack or container of salt (e.g., salt shaker or salt can).
    """


@dataclass(eq=False)
class Muesli(Food, IsPerceivable):
    """
    A pack of muesli (e.g., in a box or bag).
    """
@dataclass(eq=False)
class Corn(Food, IsPerceivable):
    """
    A can of corn.
    """


@dataclass(eq=False)
class DishwasherTab(Food, IsPerceivable):
    """
    A dishwasher tablet.
    """


@dataclass(eq=False)
class GlassCleaner(Food, IsPerceivable):
    """
    A bottle of glass cleaner.
    """


@dataclass(eq=False)
class GroundCoffee(Food, IsPerceivable):
    """
    A pack of ground coffee.
    """


@dataclass(eq=False)
class Hammer(HasBody, IsPerceivable):
    """
    A hammer.
    """


@dataclass(eq=False)
class IcedTea(Food, IsPerceivable):
    """
    A can or bottle of iced tea.
    """


@dataclass(eq=False)
class Juice(Food, IsPerceivable):
    """
    A carton or bottle of juice.
    """


@dataclass(eq=False)
class Ketchup(Food, IsPerceivable):
    """
    A bottle of ketchup.
    """


@dataclass(eq=False)
class Mayonnaise(Food, IsPerceivable):
    """
    A bottle of mayonnaise.
    """


@dataclass(eq=False)
class Mustard(Food, IsPerceivable):
    """
    A bottle of mustard.
    """


@dataclass(eq=False)
class Pear(Food, IsPerceivable):
    """
    A pear.
    """


@dataclass(eq=False)
class Pitcher(HasBody, IsPerceivable):
    """
    A pitcher (e.g., for water).
    """


@dataclass(eq=False)
class PottedMeat(Food, IsPerceivable):
    """
    A can of potted meat.
    """


@dataclass(eq=False)
class Sausage(HasBody, IsPerceivable):
    """
    A sausage (e.g., in a can).
    """


@dataclass(eq=False)
class Scissors(HasBody, IsPerceivable):
    """
    A pair of scissors.
    """


@dataclass(eq=False)
class Screwdriver(HasBody, IsPerceivable):
    """
    A screwdriver.
    """


@dataclass(eq=False)
class Skillet(HasBody, IsPerceivable):
    """
    A skillet (frying pan).
    """


@dataclass(eq=False)
class Soap(HasBody, IsPerceivable):
    """
    A bar or bottle of soap.
    """


@dataclass(eq=False)
class SoccerBall(HasBody, IsPerceivable):
    """
    A soccer ball.
    """


@dataclass(eq=False)
class Sprinkles(Food, IsPerceivable):
    """
    A pack of sprinkles.
    """


@dataclass(eq=False)
class Sugar(Food, IsPerceivable):
    """
    A box or pack of sugar.
    """


@dataclass(eq=False)
class TennisBall(HasBody, IsPerceivable):
    """
    A tennis ball.
    """


@dataclass(eq=False)
class Vegetables(Food, IsPerceivable):
    """
    A can of mixed vegetables.
    """


@dataclass(eq=False)
class Wafer(Food, IsPerceivable):
    """
    A pack of wafers.
    """


@dataclass(eq=False)
class WineGlass(HasBody, IsPerceivable):
    """
    A wine glass.
    """


@dataclass(eq=False)
class WoodBlock(HasBody, IsPerceivable):
    """
    A wooden block.
    """


# Non-Food, but physical objects
@dataclass(eq=False)
class Racquetball(HasBody, IsPerceivable):
    """
    A racquetball.
    """


@dataclass(eq=False)
class RubiksCube(HasBody, IsPerceivable):
    """
    A Rubik's cube.
    """


@dataclass(eq=False)
class Saucer(HasBody, IsPerceivable):
    """
    A saucer.
    """

@dataclass(eq=False)
class Softball(HasBody, IsPerceivable):
    """
    A softball.
    """


@dataclass(eq=False)
class Marker(HasBody, IsPerceivable):
    """
    A marker (e.g., black marker).
    """