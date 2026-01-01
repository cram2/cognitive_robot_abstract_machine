from __future__ import annotations

from dataclasses import dataclass, field
from typing import Set, Iterable, Optional

import numpy as np
from krrood.entity_query_language.entity import entity, let
from krrood.entity_query_language.quantify_entity import an
from probabilistic_model.probabilistic_circuit.rx.helper import uniform_measure_of_event
from typing_extensions import List

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
    A semantic annotation representing a fridge that has a door and a body.
    """

    container: Container
    door: Door


@dataclass(eq=False)
class Table(Furniture, HasBody):
    """
    A semantic annotation that represents a table.
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
    A semantic annotation that represents an opening in a physical entity.
    An example is like a hole in a wall that can be used to enter a room.
    """


@dataclass(eq=False)
class Door(HasBody):
    """
    A door is a physical entity that has covers an opening, has a movable body and a handle.
    """

    handle: Handle
    """
    The handle of the door.
    """


@dataclass(eq=False)
class DoubleDoor(SemanticAnnotation):
    left_door: Door
    right_door: Door


@dataclass(eq=False)
class Drawer(SemanticAnnotation):
    container: Container
    handle: Handle


############################### subclasses to Furniture
@dataclass(eq=False)
class Cabinet(Furniture, HasDrawers, HasDoors):
    container: Container


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
    A semantic annotation that represents a closed area with a specific purpose
    """

    floor: Floor
    """
    The room's floor.
    """


@dataclass(eq=False)
class Wall(SemanticAnnotation):
    body: Body

    @property
    def doors(self) -> Iterable[Door]:
        door = let(Door, self._world.semantic_annotations)
        query = an(entity(door), InsideOf(self.body, door.entry_way.region)() > 0.1)
        return query.evaluate()


class Drink(SemanticAnnotation):
    """
    A Semantic annotation representing a drink item.
    """
    body: Body

# %% Subclasses of GasDrink
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