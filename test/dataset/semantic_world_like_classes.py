from __future__ import annotations

from dataclasses import dataclass, field

from typing_extensions import List, Optional, Type, Iterable

from krrood.entity_query_language.predicate import Symbol, Predicate


@dataclass(unsafe_hash=True)
class WorldEntity(Symbol):
    world: Optional[World] = field(default=None, kw_only=True, repr=False, hash=False)


@dataclass(unsafe_hash=True)
class Body(WorldEntity):
    name: str
    size: int = field(default=1)


@dataclass(unsafe_hash=True)
class Handle(Body): ...


@dataclass(unsafe_hash=True)
class Container(Body): ...


@dataclass(unsafe_hash=True)
class Connection(WorldEntity):
    parent: Body
    child: Body


@dataclass(unsafe_hash=True)
class FixedConnection(Connection): ...


@dataclass(unsafe_hash=True)
class PrismaticConnection(Connection): ...


@dataclass(unsafe_hash=True)
class RevoluteConnection(Connection): ...


@dataclass
class World(Symbol):
    id: int = field(default=0)
    bodies: List[Body] = field(default_factory=list)
    connections: List[Connection] = field(default_factory=list)
    views: List[View] = field(default_factory=list, repr=False)

    def __hash__(self):
        return hash(self.id)

    def __eq__(self, other):
        if not isinstance(other, World):
            return False
        return self.id == other.id


@dataclass(unsafe_hash=True)
class View(WorldEntity): ...


@dataclass
class Drawer(View):
    handle: Handle
    container: Container
    correct: Optional[bool] = None

    def __hash__(self):
        return hash((self.__class__.__name__, self.handle, self.container))

    def __eq__(self, other):
        if not isinstance(other, Drawer):
            return False
        return (
            self.handle == other.handle
            and self.container == other.container
            and self.world == other.world
        )


@dataclass
class Cabinet(View):
    container: Container
    drawers: List[Drawer] = field(default_factory=list)

    def __hash__(self):
        return hash((self.__class__.__name__, self.container))

    def __eq__(self, other):
        if not isinstance(other, Cabinet):
            return False
        return (
            self.container == other.container
            and self.drawers == other.drawers
            and self.world == other.world
        )


@dataclass(unsafe_hash=True)
class Door(View):
    handle: Handle
    body: Body


@dataclass(unsafe_hash=True)
class Wardrobe(View):
    handle: Handle
    body: Body
    container: Container


@dataclass
class Apple(Body): ...


@dataclass
class FruitBox(Symbol):
    name: str
    fruits: List[Body]


@dataclass
class ContainsType(Predicate):
    """
    Predicate that checks if any object in the iterable is of the given type.
    """

    iterable: Iterable
    """
    Iterable to check for objects of the given type.
    """

    obj_type: Type
    """
    Object type to check for.
    """

    def __call__(self) -> bool:
        return any(isinstance(obj, self.obj_type) for obj in self.iterable)
