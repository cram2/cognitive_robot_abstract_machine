from __future__ import annotations

from dataclasses import dataclass, field

from typing_extensions import ClassVar, List, Mapping, Optional, Type, Iterable

from krrood.entity_query_language.predicate import Symbol, Predicate, Triple
from krrood.entity_query_language.verbalization.fragments.base import (
    VerbalizationFragment,
)
from krrood.entity_query_language.verbalization.vocabulary.english import Prepositions
from krrood.entity_query_language.verbalization.vocabulary.parts_of_speech import (
    clause,
    Noun,
    Verb,
)


@dataclass(kw_only=True, unsafe_hash=True)
class WorldEntity(Symbol):
    world: Optional[World] = field(default=None, kw_only=True, repr=False, hash=False)


@dataclass(kw_only=True, unsafe_hash=True)
class Body(WorldEntity):
    name: str
    size: int = field(default=1)


@dataclass(kw_only=True, unsafe_hash=True)
class Handle(Body): ...


@dataclass(kw_only=True, unsafe_hash=True)
class Container(Body): ...


@dataclass(kw_only=True, unsafe_hash=True)
class Connection(WorldEntity):
    parent: Body
    child: Body


@dataclass(kw_only=True, unsafe_hash=True)
class FixedConnection(Connection): ...


@dataclass(kw_only=True, unsafe_hash=True)
class PrismaticConnection(Connection): ...


@dataclass(kw_only=True, unsafe_hash=True)
class RevoluteConnection(Connection): ...


@dataclass(kw_only=True)
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


@dataclass(kw_only=True, unsafe_hash=True)
class View(WorldEntity): ...


@dataclass(kw_only=True)
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


@dataclass(kw_only=True)
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


@dataclass(kw_only=True, unsafe_hash=True)
class Door(View):
    handle: Handle
    body: Body


@dataclass(kw_only=True, unsafe_hash=True)
class Wardrobe(View):
    handle: Handle
    body: Body
    container: Container


@dataclass(kw_only=True, unsafe_hash=True)
class Apple(Body): ...


@dataclass(kw_only=True)
class FruitBox(Symbol):
    name: str
    fruits: List[Body]


@dataclass(kw_only=True)
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

    @classmethod
    def _verbalization_fragment_(
        cls, *, fields: Mapping[str, VerbalizationFragment]
    ) -> VerbalizationFragment:
        return clause(
            Noun(content=fields["iterable"]),
            Verb(lemma="contain"),
            Noun(content="instance"),
            Prepositions.OF,
            Noun(content=fields["obj_type"]),
        )


@dataclass(kw_only=True, eq=False)
class TripleDeclaringItsObjectFirst(Triple[Handle, Drawer]):
    """
    A relation whose field holding its object comes before the field holding its
    subject: a handle opens a drawer.
    """

    drawer: Drawer
    """
    The drawer the handle opens.
    """

    handle: Handle
    """
    The handle that opens the drawer.
    """

    @property
    def subject(self) -> Handle:
        return self.handle

    @property
    def object(self) -> Drawer:
        return self.drawer

    def __call__(self) -> bool:
        return self.drawer.handle is self.handle


@dataclass(kw_only=True, unsafe_hash=True)
class GraspConfig(WorldEntity):
    """
    A value object an action reaches through a relationship rather than owning directly.

    Used to test set_of() with attributes that are transitive rather than direct.
    """

    rotate_gripper: float = field(default=0.0)
    approach_direction: float = field(default=0.0)
    manipulation_offset: float = field(default=0.0)


@dataclass(kw_only=True, unsafe_hash=True)
class MoveAction(WorldEntity):
    """
    An action with direct fields alongside a relationship to a nested value object.

    Used to test set_of() with both direct and transitive attributes.
    """

    robot_x: float = field(default=0.0)
    robot_y: float = field(default=0.0)
    hip_rotation: float = field(default=0.0)
    grasp_config: GraspConfig = field(default=None)
