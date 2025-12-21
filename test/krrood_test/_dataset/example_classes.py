from __future__ import annotations

import importlib
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from types import FunctionType

from sqlalchemy import types, TypeDecorator, JSON
from typing_extensions import Dict, Any, Sequence, Self
from typing_extensions import List, Optional, Type

from krrood.adapters.json_serializer import SubclassJSONSerializer, to_json, from_json
from krrood.entity_query_language.predicate import Symbol
from krrood.ormatic.dao import AlternativeMapping, T


# check that custom enums works
class Element(Enum):
    C = "c"
    H = "h"


# Check that Types attributes work


@dataclass
class PositionTypeWrapper(Symbol):
    position_type: Type[Position]


# check that flat classes work


@dataclass
class Position(Symbol):
    x: float
    y: float
    z: float


# check that classes with optional values work


@dataclass
class Orientation(Symbol):
    x: float
    y: float
    z: float
    w: Optional[float]


# check that one to one relationship work


@dataclass
class Pose(Symbol):
    position: Position
    orientation: Orientation


# check that one to many relationship to built in types and non built in types work


@dataclass
class Positions(Symbol):
    positions: List[Position]
    some_strings: List[str]


@dataclass
class PositionsSubclassWithAnotherPosition(Positions):
    positions2: Position


# check that one to many relationships work where the many side is of the same type


@dataclass
class DoublePositionAggregator(Symbol):
    positions1: List[Position]
    positions2: List[Position]


# check that inheritance works


@dataclass
class Position4D(Position):
    w: float


# check that inheriting from an inherited class works


@dataclass
class Position5D(Position4D):
    v: float


# check with tree like classes


@dataclass
class Node(Symbol):
    parent: Optional[Node] = None


@dataclass
class NotMappedParent: ...


# check that enum references work


@dataclass
class Atom(NotMappedParent, Symbol):
    element: Element
    type: int
    charge: float
    timestamp: datetime = field(default_factory=datetime.now)


# check that custom type checks work
class PhysicalObject:
    pass


class Cup(PhysicalObject):
    pass


class Bowl(PhysicalObject):
    pass


# @dataclass
# class MultipleInheritance(Position, Orientation):
#    pass


@dataclass
class OriginalSimulatedObject(Symbol):
    concept: Optional[PhysicalObject]
    placeholder: float = field(default=0)


@dataclass
class ObjectAnnotation(Symbol):
    """
    Class for checking how classes that are explicitly mapped interact with original types.
    """

    object_reference: OriginalSimulatedObject


@dataclass
class KinematicChain(Symbol):
    name: str


@dataclass
class Torso(KinematicChain):
    """
    A Torso is a kinematic chain connecting the base of the robot with a collection of other kinematic chains.
    """

    kinematic_chains: List[KinematicChain] = field(default_factory=list)
    """
    A collection of kinematic chains that are connected to the torso.
    """


@dataclass
class Parent(Symbol):
    name: str


@dataclass
class ChildMapped(Parent):
    attribute1: int


@dataclass
class ChildNotMapped(Parent):
    attribute2: int
    unparseable: Dict[int, int]


@dataclass
class Entity(Symbol):
    name: str
    attribute_that_shouldnt_appear_at_all: float = 0


# Define a derived class


@dataclass
class DerivedEntity(Entity):
    description: str = "Default description"


@dataclass
class EntityAssociation(Symbol):
    """
    Class for checking how classes that are explicitly mapped interact with original types.
    """

    entity: Entity
    a: Sequence[str] = None


# Define an explicit mapping DAO that maps to the base entity class


@dataclass
class CustomEntity(AlternativeMapping[Entity]):
    overwritten_name: str

    @classmethod
    def create_instance(cls, obj: Entity):
        result = cls(overwritten_name=obj.name)
        return result

    def create_from_dao(self) -> T:
        return Entity(name=self.overwritten_name)


class ConceptType(TypeDecorator):
    """
    Type that casts fields that are of type `type` to their class name on serialization and converts the name
    to the class itself through the globals on load.
    """

    impl = types.String(256)

    def process_bind_param(self, value: PhysicalObject, dialect):
        return value.__class__.__module__ + "." + value.__class__.__name__

    def process_result_value(self, value: impl, dialect) -> Optional[Type]:
        if value is None:
            return None

        module_name, class_name = str(value).rsplit(".", 1)
        module = importlib.import_module(module_name)
        return getattr(module, class_name)()


@dataclass
class Reference(Symbol):
    value: int = 0
    backreference: Optional[Backreference] = None


@dataclass
class Backreference(Symbol):
    unmappable: Dict[Any, int]
    reference: Reference = None


@dataclass
class BackreferenceMapping(AlternativeMapping[Backreference]):
    values: List[int]
    reference: Reference

    @classmethod
    def create_instance(cls, obj: T):
        return cls(list(obj.unmappable.values()), obj.reference)

    def create_from_dao(self) -> T:
        return Backreference({v: v for v in self.values}, self.reference)


@dataclass
class AlternativeMappingAggregator(Symbol):
    entities1: List[Entity]
    entities2: List[Entity]


@dataclass
class ItemWithBackreference(Symbol):
    value: int = 0
    container: ContainerGeneration = None


@dataclass
class ContainerGeneration(Symbol):
    items: List[ItemWithBackreference]

    def __post_init__(self):
        for item in self.items:
            item.container = self


@dataclass
class Vector(Symbol):
    x: float


@dataclass
class VectorMapped(AlternativeMapping[Vector]):
    x: float

    @classmethod
    def create_instance(cls, obj: T):
        return VectorMapped(obj.x)

    def create_from_dao(self) -> T:
        return Vector(self.x)


@dataclass
class Rotation(Symbol):
    angle: float


@dataclass
class RotationMapped(AlternativeMapping[Rotation]):

    angle: float

    @classmethod
    def create_instance(cls, obj: T):
        return RotationMapped(obj.angle)

    def create_from_dao(self) -> T:
        pass


@dataclass
class Transformation(Symbol):
    vector: Vector
    rotation: Rotation


@dataclass
class TransformationMapped(AlternativeMapping[Transformation]):
    vector: Vector
    rotation: Rotation

    @classmethod
    def create_instance(cls, obj: T):
        return TransformationMapped(obj.vector, obj.rotation)

    def create_from_dao(self) -> T:
        return Transformation(self.vector, self.rotation)


@dataclass
class Shape(Symbol):
    name: str
    origin: Transformation


@dataclass
class Shapes(Symbol):
    shapes: List[Shape]


@dataclass
class MoreShapes(Symbol):
    shapes: List[Shapes]


@dataclass
class VectorsWithProperty(Symbol):
    _vectors: List[Vector]

    @property
    def vectors(self) -> List[Vector]:
        return self._vectors


@dataclass
class VectorsWithPropertyMapped(AlternativeMapping[VectorsWithProperty]):
    vectors: List[Vector]

    @classmethod
    def create_instance(cls, obj: T):
        return VectorsWithPropertyMapped(obj.vectors)

    def create_from_dao(self) -> T:
        return VectorsWithProperty(self.vectors)


@dataclass
class ParentBase(Symbol):
    name: str
    value: int


@dataclass
class ChildBase(ParentBase):
    pass


@dataclass
class ParentBaseMapping(AlternativeMapping[ParentBase]):
    name: str

    @classmethod
    def create_instance(cls, obj: T):
        if not isinstance(obj, Parent):
            raise TypeError(f"Expected Parent, got {type(obj)}")
        return ParentBaseMapping(obj.name)

    def create_from_dao(self) -> T:
        return ParentBase(self.name, 0)


@dataclass
class ChildBaseMapping(ParentBaseMapping, AlternativeMapping[ChildBase]):

    @classmethod
    def create_instance(cls, obj: T):
        if not isinstance(obj, ChildMapped):
            raise TypeError(f"Expected TestClass2, got {type(obj)}")
        return ChildBaseMapping(obj.name)

    def create_from_dao(self) -> T:
        return ChildBase(self.name, 0)


@dataclass
class PrivateDefaultFactory(Symbol):
    public_value: int = 0
    _private_list: List[int] = field(default_factory=list)


@dataclass
class RelationshipParent(Symbol):
    positions: Position


@dataclass
class RelationshipChild(RelationshipParent):
    """
    This class should produce a problem when reconstructed from the database as relationships must not be declared
    twice.
    """


# %% Test deep alternative mappings


@dataclass
class InheritanceBaseWithoutSymbolButAlternativelyMapped:
    """
    Test that alternative mappings that have a hierarchy of its own are correctly created.
    """

    base_attribute: float = 0


@dataclass
class InheritanceLevel1WithoutSymbolButAlternativelyMapped(
    InheritanceBaseWithoutSymbolButAlternativelyMapped
):
    level_one_attribute: float = 0


@dataclass
class InheritanceLevel2WithoutSymbolButAlternativelyMapped(
    InheritanceLevel1WithoutSymbolButAlternativelyMapped
):
    level_two_attribute: float = 0


@dataclass
class InheritanceBaseWithoutSymbolButAlternativelyMappedMapping(
    AlternativeMapping[InheritanceBaseWithoutSymbolButAlternativelyMapped]
):
    base_attribute: float = 0

    @classmethod
    def create_instance(cls, obj: T):
        return cls(obj.base_attribute)

    def create_from_dao(self) -> T:
        raise NotImplementedError


@dataclass
class InheritanceLevel1WithoutSymbolButAlternativelyMappedMapping(
    InheritanceBaseWithoutSymbolButAlternativelyMappedMapping,
    AlternativeMapping[InheritanceLevel1WithoutSymbolButAlternativelyMapped],
):
    level_one_attribute: float = 0

    @classmethod
    def create_instance(cls, obj: T):
        return cls(obj.base_attribute, obj.level_one_attribute)


@dataclass
class InheritanceLevel2WithoutSymbolButAlternativelyMappedMapping(
    InheritanceLevel1WithoutSymbolButAlternativelyMappedMapping,
    AlternativeMapping[InheritanceLevel2WithoutSymbolButAlternativelyMapped],
):
    level_two_attribute: float = 0

    @classmethod
    def create_instance(cls, obj: T):
        return cls(obj.base_attribute, obj.level_one_attribute, obj.level_two_attribute)


@dataclass
class ParentAlternativelyMapped:
    base_attribute: float = 0

    entities: List[Entity] = field(default_factory=list)


@dataclass
class ChildLevel1NormallyMapped(ParentAlternativelyMapped):
    level_one_attribute: float = 0


@dataclass
class ChildLevel2NormallyMapped(ChildLevel1NormallyMapped):
    level_two_attribute: float = 0


@dataclass
class ParentAlternativelyMappedMapping(AlternativeMapping[ParentAlternativelyMapped]):
    derived_attribute: str
    entities: List[Entity]

    @classmethod
    def create_instance(cls, obj: T) -> Self:
        return cls(str(obj.base_attribute), obj.entities)

    def create_from_dao(self) -> T:
        raise NotImplementedError


# %% Function like classes for testing


@dataclass
class CallableWrapper:
    func: FunctionType

    def custom_instance_method(self):
        return 2

    @classmethod
    def custom_class_method(cls):
        return 3

    @staticmethod
    def custom_static_method():
        return 4


def module_level_function():
    return 1


@dataclass
class UUIDWrapper:
    identification: uuid.UUID

    other_identifications: List[uuid.UUID] = field(default_factory=list)


# %% Test JSON serialization in ORM classes


@dataclass
class JSONSerializableClass(SubclassJSONSerializer):
    a: float = 0.0
    b: float = 1.0

    def to_json(self) -> Dict[str, Any]:
        return {**super().to_json(), "a": to_json(self.a), "b": to_json(self.b)}

    @classmethod
    def _from_json(cls, data: Dict[str, Any], **kwargs) -> Self:
        return cls(a=from_json(data["a"]), b=from_json(data["b"]))


@dataclass
class JSONWrapper:
    json_serializable_object: JSONSerializableClass
    more_objects: List[JSONSerializableClass] = field(default_factory=list)


# %% Multiple inheritance and MRO tests


@dataclass
class Mixin:
    mixin_attribute: str


@dataclass
class PrimaryBase:
    primary_attribute: str


@dataclass
class MultipleInheritance(PrimaryBase, Mixin):
    extra_attribute: str


# %% Test enum list
class TestEnum(Enum):
    OPTION_A = "option_a"
    OPTION_B = "option_b"
    OPTION_C = "option_c"


@dataclass
class ListOfEnum(Symbol):
    list_of_enum: List[TestEnum]


# %% Test forward reference resolution with multiple unresolved types
# This reproduces the issue where get_type_hints fails because multiple
# forward references need to be resolved iteratively


@dataclass
class ForwardRefTypeA(Symbol):
    """A simple class used as a forward reference target."""

    value: str = ""


@dataclass
class ForwardRefTypeB(Symbol):
    """Another class used as a forward reference target."""

    count: int = 0


@dataclass
class MultipleForwardRefContainer(Symbol):
    """
    A class that has multiple fields with forward reference types.
    This tests that the forward reference resolution can handle
    multiple unresolved types that need to be resolved iteratively.
    """

    ref_a: Optional[ForwardRefTypeA] = None
    ref_b: Optional[ForwardRefTypeB] = None


@dataclass
class Person:
    name: str

    knows: List[Person] = field(default_factory=list)
