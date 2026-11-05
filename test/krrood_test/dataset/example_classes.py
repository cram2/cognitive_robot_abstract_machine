from __future__ import annotations

import importlib
import uuid
from abc import ABC
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum, auto
from pathlib import Path
from types import FunctionType, NoneType
from typing import Set, Generic, TypeVar as TypingTypeVar

from sqlalchemy import types, TypeDecorator
from typing_extensions import Dict, Any, Sequence, Self
from typing_extensions import List, Optional, Type

from krrood.adapters.json_serializer import SubclassJSONSerializer, to_json, from_json
from krrood.class_diagrams.mocking import MockedClass
from krrood.entity_query_language.factories import (
    variable,
    count,
    entity,
    count_range,
)
from krrood.ormatic.data_access_objects.alternative_mappings import (
    AlternativeMapping,
    T,
)
from krrood.patterns.subclass_safe_generic import SubClassSafeGeneric
from krrood.symbol_graph.symbol_graph import Symbol
from krrood import logger

try:
    from random_events.interval import Bound, SimpleInterval
    from krrood.parametrization.feature_extraction.aggregations import (
        AggregationStatistic,
        aggregation_statistic,
    )
except ImportError as e:
    # Was added to allow this to work on Windows which random_events does not support.
    logger.debug(f"Could not import random_events: {e}")

    class AggregationStatistic(MockedClass, Generic[T]): ...

    aggregation_statistic = lambda *args: lambda *args2: args2
    Bound = NoneType
    SimpleInterval = NoneType


from ..dataset.semantic_world_like_classes import Body, Cabinet


# check that custom enums works
class Element(Enum):
    C = "c"
    H = "h"


# Check that Types attributes work


@dataclass(kw_only=True)
class KRROODPositionTypeWrapper(Symbol):
    position_type: Type[KRROODPosition]


@dataclass(kw_only=True)
class KRROODBarePositionTypeWrapper(Symbol):
    position_type: type


@dataclass(kw_only=True)
class KRROODPositionTypesWrapper(Symbol):
    position_types: List[Type[KRROODPosition]]


# check that flat classes work
@dataclass(kw_only=True, unsafe_hash=True)
class KRROODPosition(Symbol):
    x: float
    y: float
    z: float

    @classmethod
    def from_abc(cls, *, a: float, b: float, c: float) -> KRROODPosition:
        return KRROODPosition(x=a, y=b, z=c)


# check that classes with optional values work
@dataclass(kw_only=True)
class KRROODOrientation(Symbol):
    x: float
    y: float
    z: float
    w: Optional[float]


# check that the PEP 604 spelling of an optional is recognised just like Optional[...]
@dataclass(kw_only=True)
class KRROODPipeOptionalOrientation(Symbol):
    x: float
    y: float
    z: float
    w: float | None
    position: KRROODPosition | None


# check that one to one relationship work
@dataclass(kw_only=True)
class KRROODPose(Symbol):
    position: KRROODPosition
    orientation: KRROODOrientation


# check that many to many relationship to built in types and non built in types work
@dataclass(kw_only=True)
class KRROODPositions(Symbol):
    positions: List[KRROODPosition]
    some_strings: List[str]


@dataclass(kw_only=True)
class KRROODPositionsSubclassWithAnotherKRROODPosition(KRROODPositions):
    positions2: KRROODPosition


# check that one to many relationships work where the many side is of the same type
@dataclass(kw_only=True, eq=False)
class DoubleKRROODPositionAggregator(Symbol):
    positions1: List[KRROODPosition]
    positions2: List[KRROODPosition]


# check that inheritance works


@dataclass(kw_only=True)
class KRROODPosition4D(KRROODPosition):
    w: float


# check that inheriting from an inherited class works


@dataclass(kw_only=True)
class KRROODPosition5D(KRROODPosition4D):
    v: float


# check with tree like classes


@dataclass(kw_only=True)
class Node(Symbol):
    parent: Optional[Node] = None


@dataclass(kw_only=True)
class NotMappedParent: ...


# check that enum references work


@dataclass(kw_only=True)
class Atom(NotMappedParent, Symbol):
    element: Element
    type: int
    charge: float
    timestamp: datetime = field(default_factory=datetime.now)


# check that custom type checks work
class KRROODPhysicalObject:
    pass


class KRROODCup(KRROODPhysicalObject):
    pass


class KRROODBowl(KRROODPhysicalObject):
    pass


@dataclass(kw_only=True, unsafe_hash=True)
class NestedAction:
    obj: Body
    pose: Optional[KRROODPose]


@dataclass(kw_only=True)
class EnumAction:
    obj: Body
    enum: TestEnum


@dataclass(kw_only=True)
class OriginalSimulatedObject(Symbol):
    concept: Optional[KRROODPhysicalObject]
    placeholder: float = field(default=0)


@dataclass(kw_only=True)
class ObjectAnnotation(Symbol):
    """
    Class for checking how classes that are explicitly mapped interact with
    original types.
    """

    object_reference: OriginalSimulatedObject


@dataclass(kw_only=True)
class KRROODKinematicChain(Symbol):
    name: str


@dataclass(kw_only=True)
class KRROODTorso(KRROODKinematicChain):
    """
    A KRROODTorso is a kinematic chain connecting the base of the robot with a
    collection of other kinematic chains.
    """

    kinematic_chains: List[KRROODKinematicChain] = field(default_factory=list)
    """
    A collection of kinematic chains that are connected to the torso.
    """


@dataclass(kw_only=True)
class Parent(Symbol):
    name: str


@dataclass(kw_only=True)
class ChildMapped(Parent):
    attribute1: int


@dataclass(kw_only=True)
class ChildNotMapped(Parent):
    attribute2: int
    unparseable: Dict[int, int]


@dataclass(kw_only=True)
class Entity(Symbol):
    name: str
    attribute_that_shouldnt_appear_at_all: float = 0


# Define a derived class


@dataclass(kw_only=True)
class DerivedEntity(Entity):
    description: str = "Default description"


@dataclass(kw_only=True)
class EntityAssociation(Symbol):
    """
    Class for checking how classes that are explicitly mapped interact with
    original types.
    """

    entity: Entity
    a: Sequence[str] = None


# Define an explicit mapping DAO that maps to the base entity class


@dataclass(kw_only=True, eq=False)
class EntityMapping(AlternativeMapping[Entity]):
    overwritten_name: str

    @classmethod
    def from_domain_object(cls, *, obj: Entity):
        result = cls(overwritten_name=obj.name)
        return result

    def to_domain_object(self) -> T:
        return Entity(name=self.overwritten_name)


class ConceptType(TypeDecorator):
    """
    Type that casts fields that are of type `type` to their class name on
    serialization and converts the name to the class itself through the globals
    on load.
    """

    impl = types.String(256)

    def process_bind_param(self, value: KRROODPhysicalObject, dialect):
        return value.__class__.__module__ + "." + value.__class__.__name__

    def process_result_value(self, value: impl, dialect) -> Optional[Type]:
        if value is None:
            return None

        module_name, class_name = str(value).rsplit(".", 1)
        module = importlib.import_module(module_name)
        return getattr(module, class_name)()


@dataclass(kw_only=True)
class Reference(Symbol):
    value: int = 0
    backreference: Optional[Backreference] = None


@dataclass(kw_only=True)
class Backreference(Symbol):
    unmappable: Dict[Any, int]
    reference: Reference = None


@dataclass(kw_only=True, eq=False)
class BackreferenceMapping(AlternativeMapping[Backreference]):
    values: List[int]
    reference: Reference

    @classmethod
    def from_domain_object(cls, *, obj: T):
        return cls(values=list(obj.unmappable.values()), reference=obj.reference)

    def to_domain_object(self) -> T:
        return Backreference(unmappable={v: v for v in self.values}, reference=self.reference)


@dataclass(kw_only=True, eq=False)
class AlternativeMappingAggregator(Symbol):
    entities1: List[Entity]
    entities2: List[Entity]


@dataclass(kw_only=True)
class ItemWithBackreference(Symbol):
    value: int = 0
    container: ContainerGeneration = None


@dataclass(kw_only=True)
class ContainerGeneration(Symbol):
    items: List[ItemWithBackreference]

    def __post_init__(self):
        for item in self.items:
            item.container = self


@dataclass(kw_only=True)
class KRROODVector(Symbol):
    x: float


@dataclass(kw_only=True, eq=False)
class KRROODVectorMapped(AlternativeMapping[KRROODVector]):
    x: float

    @classmethod
    def from_domain_object(cls, *, obj: T):
        return KRROODVectorMapped(x=obj.x)

    def to_domain_object(self) -> T:
        return KRROODVector(x=self.x)


@dataclass(kw_only=True)
class Rotation(Symbol):
    angle: float


@dataclass(kw_only=True, eq=False)
class RotationMapped(AlternativeMapping[Rotation]):
    angle: float

    @classmethod
    def from_domain_object(cls, *, obj: T):
        return RotationMapped(angle=obj.angle)

    def to_domain_object(self) -> T:
        pass


@dataclass(kw_only=True)
class KRROODTransformation(Symbol):
    vector: KRROODVector
    rotation: Rotation


@dataclass(kw_only=True, eq=False)
class KRROODTransformationMapped(AlternativeMapping[KRROODTransformation]):
    vector: KRROODVector
    rotation: Rotation

    @classmethod
    def from_domain_object(cls, *, obj: T):
        return KRROODTransformationMapped(vector=obj.vector, rotation=obj.rotation)

    def to_domain_object(self) -> T:
        return KRROODTransformation(vector=self.vector, rotation=self.rotation)


@dataclass(kw_only=True)
class Shape(Symbol):
    name: str
    origin: KRROODTransformation


@dataclass(kw_only=True)
class Shapes(Symbol):
    shapes: List[Shape]


@dataclass(kw_only=True)
class MoreShapes(Symbol):
    shapes: List[Shapes]


@dataclass(kw_only=True)
class KRROODVectorsWithProperty(Symbol):
    _vectors: List[KRROODVector]

    @property
    def vectors(self) -> List[KRROODVector]:
        return self._vectors


@dataclass(kw_only=True, eq=False)
class KRROODVectorsWithPropertyMapped(AlternativeMapping[KRROODVectorsWithProperty]):
    vectors: List[KRROODVector]

    @classmethod
    def from_domain_object(cls, *, obj: T):
        return KRROODVectorsWithPropertyMapped(vectors=obj.vectors)

    def to_domain_object(self) -> T:
        return KRROODVectorsWithProperty(_vectors=self.vectors)


@dataclass(kw_only=True)
class ParentBase(Symbol):
    name: str
    value: int


@dataclass(kw_only=True)
class ChildBase(ParentBase):
    pass


@dataclass(kw_only=True, eq=False)
class ParentBaseMapping(AlternativeMapping[ParentBase]):
    name: str

    @classmethod
    def from_domain_object(cls, *, obj: T):
        if not isinstance(obj, Parent):
            raise TypeError(f"Expected Parent, got {type(obj)}")
        return ParentBaseMapping(name=obj.name)

    def to_domain_object(self) -> T:
        return ParentBase(name=self.name, value=0)


@dataclass(kw_only=True, eq=False)
class ChildBaseMapping(ParentBaseMapping, AlternativeMapping[ChildBase]):
    @classmethod
    def from_domain_object(cls, *, obj: T):
        if not isinstance(obj, ChildMapped):
            raise TypeError(f"Expected TestClass2, got {type(obj)}")
        return ChildBaseMapping(name=obj.name)

    def to_domain_object(self) -> T:
        return ChildBase(name=self.name, value=0)


@dataclass(kw_only=True)
class PrivateDefaultFactory(Symbol):
    public_value: int = 0
    _private_list: List[int] = field(default_factory=list)


@dataclass(kw_only=True)
class RelationshipParent(Symbol):
    positions: KRROODPosition


@dataclass(kw_only=True)
class RelationshipChild(RelationshipParent):
    """
    This class should produce a problem when reconstructed from the database as
    relationships must not be declared twice.
    """


# %% Test deep alternative mappings


@dataclass(kw_only=True)
class InheritanceBaseWithoutSymbolButAlternativelyMapped:
    """
    Test that alternative mappings that have a hierarchy of its own are
    correctly created.
    """

    base_attribute: float = 0


@dataclass(kw_only=True)
class InheritanceLevel1WithoutSymbolButAlternativelyMapped(
    InheritanceBaseWithoutSymbolButAlternativelyMapped
):
    level_one_attribute: float = 0


@dataclass(kw_only=True)
class InheritanceLevel2WithoutSymbolButAlternativelyMapped(
    InheritanceLevel1WithoutSymbolButAlternativelyMapped
):
    level_two_attribute: float = 0


@dataclass(kw_only=True, eq=False)
class InheritanceBaseWithoutSymbolButAlternativelyMappedMapping(
    AlternativeMapping[InheritanceBaseWithoutSymbolButAlternativelyMapped]
):
    base_attribute: float = 0

    @classmethod
    def from_domain_object(cls, *, obj: T):
        return cls(base_attribute=obj.base_attribute)

    def to_domain_object(self) -> T:
        raise NotImplementedError


@dataclass(kw_only=True, eq=False)
class InheritanceLevel1WithoutSymbolButAlternativelyMappedMapping(
    InheritanceBaseWithoutSymbolButAlternativelyMappedMapping,
    AlternativeMapping[InheritanceLevel1WithoutSymbolButAlternativelyMapped],
):
    level_one_attribute: float = 0

    @classmethod
    def from_domain_object(cls, *, obj: T):
        return cls(base_attribute=obj.base_attribute, level_one_attribute=obj.level_one_attribute)


@dataclass(kw_only=True, eq=False)
class InheritanceLevel2WithoutSymbolButAlternativelyMappedMapping(
    InheritanceLevel1WithoutSymbolButAlternativelyMappedMapping,
    AlternativeMapping[InheritanceLevel2WithoutSymbolButAlternativelyMapped],
):
    level_two_attribute: float = 0

    @classmethod
    def from_domain_object(cls, *, obj: T):
        return cls(base_attribute=obj.base_attribute, level_one_attribute=obj.level_one_attribute, level_two_attribute=obj.level_two_attribute)


@dataclass(kw_only=True)
class ParentAlternativelyMapped:
    base_attribute: float = 0

    entities: List[Entity] = field(default_factory=list)


@dataclass(kw_only=True)
class ChildLevel1NormallyMapped(ParentAlternativelyMapped):
    level_one_attribute: float = 0


@dataclass(kw_only=True)
class ChildLevel2NormallyMapped(ChildLevel1NormallyMapped):
    level_two_attribute: float = 0


@dataclass(kw_only=True, eq=False)
class ParentAlternativelyMappedMapping(AlternativeMapping[ParentAlternativelyMapped]):
    derived_attribute: str
    entities: List[Entity]

    @classmethod
    def from_domain_object(cls, *, obj: T) -> Self:
        return cls(derived_attribute=str(obj.base_attribute), entities=obj.entities)

    def to_domain_object(self) -> T:
        raise NotImplementedError


# %% Function like classes for testing


@dataclass(kw_only=True)
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


@dataclass(kw_only=True)
class UUIDWrapper:
    identification: uuid.UUID

    other_identifications: List[uuid.UUID] = field(default_factory=list)


# %% Test JSON serialization in ORM classes


@dataclass(kw_only=True)
class JSONSerializableClass(SubclassJSONSerializer):
    a: float = 0.0
    b: float = 1.0

    def to_json(self) -> Dict[str, Any]:
        return {**super().to_json(), "a": to_json(self.a), "b": to_json(self.b)}

    @classmethod
    def _from_json(cls, *, data: Dict[str, Any], **kwargs) -> Self:
        return cls(a=from_json(data["a"]), b=from_json(data["b"]))


@dataclass(kw_only=True)
class JSONWrapper:
    json_serializable_object: JSONSerializableClass
    more_objects: List[JSONSerializableClass] = field(default_factory=list)


JSONSerializableAnswer = TypingTypeVar("JSONSerializableAnswer")
"""
What a :class:`GenericJSONSerializableClass` answers with.
"""


@dataclass(kw_only=True)
class GenericJSONSerializableClass(
    SubclassJSONSerializer, Generic[JSONSerializableAnswer], SubClassSafeGeneric, ABC
):
    """
    A JSON-serializable base that is used as a field type without binding its parameter,
    so the field says only that some subclass of it is stored.
    """

    label: str = ""
    """
    What this value stands for.
    """

    def to_json(self) -> Dict[str, Any]:
        return {**super().to_json(), "label": to_json(self.label)}

    @classmethod
    def _from_json(cls, *, data: Dict[str, Any], **kwargs) -> Self:
        return cls(label=from_json(data["label"]))


@dataclass(kw_only=True)
class TextJSONSerializableClass(GenericJSONSerializableClass[str]): ...


@dataclass(kw_only=True)
class GenericJSONWrapper:
    """
    Holds the generic JSON value under a field that leaves the parameter free.
    """

    json_serializable_object: GenericJSONSerializableClass
    """
    The value, whose own JSON names which subclass it is.
    """


@dataclass(kw_only=True)
class HolderOfSimpleInterval:
    """
    Its sole field's type, :class:`random_events.interval.SimpleInterval`, lives in a
    package nothing else in this module references, which is exactly what is needed to
    reproduce a bug where ``WrappedTable.create_custom_type`` mapped a
    ``SubclassJSONSerializer`` field to JSON without importing that field type's own
    module, tripping a ``MappedAnnotationError`` at class-definition time.
    """

    bounds: SimpleInterval = field(
        default_factory=lambda: SimpleInterval.from_data(
            0.0, 1.0, Bound.CLOSED, Bound.CLOSED
        )
    )


# %% Multiple inheritance and MRO tests


@dataclass(kw_only=True)
class Mixin:
    mixin_attribute: str


@dataclass(kw_only=True)
class PrimaryBase:
    primary_attribute: str


@dataclass(kw_only=True)
class MultipleInheritance(PrimaryBase, Mixin):
    extra_attribute: str


# %% Test enum list
class TestEnum(Enum):
    OPTION_A = "option_a"
    OPTION_B = "option_b"
    OPTION_C = "option_c"


@dataclass(kw_only=True)
class ListOfEnum(Symbol):
    list_of_enum: List[TestEnum]


# %% Test forward reference resolution with multiple unresolved types
# This reproduces the issue where get_type_hints fails because multiple
# forward references need to be resolved iteratively


@dataclass(kw_only=True)
class ForwardRefTypeA(Symbol):
    """
    A simple class used as a forward reference target.
    """

    value: str = ""


@dataclass(kw_only=True)
class ForwardRefTypeB(Symbol):
    """
    Another class used as a forward reference target.
    """

    count: int = 0


@dataclass(kw_only=True)
class MultipleForwardRefContainer(Symbol):
    """
    A class that has multiple fields with forward reference types.

    This tests that the forward reference resolution can handle multiple
    unresolved types that need to be resolved iteratively.
    """

    ref_a: Optional[ForwardRefTypeA] = None
    ref_b: Optional[ForwardRefTypeB] = None


@dataclass(kw_only=True)
class Person:
    name: str

    knows: List[Person] = field(default_factory=list)


@dataclass(kw_only=True)
class UnderspecifiedTypesContainer:
    any_list: List[Any] = field(default_factory=list)
    any_field: Any = None


@dataclass(kw_only=True)
class TestKRROODPositionSet:
    positions: Set[KRROODPosition] = field(default_factory=set)


class PolymorphicEnum(Enum): ...


class ChildEnum1(PolymorphicEnum):
    A = auto()
    B = auto()


class ChildEnum2(PolymorphicEnum):
    B = auto()
    C = auto()


@dataclass(kw_only=True)
class PolymorphicEnumAssociation:
    value: PolymorphicEnum


@dataclass(kw_only=True, frozen=True)
class NamedNumbers:
    name: str
    numbers: List[int] = field(default_factory=list)

    def __hash__(self):
        return hash(self.name)


@dataclass(kw_only=True)
class GenericClass(Generic[T]):
    value: T
    optional_value: Optional[T] = None
    container: List[T] = field(default_factory=list)


@dataclass(kw_only=True)
class GenericClassAssociation:
    associated_value: GenericClass[float]
    associated_value_list: List[GenericClass[KRROODPosition]]

    associated_value_not_parametrized: GenericClass = None
    associated_value_not_parametrized_list: List[GenericClass] = field(
        default_factory=list
    )


# %% Test TypeVar field resolution in DAO generation
# Reproduces the issue where fields typed with a TypeVar (like
# TKinematicStructureEntity in HasRootKinematicStructureEntity) were silently
# skipped by WrappedTable.parse_field.

TTestEntity = TypingTypeVar("TTestEntity", bound=KRROODPosition)


@dataclass(kw_only=True)
class TypeVarFieldHolder(Symbol):
    """Domain class with a TypeVar-typed field to test parse_field resolution."""

    typed_field: TTestEntity = field(kw_only=True)
    name: str = ""


@dataclass(kw_only=True)
class PathAssociation:
    path: Path


@dataclass(kw_only=True)
class DurationAssociation:
    """
    Holds a duration, to test that the ORM stores a time span.
    """

    duration: timedelta
    """
    The stored time span.
    """


class SceneObjectType(Enum):
    TABLE = "table"
    CHAIR = "chair"


@dataclass(kw_only=True)
class SceneObject:
    type: SceneObjectType


@dataclass(kw_only=True)
class ApproachSceneObject:
    """
    An action standing for one whose argument is a whole entity carrying an enum, so a
    query over it has to describe that entity's own kind as well as its own fields.
    """

    target: SceneObject
    """
    The object approached, named outright rather than described.
    """

    speed: float
    """
    How fast it is approached, which a query may leave underspecified.
    """


@dataclass(kw_only=True)
class SceneRoom:
    position: KRROODPosition
    orientation: KRROODOrientation
    objects: List[SceneObject]
    type_in_need_of_preprocessing: bool = False


@dataclass(kw_only=True)
class SceneWithExchangeableParts:
    objects: List[SceneObject]
    rooms: List[SceneRoom]


TSceneObjectHolder = TypingTypeVar("TSceneObjectHolder", bound=object)


@dataclass(kw_only=True, eq=False)
class SceneObjectAggregationBase(AggregationStatistic[TSceneObjectHolder]):
    """
    Abstract base providing shared aggregation statistics over a
    ``objects: List[SceneObject]`` field.

    Concrete subclasses bind ``T`` to the owner type and are auto-registered.
    """

    @aggregation_statistic(variable(SceneRoom).objects)
    def chair_count(self) -> int:
        """
        Count of CHAIR-type objects.
        """
        type_var = variable(SceneObject, self.objects).type
        [cou] = (
            entity(count_range(type_var))
            .where(type_var == SceneObjectType.CHAIR)
            .tolist()
        )
        return cou

    @aggregation_statistic(variable(SceneRoom).objects)
    def table_count(self) -> int:
        """
        Count of TABLE-type objects.
        """
        type_var = variable(SceneObject, self.objects).type
        [cou] = (
            entity(count_range(type_var))
            .where(type_var == SceneObjectType.TABLE)
            .tolist()
        )
        return cou

    @aggregation_statistic(variable(SceneRoom).objects)
    def total_count(self) -> int:
        """
        Total number of objects.
        """
        [cou] = count(variable(SceneObject, self.objects)).tolist()
        return cou


@dataclass(kw_only=True, eq=False)
class SceneRoomAggregations(SceneObjectAggregationBase[SceneRoom]):
    """
    Aggregation statistics for :class:`SceneRoom` over its ``objects`` field.
    """


@dataclass(kw_only=True, eq=False)
class SceneWithExchangeablePartsAggregations(
    SceneObjectAggregationBase[SceneWithExchangeableParts]
):
    """
    Aggregation statistics for :class:`SceneWithExchangeableParts` over its ``objects`` and
    ``rooms`` fields.
    """

    @aggregation_statistic(variable(SceneWithExchangeableParts).rooms)
    def room_count(self) -> int:
        """
        Total number of rooms.
        """
        [cou] = count(variable(SceneRoom, self.rooms)).tolist()
        return cou


@dataclass(kw_only=True)
class SceneFloor:
    """
    A level of a building, holding rooms; a mimic of an entity that is itself an
    exchangeable part and holds exchangeable parts.
    """

    rooms: List[SceneRoom]
    """
    The rooms on this floor.
    """


@dataclass(kw_only=True)
class SceneBuilding:
    """
    A building, holding floors; the root of a three-level chain of exchangeable parts.
    """

    floors: List[SceneFloor]
    """
    The floors of this building.
    """


@dataclass(kw_only=True, eq=False)
class SceneFloorAggregations(AggregationStatistic[SceneFloor]):
    """
    Aggregation statistics for :class:`SceneFloor` over its ``rooms`` field.
    """

    @aggregation_statistic(variable(SceneFloor).rooms)
    def room_count(self) -> int:
        """
        Total number of rooms on the floor.
        """
        [cou] = count(variable(SceneRoom, self.rooms)).tolist()
        return cou


@dataclass(kw_only=True, eq=False)
class SceneBuildingAggregations(AggregationStatistic[SceneBuilding]):
    """
    Aggregation statistics for :class:`SceneBuilding` over its ``floors`` field.
    """

    @aggregation_statistic(variable(SceneBuilding).floors)
    def floor_count(self) -> int:
        """
        Total number of floors in the building.
        """
        [cou] = count(variable(SceneFloor, self.floors)).tolist()
        return cou


# %% Relational causal experiment (skill-confounded grasp attempts)


@dataclass(kw_only=True)
class GraspAttempt:
    """
    A single grasp attempt, whose arm position and outcome may be confounded by
    the robot's shared skill level.
    """

    arm: float
    """
    The arm's position at the time of this attempt.
    """

    grasped: bool
    """
    Whether this attempt succeeded.
    """


@dataclass(kw_only=True)
class PickingRobot:
    """
    A robot performing a series of grasp attempts, with a skill level shared
    across every attempt it makes.
    """

    skill: float
    """
    The robot's skill level, generated independently of any single attempt.
    """

    attempts: List[GraspAttempt]
    """
    The robot's grasp attempts.
    """


@dataclass(kw_only=True, eq=False)
class PickingRobotAggregations(AggregationStatistic[PickingRobot]):
    """
    Aggregation statistics for :class:`PickingRobot` over its ``attempts`` field.
    """

    @aggregation_statistic(variable(PickingRobot).attempts)
    def success_count(self) -> int:
        """
        Count of successful grasp attempts.
        """
        grasped_var = variable(GraspAttempt, self.attempts).grasped
        [result] = entity(count_range(grasped_var)).where(grasped_var == True).tolist()
        return result

    @aggregation_statistic(variable(PickingRobot).attempts)
    def total_count(self) -> int:
        """
        Total number of grasp attempts.
        """
        [result] = count(variable(GraspAttempt, self.attempts)).tolist()
        return result


class RobotStation(Enum):
    """
    Where a picking robot is stationed.
    """

    LAB = auto()
    FACTORY = auto()


@dataclass(kw_only=True)
class StationedPickingRobot:
    """
    A picking robot with a class-level enum attribute -- its station -- alongside its
    exchangeable grasp attempts.
    """

    station: RobotStation
    """
    Where the robot is stationed.
    """

    skill: float
    """
    The robot's skill level.
    """

    attempts: List[GraspAttempt]
    """
    The robot's grasp attempts.
    """


@dataclass(kw_only=True, eq=False)
class StationedPickingRobotAggregations(AggregationStatistic[StationedPickingRobot]):
    """
    Aggregation statistics for :class:`StationedPickingRobot` over its ``attempts``
    field.
    """

    @aggregation_statistic(variable(StationedPickingRobot).attempts)
    def success_count(self) -> int:
        """
        Count of successful grasp attempts.
        """
        grasped_var = variable(GraspAttempt, self.attempts).grasped
        [result] = entity(count_range(grasped_var)).where(grasped_var == True).tolist()
        return result


@dataclass(kw_only=True)
class ExampleInt:
    attribute: int


@dataclass(kw_only=True)
class ExampleString:
    attribute: str


@dataclass(kw_only=True)
class MissingBaseClass:
    objects: List[ExampleInt] = field(default_factory=list)


@dataclass(kw_only=True, eq=False)
class ActionWithMissingAggregationsMixin:
    """
    Action with a field whose domain type has exchangeable parts but no
    aggregation mixin.
    """

    domain_object: Cabinet
