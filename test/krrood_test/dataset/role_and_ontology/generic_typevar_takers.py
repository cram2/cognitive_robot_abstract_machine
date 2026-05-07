from dataclasses import dataclass, field
from typing import Generic, TypeVar

from krrood.entity_query_language.factories import variable_from
from krrood.patterns.role.role import Role


@dataclass(eq=False)
class BaseEntity:
    identifier: str = field(kw_only=True)


@dataclass(eq=False)
class ConcreteEntity(BaseEntity):
    data: float = field(default=0.0, kw_only=True)


TBase = TypeVar("TBase", bound=BaseEntity)


@dataclass(eq=False)
class GenericBaseMixin(Generic[TBase]):
    entity: TBase = field(kw_only=True)
    count: int = field(default=0, kw_only=True)


TConcreteEntity = TypeVar("TConcreteEntity", bound=ConcreteEntity)


@dataclass(eq=False)
class NarrowedTypeVarTaker(GenericBaseMixin[TConcreteEntity]):
    label: str = field(default="", kw_only=True)


@dataclass(eq=False)
class ConcreteTypeTaker(GenericBaseMixin[ConcreteEntity]):
    name: str = field(default="", kw_only=True)


@dataclass(eq=False)
class UnspecializedSubTaker(GenericBaseMixin):
    tag: str = field(default="", kw_only=True)


TNarrowedTypeVarTaker = TypeVar("TNarrowedTypeVarTaker", bound=NarrowedTypeVarTaker)
TConcreteTypeTaker = TypeVar("TConcreteTypeTaker", bound=ConcreteTypeTaker)
TUnspecializedSubTaker = TypeVar("TUnspecializedSubTaker", bound=UnspecializedSubTaker)


@dataclass(eq=False)
class RoleWithNarrowedTaker(Role[TNarrowedTypeVarTaker]):
    taker: TNarrowedTypeVarTaker = field(kw_only=True)

    @classmethod
    def role_taker_attribute(cls) -> TNarrowedTypeVarTaker:
        return variable_from(cls).taker


@dataclass(eq=False)
class RoleWithConcreteTaker(Role[TConcreteTypeTaker]):
    taker: TConcreteTypeTaker = field(kw_only=True)

    @classmethod
    def role_taker_attribute(cls) -> TConcreteTypeTaker:
        return variable_from(cls).taker


@dataclass(eq=False)
class RoleWithUnspecializedTaker(Role[TUnspecializedSubTaker]):
    taker: TUnspecializedSubTaker = field(kw_only=True)

    @classmethod
    def role_taker_attribute(cls) -> TUnspecializedSubTaker:
        return variable_from(cls).taker
