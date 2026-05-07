from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass

from typing_extensions import TYPE_CHECKING

if TYPE_CHECKING:
    from test.krrood_test.dataset.role_and_ontology.generic_typevar_takers import (
        ConcreteEntity,
        GenericBaseMixin,
        TBase,
        TConcreteEntity,
        TConcreteTypeTaker,
        TNarrowedTypeVarTaker,
        TUnspecializedSubTaker,
    )


@dataclass(eq=False)
class RoleForGenericBaseMixin(ABC):

    @property
    @abstractmethod
    def role_taker(self) -> GenericBaseMixin: ...

    @property
    def entity(self) -> TBase:
        return self.role_taker.entity

    @entity.setter
    def entity(self, value: TBase):
        self.role_taker.entity = value

    @property
    def count(self) -> int:
        return self.role_taker.count

    @count.setter
    def count(self, value: int):
        self.role_taker.count = value


@dataclass(eq=False)
class RoleForNarrowedTypeVarTaker(RoleForGenericBaseMixin, ABC):

    @property
    @abstractmethod
    def role_taker(self) -> TNarrowedTypeVarTaker: ...

    @property
    def entity(self) -> TConcreteEntity:
        return self.role_taker.entity

    @entity.setter
    def entity(self, value: TConcreteEntity):
        self.role_taker.entity = value

    @property
    def label(self) -> str:
        return self.role_taker.label

    @label.setter
    def label(self, value: str):
        self.role_taker.label = value


@dataclass(eq=False)
class RoleForConcreteTypeTaker(RoleForGenericBaseMixin, ABC):

    @property
    @abstractmethod
    def role_taker(self) -> TConcreteTypeTaker: ...

    @property
    def entity(self) -> ConcreteEntity:
        return self.role_taker.entity

    @entity.setter
    def entity(self, value: ConcreteEntity):
        self.role_taker.entity = value

    @property
    def name(self) -> str:
        return self.role_taker.name

    @name.setter
    def name(self, value: str):
        self.role_taker.name = value


@dataclass(eq=False)
class RoleForUnspecializedSubTaker(RoleForGenericBaseMixin, ABC):

    @property
    @abstractmethod
    def role_taker(self) -> TUnspecializedSubTaker: ...

    @property
    def tag(self) -> str:
        return self.role_taker.tag

    @tag.setter
    def tag(self, value: str):
        self.role_taker.tag = value
