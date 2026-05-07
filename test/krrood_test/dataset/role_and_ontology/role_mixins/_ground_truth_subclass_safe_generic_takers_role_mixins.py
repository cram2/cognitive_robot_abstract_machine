from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass

from typing_extensions import TYPE_CHECKING

if TYPE_CHECKING:
    from test.krrood_test.dataset.role_and_ontology.subclass_safe_generic_takers import (
        ItemHolder,
        SpecificItemTaker,
        TItem,
        TSpecificItem,
    )


@dataclass(eq=False)
class RoleForItemHolder(ABC):

    @property
    @abstractmethod
    def role_taker(self) -> ItemHolder: ...

    @property
    def item(self) -> TItem:
        return self.role_taker.item

    @item.setter
    def item(self, value: TItem):
        self.role_taker.item = value


@dataclass(eq=False)
class RoleForSpecificItemTaker(RoleForItemHolder, ABC):

    @property
    @abstractmethod
    def role_taker(self) -> SpecificItemTaker: ...

    @property
    def item(self) -> TSpecificItem:
        return self.role_taker.item

    @item.setter
    def item(self, value: TSpecificItem):
        self.role_taker.item = value

    @property
    def label(self) -> str:
        return self.role_taker.label

    @label.setter
    def label(self, value: str):
        self.role_taker.label = value
