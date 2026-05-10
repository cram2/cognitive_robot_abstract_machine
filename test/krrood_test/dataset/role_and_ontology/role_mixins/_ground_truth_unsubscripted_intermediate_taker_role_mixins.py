from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass

from typing_extensions import TYPE_CHECKING

if TYPE_CHECKING:
    from test.krrood_test.dataset.role_and_ontology.unsubscripted_intermediate_taker import (
        Box,
        Cargo,
        CargoCrate,
        TBoxItem,
        TRack,
        TRackSlot,
        TShelf,
        TShelfContent,
    )


@dataclass(eq=False)
class DelegatorForBox(ABC):

    @property
    @abstractmethod
    def delegatee(self) -> Box: ...

    @property
    def item(self) -> TBoxItem:
        return self.delegatee.item

    @item.setter
    def item(self, value: TBoxItem):
        self.delegatee.item = value


@dataclass(eq=False)
class DelegatorForCargoCrate(DelegatorForBox, ABC):

    @property
    @abstractmethod
    def delegatee(self) -> CargoCrate: ...

    @property
    def item(self) -> Cargo:
        return self.delegatee.item

    @item.setter
    def item(self, value: Cargo):
        self.delegatee.item = value


@dataclass(eq=False)
class DelegatorForShelf(DelegatorForCargoCrate, ABC):

    @property
    @abstractmethod
    def delegatee(self) -> TShelf: ...

    @property
    def slot(self) -> TShelfContent:
        return self.delegatee.slot

    @slot.setter
    def slot(self, value: TShelfContent):
        self.delegatee.slot = value


@dataclass(eq=False)
class DelegatorForRack(DelegatorForShelf, ABC):

    @property
    @abstractmethod
    def delegatee(self) -> TRack: ...

    @property
    def slot(self) -> TRackSlot:
        return self.delegatee.slot

    @slot.setter
    def slot(self, value: TRackSlot):
        self.delegatee.slot = value
