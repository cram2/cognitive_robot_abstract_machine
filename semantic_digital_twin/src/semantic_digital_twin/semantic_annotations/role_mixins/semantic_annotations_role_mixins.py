from __future__ import annotations
from abc import ABC, abstractmethod
from dataclasses import dataclass
from semantic_digital_twin.semantic_annotations.role_mixins.mixins_role_mixins import (
    DelegatorForHasCaseAsRootBody,
    DelegatorForHasStorageSpace,
)
from semantic_digital_twin.world_description.role_mixins.world_entity_role_mixins import (
    DelegatorForSemanticAnnotation,
)
from typing_extensions import TYPE_CHECKING

if TYPE_CHECKING:
    from semantic_digital_twin.semantic_annotations.semantic_annotations import (
        Bottle,
        Cabinet,
        Floor,
        Furniture,
        Room,
        TLiquid,
        TinCan,
    )


@dataclass(eq=False)
class DelegatorForFurniture(DelegatorForSemanticAnnotation, ABC):
    @property
    @abstractmethod
    def delegatee(self) -> Furniture: ...


@dataclass(eq=False)
class DelegatorForCabinet(DelegatorForFurniture, DelegatorForHasCaseAsRootBody, ABC):
    @property
    @abstractmethod
    def delegatee(self) -> Cabinet: ...


@dataclass(eq=False)
class DelegatorForRoom(DelegatorForSemanticAnnotation, ABC):
    @property
    @abstractmethod
    def delegatee(self) -> Room: ...
    @property
    def floor(self) -> Floor:
        return self.delegatee.floor

    @floor.setter
    def floor(self, value: Floor):
        self.delegatee.floor = value


@dataclass(eq=False)
class DelegatorForBottle(
    DelegatorForHasCaseAsRootBody, DelegatorForHasStorageSpace, ABC
):
    @property
    @abstractmethod
    def delegatee(self) -> Bottle: ...
    @property
    def objects(self) -> list[TLiquid]:
        return self.delegatee.objects

    @objects.setter
    def objects(self, value: list[TLiquid]):
        self.delegatee.objects = value


@dataclass(eq=False)
class DelegatorForTinCan(DelegatorForHasStorageSpace, ABC):
    @property
    @abstractmethod
    def delegatee(self) -> TinCan: ...
