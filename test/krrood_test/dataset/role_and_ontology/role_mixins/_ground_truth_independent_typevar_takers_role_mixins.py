from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass

from typing_extensions import TYPE_CHECKING

if TYPE_CHECKING:
    from test.krrood_test.dataset.role_and_ontology.independent_typevar_takers import (
        ContentHolder,
        NarrowedRootHolder,
        RootHolder,
        TContent,
        TContent2,
        TMultiTaker,
        TRoot,
        TSpecificRoot,
    )


@dataclass(eq=False)
class RoleForRootHolder(ABC):

    @property
    @abstractmethod
    def role_taker(self) -> RootHolder: ...

    @property
    def root(self) -> TRoot:
        return self.role_taker.root

    @root.setter
    def root(self, value: TRoot):
        self.role_taker.root = value


@dataclass(eq=False)
class RoleForNarrowedRootHolder(RoleForRootHolder, ABC):

    @property
    @abstractmethod
    def role_taker(self) -> NarrowedRootHolder: ...

    @property
    def root(self) -> TSpecificRoot:
        return self.role_taker.root

    @root.setter
    def root(self, value: TSpecificRoot):
        self.role_taker.root = value


@dataclass(eq=False)
class RoleForContentHolder(RoleForNarrowedRootHolder, ABC):

    @property
    @abstractmethod
    def role_taker(self) -> ContentHolder: ...

    @property
    def content(self) -> TContent:
        return self.role_taker.content

    @content.setter
    def content(self, value: TContent):
        self.role_taker.content = value


@dataclass(eq=False)
class RoleForMultiTaker(RoleForContentHolder, ABC):

    @property
    @abstractmethod
    def role_taker(self) -> TMultiTaker: ...

    @property
    def content(self) -> TContent2:
        return self.role_taker.content

    @content.setter
    def content(self, value: TContent2):
        self.role_taker.content = value
