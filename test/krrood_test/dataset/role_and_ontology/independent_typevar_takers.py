from __future__ import annotations

from dataclasses import dataclass, field
from typing_extensions import TypeVar

from krrood.entity_query_language.factories import variable_from
from krrood.patterns.role.role import Role
from krrood.patterns.subclass_safe_generic import SubClassSafeGeneric


@dataclass(eq=False)
class BaseRoot:
    name: str = field(kw_only=True)


@dataclass(eq=False)
class SpecificRoot(BaseRoot):
    code: int = field(default=0, kw_only=True)


TRoot = TypeVar("TRoot", bound=BaseRoot)


@dataclass(eq=False)
class RootHolder(SubClassSafeGeneric[TRoot]):
    """Defines a generic root field."""

    root: TRoot = field(kw_only=True)


TSpecificRoot = TypeVar("TSpecificRoot", bound=SpecificRoot)


@dataclass(eq=False)
class NarrowedRootHolder(RootHolder[TSpecificRoot]):
    """Narrows TRoot to TSpecificRoot via SubClassSafeGeneric."""


@dataclass(eq=False)
class BaseContent:
    value: str = field(kw_only=True)


TContent = TypeVar("TContent", bound=BaseContent)


@dataclass(eq=False)
class ContentHolder(NarrowedRootHolder, SubClassSafeGeneric[TContent]):
    """Introduces an independent TypeVar TContent for content.

    The root field is inherited from NarrowedRootHolder (type TSpecificRoot)
    and must remain independent of TContent.
    """

    content: TContent = field(kw_only=True)


TContent2 = TypeVar("TContent2", bound=BaseContent)


@dataclass(eq=False)
class MultiTaker(ContentHolder[TContent2]):
    """Concrete role taker: content narrows to TContent2, root must remain TSpecificRoot."""


TMultiTaker = TypeVar("TMultiTaker", bound=MultiTaker)


@dataclass(eq=False)
class MultiTakerRole(Role[TMultiTaker]):
    taker: TMultiTaker = field(kw_only=True)

    @classmethod
    def role_taker_attribute(cls) -> TMultiTaker:
        return variable_from(cls).taker
