from __future__ import annotations

from dataclasses import dataclass, field
from typing_extensions import TypeVar

from krrood.entity_query_language.factories import variable_from
from krrood.patterns.role.role import Role
from krrood.patterns.subclass_safe_generic import SubClassSafeGeneric


@dataclass(eq=False)
class BaseItem:
    name: str = field(kw_only=True)


TItem = TypeVar("TItem", bound=BaseItem)


@dataclass(eq=False)
class ItemHolder(SubClassSafeGeneric[TItem]):
    """Generic holder that keeps an item of a parameterised type."""

    item: TItem = field(kw_only=True)


TSpecificItem = TypeVar("TSpecificItem", bound=BaseItem)


@dataclass(eq=False)
class SpecificItemTaker(ItemHolder[TSpecificItem]):
    """Role taker whose item TypeVar is narrowed relative to ItemHolder."""

    label: str = field(default="", kw_only=True)


TConcreteItemTaker = TypeVar("TConcreteItemTaker", bound=SpecificItemTaker)


@dataclass(eq=False)
class ItemRole(Role[TConcreteItemTaker]):
    taker: TConcreteItemTaker = field(kw_only=True)

    @classmethod
    def role_taker_attribute(cls) -> TConcreteItemTaker:
        return variable_from(cls).taker
