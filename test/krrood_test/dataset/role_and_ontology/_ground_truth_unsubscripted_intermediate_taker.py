from __future__ import annotations

from dataclasses import dataclass, field
from typing_extensions import TypeVar

from krrood.entity_query_language.factories import variable_from
from krrood.patterns.role.role import Role
from krrood.patterns.subclass_safe_generic import SubClassSafeGeneric
from krrood.patterns.role import HasRoles
from .role_mixins.unsubscripted_intermediate_taker_role_mixins import (
    RoleForRack,
    RoleForShelf,
)


@dataclass(eq=False)
class Cargo:
    label: str = field(kw_only=True)


TBoxItem = TypeVar("TBoxItem", bound=Cargo)


@dataclass(eq=False)
class Box(SubClassSafeGeneric[TBoxItem]):
    """Generic box that holds items of a parameterised type."""

    item: TBoxItem = field(kw_only=True)


@dataclass(eq=False)
class CargoCrate(Box[Cargo]):
    """Concrete box for Cargo — narrows TBoxItem to Cargo.

    Not a role taker. Shelf inherits this WITHOUT a subscript, creating the
    unsubscripted-intermediate case that ``get_generic_type_param`` misses.
    """


TShelfContent = TypeVar("TShelfContent", bound=Cargo)


@dataclass(eq=False)
class Shelf(CargoCrate, SubClassSafeGeneric[TShelfContent], HasRoles):
    """First role taker — inherits item from CargoCrate (unsubscripted) and adds slot."""

    slot: TShelfContent = field(kw_only=True)


TShelf = TypeVar("TShelf", bound=Shelf)


@dataclass(eq=False)
class ShelfRole(Role[TShelf], RoleForShelf):
    taker: TShelf = field(kw_only=True)

    @classmethod
    def role_taker_attribute(cls) -> TShelf:
        return variable_from(cls).taker


TRackSlot = TypeVar("TRackSlot", bound=Cargo)


@dataclass(eq=False)
class Rack(Shelf[TRackSlot]):
    """Second role taker — narrows slot TypeVar to TRackSlot."""


TRack = TypeVar("TRack", bound=Rack)


@dataclass(eq=False)
class RackRole(Role[TRack], RoleForRack):
    taker: TRack = field(kw_only=True)

    @classmethod
    def role_taker_attribute(cls) -> TRack:
        return variable_from(cls).taker
