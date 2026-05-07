from __future__ import annotations

from dataclasses import dataclass, field
from typing import TypeVar

from krrood.entity_query_language.factories import variable_from
from krrood.patterns.role.role import Role


@dataclass
class BaseA:
    base_field: str = ""

    def base_method(self) -> str: ...


@dataclass
class ChildA(BaseA):
    child_field: int = 0

    def child_method(self) -> int: ...


@dataclass
class GrandchildA(ChildA):
    grandchild_field: float = 0.0

    def grandchild_method(self) -> float: ...


TGrandchildA = TypeVar("TGrandchildA", bound=GrandchildA)


@dataclass
class ChainedRole(Role[TGrandchildA]):
    taker: TGrandchildA = field(kw_only=True)

    @classmethod
    def role_taker_attribute(cls) -> TGrandchildA:
        return variable_from(cls).taker
