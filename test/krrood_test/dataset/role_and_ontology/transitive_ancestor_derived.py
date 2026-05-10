from __future__ import annotations

from dataclasses import dataclass, field
from typing import TypeVar

from krrood.entity_query_language.factories import variable_from
from krrood.patterns.role.role import Role

from .transitive_ancestor_base import AncestorBase


@dataclass
class DerivedClass(AncestorBase):
    def derived_only_method(self) -> str: ...


TDerivedClass = TypeVar("TDerivedClass", bound=DerivedClass)


@dataclass
class DerivedRole(Role[TDerivedClass]):
    taker: TDerivedClass = field(kw_only=True)

    @classmethod
    def role_taker_attribute(cls) -> TDerivedClass:
        return variable_from(cls).taker

