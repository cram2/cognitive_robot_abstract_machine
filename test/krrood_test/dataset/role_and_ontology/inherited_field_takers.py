from __future__ import annotations

from dataclasses import dataclass, field
from typing import TypeVar

from krrood.entity_query_language.factories import variable_from
from krrood.patterns.role.role import Role


@dataclass
class FieldOrigin:
    """Grandparent: where `shared_field` is first defined."""

    shared_field: str = field(default="", kw_only=True)


@dataclass
class IntermediateMixin(FieldOrigin):
    """Intermediate: inherits `shared_field` without re-annotating it."""

    def intermediate_method(self) -> str: ...


@dataclass
class TakerA(IntermediateMixin):
    """Role taker: inherits `shared_field` transitively through IntermediateMixin."""

    taker_a_field: int = field(default=0, kw_only=True)


@dataclass
class TakerB(IntermediateMixin):
    """Second role taker: same grandparent but independent branch."""

    taker_b_field: float = field(default=0.0, kw_only=True)


TTakerA = TypeVar("TTakerA", bound=TakerA)
TTakerB = TypeVar("TTakerB", bound=TakerB)


@dataclass
class RoleForTakerA(Role[TTakerA]):
    taker: TTakerA = field(kw_only=True)

    @classmethod
    def role_taker_attribute(cls) -> TTakerA:
        return variable_from(cls).taker


@dataclass
class RoleForTakerB(Role[TTakerB]):
    taker: TTakerB = field(kw_only=True)

    @classmethod
    def role_taker_attribute(cls) -> TTakerB:
        return variable_from(cls).taker
