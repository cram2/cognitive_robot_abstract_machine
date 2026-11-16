from __future__ import annotations

from dataclasses import dataclass, field

from omegaconf import MISSING
from typing_extensions import TYPE_CHECKING

from krrood.ripple_down_rules.datastructures.dataclasses import CaseConf

if TYPE_CHECKING:
    pass


@dataclass(kw_only=True)
class BodyConf:
    name: str = MISSING


@dataclass(kw_only=True)
class HandleConf(BodyConf): ...


@dataclass(kw_only=True)
class ContainerConf(BodyConf): ...


@dataclass(kw_only=True)
class Connection:
    parent: BodyConf = MISSING
    child: BodyConf = MISSING


@dataclass(kw_only=True)
class FixedConnectionConf(Connection):
    pass


@dataclass(kw_only=True)
class PrismaticConnectionConf(Connection):
    pass


@dataclass(kw_only=True)
class WorldConf(CaseConf):
    bodies: list[BodyConf] = field(default_factory=list)
    connections: list[Connection] = field(default_factory=list)
