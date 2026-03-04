from __future__ import annotations

from dataclasses import dataclass, field

from omegaconf import MISSING
from typing_extensions import List, TYPE_CHECKING

from krrood.ripple_down_rules.datastructures.dataclasses import CaseConf

if TYPE_CHECKING:
    pass


@dataclass
class BodyConf:
    name: str = MISSING


@dataclass
class HandleConf(BodyConf): ...


@dataclass
class ContainerConf(BodyConf): ...


@dataclass
class Connection:
    parent: BodyConf = MISSING
    child: BodyConf = MISSING


@dataclass
class FixedConnectionConf(Connection):
    pass


@dataclass
class PrismaticConnectionConf(Connection):
    pass


@dataclass
class WorldConf(CaseConf):
    bodies: List[BodyConf] = field(default_factory=list)
    connections: List[Connection] = field(default_factory=list)
