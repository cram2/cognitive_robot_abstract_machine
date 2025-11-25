from __future__ import annotations

from dataclasses import dataclass, field

from omegaconf import MISSING
from typing_extensions import List, Callable, Any


@dataclass
class BodyConf:
    name: str = MISSING
    size: int = field(default=1)


@dataclass
class HandleConf(BodyConf): ...


@dataclass
class ContainerConf(BodyConf): ...


@dataclass
class ConnectionConf:
    parent: BodyConf = MISSING
    child: BodyConf = MISSING


@dataclass
class FixedConnectionConf(ConnectionConf):
    pass


@dataclass
class PrismaticConnectionConf(ConnectionConf):
    pass


@dataclass
class RevoluteConnectionConf(ConnectionConf):
    pass


@dataclass
class ViewConf: ...


@dataclass
class DrawerConf(ViewConf):
    container: BodyConf = MISSING
    handle: HandleConf = MISSING


@dataclass
class CabinetConf(ViewConf):
    container: ContainerConf = MISSING
    drawers: List[DrawerConf] = field(default_factory=list)


@dataclass
class CaseConf:
    factory_method: Callable[[Any], Any] = MISSING

    def create(self) -> Any:
        return self.factory_method()


@dataclass
class WorldConf(CaseConf):
    bodies: List[BodyConf] = field(default_factory=list)
    connections: List[ConnectionConf] = field(default_factory=list)
    views: List[ViewConf] = field(default_factory=list)
