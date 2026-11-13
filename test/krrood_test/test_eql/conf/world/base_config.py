from __future__ import annotations

from dataclasses import dataclass, field

from omegaconf import MISSING
from typing_extensions import Callable, Any


@dataclass(kw_only=True)
class BodyConf:
    name: str = MISSING
    size: int = field(default=1)


@dataclass(kw_only=True)
class HandleConf(BodyConf): ...


@dataclass(kw_only=True)
class ContainerConf(BodyConf): ...


@dataclass(kw_only=True)
class ConnectionConf:
    parent: BodyConf = MISSING
    child: BodyConf = MISSING


@dataclass(kw_only=True)
class FixedConnectionConf(ConnectionConf):
    pass


@dataclass(kw_only=True)
class PrismaticConnectionConf(ConnectionConf):
    pass


@dataclass(kw_only=True)
class RevoluteConnectionConf(ConnectionConf):
    pass


@dataclass(kw_only=True)
class ViewConf: ...


@dataclass(kw_only=True)
class DrawerConf(ViewConf):
    container: BodyConf = MISSING
    handle: HandleConf = MISSING


@dataclass(kw_only=True)
class CabinetConf(ViewConf):
    container: ContainerConf = MISSING
    drawers: list[DrawerConf] = field(default_factory=list)


@dataclass(kw_only=True)
class CaseConf:
    factory_method: Callable[[Any], Any] = MISSING

    def create(self) -> Any:
        return self.factory_method()


@dataclass(kw_only=True)
class WorldConf(CaseConf):
    bodies: list[BodyConf] = field(default_factory=list)
    connections: list[ConnectionConf] = field(default_factory=list)
    views: list[ViewConf] = field(default_factory=list)
