# ===================== Possible World Configurations ========================
from dataclasses import dataclass, field

from typing_extensions import List, Callable

from .base_config import WorldConf, BodyConf, Connection, FixedConnectionConf, PrismaticConnectionConf, \
    ContainerConf, HandleConf

from ...factories.world.handles_and_containers import create_world_with_handles_and_containers


@dataclass
class Handle1(HandleConf):
    name: str = "Handle1"


@dataclass
class Handle2(HandleConf):
    name: str = "Handle2"


@dataclass
class Container1(ContainerConf):
    name: str = "Container1"


@dataclass
class Container2(ContainerConf):
    name: str = "Container2"


def bodies():
    return [
        Handle1(),
        Handle2(),
        Container1(),
        Container2()
    ]


@dataclass
class HandlesAndContainersWorld(WorldConf):
    bodies: List[BodyConf] = field(default_factory=bodies, init=False)
    connections: List[Connection] = field(default_factory=lambda: [
        FixedConnectionConf(parent=Container1(), child=Handle1()),
        PrismaticConnectionConf(parent=Container2(), child=Container1())], init=False)
    factory_method: Callable = field(default=create_world_with_handles_and_containers, init=False)


