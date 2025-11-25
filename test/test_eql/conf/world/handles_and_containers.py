# ===================== Possible World Configurations ========================
from dataclasses import dataclass, field

from typing_extensions import List, Callable

from .base_config import (
    WorldConf,
    BodyConf,
    ConnectionConf,
    FixedConnectionConf,
    PrismaticConnectionConf,
    ContainerConf,
    HandleConf,
    DrawerConf,
    CabinetConf,
    ViewConf,
)

from ...factories.world import create_world


@dataclass
class Handle1(HandleConf):
    name: str = "Handle1"


@dataclass
class Handle2(HandleConf):
    name: str = "Handle2"


@dataclass
class Handle3(HandleConf):
    name: str = "Handle3"


@dataclass
class Container1(ContainerConf):
    name: str = "Container1"


@dataclass
class Container2(ContainerConf):
    name: str = "Container2"


@dataclass
class Container3(ContainerConf):
    name: str = "Container3"


@dataclass
class Drawer1(DrawerConf):
    container: ContainerConf = field(default=Container1)
    handle: HandleConf = field(default=Handle1)


@dataclass
class Drawer2(DrawerConf):
    container: ContainerConf = field(default=Container3)
    handle: HandleConf = field(default=Handle3)


@dataclass
class Drawer3(DrawerConf):
    container: ContainerConf = field(default=Container1)
    handle: HandleConf = field(default=Handle2)


@dataclass
class Cabinet2(CabinetConf):
    container: ContainerConf = field(default=Container2)
    drawers: List[DrawerConf] = field(default_factory=lambda: [Drawer1(), Drawer2()])


@dataclass
class Cabinet1(CabinetConf):
    container: ContainerConf = field(default=Container2)
    drawers: List[DrawerConf] = field(default_factory=lambda: [Drawer3()])


def bodies():
    return [Handle1(), Handle2(), Handle3(), Container3(), Container1(), Container2()]


@dataclass
class HandlesAndContainersWorld(WorldConf):
    bodies: List[BodyConf] = field(default_factory=bodies, init=False)
    connections: List[ConnectionConf] = field(
        default_factory=lambda: [
            FixedConnectionConf(parent=Container1(), child=Container2()),
            FixedConnectionConf(parent=Container3(), child=Handle3()),
            PrismaticConnectionConf(parent=Container2(), child=Container1()),
            PrismaticConnectionConf(parent=Container2(), child=Container3()),
            FixedConnectionConf(parent=Container1(), child=Handle1()),
        ],
        init=False,
    )
    views: List[ViewConf] = field(
        default_factory=lambda: [Drawer1(), Drawer2(), Cabinet1(), Cabinet2()],
        init=False,
    )
    factory_method: Callable = field(default=create_world, init=False)
