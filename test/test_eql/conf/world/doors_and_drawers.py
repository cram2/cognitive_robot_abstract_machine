# ===================== Possible World Configurations ========================
from dataclasses import dataclass, field

from typing_extensions import List, Callable

from .base_config import (
    WorldConf,
    BodyConf,
    ConnectionConf,
    FixedConnectionConf,
    PrismaticConnectionConf,
    RevoluteConnectionConf,
    HandleConf,
)
from .handles_and_containers import (
    Handle1,
    Handle2,
    Handle3,
    Container1,
    Container2,
    Container3,
)
from ...factories.world import create_world


@dataclass
class Body1(BodyConf):
    name: str = "Body1"


@dataclass
class Body2(BodyConf):
    name: str = "Body2"
    size: int = 2


@dataclass
class Body3(BodyConf):
    name: str = "Body3"


@dataclass
class Body4(BodyConf):
    name: str = "Body4"


@dataclass
class Handle4(HandleConf):
    name: str = "Handle4"


def bodies():
    return [
        Handle1(),  # 0
        Handle2(),  # 1
        Handle3(),  # 2
        Handle4(),  # 3
        Body1(),  # 4
        Body2(),  # 5
        Body3(),  # 6
        Body4(),  # 7
        Container1(),  # 8
        Container2(),  # 9
        Container3(),  # 10
    ]


@dataclass
class DoorsAndDrawersWorld(WorldConf):
    bodies: List[BodyConf] = field(default_factory=bodies, init=False)
    connections: List[ConnectionConf] = field(
        default_factory=lambda: [
            FixedConnectionConf(parent=Container1(), child=Handle1()),
            FixedConnectionConf(parent=Body2(), child=Handle2()),
            FixedConnectionConf(parent=Body4(), child=Handle4()),
            RevoluteConnectionConf(parent=Body3(), child=Handle3()),
            RevoluteConnectionConf(parent=Container2(), child=Body4()),
            PrismaticConnectionConf(parent=Container3(), child=Container1()),
        ],
        init=False,
    )
    factory_method: Callable = field(default=create_world, init=False)
