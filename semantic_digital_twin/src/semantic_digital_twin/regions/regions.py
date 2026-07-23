from dataclasses import dataclass

from semantic_digital_twin.world_description.world_entity import Region


@dataclass(eq=False)
class Level(Region): ...


@dataclass(eq=False)
class GroundFloor(Region):
    pass


@dataclass(eq=False)
class FirstFloor(Region):
    pass


@dataclass(eq=False)
class SecondFloor(Region):
    pass
