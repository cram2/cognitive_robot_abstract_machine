from dataclasses import dataclass

from typing_extensions import List, Callable

from krrood.entity_query_language.predicate import Predicate
from semantic_digital_twin.robots.abstract_robot import Manipulator
from semantic_digital_twin.world_description.world_entity import (
    KinematicStructureEntity,
)


@dataclass
class GripperOccupancy:
    manipulator: Manipulator

    def check_man_occupancy(self, condition) -> bool:
        bodies_under_tcp = (
            self.manipulator._world.get_kinematic_structure_entities_of_branch(
                self.manipulator.tool_frame
            )
        )
        if self.manipulator.tool_frame in bodies_under_tcp:
            bodies_under_tcp.remove(self.manipulator.tool_frame)
        return condition(bodies_under_tcp)


@dataclass
class GripperIsFree(GripperOccupancy, Predicate):

    def __call__(self) -> bool:
        return self.check_man_occupancy(lambda bodies: len(bodies) == 0)


@dataclass
class GripperIsNotFree(GripperOccupancy, Predicate):

    def __call__(self) -> bool:
        return self.check_man_occupancy(lambda bodies: len(bodies) != 0)
