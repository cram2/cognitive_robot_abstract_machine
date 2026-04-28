from dataclasses import dataclass

from typing_extensions import List, Callable

from krrood.entity_query_language.predicate import Predicate
from semantic_digital_twin.robots.robot_parts import EndEffector
from semantic_digital_twin.world_description.world_entity import (
    KinematicStructureEntity,
    Body,
)


@dataclass
class GripperOccupancy:
    """
    Base class for predicates that check the gripper occupancy.
    """

    manipulator: EndEffector
    """
    Semantic annotation for the gripper that should be evaluated.
    """

    def check_man_occupancy(self, condition: Callable[List[Body], bool]) -> bool:
        """
        Checks the occupancy of the gripper against a condition.
        The condition get the list of bodies that are under the TCP in the kinematic structure and returns a boolean.

        :param condition: The condition that should be evaluated.
        :return: True if the condition is satisfied, False otherwise.
        """
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
    """
    Checks if the gripper is holding something. Checks this by looking at the kinematic structure of the manipulator.
    """

    def __call__(self) -> bool:
        return self.check_man_occupancy(lambda bodies: len(bodies) == 0)


@dataclass
class GripperIsNotFree(GripperOccupancy, Predicate):
    """
    Checks if the gripper is free at the moment, so it can be used to grab something. This is checked by looking at the
    kinematic structure.
    """

    def __call__(self) -> bool:
        return self.check_man_occupancy(lambda bodies: len(bodies) != 0)
