from dataclasses import dataclass

from typing_extensions import Optional, Tuple

from krrood.entity_query_language.predicate import symbolic_function
from pycram.datastructures.enums import Arms
from semantic_digital_twin.robots.robot_parts import (
    EndEffector,
    KinematicChain,
    AbstractRobot,
)


@dataclass
class ViewManager:

    @staticmethod
    @symbolic_function
    def get_end_effector_view(
        arm: Arms, robot_view: AbstractRobot
    ) -> Optional[EndEffector]:
        arm = ViewManager.get_arm_view(arm, robot_view)
        return arm.end_effector

    @staticmethod
    def get_arm_view(arm: Arms, robot_view: AbstractRobot) -> Optional[KinematicChain]:
        """
        Get the arm view for a given arm and robot view.

        :param arm: The arm to get the view for.
        :param robot_view: The robot view to search in.
        :return: The Manipulator object representing the arm.
        """
        all_arms = ViewManager.get_all_arm_views(arm, robot_view)
        return all_arms[0]

    @staticmethod
    def get_all_arm_views(
        arm: Arms, robot_view: AbstractRobot
    ) -> Optional[Tuple[KinematicChain]]:
        """
        Get all possible arm views for a given arm and robot view.

        :param arm: The arm to get the view for.
        :param robot_view: The robot view to search in.
        :return: The Manipulator object representing the arm.
        """
        if len(robot_view.arms) == 1:
            return (robot_view.arms[0],)
        elif arm == Arms.LEFT:
            return (robot_view.left_arm,)
        elif arm == Arms.RIGHT:
            return (robot_view.right_arm,)
        elif arm == Arms.BOTH:
            return robot_view.arms
        return None
