from dataclasses import dataclass

from typing_extensions import Optional, Tuple

from pycram.datastructures.enums import Arms
from semantic_digital_twin.robots.abstract_robot import (
    AbstractRobot,
    Manipulator,
    KinematicChain,
    Neck,
)


@dataclass
class ViewManager:

    @staticmethod
    def get_end_effector_view(
        arm: Arms, robot_view: AbstractRobot
    ) -> Optional[Manipulator]:
        arm = ViewManager.get_arm_view(arm, robot_view)
        return arm.manipulator

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
            return (robot_view.manipulator_chains[0],)
        elif arm == Arms.LEFT:
            return (robot_view.left_arm,)
        elif arm == Arms.RIGHT:
            return (robot_view.right_arm,)
        elif arm == Arms.BOTH:
            return robot_view.arms
        return None

    @staticmethod
    def get_neck_view(robot_view: AbstractRobot) -> Optional[Neck]:
        """
        Get the neck view for a given robot view.

        :param robot_view: The robot view to search in.
        :return: The Neck object representing the neck.
        """
        if getattr(robot_view, "neck", Neck):
            return robot_view.neck
        else:
            raise ValueError(f"The robot view {robot_view} has no neck.")
