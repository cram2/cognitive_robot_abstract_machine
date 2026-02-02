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

        for man in robot_view.manipulators:
            if "left" in man.name.name and arm == Arms.LEFT:
                return man
            elif "right" in man.name.name and arm == Arms.RIGHT:
                return man
        return None

    @staticmethod
    def get_arm_view(
        arm: Arms, robot_view: AbstractRobot
    ) -> Optional[Tuple[KinematicChain]]:
        """
        Get the arm view for a given arm and robot view.

        :param arm: The arm to get the view for.
        :param robot_view: The robot view to search in.
        :return: The Manipulator object representing the arm.
        """
        for arm_chain in robot_view.manipulator_chains:
            if "left" in arm_chain.name.name and arm == Arms.LEFT:
                return arm_chain
            elif "right" in arm_chain.name.name and arm == Arms.RIGHT:
                return arm_chain
            elif arm == Arms.BOTH:
                return robot_view.left_arm, robot_view.right_arm
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
