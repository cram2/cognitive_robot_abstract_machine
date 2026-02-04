from __future__ import annotations

from abc import ABC
from dataclasses import dataclass, field
from functools import cached_property
from typing import List, Type, Union

from ..reasoning.predicates import LeftOf, RightOf
from ..robots.abstract_robot import Neck, Arm


@dataclass
class HasNeck(ABC):
    """
    Mixin class for robots that have a neck.
    """

    neck: Neck = field(default=None)
    """
    The neck kinematic chain of the robot, if it exists.
    """

    def add_neck(self, neck: Neck):
        """
        Adds a neck kinematic chain to the robot.

        :param neck: The neck kinematic chain to add.
        """
        if not neck.sensors:
            raise ValueError(
                f"Neck kinematic chain {neck.name} must have at least one sensor."
            )
        if self.neck is not None:
            raise ValueError(f"Robot {self.name} already has a neck: {self.neck.name}.")
        self.neck = neck
        self.add_kinematic_chain(neck)


@dataclass
class HasArms(ABC):
    """
    Mixin class for robots that have arms.
    """

    arms: List[Arm] = field(default_factory=list)
    """
    A collection of arms in the robot.
    """

    def add_arm(self, arm: Arm):
        """
        Adds a kinematic chain to the PR2 robot's collection of kinematic chains.
        If the kinematic chain is an arm, it will be added to the left or right arm accordingly.

        :param arm: The kinematic chain to add to the PR2 robot.
        """
        if arm.manipulator is None:
            raise ValueError(f"Arm kinematic chain {arm.name} must have a manipulator.")
        self.arms.append(arm)
        self.add_kinematic_chain(arm)


@dataclass
class SpecifiesLeftRightArm(HasArms, ABC):
    """
    Mixin class for robots that have two arms and can specify which is the left and which is the right arm.
    """

    @cached_property
    def left_arm(self):
        return self._assign_left_right_arms(LeftOf)

    @cached_property
    def right_arm(self):
        return self._assign_left_right_arms(RightOf)

    def _assign_left_right_arms(self, relation: Type[Union[LeftOf, RightOf]]) -> Arm:
        """
        Assigns the left and right arms based on their position relative to the robot's root body.
        :param relation: The relation to use for determining left or right (LeftOf or RightOf).
        :return: The arm that is on the left or right side of the robot.
        """
        assert (
            len(self.arms) == 2
        ), f"Must have exactly two arms to specify left and right arm, but found {len(self.arms)}."
        pov = self.root.global_pose
        first_arm = self.arms[0]
        second_arm = self.arms[1]
        first_arm_chain = list(first_arm.bodies_with_collisions)
        second_arm_chain = list(second_arm.bodies_with_collisions)

        return (
            first_arm
            if relation(
                first_arm_chain[1],
                second_arm_chain[1],
                pov,
            )()
            else second_arm
        )
