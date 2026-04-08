from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from functools import cached_property, wraps
from typing import List, Type, Union, TYPE_CHECKING

from semantic_digital_twin.reasoning.predicates import LeftOf, RightOf
from semantic_digital_twin.world_description.world_modification import (
    synchronized_attribute_modification,
)

if TYPE_CHECKING:
    from semantic_digital_twin.robots.abstract_robot import (
        Arm,
        Torso,
        MobileBase,
    )


@dataclass(eq=False)
class HasRobotPart(ABC):

    @abstractmethod
    def _setup_robot_parts(self): ...


@dataclass(eq=False)
class HasArms(HasRobotPart, ABC):
    """
    Mixin class for robots that have arms.
    """

    arms: List[Arm] = field(default_factory=list)
    """
    A collection of arms in the robot.
    """

    @synchronized_attribute_modification
    def add_arm(self, arm: Arm):
        """
        Adds a kinematic chain to the PR2 robot's collection of kinematic chains.
        If the kinematic chain is an arm, it will be added to the left or right arm accordingly.

        :param arm: The kinematic chain to add to the PR2 robot.
        """
        self.arms.append(arm)

    def _setup_robot_parts(self):
        super()._setup_robot_parts()
        self._setup_arm_semantic_annotations()
        self._setup_arm_hardware_interfaces()
        self._setup_arm_joint_state()

    @abstractmethod
    def _setup_arm_semantic_annotations(self): ...

    @abstractmethod
    def _setup_arm_hardware_interfaces(self): ...

    @abstractmethod
    def _setup_arm_joint_state(self): ...


@dataclass(eq=False)
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
        pov = self.root.global_transform
        first_arm = self.arms[0]
        second_arm = self.arms[1]
        # the arms may share a root, but the first body after the root should be different
        world_P_first_body = first_arm.bodies[1].global_transform.to_position()
        world_P_second_body = second_arm.bodies[1].global_transform.to_position()

        return (
            first_arm
            if relation(
                world_P_first_body,
                world_P_second_body,
                pov,
            )()
            else second_arm
        )


@dataclass(eq=False)
class HasTorso(HasRobotPart, ABC):
    """
    Mixin class for robots that have a torso.
    """

    torso: Torso = field(init=False, default=None, repr=False)
    """
    The torso of the robot, represented as an arm.
    """

    @synchronized_attribute_modification
    def add_torso(self, torso: Torso):
        self.torso = torso

    def _setup_robot_parts(self):
        super()._setup_robot_parts()
        self._setup_torso_semantic_annotations()
        self._setup_torso_hardware_interfaces()
        self._setup_torso_joint_state()

    @abstractmethod
    def _setup_torso_semantic_annotations(self): ...

    @abstractmethod
    def _setup_torso_hardware_interfaces(self): ...

    @abstractmethod
    def _setup_torso_joint_state(self): ...


@dataclass(eq=False)
class HasMobileBase(HasRobotPart, ABC):

    mobile_base: MobileBase = field(init=False, default=None, repr=False)
    full_body_controlled: bool = field(default=False, kw_only=True)

    @synchronized_attribute_modification
    def add_mobile_base(self, mobile_base: MobileBase):
        self.mobile_base = mobile_base

    def _setup_robot_parts(self):
        super()._setup_robot_parts()
        self._setup_base_semantic_annotations()

    @abstractmethod
    def _setup_base_semantic_annotations(self): ...
