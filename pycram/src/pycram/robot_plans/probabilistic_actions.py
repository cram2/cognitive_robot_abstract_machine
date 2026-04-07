from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional, Any

from krrood.entity_query_language.factories import underspecified
from pycram.datastructures.dataclasses import Context
from pycram.datastructures.enums import Arms
from pycram.datastructures.grasp import GraspDescription
from pycram.plans.factories import sequential, execute_single
from pycram.plans.plan import Plan
from pycram.robot_plans import MoveToolCenterPointMotion
from pycram.robot_plans.actions.base import ActionDescription
from pycram.robot_plans.actions.core.navigation import NavigateAction
from semantic_digital_twin.robots.abstract_robot import AbstractRobot
from semantic_digital_twin.spatial_types.spatial_types import Pose
from semantic_digital_twin.world import World
from semantic_digital_twin.world_description.world_entity import Body


@dataclass
class LearnableAction(ActionDescription, ABC):

    @classmethod
    @abstractmethod
    def minimal_setup(cls) -> Plan:
        """
        Create a plan with
        :return:
        """


@dataclass
class MoveToReach(LearnableAction):
    """
    Let the robot reach a specific pose.
    """

    standing_pose: Pose
    """
    The pose that the robot should stand at.
    """

    target_pose: Pose
    """
    Pose that should be reached.
    """

    arm: Arms
    """
    The arm that should be used for the reaching
    """

    def execute(self):
        self.add_subplan(
            sequential(
                [
                    NavigateAction(self.standing_pose),
                    MoveToolCenterPointMotion(
                        self.target_pose, self.arm, allow_gripper_collision=False
                    ),
                ]
            )
        ).perform()

    @classmethod
    def minimal_setup(cls, robot: AbstractRobot) -> Plan:
        world = World()
        target_pose = Pose.from_xyz_rpy(x=0, y=0, z=0)

        context = Context(robot=robot, world=world)

        plan = execute_single(
            underspecified(MoveToReach)(
                target_pose=target_pose,
                standing_pose=underspecified(Pose.from_xyz_rpy)(x=..., y=..., z=...),
                arm=...,
            ),
            context=context,
        ).plan
        return plan
