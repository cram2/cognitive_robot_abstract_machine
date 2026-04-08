from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional, Any

from krrood.entity_query_language.backends import ProbabilisticBackend
from krrood.entity_query_language.factories import (
    underspecified,
    variable_from,
    variable,
)
from pycram.datastructures.dataclasses import Context
from pycram.datastructures.enums import Arms
from pycram.datastructures.grasp import GraspDescription
from pycram.plans.factories import sequential, execute_single
from pycram.plans.plan import Plan
from pycram.robot_plans import MoveToolCenterPointMotion
from pycram.robot_plans.actions.base import ActionDescription
from pycram.robot_plans.actions.core.navigation import NavigateAction
from semantic_digital_twin.robots.abstract_robot import AbstractRobot, Manipulator
from semantic_digital_twin.spatial_types.spatial_types import Pose
from semantic_digital_twin.world import World
from semantic_digital_twin.world_description.world_entity import Body


@dataclass
class TrainingEnvironment:
    """
    Represents a training environment for learning the parametrization of actions.
    """

    plan: Plan
    """
    The plan that should be used for training.
    """

    def generate_episode(self) -> Any:
        self.plan.perform()


@dataclass
class LearnableAction(ActionDescription, ABC):

    @classmethod
    @abstractmethod
    def training_environment(cls, **kwargs) -> TrainingEnvironment: ...


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

    grasp_description: GraspDescription
    """
    The semantic description that should be used for the reaching
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
    def training_environment(cls, robot: AbstractRobot) -> TrainingEnvironment:
        world = World()

        world.merge_world(robot._world)

        target_pose = Pose.from_xyz_rpy(x=0, y=0, z=0)

        context = Context(
            robot=robot, world=world, query_backend=ProbabilisticBackend()
        )

        move_to_reach = underspecified(MoveToReach)(
            target_pose=target_pose,
            standing_pose=underspecified(Pose.from_xyz_rpy)(
                x=..., y=..., z=..., roll=0, pitch=0, yaw=...
            ),
            grasp_description=underspecified(GraspDescription)(
                approach_direction=...,
                vertical_alignment=...,
                manipulator=variable(Manipulator, world.semantic_annotations),
                rotate_gripper=...,
                manipulation_offset=...,
            ),
        )

        move_to_reach.expression.limit(50)

        plan = execute_single(
            move_to_reach,
            context=context,
        ).plan
        return TrainingEnvironment(plan=plan)
