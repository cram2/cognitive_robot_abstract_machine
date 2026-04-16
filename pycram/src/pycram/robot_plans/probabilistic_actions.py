from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Optional, Any, Callable

import numpy as np

from krrood.entity_query_language.backends import ProbabilisticBackend
from krrood.entity_query_language.factories import underspecified, variable
from pycram.datastructures.dataclasses import Context
from pycram.datastructures.grasp import GraspDescription
from pycram.motion_executor import simulated_robot
from pycram.plans.factories import sequential, execute_single
from pycram.plans.failures import (
    NavigationGoalNotReachedError,
    ManipulatorDidNotReachTarget,
    PlanFailure,
)
from pycram.plans.plan import Plan
from pycram.plans.plan_node import UnderspecifiedNode
from pycram.robot_plans import MoveManipulatorMotion
from pycram.robot_plans.actions.base import ActionDescription
from pycram.robot_plans.actions.core.navigation import NavigateAction
from semantic_digital_twin.robots.abstract_robot import (
    Manipulator,
    AbstractRobot,
)
from semantic_digital_twin.spatial_types.spatial_types import Pose


@dataclass
class TrainingEnvironment:
    """
    Represents a training environment for learning the parametrization of actions.
    """

    root_node_generator: Callable[[], UnderspecifiedNode] = None

    executed_plans: list[Plan] = field(default_factory=list)

    def generate_episode(self) -> Any:
        limit = self.plan.root.underspecified_action.expression._limit_

        with simulated_robot:
            try:
                self.plan.perform()
            except PlanFailure:
                pass


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

    manipulator: Manipulator
    """
    The Manipulator to move to the target pose
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
                    MoveManipulatorAction(
                        self.target_pose,
                        self.manipulator,
                        allow_gripper_collision=False,
                    ),
                ]
            )
        ).perform()

    def validate_postcondition(self, result: Optional[Any] = None):

        root_T_robot_base = self.world.transform(self.standing_pose, self.world.root)

        if not np.allclose(
            root_T_robot_base, self.robot.base.root.global_pose.to_np(), atol=0.1
        ):
            raise NavigationGoalNotReachedError(
                goal_pose=root_T_robot_base,
                current_pose=self.robot.base.root.global_pose,
            )

    @classmethod
    def training_environment(
        cls, robot: AbstractRobot, limit: int = 10
    ) -> TrainingEnvironment:

        return TrainingEnvironment(cls.generate_root_for_training)

    @classmethod
    def generate_root_for_training(
        cls, robot: AbstractRobot, limit: int = 10
    ) -> UnderspecifiedNode:
        world = robot._world
        target_pose = Pose.from_xyz_rpy(
            x=1,
            y=1,
            z=1,
            reference_frame=world.root,
        )

        move_to_reach = underspecified(MoveToReach)(
            target_pose=target_pose,
            standing_pose=underspecified(Pose.from_xyz_rpy)(
                x=...,
                y=...,
                z=float(robot.root.global_pose.z),
                roll=0.0,
                pitch=0.0,
                yaw=...,
                reference_frame=target_pose.reference_frame,
            ),
            manipulator=variable(Manipulator, world.semantic_annotations),
            grasp_description=underspecified(GraspDescription)(
                approach_direction=...,
                vertical_alignment=...,
                manipulator=variable(Manipulator, world.semantic_annotations),
                rotate_gripper=...,
                manipulation_offset=...,
            ),
        )

        move_to_reach.expression.limit(limit)

        return execute_single(move_to_reach)


@dataclass
class MoveManipulatorAction(ActionDescription):
    target_pose: Pose
    manipulator: Manipulator
    allow_gripper_collision: bool

    def execute(self):
        self.add_subplan(
            execute_single(
                MoveManipulatorMotion(
                    self.target_pose,
                    self.manipulator,
                    self.allow_gripper_collision,
                )
            )
        ).perform()

    def validate_postcondition(self, result: Optional[Any] = None):

        if not np.allclose(
            self.manipulator.tool_frame.global_pose.to_np(),
            self.target_pose.to_np(),
            atol=0.1,
        ):
            raise ManipulatorDidNotReachTarget(self.manipulator, self.target_pose)
