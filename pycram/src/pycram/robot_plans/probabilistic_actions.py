from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional, Any

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

    plan: Plan
    """
    The plan that should be used for training.
    """

    def generate_episode(self) -> Any:
        with simulated_robot:
            try:
                self.plan.perform()
            except (PlanFailure, StopIteration):
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
                    MoveManipulatorMotion(
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

        if not np.allclose(
            self.manipulator.tool_frame.global_pose.to_np(),
            self.target_pose.to_np(),
            atol=0.1,
        ):
            raise ManipulatorDidNotReachTarget(self.manipulator, self.target_pose)

    @classmethod
    def training_environment(
        cls, robot: AbstractRobot, limit: int = 10
    ) -> TrainingEnvironment:
        world = robot._world
        target_pose = Pose.from_xyz_rpy(
            x=1,
            y=1,
            z=1,
            reference_frame=world.root,
        )

        context = Context(
            robot=robot, world=world, query_backend=ProbabilisticBackend()
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

        plan = execute_single(
            move_to_reach,
            context=context,
        ).plan
        return TrainingEnvironment(plan=plan)
