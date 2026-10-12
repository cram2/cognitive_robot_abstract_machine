from __future__ import annotations

from dataclasses import dataclass
from datetime import timedelta
from typing import Tuple, List

from typing_extensions import Optional, Dict, Any

from coraplex.plans.plan_node import PlanNode
from krrood.entity_query_language.core.base_expressions import SymbolicExpression
from krrood.entity_query_language.core.variable import Variable
from coraplex.datastructures.dataclasses import Context
from coraplex.robot_plans import MoveManipulatorMotion
from krrood.entity_query_language.factories import variable_from
from semantic_digital_twin.reasoning.predicates import allclose
from semantic_digital_twin.robots.robot_parts import Arm, EndEffector
from semantic_digital_twin.spatial_types.spatial_types import Pose

from coraplex.datastructures.trajectory import PoseTrajectory
from coraplex.plans.factories import execute_single
from coraplex.robot_plans.actions.base import ActionDescription, DescriptionType
from coraplex.robot_plans.mixins import HasMaxJointVelocity, HasTcpGoalThresholds
from coraplex.robot_plans.motions.gripper import (
    MoveGripperMotion,
    MoveTCPWaypointsMotion,
)
from coraplex.robot_plans.motions.robot_body import MoveJointsMotion
from coraplex.validation.goal_validator import create_multiple_joint_goal_validator
from semantic_digital_twin.datastructures.definitions import (
    TorsoState,
    GripperState,
    StaticJointState,
)


@dataclass
class MoveTorsoAction(ActionDescription):
    """
    Move the torso of the robot up and down.
    """

    torso_state: TorsoState
    """
    The state of the torso that should be set
    """

    @property
    def _action_plan(self) -> PlanNode:
        joint_state = self.robot.get_torso().get_joint_state_by_type(self.torso_state)
        return execute_single(
            MoveJointsMotion(
                [c.name.name for c in joint_state.connections],
                joint_state.target_values,
            ),
        )

    @staticmethod
    def post_condition(
        variables: Dict[str, Variable], context: Context, kwargs: Dict[str, Any]
    ) -> SymbolicExpression | bool:
        """
        The target joint state for the torso needs to be achieved.
        """
        joint_state = context.robot.get_torso().get_joint_state_by_type(
            kwargs["torso_state"]
        )
        return variable_from(joint_state).is_achieved()


@dataclass
class SetGripperAction(ActionDescription):
    """
    Set the gripper state of the robot.
    """

    gripper: EndEffector
    """
    The gripper that should be set.
    """

    motion: GripperState
    """
    The motion that should be set on the gripper.
    """

    @property
    def _action_plan(self) -> PlanNode:
        return execute_single(
            MoveGripperMotion(gripper=self.gripper, motion=self.motion)
        )


@dataclass
class ParkArmsAction(ActionDescription, HasMaxJointVelocity):
    """
    Park the arms of the robot.
    """

    arms: List[Arm]
    """
    The arms that should be parked.
    """

    @property
    def _action_plan(self) -> PlanNode:
        joint_names, joint_poses = self.get_joint_poses()

        return execute_single(
            MoveJointsMotion(
                names=joint_names,
                positions=joint_poses,
                max_joint_velocity=self.max_joint_velocity,
            )
        )

    def get_joint_poses(self) -> Tuple[List[str], List[float]]:
        """
        :return: The joint positions that should be set for the arm to be in the park position.
        """
        names = []
        values = []
        for arm in self.arms:
            joint_state = arm.get_joint_state_by_type(StaticJointState.PARK)
            names.extend([c.name.name for c in joint_state.connections])
            values.extend(joint_state.target_values)
        return names, values


@dataclass
class FollowToolCenterPointPathAction(ActionDescription, HasTcpGoalThresholds):
    """
    Represents an action to move a robotic arm's TCP (Tool Center Point) along a path of
    poses.
    """

    target_locations: PoseTrajectory
    """
    Path poses for the TCP motion.
    """

    arm: Arm
    """
    The arm to use.
    """

    @property
    def _action_plan(self) -> PlanNode:
        target_locations = list(self.target_locations.poses)

        motion = MoveTCPWaypointsMotion(
            target_locations,
            self.arm,
            allow_gripper_collision=True,
            position_threshold=self.position_threshold,
            orientation_threshold=self.orientation_threshold,
        )

        return execute_single(motion)

    def validate(
        self,
        result: Optional[Any] = None,
        max_wait_time: timedelta = timedelta(seconds=2),
    ):
        pass


@dataclass
class MoveManipulatorAction(ActionDescription, HasTcpGoalThresholds):
    """
    Move the end_effector to a specific pose.
    """

    target_pose: Pose
    """
    The pose where the end_effector should be moved to.
    """

    end_effector: EndEffector
    """
    The end_effector that should be moved.
    """

    allow_gripper_collision: bool
    """
    If the gripper can collide with something.
    """

    @property
    def _action_plan(self) -> PlanNode:
        return execute_single(
            MoveManipulatorMotion(
                self.target_pose,
                self.end_effector,
                self.allow_gripper_collision,
                position_threshold=self.position_threshold,
                orientation_threshold=self.orientation_threshold,
            )
        )

    @staticmethod
    def post_condition(
        variables: Dict[str, Variable], context: Context, kwargs: Dict[str, Any]
    ) -> SymbolicExpression:
        end_effector = variables["end_effector"]
        target_pose = variables["target_pose"]
        return allclose(
            end_effector.tool_frame.global_pose.to_np(),
            target_pose.to_np(),
            atol=0.1,
        )
