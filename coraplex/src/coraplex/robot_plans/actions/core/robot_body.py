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
from semantic_digital_twin.robots.robot_parts import Arm

from coraplex.datastructures.trajectory import PoseTrajectory
from coraplex.plans.factories import execute_single
from coraplex.robot_plans.actions.base import ActionDescription, DescriptionType
from coraplex.robot_plans.mixins import (
    ArmGoalParameters,
    EndEffectorPoseParameters,
    GripperActuationParameters,
    MaxJointVelocityParameter,
    TorsoStateParameter,
)
from coraplex.robot_plans.motions.gripper import (
    MoveGripperMotion,
    MoveTCPWaypointsMotion,
)
from coraplex.robot_plans.motions.robot_body import MoveJointsMotion
from coraplex.validation.goal_validator import create_multiple_joint_goal_validator
from semantic_digital_twin.datastructures.definitions import (
    StaticJointState,
)


@dataclass
class MoveTorsoAction(ActionDescription, TorsoStateParameter):
    """
    Move the torso of the robot up and down.
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
class SetGripperAction(ActionDescription, GripperActuationParameters):
    """
    Set the gripper state of the robot.
    """

    @property
    def _action_plan(self) -> PlanNode:
        return execute_single(
            MoveGripperMotion(end_effector=self.end_effector, motion=self.motion)
        )


@dataclass
class ParkArmsAction(ActionDescription, MaxJointVelocityParameter):
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
class FollowToolCenterPointPathAction(ActionDescription, ArmGoalParameters):
    """
    Represents an action to move a robotic arm's TCP (Tool Center Point) along a path of
    poses.
    """

    target_locations: PoseTrajectory
    """
    Path poses for the TCP motion.
    """

    @property
    def _action_plan(self) -> PlanNode:
        target_locations = list(self.target_locations.poses)

        motion = MoveTCPWaypointsMotion(
            target_locations,
            arm=self.arm,
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
class MoveManipulatorAction(ActionDescription, EndEffectorPoseParameters):
    """
    Move the end_effector to a specific pose.
    """

    @property
    def _action_plan(self) -> PlanNode:
        return execute_single(
            MoveManipulatorMotion(
                target_pose=self.target_pose,
                end_effector=self.end_effector,
                allow_gripper_collision=self.allow_gripper_collision,
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
