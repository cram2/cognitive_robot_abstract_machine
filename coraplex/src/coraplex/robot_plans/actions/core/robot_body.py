from __future__ import annotations

from dataclasses import dataclass
from datetime import timedelta

from typing_extensions import Any

from semantic_digital_twin.collision_checking.collision_matrix import CollisionRule
from semantic_digital_twin.collision_checking.collision_rules import (
    AllowCollisionForEndEffector,
)
from semantic_digital_twin.robots.robot_parts import Arm
from krrood_spatial_math.spatial_types import Pose

from coraplex.datastructures.trajectory import PoseTrajectory
from coraplex.robot_plans.actions.base import (
    GoalAction,
    ParallelAction,
    SequentialAction,
)
from coraplex.robot_plans.actions.core.exceptions import WidthBeyondGripperOpening
from coraplex.robot_plans.mixins import (
    HasArmParameter,
    HasGripperCollisionParameter,
    HasGripperConfigurationParameter,
    HasTargetPoseParameter,
    HasTorsoStateParameter,
    MovesToolCenterPoint,
    OperatesGripper,
)
from cramph.composites import Sequence
from cramph.node import StatechartNode
from giskardpy.motion_statechart.binding_policy import GoalBindingPolicy
from giskardpy.motion_statechart.mixins import (
    HasJointVelocityLimitParameter,
    HasThresholdParameter,
    HasTwoFingerGripperParameters,
)
from giskardpy.motion_statechart.tasks.cartesian_tasks import (
    CartesianPose,
)
from giskardpy.motion_statechart.tasks.joint_tasks import (
    JointPositionList,
    JointVelocityLimit,
)
from semantic_digital_twin.datastructures.joint_state import JointState
from giskardpy.motion_statechart.goals.gripper import MoveGripper
from semantic_digital_twin.datastructures.definitions import StaticJointState


@dataclass(eq=False, repr=False, kw_only=True)
class MoveTorsoAction(SequentialAction, HasTorsoStateParameter):
    """
    Move the torso of the robot up and down.
    """

    def create_steps(self) -> list[StatechartNode]:
        joint_state = self.robot.get_torso().get_joint_state_by_type(self.torso_state)
        return [JointPositionList(goal=joint_state)]


@dataclass(eq=False, repr=False, kw_only=True)
class SetGripperAction(OperatesGripper, GoalAction, HasGripperConfigurationParameter):
    """
    Moves a gripper to the given configuration.
    """

    def create_goal(self) -> StatechartNode:
        return self.gripper_goal_type.moving_gripper_to(self.configuration)


@dataclass(eq=False, repr=False, kw_only=True)
class MoveFingersToWidthAction(
    GoalAction, HasTwoFingerGripperParameters, HasThresholdParameter
):
    """
    Closes a two-finger gripper until its fingertips stand a given distance apart,
    rather than driving it to a gripper state of its own.

    Closing a gripper all the way onto an object that is not centred between the
    fingers wedges it sideways instead of holding it, so an object held by contact alone
    is better met at its own width. Giskard is given the distance between the fingertips
    as the goal and finds the joint positions that realise it.

    .. note:: This only means anything where something pushes back, see
        :class:`~coraplex.robot_plans.plan_transformations.CloseFingersToObjectWidth`.
    """

    squeeze: float = 0.0
    """
    How far past that width each fingertip is sent, in metres, so the fingers press into
    an object rather than merely touching it. The object stops them short of that, so
    fingers squeezing it count as done once they stopped approaching it.

    Keep it small, since it is commanded penetration into a body that does not yield.
    """

    threshold: float = 0.0005
    """
    How close, in metres, the fingertips have to come to the goal for the action to
    count as done: well under a squeeze, so the squeeze is not lost to the tolerance.
    """

    def create_goal(self) -> StatechartNode:
        """
        :raises WidthBeyondGripperOpening: If :attr:`width` is wider than the gripper
            opens.
        """
        if self.width > self.gripper.separation_when_open:
            raise WidthBeyondGripperOpening(gripper=self.gripper, width=self.width)
        return MoveGripper(
            gripper=self.gripper,
            width=max(0.0, self.width - 2 * self.squeeze),
            threshold=self.threshold,
            tolerate_finger_stall=self.squeeze > 0.0,
            name="MoveFingersToWidth",
        )


@dataclass(eq=False, repr=False, kw_only=True)
class ParkArmsAction(ParallelAction, HasJointVelocityLimitParameter):
    """
    Park the arms of the robot.
    """

    arms: list[Arm]
    """
    The arms that should be parked.
    """

    def create_steps(self) -> list[StatechartNode]:
        park_state = self.park_joint_state()
        joint_goal = JointPositionList(goal=park_state)
        if self.max_joint_velocity is None:
            return [joint_goal]
        return [
            joint_goal,
            JointVelocityLimit(
                connections=list(park_state.connections),
                max_joint_velocity=self.max_joint_velocity,
            ),
        ]

    def park_joint_state(self) -> JointState:
        """
        :return: The joint state that puts every arm this action parks into its park
            position.
        """
        connections = []
        target_values = []
        for arm in self.arms:
            joint_state = arm.get_joint_state_by_type(StaticJointState.PARK)
            connections.extend(joint_state.connections)
            target_values.extend(joint_state.target_values)
        return JointState(connections=connections, target_values=target_values)


@dataclass(eq=False, repr=False, kw_only=True)
class FollowToolCenterPointPathAction(
    MovesToolCenterPoint, ParallelAction, HasArmParameter
):
    """
    Represents an action to move a robotic arm's TCP (Tool Center Point) along a path of
    poses.
    """

    target_locations: PoseTrajectory
    """
    Path poses for the TCP motion.
    """

    position_threshold: float = CartesianPose.position_threshold
    """
    Distance threshold in meters within which the tool center point passes a waypoint.
    """

    orientation_threshold: float = CartesianPose.orientation_threshold
    """
    Rotation threshold in rad within which the tool center point passes a waypoint.
    """

    def create_steps(self) -> list[StatechartNode]:
        waypoints = Sequence(
            nodes=[self._waypoint_goal(pose) for pose in self.target_locations.poses]
        )
        return [waypoints, *self.keep_base_still()]

    def _waypoint_goal(self, target: Pose) -> CartesianPose:
        """
        :param target: The waypoint the tool center point passes through.
        :return: The task reaching that waypoint.
        """
        return CartesianPose(
            root_link=self.controlled_root,
            tip_link=self.arm.end_effector.tool_frame,
            goal=target,
            position_threshold=self.position_threshold,
            orientation_threshold=self.orientation_threshold,
        )

    def validate(
        self,
        result: Any | None = None,
        max_wait_time: timedelta = timedelta(seconds=2),
    ):
        pass


@dataclass(eq=False, repr=False, kw_only=True)
class MoveManipulatorAction(
    MovesToolCenterPoint,
    GoalAction,
    HasArmParameter,
    HasTargetPoseParameter,
    HasGripperCollisionParameter,
):
    """
    Move the end effector of :attr:`arm` to :attr:`target_pose`.
    """

    def create_temporary_collision_rules(self) -> list[CollisionRule]:
        """
        :return: An allowance for the end effector of :attr:`arm` to touch its
            surroundings if :attr:`allow_gripper_collision`, none otherwise.
        """
        if not self.allow_gripper_collision:
            return []
        return [AllowCollisionForEndEffector(end_effector=self.arm.end_effector)]

    def create_goal(self) -> StatechartNode:
        goal = CartesianPose(
            root_link=self.controlled_root,
            tip_link=self.arm.end_effector.tool_frame,
            goal=self.target_pose,
            position_threshold=self.position_threshold,
            orientation_threshold=self.orientation_threshold,
            binding_policy=GoalBindingPolicy.Bind_on_start,
        )
        for node in self.keep_base_still():
            goal.add_node(node)
        return goal
