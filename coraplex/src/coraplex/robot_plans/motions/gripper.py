from __future__ import annotations

from dataclasses import dataclass, field
from typing import Generic

from giskardpy.motion_statechart.binding_policy import GoalBindingPolicy
from giskardpy.motion_statechart.data_types import DefaultWeights
from giskardpy.motion_statechart.goals.templates import Parallel, Sequence
from giskardpy.motion_statechart.graph_node import MotionStatechartNode, Task
from giskardpy.motion_statechart.monitors.monitors import LocalMinimumReached
from giskardpy.motion_statechart.tasks.align_planes import AlignPlanes
from giskardpy.motion_statechart.tasks.cartesian_tasks import (
    CartesianPose,
    CartesianPosition,
    CartesianPositionTrajectory,
    CartesianPositionVelocityLimit,
    CartesianRotationVelocityLimit,
)
from giskardpy.motion_statechart.tasks.joint_tasks import (
    JointPositionList,
    JointVelocityLimit,
)
from krrood.patterns.subclass_safe_generic import SubClassSafeGeneric
from semantic_digital_twin.datastructures.alignment import AlignmentPair
from semantic_digital_twin.datastructures.robots.gripper_configuration import (
    TGripperConfiguration,
)
from semantic_digital_twin.robots.justin import Justin
from semantic_digital_twin.robots.robot_part_mixins import HasMobileBase
from semantic_digital_twin.robots.robot_parts import Arm, EndEffector
from semantic_digital_twin.spatial_types import Point3, Vector3
from semantic_digital_twin.spatial_types.spatial_types import Pose
from semantic_digital_twin.world_description.world_entity import Body

from coraplex.datastructures.enums import (
    MovementType,
    WaypointsMovementType,
)
from coraplex.exceptions import MissingToolFrame, MissingWaypoints
from coraplex.robot_plans.mixins import (
    CartesianVelocityLimitParameters,
    GripperStallToleranceParameters,
    HasTcpGoalThresholds,
)
from coraplex.robot_plans.motions.base import BaseMotion


@dataclass
class MoveGripperMotion(
    BaseMotion,
    Generic[TGripperConfiguration],
    SubClassSafeGeneric,
    GripperStallToleranceParameters,
):
    """
    Moves a gripper into the configuration it describes.
    """

    configuration: TGripperConfiguration
    """
    The gripper configuration to command.
    """

    allow_gripper_collision: bool | None = None
    """
    If the gripper is allowed to collide with something.
    """

    def perform(self):
        return

    @property
    def _motion_chart(self):
        """
        :return: The chart driving the gripper's connections to the joint state the
            configuration commands, with the velocity limit and collision rules the
            configuration and parameters ask for.
        """
        goal_state = self.configuration.joint_state
        name = goal_state.name.name
        joint_task = JointPositionList(goal_state=goal_state, name=name)

        done_node = joint_task
        if self.tolerate_stall:
            stall_monitor = LocalMinimumReached(
                degrees_of_freedom=[
                    connection.raw_dof for connection in goal_state.connections
                ],
                minimum_time=(
                    self.stall_minimum_time
                    if self.stall_minimum_time is not None
                    else 1.0
                ),
                measure_from_own_start=True,
            )
            done_node = Parallel(
                [joint_task, stall_monitor], minimum_success=1, name=name
            )

        nodes = [done_node]

        finger_velocity = self.configuration.finger_velocity
        if finger_velocity is not None:
            nodes.append(
                JointVelocityLimit(
                    connections=list(goal_state.connections),
                    max_velocity=finger_velocity,
                )
            )

        if self.allow_gripper_collision:
            nodes.extend(
                self._only_allow_gripper_collision_rules(
                    self.configuration.end_effector
                )
            )

        if len(nodes) == 1:
            return nodes[0]
        return Parallel(nodes, name=name)

    @classmethod
    def handles(cls, motion: MoveGripperMotion) -> bool:
        """
        Whether this alternative can be built from the given motion's configuration.

        :param motion: The motion whose configuration is checked.
        :return: True if the motion's configuration is an instance of the configuration
            type this alternative is bound to. Also True when this class binds no
            concrete configuration type.
        """
        bound_configuration_type = cls.get_generic_type_parameters()
        if not bound_configuration_type:
            return True
        configuration_type = bound_configuration_type[0]
        if not isinstance(configuration_type, type):
            # The generic parameter is still an unresolved type variable, so this class
            # binds no concrete configuration type it could check against.
            return True
        return isinstance(motion.configuration, configuration_type)


@dataclass
class MoveToolCenterPointMotion(
    BaseMotion, CartesianVelocityLimitParameters, HasTcpGoalThresholds
):
    """
    Moves the Tool center point (TCP) of the robot.
    """

    target: Pose
    """
    Target pose to which the TCP should be moved.
    """

    arm: Arm
    """
    Arm with the TCP that should be moved to the target.
    """

    allow_gripper_collision: bool | None = None
    """
    If the gripper can collide with something.
    """

    movement_type: MovementType | None = MovementType.CARTESIAN
    """
    The type of movement that should be performed.
    """

    def perform(self):
        return

    def _velocity_limit_nodes(self, root: Body, tip: Body) -> list[Task]:
        """
        :return: The :class:`CartesianPositionVelocityLimit`/
            :class:`CartesianRotationVelocityLimit` nodes requested via
            :attr:`max_linear_velocity`/:attr:`max_angular_velocity`, if any.
        """
        nodes = []
        if self.max_linear_velocity is not None:
            nodes.append(
                CartesianPositionVelocityLimit(
                    root_link=root,
                    tip_link=tip,
                    max_linear_velocity=self.max_linear_velocity,
                )
            )
        if (
            self.max_angular_velocity is not None
            and self.movement_type != MovementType.TRANSLATION
        ):
            nodes.append(
                CartesianRotationVelocityLimit(
                    root_link=root,
                    tip_link=tip,
                    max_angular_velocity=self.max_angular_velocity,
                )
            )
        return nodes

    @property
    def _motion_chart(self):
        tip = self.arm.end_effector.tool_frame
        root = (
            self.world.root
            if isinstance(self.robot, HasMobileBase)
            and self.robot.mobile_base.full_body_controlled
            else self.robot.root
        )
        if self.movement_type == MovementType.TRANSLATION:
            task = CartesianPosition(
                root_link=root,
                tip_link=tip,
                goal_point=self.target.position,
                name="MoveTCP",
                weight=DefaultWeights.WEIGHT_BELOW_COLLISION_AVOIDANCE,
                threshold=self.resolved_position_threshold(),
            )
        else:
            task = CartesianPose(
                root_link=root,
                tip_link=tip,
                goal_pose=self.target,
                name="MoveTCP",
                weight=DefaultWeights.WEIGHT_BELOW_COLLISION_AVOIDANCE,
                translation_threshold=self.resolved_position_threshold(),
                orientation_threshold=self.resolved_orientation_threshold(),
            )
        accompanying_nodes: list[MotionStatechartNode] = list(
            self._velocity_limit_nodes(root, tip)
        )
        if self.allow_gripper_collision:
            accompanying_nodes.extend(
                self._only_allow_gripper_collision_rules(self.arm.end_effector)
            )
        if not accompanying_nodes:
            return task
        return Parallel([task, *accompanying_nodes], name="MoveTCP")


@dataclass
class MoveTCPWaypointsMotion(BaseMotion, HasTcpGoalThresholds):
    """
    Moves the Tool center point (TCP) of the robot.
    """

    waypoints: list[Pose]
    """
    Waypoints the TCP should move along.
    """

    arm: Arm
    """
    Arm with the TCP that should be moved to the target.
    """

    allow_gripper_collision: bool | None = None
    """
    If the gripper can collide with something.
    """

    movement_type: WaypointsMovementType = (
        WaypointsMovementType.ENFORCE_ORIENTATION_FINAL_POINT
    )
    """
    The type of movement that should be performed.
    """

    def perform(self):
        return

    @property
    def _motion_chart(self):
        tip = self.arm.end_effector.tool_frame
        root = (
            self.world.root
            if isinstance(self.robot, HasMobileBase)
            and self.robot.mobile_base.full_body_controlled
            else self.robot.root
        )
        task_kwargs = dict(root_link=root, tip_link=tip)
        if self.position_threshold is not None:
            task_kwargs["translation_threshold"] = self.position_threshold
        if self.orientation_threshold is not None:
            task_kwargs["orientation_threshold"] = self.orientation_threshold
        nodes = [
            CartesianPose(
                goal_pose=pose,
                **task_kwargs,
            )
            for pose in self.waypoints
        ]
        return Sequence(nodes=nodes)


@dataclass
class MoveTCPWaypointsAlignedMotion(BaseMotion, HasTcpGoalThresholds):
    """
    Moves the tool center point (TCP) of the robot along waypoints while keeping the
    given plane alignments.
    """

    waypoints: list[Point3]
    """
    Waypoints the TCP should move along.
    """

    arm: Arm
    """
    Arm with the TCP that should be moved along the waypoints.
    """

    alignment_pairs: list[AlignmentPair] = field(default_factory=list)
    """
    Normal pairs kept aligned during the motion.
    """

    allow_gripper_collision: bool | None = None
    """
    If the gripper can collide with something.
    """

    tip: Body | None = None
    """
    The body that should follow the waypoints.

    Defaults to the arm's tool frame.
    """

    def perform(self):
        return

    def _resolve_tip(self) -> Body:
        """
        :return: The body that follows the waypoints: the explicit tip if given,
            otherwise the arm's tool frame.
        :raises MissingToolFrame: If no tip is given and the arm has no tool frame.
        """
        if self.tip is not None:
            return self.tip
        tool_frame = self.arm.end_effector.tool_frame
        if tool_frame is None:
            raise MissingToolFrame(self.arm, self.robot)
        return tool_frame

    def _upright_torso_task(self, tip_link: Body, root_link: Body) -> AlignPlanes:
        """
        :return: A task that keeps Justin's torso upright during the motion.
        """
        torso_tip = self.robot.mobile_base.torso.tip
        return AlignPlanes(
            tip_link=torso_tip,
            root_link=root_link,
            tip_normal=Vector3.X(torso_tip),
            goal_normal=Vector3.Z(root_link),
            weight=DefaultWeights.WEIGHT_ABOVE_COLLISION_AVOIDANCE.value,
        )

    @property
    def _motion_chart(self):
        if not self.waypoints:
            raise MissingWaypoints(self)

        tip_link = self._resolve_tip()
        root_link = (
            self.world.root
            if isinstance(self.robot, HasMobileBase)
            and self.robot.mobile_base.full_body_controlled
            else self.robot.root
        )
        trajectory_kwargs = dict(
            root_link=root_link,
            tip_link=tip_link,
            goal_points=self.waypoints,
            maximum_skip_ahead=2,
            weight=float(DefaultWeights.WEIGHT_BELOW_COLLISION_AVOIDANCE),
            name="MoveTCPWaypointsAligned",
        )
        if self.position_threshold is not None:
            trajectory_kwargs["threshold"] = self.position_threshold
        tasks = [CartesianPositionTrajectory(**trajectory_kwargs)]
        tasks.extend(
            AlignPlanes(
                tip_link=tip_link,
                root_link=root_link,
                tip_normal=pair.tip_normal,
                goal_normal=pair.goal_normal,
                weight=DefaultWeights.WEIGHT_BELOW_COLLISION_AVOIDANCE.value,
            )
            for pair in self.alignment_pairs
        )
        if isinstance(self.robot, Justin):
            tasks.append(self._upright_torso_task(tip_link, root_link))
        motion_statechart_nodes = (
            self._only_allow_gripper_collision_rules(self.arm.end_effector)
            if self.allow_gripper_collision
            else []
        )
        motion_statechart_nodes.append(Parallel(tasks))
        return Parallel(motion_statechart_nodes)


@dataclass
class MoveManipulatorMotion(BaseMotion, HasTcpGoalThresholds):
    """
    Moves the Tool center point (TCP) of the robot.
    """

    target: Pose
    """
    Target pose to which the TCP should be moved.
    """

    end_effector: EndEffector
    """
    The end effector to move to the target pose.
    """

    allow_gripper_collision: bool = False
    """
    If the gripper can collide with something.
    """

    @property
    def _motion_chart(self):
        robot = self.robot
        full_body_controlled = (
            robot.mobile_base.full_body_controlled
            if isinstance(robot, HasMobileBase)
            else False
        )
        root = self.world.root if full_body_controlled else robot.root
        task = CartesianPose(
            root_link=root,
            tip_link=self.end_effector.tool_frame,
            goal_pose=self.target,
            translation_threshold=self.resolved_position_threshold(),
            orientation_threshold=self.resolved_orientation_threshold(),
            binding_policy=GoalBindingPolicy.Bind_on_start,
            name=self.__class__.__name__,
        )
        return task
