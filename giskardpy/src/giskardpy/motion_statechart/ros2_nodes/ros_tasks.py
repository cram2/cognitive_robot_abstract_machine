from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field

from geometry_msgs.msg import (
    PoseStamped as ROSPoseStamped,
    Pose as ROSPose,
    Point as ROSPoint,
    Quaternion as ROSQuaternion,
)

from griplink_interfaces.action import Grip, Release, Flexgrip, Flexrelease


from control_msgs.action import ParallelGripperCommand
from coraplex.datastructures.enums import WPGGripPreset
from semantic_digital_twin.datastructures.definitions import GripperState
from semantic_digital_twin.robots.robot_parts import EndEffector

try:
    from nav2_msgs.action import NavigateToPose
except ModuleNotFoundError:
    NavigateToPose = None
from rclpy.action import ActionClient
from std_msgs.msg import Header
from typing_extensions import Type, TypeVar, Generic

import krrood.symbolic_math.symbolic_math as sm
from giskardpy.motion_statechart.context import MotionStatechartContext
from giskardpy.motion_statechart.data_types import ObservationStateValues
from giskardpy.motion_statechart.graph_node import MotionStatechartNode, NodeArtifacts
from giskardpy.motion_statechart.ros_context import RosContextExtension
from semantic_digital_twin.spatial_types.spatial_types import Pose
from semantic_digital_twin.world_description.world_entity import Body

try:
    from control_msgs.action import ParallelGripperCommand
except ModuleNotFoundError:
    ParallelGripperCommand = None

logger = logging.getLogger(__name__)

Action = TypeVar("Action")
ActionGoal = TypeVar("ActionGoal")
ActionResult = TypeVar("ActionResult")
ActionFeedback = TypeVar("ActionFeedback")


@dataclass
class ActionServerTask(
    MotionStatechartNode,
    ABC,
    Generic[Action, ActionGoal, ActionResult, ActionFeedback],
):
    """
    Abstract base class for tasks that call a ROS2 action server.
    """

    action_topic: str
    """
    Topic name for the action server.
    """

    message_type: Type[Action]
    """
    Fully specified goal message that can be send out. 
    """

    _action_client: ActionClient = field(init=False)
    """
    ROS action client, is created in `build`.
    """

    _msg: ActionGoal = field(init=False, default=None)
    """
    ROS message to send to the action server.
    """

    _result: ActionResult = field(init=False, default=None)
    """
    ROS action server result.
    """

    @abstractmethod
    def build_msg(self, context: MotionStatechartContext):
        """
        Build the action server message and returns it.
        """
        ...

    def build(self, context: MotionStatechartContext) -> NodeArtifacts:
        """
        Creates the action client.
        """
        ros_context_extension = context.require_extension(RosContextExtension)
        self._action_client = ActionClient(
            ros_context_extension.ros_node, self.message_type, self.action_topic
        )
        self.build_msg(context)
        logger.info(f"Waiting for action server {self.action_topic}")
        self._action_client.wait_for_server()
        return NodeArtifacts()

    def on_start(self, context: MotionStatechartContext):
        """
        Creates a goal and sends it to the action server asynchronously.
        """
        future = self._action_client.send_goal_async(self._msg)
        future.add_done_callback(self.result_callback)

    def result_callback(self, future):
        self._result = future.result().result
        logger.info(
            f"Action server {self.action_topic} returned result: {self._result}"
        )


@dataclass
class NavigateActionServerTask(
    ActionServerTask[
        NavigateToPose,
        NavigateToPose.Goal,
        NavigateToPose.Result,
        NavigateToPose.Feedback,
    ]
):
    """
    Node for calling a Navigation2 ROS2 action server to navigate to a given pose.1
    """

    target_pose: Pose
    """
    Target pose to which the robot should navigate.
    """

    base_link: Body
    """
    Base link of the robot, used for estimating the distance to the goal
    """

    action_topic: str
    """
    Topic name for the navigation action server.
    """

    def build_msg(self, context: MotionStatechartContext):
        root_p_goal = context.world.transform(
            target_frame=context.world.root, spatial_object=self.target_pose
        )
        position = root_p_goal.to_position().to_np()
        orientation = root_p_goal.to_quaternion().to_np()
        pose_stamped = ROSPoseStamped(
            header=Header(frame_id="map"),
            pose=ROSPose(
                position=ROSPoint(x=position[0], y=position[1], z=position[2]),
                orientation=ROSQuaternion(
                    x=orientation[0],
                    y=orientation[1],
                    z=orientation[2],
                    w=orientation[3],
                ),
            ),
        )
        self._msg = NavigateToPose.Goal(pose=pose_stamped)

    def build(self, context: MotionStatechartContext) -> NodeArtifacts:
        """
        Builds the motion state node this includes creating the action client and setting the observation expression.
        The observation is true if the robot is within 1cm of the target pose.
        """
        super().build_msg(context)
        artifacts = NodeArtifacts()
        root_T_goal = context.world.transform(
            target_frame=context.world.root, spatial_object=self.target_pose
        )
        root_T_current = context.world.compose_forward_kinematics_expression(
            context.world.root, self.base_link
        )

        position_error = root_T_goal.to_position().euclidean_distance(
            root_T_current.to_position()
        )
        rotation_error = root_T_goal.to_rotation_matrix().rotational_error(
            root_T_current.to_rotation_matrix()
        )

        artifacts.observation = sm.trinary_logic_and(
            position_error < 0.01, sm.abs(rotation_error) < 0.01
        )

        logger.info(f"Waiting for action server {self.action_topic}")
        self._action_client.wait_for_server()

        return artifacts

    def on_tick(self, context: MotionStatechartContext) -> ObservationStateValues:
        if self._result:
            return (
                ObservationStateValues.TRUE
                if self._result.error_code == NavigateToPose.Result.NONE
                else ObservationStateValues.FALSE
            )
        return ObservationStateValues.UNKNOWN


@dataclass(eq=False, repr=False)
class WPGGripperActionServerTask(
    ActionServerTask[Grip, Grip.Goal, Grip.Result, Grip.Feedback]
):
    """
    Node for calling a WPG-300 ROS2 action server to grip the object.
    """

    grip_preset: WPGGripPreset = WPGGripPreset.PRESET_0
    """
    Grip preset
    """

    grip_position: int = None
    """
    Opening width of the gripper [-5..120 mm].
    """

    grip_force: int = None
    """
    Force the gripper applies to the object [30..300 N].
    """

    grip_speed: int = None
    """
    Motion speed of the gripper [5..350 mm/s].
    """

    grip_acceleration: int = None
    """
    Motion acceleration of the gripper [100..4000 mm/s^2].
    """

    def build_msg(self, context: MotionStatechartContext):
        """
        Creates and returns a message based on the provided MotionStatechartContext.

        The method processes the given context to construct a specific message
        that can be utilized for further communication or logging purposes. The
        context determines the message's content and structure.

        Parameters:
            context: MotionStatechartContext
                The context from which the message is built. It contains information
                necessary to construct the message.

        Returns:
            str: The constructed message based on the provided context.
        """
        super().build_msg(context)

        if self.message_type == Flexgrip:
            self._msg = Flexgrip.Goal(
                port=0,
                position=self.grip_position,
                force=self.grip_force,
                speed=self.grip_speed,
                acceleration=self.grip_acceleration,
            )
        elif self.message_type == Flexrelease:
            self._msg = Flexrelease.Goal(
                port=0,
                position=self.grip_position,
                speed=self.grip_speed,
                acceleration=self.grip_acceleration,
            )
        elif self.message_type == Grip:
            self._msg = Grip.Goal(
                port=0,
                index=self.grip_preset.value,
            )
        elif self.message_type == Release:
            self._msg = Release.Goal(
                port=0,
                index=self.grip_preset.value,
            )
        else:
            raise ValueError(f"Unknown message type: {self.message_type}")

    def on_tick(self, context: MotionStatechartContext) -> ObservationStateValues:
        if self._result:
            if self.message_type == Flexgrip:
                return (
                    ObservationStateValues.TRUE
                    if self._result.status == 0
                    else ObservationStateValues.FALSE
                )
            elif self.message_type == Flexrelease:
                return (
                    ObservationStateValues.TRUE
                    if self._result.status == 0
                    else ObservationStateValues.FALSE
                )
            elif self.message_type == Grip:
                return (
                    ObservationStateValues.TRUE
                    if self._result.status == 0
                    else ObservationStateValues.FALSE
                )
            elif self.message_type == Release:
                return (
                    ObservationStateValues.TRUE
                    if self._result.status == 0
                    else ObservationStateValues.FALSE
                )
            else:
                raise ValueError(f"Unknown message type: {self.message_type}")
        return ObservationStateValues.UNKNOWN


@dataclass(eq=False, repr=False)
class RobotiqGripperActionServerTask(
    ActionServerTask[ParallelGripperCommand, ParallelGripperCommand.Goal, ParallelGripperCommand.Result, ParallelGripperCommand.Feedback]
):
    
    """
    Node for calling a Robotiq ROS2 action server to control the gripper using 
    control_msgs/ParallelGripperCommand interface.
    """

    target_position: float = 0.0
    """
    Target position/opening width of the gripper in meters. (e.g., 0.0 for fully closed, 0.085 for fully open)
    """

    target_velocity: float = 10.0
    """
    Velocity at which the gripper should move to the target position in meters per second.
    """

    target_effort: float = 5.0
    """
    Effort (force) that the gripper should apply in Newtons.
    """

    def build_msg(self, context: MotionStatechartContext):
        """
        Creates and returns a message based on the provided MotionStatechartContext.

        The method processes the given context to construct a specific message
        that can be utilized for further communication or logging purposes. The
        context determines the message's content and structure.

        Parameters:
            context: MotionStatechartContext
                The context from which the message is built. It contains information
                necessary to construct the message.

        Returns:
            str: The constructed message based on the provided context.
        """
        super().build_msg(context)

        self._msg = ParallelGripperCommand.Goal()
        self._msg.command.position = [self.target_position]
        self._msg.command.velocity = [self.target_velocity]
        self._msg.command.effort = [self.target_effort]

    def on_tick(self, context: MotionStatechartContext) -> ObservationStateValues:
        """
        Evaluates the action result to track completion.
        """
        if self._result:
            # ParallelGripperCommand.Result contains a boolean 'stalled' flag and a 'reached_goal' flag.
            # Usually, reaching the goal or stalling out (gripping an object) counts as a successful execution.
            if hasattr(self._result, "reached_goal") and self._result.reached_goal:
                return ObservationStateValues.TRUE
            elif hasattr(self._result, "stalled") and self._result.stalled:
                # Stalled means it met resistance (successfully grasped something)
                return ObservationStateValues.TRUE
            else:
                return ObservationStateValues.FALSE

        return ObservationStateValues.UNKNOWN