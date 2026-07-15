from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field

from control_msgs.action import ParallelGripperCommand
from control_msgs.msg import GripperCommand, Float64Values

from geometry_msgs.msg import (
    Point as ROSPoint,
)
from geometry_msgs.msg import (
    Pose as ROSPose,
)
from geometry_msgs.msg import (
    PoseStamped as ROSPoseStamped,
)
from geometry_msgs.msg import (
    Quaternion as ROSQuaternion,
)
from semantic_digital_twin.datastructures.definitions import GripperState
from semantic_digital_twin.robots.robot_parts import EndEffector

try:
    from nav2_msgs.action import NavigateToPose
except ModuleNotFoundError:
    NavigateToPose = None
import krrood.symbolic_math.symbolic_math as sm
from rclpy.action import ActionClient
from semantic_digital_twin.spatial_types.spatial_types import Pose
from semantic_digital_twin.world_description.world_entity import Body
from std_msgs.msg import Header
from typing_extensions import Generic, Type, TypeVar

from giskardpy.motion_statechart.context import MotionStatechartContext
from giskardpy.motion_statechart.data_types import ObservationStateValues
from giskardpy.motion_statechart.graph_node import MotionStatechartNode, NodeArtifacts
from giskardpy.motion_statechart.ros_context import RosContextExtension

logger = logging.getLogger(__name__)

Action = TypeVar("Action")
ActionGoal = TypeVar("ActionGoal")
ActionResult = TypeVar("ActionResult")
ActionFeedback = TypeVar("ActionFeedback")


@dataclass(eq=False, repr=False)
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
        future.add_done_callback(self.goal_response_callback)

    def goal_response_callback(self, future):
        """
        Handles the server's response to the goal submission.

        On rejection a failure sentinel is stored so that :meth:`on_tick` can
        return :attr:`~ObservationStateValues.FALSE` immediately.
        """
        goal_handle = future.result()
        if not goal_handle.accepted:
            logger.error("Goal rejected by action server")
            return

        logger.info("Sent query to action server ")

        result_future = goal_handle.get_result_async()
        result_future.add_done_callback(self.result_callback)

    def result_callback(self, future):
        self._result = future.result()
        logger.info(f"Action server {self.action_topic} done.")


@dataclass(eq=False, repr=False)
class NavigateActionServerTask(
    ActionServerTask[
        NavigateToPose,
        NavigateToPose.Goal,
        NavigateToPose.Result,
        NavigateToPose.Feedback,
    ]
):
    """
    Node for calling a Navigation2 ROS2 action server to navigate to a given pose.
    """

    target_pose: Pose
    """
    Target pose to which the robot should navigate.
    """

    base_link: Body
    """
    Base link of the robot, used for estimating the distance to the goal
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
        super().build(context)
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

        return artifacts

    def result_callback(self, future):
        """
        Stores the navigation result returned by the action server.
        """
        # TODO: Check the ._result.result calls against a running action server because I'm not sure if all of them are correct
        super().result_callback(future)
        logger.info(
            f"Finished navigation with response status: {self._result.result.status} and result code: {self._result.error_code}"
        )

    def on_tick(self, context: MotionStatechartContext) -> ObservationStateValues:
        if self._result.result:
            return (
                ObservationStateValues.TRUE
                if self._result.error_code == NavigateToPose.Result.NONE
                else ObservationStateValues.FALSE
            )
        return ObservationStateValues.UNKNOWN


@dataclass(eq=False, repr=False)
class RobotiqGripperActionServerTask(
    ActionServerTask[
        ParallelGripperCommand,
        ParallelGripperCommand.Goal,
        ParallelGripperCommand.Result,
        ParallelGripperCommand.Feedback,
    ]
):
    """
    Node for calling a Robotiq ROS2 action server using the
    control_msgs/ParallelGripperCommand interface.
    """

    target_position: float
    """
    Desired gripper opening in meters.

    Examples:
        0.0   -> Fully closed
        0.7 -> Fully open
    """

    target_velocity: float = 10.0
    """
    Desired gripper velocity.
    """

    target_effort: float = 10.0
    """
    Maximum gripping effort.
    """

    def build_msg(self, context: MotionStatechartContext):
        """
        Builds the ParallelGripperCommand goal message.

        """

        self._msg = ParallelGripperCommand.Goal()

        self._msg.command.position = [float(self.target_position)]
        self._msg.command.velocity = [float(self.target_velocity)]
        self._msg.command.effort = [float(self.target_effort)]

    def result_callback(self, future):
        """
        Stores the gripper action result returned by the action server.
        """
        super().result_callback(future)

        logger.info(
            f"Gripper action finished. "
            f"Reached goal: {self._result.result.reached_goal}, "
            f"Stalled: {self._result.result.stalled}"
        )

    def on_tick(self, context: MotionStatechartContext) -> ObservationStateValues:
        """
        Returns:
            TRUE    -> Goal reached or gripper stalled while grasping.
            FALSE   -> Action finished unsuccessfully.
            UNKNOWN -> Action still running.
        """

        if self._result is None:
            return ObservationStateValues.UNKNOWN

        result = self._result.result

        if result.reached_goal:
            return ObservationStateValues.TRUE

        if result.stalled:
            # Stalled usually means the gripper contacted an object.
            return ObservationStateValues.TRUE

        return ObservationStateValues.FALSE
