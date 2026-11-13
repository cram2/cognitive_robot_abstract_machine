from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field

from control_msgs.action import ParallelGripperCommand
from geometry_msgs.msg import (
    PoseStamped as ROSPoseStamped,
    Pose as ROSPose,
    Point as ROSPoint,
    Quaternion as ROSQuaternion,
)

try:
    from nav2_msgs.action import NavigateToPose
except ModuleNotFoundError:
    NavigateToPose = None
from rclpy.action import ActionClient
from std_msgs.msg import Header
from typing_extensions import TypeVar, Generic
import krrood.symbolic_math.symbolic_math as sm
from krrood.ormatic.utils import classproperty
from cramph.node import EndedByOwner
from cramph.context import ContextExtension
from cramph.data_types import ObservationStateValues
from giskardpy.motion_statechart.graph_node import MotionStatechartNode
from cramph.node import NodeArtifacts
from giskardpy.motion_statechart.ros_context import RosContextExtension
from krrood_spatial_math.spatial_types import Pose
from semantic_digital_twin.world_description.world_entity import Body
from krrood.patterns.called_positionally import called_positionally

logger = logging.getLogger(__name__)


Action = TypeVar("Action")
ActionGoal = TypeVar("ActionGoal")
ActionResult = TypeVar("ActionResult")
ActionFeedback = TypeVar("ActionFeedback")


@dataclass(eq=False, repr=False, kw_only=True)
class ActionServerTask(
    EndedByOwner,
    MotionStatechartNode,
    ABC,
    Generic[Action, ActionGoal, ActionResult, ActionFeedback],
):
    """
    Abstract base class for tasks that call a ROS2 action server.
    """

    action_topic: str = field(kw_only=True)
    """
    Topic name for the action server.
    """

    message_type: type[Action] = field(kw_only=True)
    """
    Fully specified goal message that can be send out.
    """

    _action_client: ActionClient = field(init=False)
    """
    ROS action client, is created in `set_up`.
    """

    _action_goal: ActionGoal | None = field(init=False, default=None)
    """
    Goal to send to the action server.
    """

    _result: ActionResult = field(init=False, default=None)
    """
    ROS action server result.
    """

    @classproperty
    def required_context_extensions(cls) -> tuple[type[ContextExtension], ...]:
        return super().required_context_extensions + (RosContextExtension,)

    @abstractmethod
    def build_action_goal(self) -> None:
        """
        Build the goal message this node sends to the action server.
        """
        ...

    def set_up(self) -> None:
        """
        Creates the action client and the message to send.
        """
        super().set_up()
        ros_context_extension = self.context.require_extension(
            extension_type=RosContextExtension
        )
        self._action_client = ros_context_extension.get_or_create_action_client(
            message_type=self.message_type, action_topic=self.action_topic
        )
        self.build_action_goal()
        logger.info(f"Waiting for action server {self.action_topic}")
        self._action_client.wait_for_server()

    def on_start(self) -> None:
        """
        Creates a goal and sends it to the action server asynchronously.
        """
        future = self._action_client.send_goal_async(self._action_goal)
        future.add_done_callback(self.goal_response_callback)

    @called_positionally
    def goal_response_callback(self, future):
        """
        Handles the server's response to the goal submission.

        On rejection a failure sentinel is stored so that :meth:`on_tick` can return
        :attr:`~ObservationStateValues.FALSE` immediately.
        """
        goal_handle = future.result()
        if not goal_handle.accepted:
            logger.error("Goal rejected by action server")
            return

        logger.info("Sent query to action server ")

        result_future = goal_handle.get_result_async()
        result_future.add_done_callback(self.result_callback)

    @called_positionally
    def result_callback(self, future):
        self._result = future.result()
        logger.info(f"Action server {self.action_topic} done.")


@dataclass(eq=False, repr=False, kw_only=True)
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
    Base link of the robot, used for estimating the distance to the goal.
    """

    def build_action_goal(self) -> None:
        root_T_goal = self.context.world.transform(
            target_frame=self.context.world.root, spatial_object=self.target_pose
        )
        position = root_T_goal.position.to_np()
        orientation = root_T_goal.quaternion.to_np()
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
        self._action_goal = NavigateToPose.Goal(pose=pose_stamped)

    def build_artifacts(self) -> NodeArtifacts:
        """
        Observes whether the robot is within 1cm of the target pose.
        """
        artifacts = NodeArtifacts()
        root_T_goal = self.context.world.transform(
            target_frame=self.context.world.root, spatial_object=self.target_pose
        )
        root_T_current = self.context.world.compose_forward_kinematics_expression(
            self.context.world.root, self.base_link
        )

        position_error = root_T_goal.position.euclidean_distance(
            root_T_current.position
        )
        rotation_error = root_T_goal.rotation_matrix.rotational_distance(
            root_T_current.rotation_matrix
        )

        artifacts.observation = sm.trinary_logic_and(
            position_error < 0.01, sm.abs(rotation_error) < 0.01
        )

        return artifacts

    @called_positionally
    def result_callback(self, future):
        """
        Stores the navigation result returned by the action server.
        """
        # TODO: Check the ._result.result calls against a running action server because I'm not sure if all of them are correct
        super().result_callback(future=future)
        logger.info(
            f"Finished navigation with response status: {self._result.result.status} and result code: {self._result.error_code}"
        )

    def on_tick(self) -> ObservationStateValues:
        if self._result.result:
            return (
                ObservationStateValues.TRUE
                if self._result.error_code == NavigateToPose.Result.NONE
                else ObservationStateValues.FALSE
            )
        return ObservationStateValues.UNKNOWN


@dataclass(eq=False, repr=False, kw_only=True)
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

    message_type: type[ParallelGripperCommand] = field(
        default=ParallelGripperCommand, kw_only=True
    )
    """
    The ROS2 action interface type.
    """

    target_position: float
    """
    Desired gripper opening width in meters (e.g., 0.0 for fully open, 0.085 for fully
    closed).
    """

    target_velocity: float = 10.0
    """
    Desired gripper joint velocity in radians per second.
    """

    target_effort: float = 50.0
    """
    Maximum gripping force or effort applied by the fingers (in Newtons / effort units,
    up to 125.0).
    """

    _goal_accepted: bool = field(init=False, default=False)
    _done: bool = field(init=False, default=False)

    def build_action_goal(self) -> None:
        """
        Build the ParallelGripperCommand goal message to send to the action server.
        """
        self._action_goal = ParallelGripperCommand.Goal()
        self._action_goal.command.position = [float(self.target_position)]
        self._action_goal.command.velocity = [float(self.target_velocity)]
        self._action_goal.command.effort = [float(self.target_effort)]
        self._goal_accepted = False
        self._done = False

    @called_positionally
    def goal_response_callback(self, future):
        """
        Handles the server's response to the goal submission.
        """
        goal_handle = future.result()
        if not goal_handle.accepted:
            logger.error("Gripper goal rejected by action server")
            self._goal_accepted = False
            self._done = True
            return

        self._goal_accepted = True
        logger.info("Gripper goal accepted, waiting for result...")

        result_future = goal_handle.get_result_async()
        result_future.add_done_callback(self.result_callback)

    @called_positionally
    def result_callback(self, future):
        """
        Stores the action result and marks execution as done.
        """
        super().result_callback(future=future)
        self._done = True
        if self._result and hasattr(self._result, "result"):
            logger.info(
                f"Gripper action finished. "
                f"Reached goal: {self._result.result.reached_goal}, "
                f"Stalled: {self._result.result.stalled}"
            )

    def on_tick(self) -> ObservationStateValues:
        """
        Observes whether the gripper action succeeded (reached goal or stalled on
        object).
        """
        # If goal was rejected, fail immediately
        if self._done and not self._goal_accepted:
            return ObservationStateValues.FALSE

        # If completed and result was received
        if self._done and self._result is not None:
            action_result = getattr(self._result, "result", self._result)
            if action_result is not None:
                if action_result.reached_goal or action_result.stalled:
                    return ObservationStateValues.TRUE
                return ObservationStateValues.FALSE

        return ObservationStateValues.UNKNOWN
