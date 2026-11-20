"""
Motion statechart tasks that send a goal to a ROS 2 action server and end with its
result.
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from functools import partial
from threading import Lock

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
from rclpy.action.client import ClientGoalHandle
from rclpy.task import Future
from std_msgs.msg import Header
from typing_extensions import Callable, TypeVar, Generic

import krrood.symbolic_math.symbolic_math as sm
from krrood.patterns.classproperty import classproperty
from cramph.node import EndedByOwner
from cramph.context import ContextExtension
from cramph.data_types import LifeCycleValue, ObservationStateValue
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


# %% goals sent to an action server


@dataclass(kw_only=True)
class SentGoal:
    """
    One goal an :class:`ActionServerTask` sent to its action server.

    Once cancelled, the server's responses to it are no longer passed on, and the server
    is asked to cancel it as soon as it accepted it. A goal the server answered can no
    longer be cancelled.
    """

    goal_handle: ClientGoalHandle | None = field(init=False, default=None)
    """
    The client's handle of the goal, None until the server accepted it.
    """

    is_cancelled: bool = field(init=False, default=False)
    """
    Whether the task stopped pursuing the goal before the server answered it.
    """

    is_answered: bool = field(init=False, default=False)
    """
    Whether the server sent the goal's result.
    """

    _lock: Lock = field(init=False, default_factory=Lock, repr=False)
    """
    Orders cancelling, which runs on the ticking thread, against the server's responses,
    which arrive on the ROS executor's thread.
    """

    def cancel(self) -> None:
        """
        Stops pursuing the goal unless the server already answered it, and asks the
        server to cancel it if the server accepted it.
        """
        with self._lock:
            if self.is_cancelled or self.is_answered:
                return
            self.is_cancelled = True
            if self.goal_handle is not None:
                self.goal_handle.cancel_goal_async()

    def receive_goal_response(
        self, *, goal_handle: ClientGoalHandle, record_response: Callable[[], None]
    ) -> None:
        """
        Calls `record_response` unless the goal is cancelled.

        A goal the server accepted after it was cancelled is cancelled on the server.
        """
        with self._lock:
            if goal_handle.accepted:
                self.goal_handle = goal_handle
            if not self.is_cancelled:
                record_response()
                return
            if goal_handle.accepted:
                goal_handle.cancel_goal_async()

    def receive_result(self, *, store_result: Callable[[], None]) -> None:
        """
        Marks the goal as answered and calls `store_result` unless the goal is
        cancelled.
        """
        with self._lock:
            self.is_answered = True
            if not self.is_cancelled:
                store_result()


# %% tasks calling an action server


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

    _result: ActionResult | None = field(init=False, default=None)
    """
    The result of the goal last sent, None until it arrived.
    """

    _sent_goal: SentGoal | None = field(init=False, default=None)
    """
    The goal last sent to the action server, None before the first one.
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
        Differs from :meth:`StatechartNode.on_start` in calling
        :meth:`align_goal_with_life_cycle`.
        """
        self.align_goal_with_life_cycle()

    def on_pause(self) -> None:
        """
        Differs from :meth:`StatechartNode.on_pause` in calling
        :meth:`align_goal_with_life_cycle`.
        """
        self.align_goal_with_life_cycle()

    def on_unpause(self) -> None:
        """
        Differs from :meth:`StatechartNode.on_unpause` in calling
        :meth:`align_goal_with_life_cycle`.
        """
        self.align_goal_with_life_cycle()

    def on_end(self) -> None:
        """
        Differs from :meth:`StatechartNode.on_end` in calling
        :meth:`align_goal_with_life_cycle`.
        """
        self.align_goal_with_life_cycle()

    def on_reset(self) -> None:
        """
        Differs from :meth:`StatechartNode.on_reset` in calling
        :meth:`align_goal_with_life_cycle`.
        """
        self.align_goal_with_life_cycle()

    def cleanup(self) -> None:
        """
        Differs from :meth:`StatechartNode.cleanup` in cancelling the goal last sent.
        """
        super().cleanup()
        self.cancel_goal()

    def align_goal_with_life_cycle(self) -> None:
        """
        Sends a new goal if this task runs and pursues none, cancels the goal last sent
        if it does not run, and forgets that goal once it is reset.

        .. note:: A task that starts paused therefore sends its goal only once it
            continues, and a task paused after its goal was answered keeps the answer.
        """
        match self.life_cycle_state:
            case LifeCycleValue.RUNNING:
                if self._sent_goal is None or self._sent_goal.is_cancelled:
                    self.send_goal()
            case LifeCycleValue.NOT_STARTED:
                self.cancel_goal()
                self._sent_goal = None
            case _:
                self.cancel_goal()

    def cancel_goal(self) -> None:
        """
        Stops pursuing the goal last sent, see :meth:`SentGoal.cancel`.
        """
        if self._sent_goal is not None:
            self._sent_goal.cancel()

    def send_goal(self) -> None:
        """
        Sends :attr:`_action_goal` to the action server as a new goal, forgetting the
        result of the goal sent before.
        """
        self._result = None
        sent_goal = SentGoal()
        self._sent_goal = sent_goal
        future = self._action_client.send_goal_async(self._action_goal)
        future.add_done_callback(
            partial(self.goal_response_callback, sent_goal=sent_goal)
        )

    @called_positionally
    def goal_response_callback(self, future: Future, *, sent_goal: SentGoal) -> None:
        """
        Handles the server's response to `sent_goal`, and waits for the goal's result
        once the server accepted it, unless the goal is cancelled.
        """
        goal_handle = future.result()
        sent_goal.receive_goal_response(
            goal_handle=goal_handle,
            record_response=partial(
                self.record_goal_response, accepted=goal_handle.accepted
            ),
        )
        if not goal_handle.accepted or sent_goal.is_cancelled:
            return
        goal_handle.get_result_async().add_done_callback(
            partial(self.goal_result_callback, sent_goal=sent_goal)
        )

    @called_positionally
    def goal_result_callback(self, future: Future, *, sent_goal: SentGoal) -> None:
        """
        Handles the result the server sent for `sent_goal`.
        """
        sent_goal.receive_result(
            store_result=partial(self.result_callback, future=future)
        )

    def record_goal_response(self, *, accepted: bool) -> None:
        """
        Records whether the server accepted the goal last sent.
        """
        if accepted:
            logger.info(f"Action server {self.action_topic} accepted the goal.")
            return
        logger.error(f"Action server {self.action_topic} rejected the goal.")

    def result_callback(self, *, future: Future) -> None:
        """
        Stores the result of the goal last sent.
        """
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

    def result_callback(self, *, future: Future) -> None:
        """
        Differs from :meth:`ActionServerTask.result_callback` in logging the navigation
        status and error code.
        """
        # TODO: Check the ._result.result calls against a running action server because I'm not sure if all of them are correct
        super().result_callback(future=future)
        logger.info(
            f"Finished navigation with response status: {self._result.result.status} and result code: {self._result.error_code}"
        )

    def on_tick(self) -> ObservationStateValue:
        if self._result.result:
            return (
                ObservationStateValue.TRUE
                if self._result.error_code == NavigateToPose.Result.NONE
                else ObservationStateValue.FALSE
            )
        return ObservationStateValue.UNKNOWN
