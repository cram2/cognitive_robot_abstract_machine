from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Optional

import rclpy
from geometry_msgs.msg import PoseStamped, Pose, Point, Quaternion
from nav2_msgs.action import NavigateToPose
from rclpy.action import ActionClient
from std_msgs.msg import Header
from typing_extensions import Any, Type, TypeVar

from giskardpy.motion_statechart.context import ExecutionContext, BuildContext
from giskardpy.motion_statechart.data_types import ObservationStateValues
from giskardpy.motion_statechart.graph_node import (
    MotionStatechartNode,
    NodeArtifacts,
)
from semantic_digital_twin.spatial_types import HomogeneousTransformationMatrix
from semantic_digital_twin.world_description.world_entity import Body

logger = logging.getLogger(__name__)


T = TypeVar("T")


@dataclass
class ActionServerTask(MotionStatechartNode, ABC):
    """
    Abstract base class for tasks that call a ROS2 action server.
    """

    action_topic: str
    """
    Topic name for the action server.
    """

    message_type: Type[T]
    """
    Fully specified goal message that can be send out. 
    """

    node_handle: rclpy.node.Node
    """
    A ROS node to create the action client.
    """

    _action_client: ActionClient = field(init=False)
    """
    ROS action client, is created in `build`.
    """

    _msg: T.Goal = field(init=False, default=None)
    """
    ROS message to send to the action server.
    """

    _result: T.Result = field(init=False, default=None)
    """
    ROS action server result.
    """

    @abstractmethod
    def build_msg(self, context: BuildContext):
        """
        Build the action server message and returns it.
        """
        ...

    def build(self, context: BuildContext) -> NodeArtifacts:
        """
        Creates the action client.
        """
        self._action_client = ActionClient(
            self.node_handle, self.message_type, self.action_topic
        )
        self.build_msg(context)
        logger.info(f"Waiting for action server {self.action_topic}")
        self._action_client.wait_for_server()
        return NodeArtifacts()

    def on_start(self, context: ExecutionContext):
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
class NavigateActionServerTask(ActionServerTask):
    """
    Node for calling a Navigation2 ROS2 action server to navigate to a given pose.1
    """

    target_pose: HomogeneousTransformationMatrix
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

    def build_msg(self, context: BuildContext):
        root_p_goal = context.world.transform(
            target_frame=context.world.root, spatial_object=self.target_pose
        )
        position = root_p_goal.to_position().to_np()
        orientation = root_p_goal.to_quaternion().to_np()
        pose_stamped = PoseStamped(
            header=Header(frame_id="map"),
            pose=Pose(
                position=Point(x=position[0], y=position[1], z=position[2]),
                orientation=Quaternion(
                    x=orientation[0],
                    y=orientation[1],
                    z=orientation[2],
                    w=orientation[3],
                ),
            ),
        )
        self._msg = NavigateToPose.Goal(pose=pose_stamped)

    def build(self, context: BuildContext) -> NodeArtifacts:
        """
        Builds the motion state node this includes creating the action client and setting the observation expression.
        The observation is true if the robot is within 1cm of the target pose.
        """
        super()
        self._action_client = ActionClient(
            self.node_handle, NavigateToPose, self.action_topic
        )
        artifacts = NodeArtifacts()
        root_p_goal = context.world.transform(
            target_frame=context.world.root, spatial_object=self.target_pose
        )
        r_P_c = context.world.compose_forward_kinematics_expression(
            context.world.root, self.base_link
        ).to_position()

        artifacts.observation = (
            root_p_goal.to_position().euclidean_distance(r_P_c) < 0.01
        )

        logger.info(f"Waiting for action server {self.action_topic}")
        self._action_client.wait_for_server()

        return artifacts

    def on_tick(self, context: ExecutionContext) -> ObservationStateValues:
        if self._result:
            return (
                ObservationStateValues.TRUE
                if self._result.error_code == NavigateToPose.Result.NONE
                else ObservationStateValues.FALSE
            )
        return ObservationStateValues.UNKNOWN
