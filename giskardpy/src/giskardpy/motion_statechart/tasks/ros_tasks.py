from __future__ import annotations

import logging
from dataclasses import dataclass, field

import rclpy
from geometry_msgs.msg import PoseStamped, Pose, Point, Quaternion
from nav2_msgs.action import NavigateToPose
from rclpy.action import ActionClient
from std_msgs.msg import Header
from typing_extensions import Any

from giskardpy.motion_statechart.context import ExecutionContext, BuildContext
from giskardpy.motion_statechart.graph_node import (
    MotionStatechartNode,
    NodeArtifacts,
)
from semantic_digital_twin.spatial_types import TransformationMatrix
from semantic_digital_twin.world_description.world_entity import Body

logger = logging.getLogger(__name__)


@dataclass
class ActionServerTask(MotionStatechartNode):
    """
    Motion node for a ROS action server. Allows to send a goal to an action server while being part of a motion statechart.
    """

    action_topic: str
    """
    Topic name for the action server.
    """

    goal_msg: Any
    """
    Fully specified goal message that can be send out. 
    """

    node_handle: rclpy.node.Node
    """
    A ROS node to create the action client.
    """

    _action_client: ActionClient = field(init=False)
    """
    ROS action client, is created in `on_start`.
    """

    def build(self, context: BuildContext) -> NodeArtifacts:
        """
        Creates the action client.
        """
        self._action_client = ActionClient(
            self.node_handle, self.goal_msg.__class__, self.action_topic
        )
        return NodeArtifacts()

    def on_start(self, context: ExecutionContext):
        logger.info(f"Waiting for action server {self.action_topic}")
        self._action_client.wait_for_server()
        logger.debug("Sending goal to action server")
        future = self._action_client.send_goal_async(self.goal_msg)
        future.add_done_callback(self.result_callback)

    def result_callback(self, future):
        result = future.result().result
        logger.info(f"Action server {self.action_topic} returned result: {result}")


@dataclass
class NavigateActionServerTask(MotionStatechartNode):
    """
    Node for calling a Navigation2 ROS2 action server to navigate to a given pose.1
    """

    target_pose: TransformationMatrix
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

    node_handle: rclpy.node.Node
    """
    A ROS node to create the action client.
    """

    _action_client: ActionClient = field(init=False)
    """
    ROS Action client, is created in `on_start`.
    """

    def build(self, context: BuildContext) -> NodeArtifacts:
        """
        Builds the motion state node this includes creating the action client and setting the observation expression.
        The observation is true if the robot is within 1cm of the target pose.
        """
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

        artifacts.observation = root_p_goal.eucluidian_distance(r_P_c) < 0.01

        return artifacts

    def on_start(self, context: ExecutionContext):
        """
        Creates a goal and sends it to the action server asynchronously.
        """
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
        self._action_client.wait_for_server()
        future = self._action_client.send_goal_async(
            NavigateToPose.Goal(pose=pose_stamped)
        )
        future.add_done_callback(self.result_callback)

    def result_callback(self, future):
        result = future.result().result
        logger.info(f"Action server {self.action_topic} returned result: {result}")
