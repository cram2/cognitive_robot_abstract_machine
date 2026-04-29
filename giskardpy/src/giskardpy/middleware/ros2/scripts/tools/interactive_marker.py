from __future__ import annotations

from time import sleep
from dataclasses import dataclass, field
from typing import Dict, List

import rclpy
from interactive_markers.interactive_marker_server import InteractiveMarkerServer
from rclpy import Parameter
from semantic_digital_twin.world_description.world_entity import (
    KinematicStructureEntity,
)
from visualization_msgs.msg import InteractiveMarker, InteractiveMarkerControl, Marker
from visualization_msgs.msg import InteractiveMarkerFeedback

from giskardpy.motion_statechart.graph_node import EndMotion
from giskardpy.motion_statechart.monitors.payload_monitors import CountSeconds
from giskardpy.motion_statechart.motion_statechart import MotionStatechart
from giskardpy.motion_statechart.tasks.cartesian_tasks import CartesianPose
from giskardpy.middleware.ros2.python_interface import GiskardWrapper
from giskardpy.middleware.ros2 import rospy
from semantic_digital_twin.exceptions import WorldEntityNotFoundError
from semantic_digital_twin.spatial_types import HomogeneousTransformationMatrix


@dataclass
class InteractiveMarkerNode:
    """
    ROS 2 node that manages interactive markers for Cartesian pose control.

    This node creates interactive markers for specified kinematic chains and allows
    users to manipulate them via RViz. When a marker is moved, it generates motion
    goals that are sent to Giskard for execution.

    Node parameters for launching the node:

    - root_links: List of root link names for kinematic chains
    - tip_links: List of tip link names corresponding to root_links

    Example Node for .launch.py:

    .. code-block:: python

        Node(
            package="giskardpy_ros",
            executable="interactive_marker",
            name="giskard_interactive_marker",
            parameters=[
                {
                    "root_links": ["map", "map", "map"],
                    "tip_links": [
                        "r_gripper_tool_frame",
                        "l_gripper_tool_frame",
                        "base_footprint",
                    ],
                }
            ],
            output="screen",
        ),
    """

    motion_timeout_seconds: float = 20
    """
    Timeout in seconds for motion execution.
    """
    giskard: GiskardWrapper = field(init=False)
    """
    Wrapper for Giskard motion planner.
    """
    markers: Dict[str, KinematicChainMarker] = field(init=False)
    """
    Dictionary mapping marker names to KinematicChainMarker instances.
    """
    server: InteractiveMarkerServer | None = field(init=False)
    """
    Interactive marker server for RViz.
    """
    root_links: List[str] = field(init=False)
    """
    List of root link names from parameters.
    """
    tip_links: List[str] = field(init=False)
    """
    List of tip link names from parameters.
    """

    def __post_init__(self) -> None:
        """
        Sets up the Giskard wrapper, reads parameters, creates kinematic chain markers,
        and initializes the interactive marker server. Retries up to world_entity_retry_attempts
        times if world entities are not found initially.

        :raises WorldEntityNotFoundError: If kinematic structure entities cannot be found
            after all retry attempts.
        """
        self.giskard = GiskardWrapper(
            node_handle=rospy.node, giskard_node_name="giskard"
        )
        self.markers = {}
        self.server = None

        self.giskard.node_handle.declare_parameters(
            namespace="",
            parameters=[
                ("root_links", Parameter.Type.STRING_ARRAY),
                ("tip_links", Parameter.Type.STRING_ARRAY),
            ],
        )
        self.root_links = self.giskard.node_handle.get_parameter("root_links").value
        self.tip_links = self.giskard.node_handle.get_parameter("tip_links").value

        self._initialize_markers()
        self._setup_marker_server()

    def _initialize_markers(self) -> None:
        """
        Attempts to find kinematic structure entities and create markers for them.

        :raises WorldEntityNotFoundError: If entities cannot be found after all retries.
        """
        try:
            for root, tip in zip(self.root_links, self.tip_links):
                root_body = self.giskard.world.get_kinematic_structure_entity_by_name(
                    root
                )
                tip_body = self.giskard.world.get_kinematic_structure_entity_by_name(
                    tip
                )
                kinematic_chain = KinematicChainMarker(root, tip, root_body, tip_body)
                self.markers[kinematic_chain.name] = kinematic_chain

            return
        except WorldEntityNotFoundError as error:
            raise error

    def _setup_marker_server(self) -> None:
        """
        Set up the interactive marker server and register all markers.

        Creates the server, configures each marker with controls,
        and registers feedback callbacks.
        """
        self.server = InteractiveMarkerServer(rospy.node, "cartesian_goals")

        for marker in self.markers.values():
            marker.create_marker()
            self.server.insert(marker.interactive_marker_message)
            self.server.setCallback(marker.name, self.process_feedback)

        self.server.applyChanges()

    def process_feedback(self, feedback: InteractiveMarkerFeedback) -> None:
        """
        Process feedback from interactive markers when user interactions occur.

        When a marker is released (MOUSE_UP event), this method extracts the new pose,
        creates a motion goal with a timeout, and sends it to Giskard for execution.
        The marker is then reset to its default pose.

        :param feedback: The interactive marker feedback containing pose and event information.
        """
        if feedback.event_type != InteractiveMarkerFeedback.MOUSE_UP:
            return

        self.giskard.node_handle.get_logger().info(
            f"Marker feedback received: {feedback.event_type}"
        )

        kinematic_chain_marker = self.markers[feedback.marker_name]

        goal_transformation = HomogeneousTransformationMatrix.from_xyz_quaternion(
            pos_x=feedback.pose.position.x,
            pos_y=feedback.pose.position.y,
            pos_z=feedback.pose.position.z,
            quat_x=feedback.pose.orientation.x,
            quat_y=feedback.pose.orientation.y,
            quat_z=feedback.pose.orientation.z,
            quat_w=feedback.pose.orientation.w,
            reference_frame=kinematic_chain_marker.tip_body,
        )

        motion_statechart = MotionStatechart()
        motion_statechart.add_nodes(
            [
                goal_transformation := CartesianPose(
                    root_link=kinematic_chain_marker.root_body,
                    tip_link=kinematic_chain_marker.tip_body,
                    goal_pose=goal_transformation,
                ),
                motion_timeout := CountSeconds(seconds=self.motion_timeout_seconds),
            ]
        )
        motion_statechart.add_node(
            EndMotion.when_any_true([goal_transformation, motion_timeout])
        )
        self.giskard.execute_async(motion_statechart)

        # Reset marker pose
        kinematic_chain_marker.reset_pose()

        # Update marker in server
        self.server.insert(kinematic_chain_marker.interactive_marker_message)
        self.server.applyChanges()


@dataclass
class KinematicChainMarker:
    """
    Represents an interactive marker for a kinematic chain in RViz.

    This dataclass encapsulates all data and functionality needed to create and manage
    an interactive marker for a robot kinematic chain. It handles marker visualization,
    control setup (translation and rotation), and pose representation.
    """

    root_link: str
    """
    Name of the root link in the kinematic chain.
    """
    tip_link: str
    """
    Name of the tip/end-effector link in the kinematic chain.
    """
    root_body: KinematicStructureEntity
    """
    Kinematic structure entity representing the root body.
    """
    tip_body: KinematicStructureEntity
    """
    Kinematic structure entity representing the tip body.
    """
    marker_scale: float = 0.25
    """
    Scale of the interactive marker in meters.
    """
    marker_box_size: float = 0.175
    """
    Size of the marker box along each axis in meters.
    """
    marker_color_value: float = 0.5
    """
    RGB color value for the marker box (0.0 to 1.0).
    """
    marker_color_alpha: float = 0.5
    """
    Alpha transparency value for the marker box (0.0 to 1.0).
    """
    name: str = field(init=False)
    """
    Formatted name combining root and tip links.
    """
    interactive_marker_message: InteractiveMarker = field(init=False)
    """
    ROS InteractiveMarker message object.
    """

    def __post_init__(self) -> None:
        """
        Creates the name attribute and initializes an empty InteractiveMarker object.
        """
        self.name = f"{self.root_link}/{self.tip_link}"
        self.interactive_marker_message = InteractiveMarker()

    def create_marker(self) -> None:
        """
        Configure and create the interactive marker with all controls.

        Sets up marker frame, appearance (cube visualization), and movement and
        rotation controls for all axes.
        """
        self._setup_marker_properties()
        self._add_box_control()
        self._add_movement_controls()
        self._add_rotation_controls()

    def _setup_marker_properties(self) -> None:
        """
        Set up basic marker properties and appearance.
        """
        self.interactive_marker_message.header.frame_id = str(self.tip_body.name)
        self.interactive_marker_message.name = self.name
        self.interactive_marker_message.scale = self.marker_scale
        self.interactive_marker_message.pose.orientation.w = 1.0

    def _add_box_control(self) -> None:
        """
        Add the visual box control to the marker.
        """
        box_marker = Marker()
        box_marker.type = Marker.CUBE
        box_marker.scale.x = self.marker_box_size
        box_marker.scale.y = self.marker_box_size
        box_marker.scale.z = self.marker_box_size
        box_marker.color.r = self.marker_color_value
        box_marker.color.g = self.marker_color_value
        box_marker.color.b = self.marker_color_value
        box_marker.color.a = self.marker_color_alpha

        box_control = InteractiveMarkerControl()
        box_control.always_visible = True
        box_control.markers.append(box_marker)
        box_control.interaction_mode = InteractiveMarkerControl.MOVE_PLANE

        self.interactive_marker_message.controls.append(box_control)

    def _add_movement_controls(self) -> None:
        """
        Add translation controls for X, Y, Z axes.
        """
        self.add_control(
            "move_x", InteractiveMarkerControl.MOVE_AXIS, 1.0, 0.0, 0.0, 1.0
        )
        self.add_control(
            "move_y", InteractiveMarkerControl.MOVE_AXIS, 0.0, 1.0, 0.0, 1.0
        )
        self.add_control(
            "move_z", InteractiveMarkerControl.MOVE_AXIS, 0.0, 0.0, 1.0, 1.0
        )

    def _add_rotation_controls(self) -> None:
        """
        Add rotation controls for X, Y, Z axes.
        """
        self.add_control(
            "rotate_x",
            InteractiveMarkerControl.ROTATE_AXIS,
            1.0,
            0.0,
            0.0,
            1.0,
        )
        self.add_control(
            "rotate_y",
            InteractiveMarkerControl.ROTATE_AXIS,
            0.0,
            1.0,
            0.0,
            1.0,
        )
        self.add_control(
            "rotate_z",
            InteractiveMarkerControl.ROTATE_AXIS,
            0.0,
            0.0,
            1.0,
            1.0,
        )

    def add_control(
        self,
        name: str,
        interaction_mode: int,
        x: float,
        y: float,
        z: float,
        w: float,
    ) -> None:
        """
        Add a single interactive marker control for translation or rotation.

        :param name: Identifier for the control (e.g., "move_x", "rotate_z")
        :param interaction_mode: InteractiveMarkerControl mode (MOVE_AXIS or ROTATE_AXIS)
        :param x: X component of the control orientation quaternion
        :param y: Y component of the control orientation quaternion
        :param z: Z component of the control orientation quaternion
        :param w: W component of the control orientation quaternion
        """
        control = InteractiveMarkerControl()
        control.name = name
        control.interaction_mode = interaction_mode
        control.orientation.w = w
        control.orientation.x = x
        control.orientation.y = y
        control.orientation.z = z
        self.interactive_marker_message.controls.append(control)

    def reset_pose(self) -> None:
        """
        Reset the marker pose to the default identity position and orientation.

        Sets position to origin and orientation to identity quaternion.
        """
        self.interactive_marker_message.pose.position.x = 0.0
        self.interactive_marker_message.pose.position.y = 0.0
        self.interactive_marker_message.pose.position.z = 0.0
        self.interactive_marker_message.pose.orientation.x = 0.0
        self.interactive_marker_message.pose.orientation.y = 0.0
        self.interactive_marker_message.pose.orientation.z = 0.0
        self.interactive_marker_message.pose.orientation.w = 1.0


def main(args: None = None) -> None:
    """
    Main entry point for the interactive marker ROS 2 node.

    Initializes the ROS 2 node, creates the InteractiveMarkerNode instance,
    logs a startup message, and keeps the node running until shutdown is requested.

    :param args: Optional command-line arguments (currently unused)
    """
    rospy.init_node("interactive_marker")
    node = InteractiveMarkerNode()
    node.giskard.node_handle.get_logger().info("interactive marker server running")
    rospy.spinner_thread.join()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
