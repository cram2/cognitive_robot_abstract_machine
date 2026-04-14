from time import sleep

import rclpy
from interactive_markers.interactive_marker_server import InteractiveMarkerServer
from rclpy import Parameter
from visualization_msgs.msg import InteractiveMarker, InteractiveMarkerControl, Marker
from visualization_msgs.msg import InteractiveMarkerFeedback

from giskardpy.motion_statechart.graph_node import EndMotion
from giskardpy.motion_statechart.monitors.payload_monitors import CountSeconds
from giskardpy.motion_statechart.motion_statechart import MotionStatechart
from giskardpy.motion_statechart.tasks.cartesian_tasks import CartesianPose
from giskardpy.middleware.ros2.python_interface import (
    GiskardWrapper,
)
from giskardpy.middleware.ros2 import rospy
from semantic_digital_twin.exceptions import WorldEntityNotFoundError
from semantic_digital_twin.spatial_types import HomogeneousTransformationMatrix


class InteractiveMarkerNode:
    def __init__(self) -> None:
        super().__init__()
        self.giskard = GiskardWrapper(
            node_handle=rospy.node, giskard_node_name="giskard"
        )

        self.giskard.node_handle.declare_parameters(
            namespace="",
            parameters=[
                ("root_link", Parameter.Type.STRING),
                ("tip_link", Parameter.Type.STRING),
            ],
        )
        self.root_link = self.giskard.node_handle.get_parameter("root_link").value
        self.tip_link = self.giskard.node_handle.get_parameter("tip_link").value
        for i in range(100):
            try:
                self.root_body = (
                    self.giskard.world.get_kinematic_structure_entity_by_name(
                        self.root_link
                    )
                )
                self.tip_body = (
                    self.giskard.world.get_kinematic_structure_entity_by_name(
                        self.tip_link
                    )
                )
                break
            except WorldEntityNotFoundError as e:
                sleep(0.5)
                self.giskard.node_handle.get_logger().error(
                    "failed to find bodies in world, retrying in 0.5s..."
                )
        else:
            raise e

        # Create an interactive marker server
        self.server = InteractiveMarkerServer(rospy.node, "cartesian_goals")

        # Create an interactive marker
        int_marker = InteractiveMarker()
        int_marker.header.frame_id = str(self.tip_body.name)
        int_marker.name = f"{self.root_link}/{self.tip_link}"
        int_marker.scale = 0.25
        int_marker.pose.orientation.w = 1.0

        # Create a marker for the interactive marker
        box_marker = Marker()
        box_marker.type = Marker.CUBE
        box_marker.scale.x = 0.175
        box_marker.scale.y = 0.175
        box_marker.scale.z = 0.175
        box_marker.color.r = 0.5
        box_marker.color.g = 0.5
        box_marker.color.b = 0.5
        box_marker.color.a = 0.5

        # Create a control that contains the marker
        box_control = InteractiveMarkerControl()
        box_control.always_visible = True
        box_control.markers.append(box_marker)
        box_control.interaction_mode = InteractiveMarkerControl.MOVE_PLANE

        # Add the control to the interactive marker
        int_marker.controls.append(box_control)

        # Create controls to move the marker along all axes
        self.add_control(
            int_marker, "move_x", InteractiveMarkerControl.MOVE_AXIS, 1.0, 0.0, 0.0, 1.0
        )
        self.add_control(
            int_marker, "move_y", InteractiveMarkerControl.MOVE_AXIS, 0.0, 1.0, 0.0, 1.0
        )
        self.add_control(
            int_marker, "move_z", InteractiveMarkerControl.MOVE_AXIS, 0.0, 0.0, 1.0, 1.0
        )

        # Create controls to rotate the marker around all axes
        self.add_control(
            int_marker,
            "rotate_x",
            InteractiveMarkerControl.ROTATE_AXIS,
            1.0,
            0.0,
            0.0,
            1.0,
        )
        self.add_control(
            int_marker,
            "rotate_y",
            InteractiveMarkerControl.ROTATE_AXIS,
            0.0,
            1.0,
            0.0,
            1.0,
        )
        self.add_control(
            int_marker,
            "rotate_z",
            InteractiveMarkerControl.ROTATE_AXIS,
            0.0,
            0.0,
            1.0,
            1.0,
        )

        # Add the interactive marker to the server
        self.server.insert(int_marker)

        # Set the callback for marker feedback
        self.server.setCallback(int_marker.name, self.process_feedback)

        # 'commit' changes and send to all clients
        self.server.applyChanges()

        self.int_marker = int_marker

    def add_control(
        self,
        int_marker: InteractiveMarker,
        name: str,
        interaction_mode: int,
        x: float,
        y: float,
        z: float,
        w: float,
    ) -> None:
        control = InteractiveMarkerControl()
        control.name = name
        control.interaction_mode = interaction_mode
        control.orientation.w = w
        control.orientation.x = x
        control.orientation.y = y
        control.orientation.z = z
        int_marker.controls.append(control)

    def process_feedback(self, feedback: InteractiveMarkerFeedback) -> None:
        if feedback.event_type == InteractiveMarkerFeedback.MOUSE_UP:
            self.giskard.node_handle.get_logger().info(
                f"Marker feedback received: {feedback.event_type}"
            )

            goal = HomogeneousTransformationMatrix.from_xyz_quaternion(
                pos_x=feedback.pose.position.x,
                pos_y=feedback.pose.position.y,
                pos_z=feedback.pose.position.z,
                quat_x=feedback.pose.orientation.x,
                quat_y=feedback.pose.orientation.y,
                quat_z=feedback.pose.orientation.z,
                quat_w=feedback.pose.orientation.w,
                reference_frame=self.tip_body,
            )

            msc = MotionStatechart()
            msc.add_nodes(
                [
                    goal := CartesianPose(
                        root_link=self.root_body,
                        tip_link=self.tip_body,
                        goal_pose=goal,
                    ),
                    time_out := CountSeconds(seconds=20),
                ]
            )
            msc.add_node(EndMotion.when_any_true([goal, time_out]))
            self.giskard.execute_async(msc)
            # reset marker pose
            self.int_marker.pose.position.x = 0.0
            self.int_marker.pose.position.y = 0.0
            self.int_marker.pose.position.z = 0.0
            self.int_marker.pose.orientation.x = 0.0
            self.int_marker.pose.orientation.y = 0.0
            self.int_marker.pose.orientation.z = 0.0
            self.int_marker.pose.orientation.w = 1.0
            # self.server.clear()
            self.server.insert(self.int_marker)
            self.server.applyChanges()


def main(args: None = None) -> None:
    rospy.init_node("interactive_marker")
    node = InteractiveMarkerNode()
    node.giskard.node_handle.get_logger().info("interactive marker server running")
    rospy.spinner_thread.join()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
