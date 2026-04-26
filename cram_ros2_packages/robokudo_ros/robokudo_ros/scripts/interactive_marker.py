#!/usr/bin/env python3

# sudo apt install ros-jazzy-interactive-markers
import numpy as np
import rclpy
from interactive_markers.interactive_marker_server import InteractiveMarkerServer
from rcl_interfaces.msg import ParameterDescriptor, ParameterType
from rclpy.node import Node
from scipy.spatial.transform import Rotation as R
from visualization_msgs.msg import (
    InteractiveMarker,
    InteractiveMarkerControl,
    InteractiveMarkerFeedback,
)


def invert_transform_scipy(p: np.ndarray, q_xyzw: np.ndarray):
    """
    p: [x, y, z]
    q_xyzw: [x, y, z, w]
    T = (R(q), p)
    T^-1: R^-1 = R^T, p^-1 = -R^T p, q^-1 = conjugate(q) for unit q
    """
    rot = R.from_quat(q_xyzw)  # expects [x, y, z, w]
    rot_inv = rot.inv()

    p_inv = rot_inv.apply(-p)  # -R^T p
    q_inv = rot_inv.as_quat()  # [x, y, z, w]
    return p_inv, q_inv


class AlignerNode(Node):
    def __init__(self):
        super().__init__("aligner_node")

        # Frame parameter
        self.declare_parameter(
            "frame_id",
            "map",
            ParameterDescriptor(
                description="Parent frame (e.g. map, world or camera_color_optical_frame)"
            ),
        )
        self.frame_id = (
            self.get_parameter("frame_id").get_parameter_value().string_value
        )

        # Scale parameter, default 0.4
        self.declare_parameter(
            "marker_scale",
            0.4,
            ParameterDescriptor(
                type=ParameterType.PARAMETER_DOUBLE,
                description="Marker size scale (0.1-2.0 recommended)",
            ),
        )
        self.marker_scale = (
            self.get_parameter("marker_scale").get_parameter_value().double_value
        )

        self.get_logger().info(
            f"Using parent frame: {self.frame_id}, scale: {self.marker_scale}"
        )

        # Interactive marker server
        self.server = InteractiveMarkerServer(self, "aligner")

        # Create the interactive marker
        self.int_marker = InteractiveMarker()
        self.int_marker.header.frame_id = self.frame_id
        self.int_marker.pose.position.z = 1.0
        self.int_marker.scale = float(self.marker_scale)
        self.int_marker.name = "camera_guess"
        self.int_marker.description = "Drag axes/arrows/rings to align camera"

        # Keep this off by default to avoid "extra ring" confusion
        enable_view_facing = False
        if enable_view_facing:
            view_control = InteractiveMarkerControl()
            view_control.orientation_mode = InteractiveMarkerControl.VIEW_FACING
            view_control.interaction_mode = InteractiveMarkerControl.MOVE_ROTATE
            view_control.always_visible = True
            self.int_marker.controls.append(view_control)

        def add_axis_controls(qw, qx, qy, qz):
            for mode in (
                InteractiveMarkerControl.MOVE_AXIS,
                InteractiveMarkerControl.ROTATE_AXIS,
            ):
                c = InteractiveMarkerControl()
                c.orientation.w = float(qw)
                c.orientation.x = float(qx)
                c.orientation.y = float(qy)
                c.orientation.z = float(qz)
                c.interaction_mode = mode
                self.int_marker.controls.append(c)

        # X axis (red): move + rotate (normalized quaternion)
        add_axis_controls(0.7071, 0.7071, 0.0, 0.0)
        # Y axis (green): move + rotate
        add_axis_controls(0.7071, 0.0, 0.0, 0.7071)
        # Z axis (blue): move + rotate
        add_axis_controls(0.7071, 0.0, 0.7071, 0.0)

        # Insert and activate
        self.server.insert(self.int_marker, feedback_callback=self.feedback_cb)
        self.server.applyChanges()

        self.get_logger().info(
            f"Marker ready (scale={self.marker_scale}). RViz: Add -> InteractiveMarkers -> /aligner"
        )

    def feedback_cb(self, feedback: InteractiveMarkerFeedback):
        if feedback.event_type is not InteractiveMarkerFeedback.MOUSE_UP:
            return

        # Do NOT call applyChanges() here (causes jitter)
        pose = feedback.pose

        p = np.array([pose.position.x, pose.position.y, pose.position.z], dtype=float)
        q = np.array(
            [
                pose.orientation.x,
                pose.orientation.y,
                pose.orientation.z,
                pose.orientation.w,
            ],
            dtype=float,
        )

        p_inv, q_inv = invert_transform_scipy(p, q)

        self.get_logger().info(
            f"T ({feedback.header.frame_id} -> {self.int_marker.name}): "
            f"pos=({p[0]:.3f}, {p[1]:.3f}, {p[2]:.3f}), "
            f"quat=({q[0]:.3f}, {q[1]:.3f}, {q[2]:.3f}, {q[3]:.3f}) | \n"
            f"T^-1 ({self.int_marker.name} -> {feedback.header.frame_id}): "
            f"pos=({p_inv[0]:.3f}, {p_inv[1]:.3f}, {p_inv[2]:.3f}), "
            f"quat=({q_inv[0]:.3f}, {q_inv[1]:.3f}, {q_inv[2]:.3f}, {q_inv[3]:.3f})"
        )

        self.server.setPose(self.int_marker.name, feedback.pose, feedback.header)
        self.server.applyChanges()


def main(args=None):
    rclpy.init(args=args)
    node = AlignerNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.server.clear()
        node.server.applyChanges()
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
