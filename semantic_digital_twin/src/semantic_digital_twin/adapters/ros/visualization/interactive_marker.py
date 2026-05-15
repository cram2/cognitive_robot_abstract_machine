from __future__ import annotations
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Dict, List, Any, Optional
import numpy as np

import rclpy
from rclpy.node import Node
from interactive_markers.interactive_marker_server import InteractiveMarkerServer
from visualization_msgs.msg import (
    InteractiveMarker,
    InteractiveMarkerControl,
    Marker,
    InteractiveMarkerFeedback,
)

from semantic_digital_twin.datastructures.prefixed_name import PrefixedName
from semantic_digital_twin.callbacks.callback import (
    ModelChangeCallback,
    StateChangeCallback,
)
from semantic_digital_twin.world_description.connections import (
    FixedConnection,
    ActiveConnection1DOF,
    RevoluteConnection,
    PrismaticConnection,
    Connection6DoF,
)
from semantic_digital_twin.adapters.ros.semdt_to_ros2_converters import (
    PoseToRos2Converter,
)
from semantic_digital_twin.spatial_types import (
    HomogeneousTransformationMatrix,
    Quaternion,
)

if TYPE_CHECKING:
    from semantic_digital_twin.world import World


@dataclass(eq=False)
class InteractiveMarkerPublisher(ModelChangeCallback):
    """
    Publishes interactive markers for every non-fixed connection in the world.
    Allows manipulating the state of the degrees of freedom in Rviz2.
    """

    node: Node = field(kw_only=True)
    """
    The ROS2 node used for the interactive marker server.
    """

    server_name: str = "/semworld/interactive_markers"
    """
    The name of the interactive marker server.
    """

    server: InteractiveMarkerServer = field(init=False)
    """
    The interactive marker server.
    """

    state_callback: _InternalStateChangeCallback = field(init=False)
    """
    Internal callback to handle state changes.
    """

    def __post_init__(self):
        super().__post_init__()
        self.server = InteractiveMarkerServer(self.node, self.server_name)
        self.state_callback = _InternalStateChangeCallback(
            _world=self._world, parent=self
        )
        self._rebuild_markers()

    def stop(self):
        """
        Stops the publisher and clears the server.
        """
        super().stop()
        self.state_callback.stop()
        self.server.clear()
        self.server.applyChanges()

    def _notify(self, **kwargs):
        """
        Rebuilds markers when the world model changes.
        """
        self._rebuild_markers()

    def _rebuild_markers(self):
        """
        Clears and recreates all interactive markers.
        """
        self.server.clear()
        for connection in self._world.connections:
            if isinstance(connection, FixedConnection):
                continue
            self._create_interactive_marker(connection)
        self.server.applyChanges()

    def _create_interactive_marker(self, connection: Any):
        """
        Creates an interactive marker for a connection.
        """
        int_marker = InteractiveMarker()
        int_marker.header.frame_id = str(connection.parent.name)
        int_marker.name = str(connection.name)
        int_marker.description = f"Control {connection.name}"
        int_marker.pose = PoseToRos2Converter.convert(connection.origin.to_pose())
        int_marker.scale = 0.2

        # Visual indicator
        visual_control = InteractiveMarkerControl()
        visual_control.always_visible = True
        marker = Marker()
        marker.type = Marker.SPHERE
        marker.scale.x = 0.05
        marker.scale.y = 0.05
        marker.scale.z = 0.05
        marker.color.r = 1.0
        marker.color.g = 1.0
        marker.color.b = 0.0
        marker.color.a = 0.5
        visual_control.markers.append(marker)
        int_marker.controls.append(visual_control)

        if isinstance(connection, ActiveConnection1DOF):
            self._add_1dof_controls(int_marker, connection)
        elif isinstance(connection, Connection6DoF):
            self._add_6dof_controls(int_marker)

        self.server.insert(int_marker, feedback_callback=self._process_feedback)

    def _add_1dof_controls(
        self, int_marker: InteractiveMarker, connection: ActiveConnection1DOF
    ):
        """
        Adds controls for a 1DOF connection.
        """
        control = InteractiveMarkerControl()
        axis = connection.axis.to_np()[:3]
        q = self._get_rotation_to_axis(np.array([1.0, 0.0, 0.0]), axis)
        control.orientation.x = float(q.x)
        control.orientation.y = float(q.y)
        control.orientation.z = float(q.z)
        control.orientation.w = float(q.w)

        if isinstance(connection, RevoluteConnection):
            control.interaction_mode = InteractiveMarkerControl.ROTATE_AXIS
        elif isinstance(connection, PrismaticConnection):
            control.interaction_mode = InteractiveMarkerControl.MOVE_AXIS

        int_marker.controls.append(control)

    def _add_6dof_controls(self, int_marker: InteractiveMarker):
        """
        Adds 6DOF controls.
        """
        for mode in [
            InteractiveMarkerControl.ROTATE_AXIS,
            InteractiveMarkerControl.MOVE_AXIS,
        ]:
            # X
            c = InteractiveMarkerControl()
            c.orientation.w = 1.0
            c.orientation.x = 1.0
            c.interaction_mode = mode
            int_marker.controls.append(c)
            # Y
            c = InteractiveMarkerControl()
            c.orientation.w = 1.0
            c.orientation.z = 1.0
            c.interaction_mode = mode
            int_marker.controls.append(c)
            # Z
            c = InteractiveMarkerControl()
            c.orientation.w = 1.0
            c.orientation.y = 1.0
            c.interaction_mode = mode
            int_marker.controls.append(c)

    def _get_rotation_to_axis(
        self, from_axis: np.ndarray, to_axis: np.ndarray
    ) -> Quaternion:
        """
        Calculates the rotation from one axis to another.
        """
        from_axis = from_axis / np.linalg.norm(from_axis)
        to_axis = to_axis / np.linalg.norm(to_axis)
        dot = np.dot(from_axis, to_axis)
        if dot > 0.999999:
            return Quaternion(x=0, y=0, z=0, w=1)
        if dot < -0.999999:
            ortho = (
                np.array([1, 0, 0]) if abs(from_axis[0]) < 0.9 else np.array([0, 1, 0])
            )
            axis = np.cross(from_axis, ortho)
            axis = axis / np.linalg.norm(axis)
            return Quaternion.from_axis_angle(axis, np.pi)

        axis = np.cross(from_axis, to_axis)
        angle = np.arccos(dot)
        return Quaternion.from_axis_angle(axis, angle)

    def _process_feedback(self, feedback: InteractiveMarkerFeedback):
        """
        Processes feedback from the interactive markers.
        """
        if feedback.event_type != InteractiveMarkerFeedback.POSE_UPDATE:
            return

        connection_name = PrefixedName.from_string(feedback.marker_name)
        connection = self._world.get_connection_by_name(connection_name)
        if not connection:
            return

        F = HomogeneousTransformationMatrix.from_xyz_quaternion(
            pos_x=feedback.pose.position.x,
            pos_y=feedback.pose.position.y,
            pos_z=feedback.pose.position.z,
            quat_x=feedback.pose.orientation.x,
            quat_y=feedback.pose.orientation.y,
            quat_z=feedback.pose.orientation.z,
            quat_w=feedback.pose.orientation.w,
            reference_frame=connection.parent,
        )

        parent_T_conn = connection.parent_T_connection_expression
        conn_T_child = connection.connection_T_child_expression
        joint_transform = parent_T_conn.inverse() @ F @ conn_T_child.inverse()

        if isinstance(connection, ActiveConnection1DOF):
            self._update_1dof_connection(connection, joint_transform)
        elif isinstance(connection, Connection6DoF):
            self._update_6dof_connection(connection, joint_transform)

    def _update_1dof_connection(
        self,
        connection: ActiveConnection1DOF,
        joint_transform: HomogeneousTransformationMatrix,
    ):
        """
        Updates a 1DOF connection state.
        """
        if isinstance(connection, RevoluteConnection):
            axis, angle = joint_transform.to_rotation_matrix().to_axis_angle()
            if np.dot(axis.to_np()[:3], connection.axis.to_np()[:3]) < 0:
                angle = -angle
            q = float(angle)
        elif isinstance(connection, PrismaticConnection):
            translation = joint_transform.to_position().to_np()[:3]
            q = float(np.dot(translation, connection.axis.to_np()[:3]))
        else:
            return

        # Respect joint limits
        dof = connection.raw_dof
        # Raw position from the target q
        raw_q = (q - connection.offset) / connection.multiplier

        if dof.limits.lower.position is not None:
            raw_q = max(raw_q, dof.limits.lower.position)
        if dof.limits.upper.position is not None:
            raw_q = min(raw_q, dof.limits.upper.position)

        # Set position back through the connection (handles multiplier/offset)
        connection.position = raw_q * connection.multiplier + connection.offset

    def _update_6dof_connection(
        self,
        connection: Connection6DoF,
        joint_transform: HomogeneousTransformationMatrix,
    ):
        """
        Updates a 6DOF connection state.
        """
        pos = joint_transform.to_position().to_np()
        quat = joint_transform.to_rotation_matrix().to_quaternion().to_np()

        def clamp(val, dof_id):
            dof = self._world.get_degree_of_freedom_by_id(dof_id)
            if dof.limits.lower.position is not None:
                val = max(val, dof.limits.lower.position)
            if dof.limits.upper.position is not None:
                val = min(val, dof.limits.upper.position)
            return val

        state = self._world.state
        state[connection.x.id].position = float(clamp(pos[0], connection.x.id))
        state[connection.y.id].position = float(clamp(pos[1], connection.y.id))
        state[connection.z.id].position = float(clamp(pos[2], connection.z.id))

        # Orientations usually don't have limits for 6DOF, but if they do, we clamp
        # Note: Clamping individual quaternion components can lead to non-unit quaternions.
        # We re-normalize afterwards if any were clamped.
        q_original = quat.copy()
        quat[0] = clamp(quat[0], connection.qx.id)
        quat[1] = clamp(quat[1], connection.qy.id)
        quat[2] = clamp(quat[2], connection.qz.id)
        quat[3] = clamp(quat[3], connection.qw.id)

        if not np.allclose(q_original, quat):
            norm = np.linalg.norm(quat)
            if norm > 1e-6:
                quat /= norm
            else:
                quat = q_original  # Fallback to original if clamping made it zero

        state[connection.qx.id].position = float(quat[0])
        state[connection.qy.id].position = float(quat[1])
        state[connection.qz.id].position = float(quat[2])
        state[connection.qw.id].position = float(quat[3])
        self._world.notify_state_change()

    def _update_marker_poses(self):
        """
        Updates marker poses to match the current world state.
        """
        for connection in self._world.connections:
            if isinstance(connection, FixedConnection):
                continue
            pose = PoseToRos2Converter.convert(connection.origin.to_pose())
            self.server.setPose(str(connection.name), pose)
        self.server.applyChanges()


@dataclass(eq=False)
class _InternalStateChangeCallback(StateChangeCallback):
    """
    Internal callback for handling state changes in InteractiveMarkerPublisher.
    """

    parent: InteractiveMarkerPublisher = field(kw_only=True)
    """
    Reference to the parent publisher.
    """

    def _notify(self, **kwargs):
        self.parent._update_marker_poses()
