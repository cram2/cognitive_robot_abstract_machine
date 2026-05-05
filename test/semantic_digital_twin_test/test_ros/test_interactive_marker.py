import pytest
import numpy as np
from visualization_msgs.msg import InteractiveMarkerFeedback
from semantic_digital_twin.adapters.ros.visualization.interactive_marker import (
    InteractiveMarkerPublisher,
)
from semantic_digital_twin.world_description.connections import (
    RevoluteConnection,
    PrismaticConnection,
    Connection6DoF,
)
from semantic_digital_twin.spatial_types import HomogeneousTransformationMatrix, Vector3


def test_interactive_marker_revolute_feedback(rclpy_node, self_collision_bot_world):
    interactive_publisher = InteractiveMarkerPublisher(
        _world=self_collision_bot_world, node=rclpy_node
    )

    # Find a revolute connection
    revolute_conn = None
    for conn in self_collision_bot_world.connections:
        if isinstance(conn, RevoluteConnection):
            revolute_conn = conn
            break

    assert revolute_conn is not None

    # Simulate feedback
    feedback = InteractiveMarkerFeedback()
    feedback.marker_name = str(revolute_conn.name)
    feedback.event_type = InteractiveMarkerFeedback.POSE_UPDATE

    q_new = 0.5
    joint_transform = HomogeneousTransformationMatrix.from_xyz_axis_angle(
        axis=revolute_conn.axis, angle=q_new
    )
    # The interactive marker's pose is representing parent_T_conn @ joint_transform @ conn_T_child
    new_pose = (
        revolute_conn.parent_T_connection_expression
        @ joint_transform
        @ revolute_conn.connection_T_child_expression
    )

    feedback.pose.position.x = float(new_pose.to_position().x)
    feedback.pose.position.y = float(new_pose.to_position().y)
    feedback.pose.position.z = float(new_pose.to_position().z)
    quat = new_pose.to_rotation_matrix().to_quaternion()
    feedback.pose.orientation.x = float(quat.x)
    feedback.pose.orientation.y = float(quat.y)
    feedback.pose.orientation.z = float(quat.z)
    feedback.pose.orientation.w = float(quat.w)

    interactive_publisher._process_feedback(feedback)

    assert np.isclose(revolute_conn.position, q_new, atol=1e-5)


def test_interactive_marker_6dof_feedback(rclpy_node, self_collision_bot_world):
    # Add a 6DOF connection to the world
    with self_collision_bot_world.modify_world():
        child = self_collision_bot_world.get_body_by_name("l_shoulder")
        parent = self_collision_bot_world.get_body_by_name("map")
        conn_6dof = Connection6DoF.create_with_dofs(
            parent=parent, child=child, world=self_collision_bot_world
        )
        # We need to remove the old connection first or just use a new body
        # For simplicity, let's just create a new body
        from semantic_digital_twin.world_description.world_entity import Body
        from semantic_digital_twin.datastructures.prefixed_name import PrefixedName

        new_body = Body(name=PrefixedName("test_6dof_body"))
        self_collision_bot_world.add_body(new_body)
        conn_6dof = Connection6DoF.create_with_dofs(
            parent=parent, child=new_body, world=self_collision_bot_world
        )
        self_collision_bot_world.add_connection(conn_6dof)

    interactive_publisher = InteractiveMarkerPublisher(
        _world=self_collision_bot_world, node=rclpy_node
    )

    # Simulate feedback
    feedback = InteractiveMarkerFeedback()
    feedback.marker_name = str(conn_6dof.name)
    feedback.event_type = InteractiveMarkerFeedback.POSE_UPDATE

    pos_new = np.array([1.0, 2.0, 3.0])
    q_new = np.array([0.0, 0.0, 0.0, 1.0])  # identity
    joint_transform = HomogeneousTransformationMatrix.from_xyz_quaternion(
        pos_x=pos_new[0],
        pos_y=pos_new[1],
        pos_z=pos_new[2],
        quat_x=q_new[0],
        quat_y=q_new[1],
        quat_z=q_new[2],
        quat_w=q_new[3],
    )
    new_pose = (
        conn_6dof.parent_T_connection_expression
        @ joint_transform
        @ conn_6dof.connection_T_child_expression
    )

    feedback.pose.position.x = float(new_pose.to_position().x)
    feedback.pose.position.y = float(new_pose.to_position().y)
    feedback.pose.position.z = float(new_pose.to_position().z)
    quat = new_pose.to_rotation_matrix().to_quaternion()
    feedback.pose.orientation.x = float(quat.x)
    feedback.pose.orientation.y = float(quat.y)
    feedback.pose.orientation.z = float(quat.z)
    feedback.pose.orientation.w = float(quat.w)

    interactive_publisher._process_feedback(feedback)

    assert np.isclose(
        self_collision_bot_world.state[conn_6dof.x.id].position, pos_new[0]
    )
    assert np.isclose(
        self_collision_bot_world.state[conn_6dof.y.id].position, pos_new[1]
    )
    assert np.isclose(
        self_collision_bot_world.state[conn_6dof.z.id].position, pos_new[2]
    )
