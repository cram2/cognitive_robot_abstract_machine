import unittest
from itertools import chain, repeat

from unittest.mock import patch, MagicMock

import pytest
import rclpy.publisher
from sensor_msgs.msg import JointState
from coraplex.ros_utils.joint_state_publisher import JointStatePublisher
from semantic_digital_twin.robots.pr2 import PR2Joint


class DummyRobot:
    def __init__(self):
        self.joint_name_to_id = {"joint1": 0, "joint2": 1}
        self.joint_states = {"joint1": 1.0, "joint2": 2.0}

    def get_joint_position(self, joint_name):
        return self.joint_states[joint_name]


def test_initialization(immutable_model_world, rclpy_node):
    world, robot_view, context = immutable_model_world
    node = rclpy_node
    publisher = JointStatePublisher(
        world, node, joint_state_topic="/test_topic", interval=0.05
    )
    assert publisher.interval == 0.05
    assert isinstance(publisher.joint_state_pub, rclpy.publisher.Publisher)

    publisher._stop_publishing()


def test_publish_sends_joint_state(immutable_model_world, rclpy_node):
    world, robot_view, context = immutable_model_world
    node = rclpy_node
    mock_publisher = MagicMock()
    publisher = JointStatePublisher(world, node)
    publisher.joint_state_pub = mock_publisher
    publisher.interval = 0.1
    publisher.kill_event = MagicMock()
    publisher.kill_event.is_set.side_effect = chain([False], repeat(True))

    publisher._publish()

    assert mock_publisher.publish.called
    msg = mock_publisher.publish.call_args[0][0]
    assert isinstance(msg, JointState)
    assert PR2Joint.TORSO_LIFT in msg.name
    assert PR2Joint.RIGHT_SHOULDER_PAN in msg.name
    joint_to_position = dict(zip(msg.name, msg.position))
    assert joint_to_position[PR2Joint.RIGHT_WRIST_ROLL] == pytest.approx(
        world.state[
            world.get_degree_of_freedom_by_name(PR2Joint.RIGHT_WRIST_ROLL).id
        ].position,
        abs=0.01,
    )

    assert joint_to_position[PR2Joint.RIGHT_SHOULDER_PAN] == pytest.approx(
        world.state[
            world.get_degree_of_freedom_by_name(PR2Joint.RIGHT_SHOULDER_PAN).id
        ].position,
        abs=0.01,
    )


def test_stop_publishing(immutable_model_world, rclpy_node):
    world, robot_view, context = immutable_model_world
    node = rclpy_node
    publisher = JointStatePublisher(world, node)
    publisher.kill_event = MagicMock()
    publisher.thread = MagicMock()

    publisher._stop_publishing()
    publisher.kill_event.set.assert_called_once()
    publisher.thread.join.assert_called_once()
