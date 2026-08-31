"""
Tests for Giskard's ownership of the process-local ROS node lifecycle.
"""

from __future__ import annotations

from collections.abc import Iterator

import pytest

from giskardpy.middleware.ros2 import rospy
from semantic_digital_twin.adapters.ros.node_registry import ROSNodeRegistry


@pytest.fixture()
def initialized_rospy() -> Iterator[ROSNodeRegistry]:
    """
    Start Giskard's ROS runtime and stop it after the test.
    """
    registry = ROSNodeRegistry()
    registry.clear()
    rospy.init_node("giskard_registry_test")
    yield registry
    rospy.shutdown()
    registry.clear()


def test_init_node_registers_giskards_node(
    initialized_rospy: ROSNodeRegistry,
) -> None:
    assert rospy.get_node() is initialized_rospy.get()


def test_init_node_does_not_replace_an_initialized_node(
    initialized_rospy: ROSNodeRegistry,
) -> None:
    initialized_node = initialized_rospy.get()

    rospy.init_node("replacement")

    assert initialized_rospy.get() is initialized_node


def test_shutdown_clears_the_node_registration() -> None:
    registry = ROSNodeRegistry()
    registry.clear()
    rospy.init_node("giskard_registry_test")

    rospy.shutdown()

    assert not registry.has_node()
