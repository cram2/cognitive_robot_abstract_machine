"""Regression tests proving threaded ROS publishers are not retained after they stop.

The publishers spawn a background thread and previously registered a bound method with
``atexit``. That registration is a process-global strong reference to the publisher (and, through
it, to the :class:`~semantic_digital_twin.world.World` it reads from), so every publisher ever
constructed stayed alive for the whole interpreter session and pinned its world -- surfacing as the
coraplex "leaking worlds" failures. These tests assert that once a publisher's thread has stopped,
nothing keeps the publisher alive.
"""

import gc
import weakref
from unittest.mock import MagicMock, patch

from coraplex.ros_utils.force_torque_sensor import ForceTorqueSensorSimulated
from coraplex.ros_utils.joint_state_publisher import JointStatePublisher


def _assert_collectable(publisher) -> None:
    """Stop the publisher's thread and assert the publisher is garbage-collectable."""
    reference = weakref.ref(publisher)
    publisher._stop_publishing()
    del publisher
    gc.collect()
    assert reference() is None, "publisher was retained after its thread stopped"


@patch("coraplex.ros_utils.joint_state_publisher.create_publisher")
def test_joint_state_publisher_is_not_retained_after_stop(mock_create_publisher):
    publisher = JointStatePublisher(MagicMock(name="world"), MagicMock(name="node"))
    _assert_collectable(publisher)


@patch("coraplex.ros_utils.force_torque_sensor.create_publisher")
def test_force_torque_sensor_is_not_retained_after_stop(mock_create_publisher):
    sensor = ForceTorqueSensorSimulated("joint", MagicMock(name="world"))
    _assert_collectable(sensor)
