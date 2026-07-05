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
from typing import Callable

from unittest.mock import MagicMock, patch

from coraplex.ros_utils.force_torque_sensor import ForceTorqueSensorSimulated
from coraplex.ros_utils.joint_state_publisher import JointStatePublisher


def _weakref_to_stopped_publisher(construct: Callable) -> weakref.ref:
    """Construct a publisher, stop its thread, and return only a weakref to it.

    The publisher is built and released entirely within this frame so the caller keeps no strong
    reference; whether it survives a following ``gc.collect()`` then depends solely on process-global
    anchors such as an ``atexit`` registration.
    """
    publisher = construct()
    publisher._stop_publishing()
    return weakref.ref(publisher)


@patch("coraplex.ros_utils.joint_state_publisher.create_publisher")
def test_joint_state_publisher_is_not_retained_after_stop(mock_create_publisher):
    reference = _weakref_to_stopped_publisher(
        lambda: JointStatePublisher(MagicMock(name="world"), MagicMock(name="node"))
    )
    gc.collect()
    assert reference() is None, "publisher was retained after its thread stopped"


@patch("coraplex.ros_utils.force_torque_sensor.create_publisher")
def test_force_torque_sensor_is_not_retained_after_stop(mock_create_publisher):
    reference = _weakref_to_stopped_publisher(
        lambda: ForceTorqueSensorSimulated("joint", MagicMock(name="world"))
    )
    gc.collect()
    assert reference() is None, "publisher was retained after its thread stopped"
