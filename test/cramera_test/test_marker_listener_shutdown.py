"""
ROS listener termination when the owning runtime shuts down.
"""

from unittest.mock import Mock

from rclpy.executors import ExternalShutdownException

from cramera.live.bridge import Bridge
from cramera.live.ros_markers import RosMarkerListener


# %% external context shutdown


def test_external_shutdown_finishes_the_listener_thread():
    """
    The ROS shutdown signal ends the listener without an unhandled exception.
    """
    executor = Mock()
    executor.spin.side_effect = ExternalShutdownException
    listener = RosMarkerListener(bridge=Bridge())

    assert listener._spin(executor) is None
