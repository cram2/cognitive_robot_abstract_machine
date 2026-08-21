from dataclasses import dataclass

import pytest

from robokudo.exceptions import RoboKudoROSNodeMissing
from robokudo.io import camera_interface

# %% ROS node ownership


@dataclass
class CameraConfigWithoutTF:
    """
    Minimal camera config that does not request TF lookup.
    """

    interface_type: str = "ROSCameraInterface"


class RegisteredNode:
    """
    Mimic a registered ROS node.
    """


def fail_private_node_creation(node_name: str):
    raise AssertionError("ROSCameraInterface must use the registered RoboKudo node")


def test_ros_camera_interface_uses_registered_robokudo_node(monkeypatch):
    registered_node = RegisteredNode()
    monkeypatch.setattr(camera_interface, "get_node", lambda: registered_node)
    monkeypatch.setattr(camera_interface, "Node", fail_private_node_creation)

    interface = camera_interface.ROSCameraInterface(CameraConfigWithoutTF())

    assert interface.node is registered_node


def test_ros_camera_interface_raises_when_no_robokudo_node_exists(monkeypatch):
    monkeypatch.setattr(
        camera_interface,
        "get_node",
        lambda: (_ for _ in ()).throw(RoboKudoROSNodeMissing()),
    )
    monkeypatch.setattr(camera_interface, "Node", fail_private_node_creation)

    with pytest.raises(RoboKudoROSNodeMissing):
        camera_interface.ROSCameraInterface(CameraConfigWithoutTF())
