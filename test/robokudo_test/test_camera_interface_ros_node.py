from collections.abc import Iterator
from dataclasses import dataclass

import pytest

from robokudo.io import camera_interface
from semantic_digital_twin.adapters.ros.node_registry import ROSNodeRegistry
from semantic_digital_twin.exceptions import ROSNodeNotRegisteredError

# %% ROS node ownership


@dataclass
class CameraConfigWithoutTF:
    """
    Minimal camera config that does not request TF lookup.
    """

    interface_type: str = "ROSCameraInterface"
    """
    Selects the ROS camera interface implementation.
    """


@dataclass
class RegisteredNode:
    """
    Mimic a registered ROS node.
    """


@pytest.fixture
def node_registry() -> Iterator[ROSNodeRegistry]:
    """
    Provide the empty process registry required by the camera integration tests.
    """
    registry = ROSNodeRegistry()
    registry.clear()
    yield registry
    registry.clear()


def test_ros_camera_interface_uses_registered_node(
    node_registry: ROSNodeRegistry,
) -> None:
    registered_node = RegisteredNode()
    node_registry.register(registered_node)

    interface = camera_interface.ROSCameraInterface(CameraConfigWithoutTF())

    assert interface.node is registered_node


def test_ros_camera_interface_raises_when_no_node_is_registered(
    node_registry: ROSNodeRegistry,
) -> None:
    with pytest.raises(ROSNodeNotRegisteredError):
        camera_interface.ROSCameraInterface(CameraConfigWithoutTF())
