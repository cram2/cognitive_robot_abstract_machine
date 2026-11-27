"""
Tests that a ROS camera interface uses the ROS node registered for its process.
"""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass

import pytest

from robokudo.descriptors.camera_configs.components import TfComponent
from robokudo.io import camera_interface
from semantic_digital_twin.adapters.ros.node_registry import ROSNodeRegistry
from semantic_digital_twin.exceptions import ROSNodeNotRegisteredError

# %% ROS node ownership


@dataclass(kw_only=True)
class CameraConfigWithoutTF(TfComponent):
    """
    Minimal camera config that does not request TF lookup.
    """

    tf_from: str = "camera_optical_frame"
    """
    Optical frame of the camera.
    """

    lookup_viewpoint: bool = False
    """
    Whether to look the camera pose up through TF.
    """

    interface_type: str = "ROSCameraInterface"
    """
    Selects the ROS camera interface implementation.
    """


@dataclass(kw_only=True)
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
    node_registry.register(node=registered_node)

    interface = camera_interface.ROSCameraInterface(CameraConfigWithoutTF())

    assert interface.node is registered_node


def test_ros_camera_interface_raises_when_no_node_is_registered(
    node_registry: ROSNodeRegistry,
) -> None:
    with pytest.raises(ROSNodeNotRegisteredError):
        camera_interface.ROSCameraInterface(CameraConfigWithoutTF())
