"""
Contracts for interchangeable ROS stereo depth producers.
"""

from __future__ import annotations

# %% Imports

from copy import deepcopy
from unittest.mock import Mock

import numpy as np
import pytest
from rclpy.node import Node
from rclpy.time import Time
from sensor_msgs.msg import CameraInfo

from robokudo.cas import CAS, CASViews
from robokudo.descriptors.camera_configs.config_stereo import StereoCameraConfig
from robokudo.descriptors.camera_configs.config_foundation_stereo import (
    FoundationStereoCameraConfig,
)
from robokudo.descriptors.camera_configs.config_zed import ZedCameraConfig
from robokudo.descriptors.factories.cr_descriptor_factory import (
    CollectionReaderDescriptorFactory,
)
from robokudo.exceptions import InvalidCameraObservation
from robokudo.io.exceptions import InvalidStereoCameraConfiguration
from robokudo.io.rectified_depth_camera_interface import (
    ColorDepthImages,
    RectifiedDepthCameraInterface,
    RectifiedDepthFrame,
)
from robokudo.utils.cv_bridge_workaround import CVBridgeWorkaround

# %% Acquisitions


@pytest.fixture
def stereo_frame(node: Node) -> RectifiedDepthFrame:
    """
    Provide aligned images whose raw and rectified intrinsics differ.
    """
    bridge = CVBridgeWorkaround()
    color = bridge.cv2_to_imgmsg(np.zeros((2, 3, 4), dtype=np.uint8), "bgra8")
    depth = bridge.cv2_to_imgmsg(
        np.array([[1.5, np.nan, np.inf], [0, -1, 20]], dtype=np.float32), "32FC1"
    )
    calibration = CameraInfo(width=3, height=2)
    calibration.k = [200.0, 0.0, 1.0, 0.0, 200.0, 1.0, 0.0, 0.0, 1.0]
    calibration.p = [100.0, 0.0, 1.0, 0.0, 0.0, 100.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0]
    calibration.d = [0.1] * 5
    for message in (color, depth, calibration):
        message.header.frame_id = "left_optical_frame"
        message.header.stamp = node.get_clock().now().to_msg()
    depth.header = deepcopy(color.header)
    calibration.header = deepcopy(color.header)
    return RectifiedDepthFrame(
        images=ColorDepthImages(color=color, depth=depth), calibration=calibration
    )


def test_rectified_depth_uses_projection_and_millimeters(
    node: Node, stereo_frame: RectifiedDepthFrame
) -> None:
    """
    Consumers receive rectified calibration and sanitized millimeter depth.
    """
    interface = RectifiedDepthCameraInterface(
        camera_config=StereoCameraConfig(lookup_viewpoint=False), node=node
    )
    original = deepcopy(stereo_frame.calibration)
    interface.receive_frame(stereo_frame.images, stereo_frame.calibration)
    cas = CAS()
    assert interface.set_data(cas) is True
    np.testing.assert_array_equal(
        cas.get(CASViews.DEPTH_IMAGE), [[1500, 0, 0], [0, 0, 0]]
    )
    assert cas.get(CASViews.COLOR_IMAGE).shape == (2, 3, 3)
    assert cas.camera_observation.effective_camera_model.focal_length_x == original.p[0]
    assert stereo_frame.calibration == original
    assert interface.has_new_data() is False


def test_rectified_depth_rejects_mismatched_acquisitions(
    stereo_frame: RectifiedDepthFrame,
) -> None:
    """
    Depth from another acquisition cannot describe the current image.
    """
    stereo_frame.images.depth.header.stamp.sec += 1
    with pytest.raises(InvalidCameraObservation):
        stereo_frame.validate(optical_frame="left_optical_frame")


def test_rectified_depth_keeps_newest_frame(
    node: Node, stereo_frame: RectifiedDepthFrame
) -> None:
    """
    A repeated acquisition cannot overwrite pending newer data.
    """
    interface = RectifiedDepthCameraInterface(
        camera_config=StereoCameraConfig(lookup_viewpoint=False), node=node
    )
    interface.receive_frame(stereo_frame.images, stereo_frame.calibration)
    interface.receive_frame(stereo_frame.images, stereo_frame.calibration)
    assert interface.statistics.obsolete_frames == 1
    assert (
        interface.last_timestamp
        == Time.from_msg(stereo_frame.images.header.stamp).nanoseconds
    )


def test_rectified_depth_records_configured_static_pose(
    node: Node, stereo_frame: RectifiedDepthFrame
) -> None:
    """
    A fixed camera observation binds its configured pose to the optical root.
    """
    configuration = StereoCameraConfig(
        lookup_viewpoint=False, static_camera_transform_enabled=True
    )
    interface = RectifiedDepthCameraInterface(camera_config=configuration, node=node)
    interface.receive_frame(stereo_frame.images, stereo_frame.calibration)
    cas = CAS()
    assert interface.set_data(cas) is True
    observation = cas.camera_observation
    np.testing.assert_array_equal(
        observation.world_T_camera, configuration.static_world_T_camera
    )
    assert observation.world_T_camera.child_frame is observation.camera.root
    assert (
        observation.world_T_camera.reference_frame.name.name
        == configuration.static_world_frame
    )


# %% Synchronization and delayed consumption


def test_rectified_depth_matches_delayed_depth_exactly(
    node: Node, stereo_frame: RectifiedDepthFrame
) -> None:
    """
    Only depth with the original color acquisition stamp completes a frame.
    """
    interface = RectifiedDepthCameraInterface(
        camera_config=StereoCameraConfig(lookup_viewpoint=False), node=node
    )
    other_depth = deepcopy(stereo_frame.images.depth)
    other_depth.header.stamp.sec -= 1
    interface.color_subscriber.signalMessage(stereo_frame.images.color)
    interface.depth_subscriber.signalMessage(other_depth)
    interface.camera_info_subscriber.signalMessage(stereo_frame.calibration)
    assert interface.has_new_data() is False
    interface.depth_subscriber.signalMessage(stereo_frame.images.depth)
    assert interface.has_new_data() is True
    assert interface.pending_frame.images.depth is stereo_frame.images.depth


def test_rectified_depth_accepts_old_acquisition(
    node: Node, stereo_frame: RectifiedDepthFrame
) -> None:
    """
    A historical acquisition retains its original timestamp when delivered.
    """
    for message in (
        stereo_frame.images.color,
        stereo_frame.images.depth,
        stereo_frame.calibration,
    ):
        message.header.stamp = Time().to_msg()
    interface = RectifiedDepthCameraInterface(
        camera_config=StereoCameraConfig(lookup_viewpoint=False), node=node
    )
    interface.receive_frame(stereo_frame.images, stereo_frame.calibration)
    assert interface.has_new_data() is True
    cas = CAS()
    assert interface.set_data(cas) is True
    assert cas.camera_observation.timestamp_nanoseconds == Time().nanoseconds


def test_rectified_depth_accepts_delayed_consumption(
    node: Node, stereo_frame: RectifiedDepthFrame, monkeypatch: pytest.MonkeyPatch
) -> None:
    """
    Waiting before consumption preserves a valid pending acquisition.
    """
    interface = RectifiedDepthCameraInterface(
        camera_config=StereoCameraConfig(lookup_viewpoint=False), node=node
    )
    interface.receive_frame(stereo_frame.images, stereo_frame.calibration)
    clock = Mock()
    clock.now.return_value = Time(
        nanoseconds=interface.last_timestamp,
        seconds=60,
    )
    monkeypatch.setattr(node, "get_clock", lambda: clock)
    cas = CAS()
    assert interface.set_data(cas) is True
    assert cas.camera_observation.timestamp_nanoseconds == interface.last_timestamp
    assert interface.pending_frame is None


def test_rectified_depth_waits_for_acquisition_transform(
    node: Node, stereo_frame: RectifiedDepthFrame, monkeypatch: pytest.MonkeyPatch
) -> None:
    """
    Missing acquisition-time TF drops a frame instead of using a later pose.
    """
    interface = RectifiedDepthCameraInterface(
        camera_config=StereoCameraConfig(), node=node
    )
    lookup = Mock(return_value=False)
    monkeypatch.setattr(interface, "lookup_transform", lookup)
    interface.receive_frame(stereo_frame.images, stereo_frame.calibration)
    assert interface.set_data(CAS()) is False
    lookup.assert_called_once_with(Time.from_msg(stereo_frame.images.header.stamp))
    assert interface.statistics.missing_transforms == 1


# %% Source selection


@pytest.mark.parametrize(
    "configuration_type",
    [StereoCameraConfig, FoundationStereoCameraConfig, ZedCameraConfig],
)
def test_stereo_sources_share_interface(
    node: Node, configuration_type: type[StereoCameraConfig]
) -> None:
    """
    All source presets select the same backend-independent acquisition code.
    """
    descriptor = CollectionReaderDescriptorFactory.create_descriptor(
        configuration_type.registry_name, lookup_viewpoint=False
    )
    assert type(descriptor.parameters.camera_interface) is RectifiedDepthCameraInterface
    assert type(descriptor.parameters.camera_config) is configuration_type


def test_stereo_configuration_rejects_conflicting_pose_sources() -> None:
    """
    A configured static pose cannot compete with acquisition-time TF.
    """
    with pytest.raises(InvalidStereoCameraConfiguration):
        StereoCameraConfig(static_camera_transform_enabled=True)


@pytest.mark.parametrize(
    "mismatch", ["frame", "dimensions", "encoding", "projection", "binning"]
)
def test_rectified_depth_rejects_incompatible_geometry(
    stereo_frame: RectifiedDepthFrame, mismatch: str
) -> None:
    """
    Unsupported geometry is rejected before image publication.
    """
    if mismatch == "frame":
        stereo_frame.images.depth.header.frame_id = "right_optical_frame"
    elif mismatch == "dimensions":
        stereo_frame.images.depth.width += 1
    elif mismatch == "encoding":
        stereo_frame.images.depth.encoding = "16UC1"
    elif mismatch == "projection":
        stereo_frame.calibration.p[3] = -1
    else:
        stereo_frame.calibration.binning_x = 2
    with pytest.raises(InvalidCameraObservation):
        stereo_frame.validate(optical_frame="left_optical_frame")
