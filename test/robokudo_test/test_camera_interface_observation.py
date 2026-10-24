from __future__ import annotations
from collections.abc import Iterator
from dataclasses import dataclass, field
from unittest.mock import Mock
from geometry_msgs.msg import TransformStamped
from rclpy.time import Time as ROSClockTime
from tf2_ros import LookupException
import logging
from threading import Lock

import cv2
import numpy as np
import pytest
from builtin_interfaces.msg import Time
from py_trees.common import Status
from sensor_msgs.msg import CameraInfo

import robokudo.world as robokudo_world
from robokudo.annotators.static_camera_transform import StaticCameraTransformAnnotator
from robokudo.types.cv import ColorToDepthRatio
from robokudo.cas import CAS
from robokudo.types.image import ImageRotation
from robokudo.descriptors.camera_configs.base_camera_config import BaseCameraConfig
from robokudo.exceptions import CameraPoseMissing, InvalidCameraObservation
from robokudo.io.camera_interface import (
    CameraInterface,
    KinectCameraInterface,
    ROSCameraInterface,
)
from robokudo.io.camera_model_adapters import RosCameraModelAdapter
from robokudo.io.camera_without_depth_interface import (
    OpenCVCameraWithoutDepthInterface,
)
from robokudo.io.ros_camera_without_depth_interface import (
    ROSCameraWithoutDepthInterface,
)
from semantic_digital_twin.datastructures.camera_model import (
    CameraDistortionModel,
    CameraModality,
    PinholeCameraModel,
    RationalPolynomialCameraDistortion,
)
from semantic_digital_twin.datastructures.camera_resolution import CameraResolution
from semantic_digital_twin.datastructures.field_of_view import FieldOfView
from semantic_digital_twin.datastructures.prefixed_name import PrefixedName
from semantic_digital_twin.robots.robot_parts import Camera
from semantic_digital_twin.spatial_types import HomogeneousTransformationMatrix
from semantic_digital_twin.world import World
from semantic_digital_twin.world_description.connections import Connection6DoF
from semantic_digital_twin.world_description.world_entity import Body


@dataclass
class SynchronizedImageConfiguration:
    """
    Provide the options consumed by the RGB-D reader's frame conversion.
    """

    tf_from: str = "color_optical_frame"
    """
    Camera optical frame used when the message does not declare one.
    """

    hi_res_mode: bool = False
    """
    Whether the legacy high-resolution crop is active.
    """

    color_to_depth_ratio: ColorToDepthRatio = field(default_factory=ColorToDepthRatio)
    """
    Scale from color resolution to aligned depth resolution.
    """


@dataclass
class LiveColorCameraConfig:
    """
    Provide the options consumed by the RGB-only reader's frame conversion.
    """

    tf_from: str = "color_optical_frame"
    """
    Camera optical frame used when the message does not declare one.
    """

    rotate_image: ImageRotation | None = None
    """
    Optional rotation applied before publishing the frame to the CAS.
    """


@dataclass
class OpenCVCameraTestConfig:
    """
    Provide the options consumed by one OpenCV frame conversion.
    """

    camera_info: CameraInfo
    """
    Calibration for the configured capture resolution.
    """

    camera_frame: str = "opencv_optical_frame"
    """
    Fallback optical frame when the calibration declares no frame.
    """

    normalize_rgb: bool = False
    """
    Whether to normalize the captured color values.
    """

    depth: np.ndarray | None = None
    """
    Optional static depth image.
    """

    camera_intrinsic: object | None = None
    """
    Legacy intrinsic value replaced from the declared calibration.
    """

    color_to_depth_ratio: ColorToDepthRatio = field(default_factory=ColorToDepthRatio)
    """
    Scale from color resolution to optional depth resolution.
    """

    update_global_with_depth_parameter: bool = False
    """
    Whether to change the process-wide depth setting.
    """


@dataclass
class SeekableImageSequence:
    """
    Provide the metadata operations used after an OpenCV frame is retrieved.
    """

    frame_position: float = 1.0
    """
    Current playback frame position.
    """

    frame_count: float = 1.0
    """
    Number of frames in the source.
    """

    def get(self, property_identifier: int) -> float:
        """
        Return the requested playback metadata.

        :param property_identifier: OpenCV capture property identifier.
        :return: Stored value for the requested property.
        """
        if property_identifier == cv2.CAP_PROP_POS_FRAMES:
            return self.frame_position
        return self.frame_count

    def set(self, property_identifier: int, value: float) -> bool:
        """
        Accept a playback-position update.

        :param property_identifier: OpenCV capture property identifier.
        :param value: Requested property value.
        :return: Whether the property is supported.
        """
        if property_identifier != cv2.CAP_PROP_POS_FRAMES:
            return False
        self.frame_position = value
        return True


# %% Fixtures


@pytest.fixture
def camera_info() -> CameraInfo:
    """
    Provide a calibrated ROS camera-info message.
    """
    message = CameraInfo()
    message.header.frame_id = "color_optical_frame"
    message.width = 640
    message.height = 480
    message.k = [500.0, 0.0, 321.0, 0.0, 510.0, 239.0, 0.0, 0.0, 1.0]
    message.distortion_model = CameraDistortionModel.RATIONAL_POLYNOMIAL.value
    message.d = [0.1, -0.2, 0.01, 0.02, 0.03, 0.04, 0.05, 0.06]
    return message


# %% Camera model conversion


def test_camera_info_conversion_preserves_effective_calibration(
    camera_info: CameraInfo,
) -> None:
    """
    The semantic pinhole model represents every relevant CameraInfo value.
    """
    camera_model = RosCameraModelAdapter.from_camera_info(camera_info)

    assert camera_model.resolution == CameraResolution(width=640, height=480)
    assert camera_model.focal_length_x == camera_info.k[0]
    assert camera_model.focal_length_y == camera_info.k[4]
    assert float(camera_model.principal_point.x) == camera_info.k[2]
    assert float(camera_model.principal_point.y) == camera_info.k[5]
    assert isinstance(camera_model.distortion, RationalPolynomialCameraDistortion)
    assert camera_model.distortion.radial_coefficient_1 == camera_info.d[0]
    assert camera_model.distortion.tangential_coefficient_1 == camera_info.d[2]
    assert camera_model.distortion.to_ordered_coefficients() == tuple(camera_info.d)


def test_camera_model_conversion_preserves_ros_calibration(
    camera_info: CameraInfo,
) -> None:
    """
    The reverse conversion recreates the ROS calibration values.
    """
    camera_model = RosCameraModelAdapter.from_camera_info(camera_info)

    converted_info = RosCameraModelAdapter.to_camera_info(
        camera_model,
        frame_id=camera_info.header.frame_id,
    )

    assert converted_info.header.frame_id == camera_info.header.frame_id
    assert converted_info.width == camera_info.width
    assert converted_info.height == camera_info.height
    assert converted_info.distortion_model == camera_info.distortion_model
    assert converted_info.d == camera_info.d
    np.testing.assert_allclose(converted_info.k, camera_info.k)


# %% Semantic camera resolution


def _mounted_camera_bodies(world: World) -> tuple[Body, Body]:
    """
    Create a world-anchored camera body beneath a robot body.
    """
    reference = Body(name=PrefixedName(name="map"))
    robot_body = Body(name=PrefixedName(name="robot_head"))
    camera_body = Body(name=PrefixedName(name="color_optical_frame"))
    with world.modify_world():
        for body in (reference, robot_body, camera_body):
            world.add_body(body)
        for parent, child in (
            (reference, robot_body),
            (robot_body, camera_body),
        ):
            world.add_connection(
                Connection6DoF.create_with_dofs(parent=parent, child=child, world=world)
            )
    return reference, camera_body


def test_ros_frame_alias_reuses_robot_optical_body(
    runtime_world: World,
    camera_info: CameraInfo,
) -> None:
    """
    A leading slash in CameraInfo does not duplicate a robot optical body.
    """
    _, camera_body = _mounted_camera_bodies(runtime_world)
    camera_info.header.frame_id = "/color_optical_frame"
    interface = CameraInterface(BaseCameraConfig(interface_type="test"))
    cas = CAS()

    interface.store_camera_observation(
        cas=cas,
        camera_model=RosCameraModelAdapter.from_camera_info(camera_info),
        camera_frame=camera_info.header.frame_id,
        timestamp_nanoseconds=123,
        modalities=(CameraModality.COLOR,),
        world_T_camera=None,
    )

    assert cas.require_camera_observation().camera.root is camera_body
    assert len(runtime_world.bodies) == 3


def test_tf_binding_preserves_robot_camera_mount(runtime_world: World) -> None:
    """
    A sampled TF pose does not add a parent to a robot optical body.
    """
    reference, camera_body = _mounted_camera_bodies(runtime_world)
    original_parent = camera_body.parent_connection
    numeric_pose = HomogeneousTransformationMatrix.from_xyz_quaternion(
        pos_x=0.4,
        pos_y=0.2,
        pos_z=1.3,
        quat_x=0.0,
        quat_y=0.0,
        quat_z=0.0,
        quat_w=1.0,
    )

    pose = CameraInterface.bind_world_T_camera(
        world_frame="map",
        camera_frame="/color_optical_frame",
        world_T_camera=numeric_pose,
    )

    assert pose.reference_frame is reference
    assert pose.child_frame is camera_body
    assert camera_body.parent_connection is original_parent
    assert len(runtime_world.connections) == 2
    np.testing.assert_allclose(pose.to_np(), numeric_pose.to_np())


def test_mismatched_pose_child_is_rejected_at_acquisition(
    runtime_world: World,
    camera_info: CameraInfo,
) -> None:
    """
    Do not label another physical frame's pose as the image camera pose.
    """
    reference, camera_body = _mounted_camera_bodies(runtime_world)
    other_frame = camera_body.parent_connection.parent
    pose = HomogeneousTransformationMatrix(
        reference_frame=reference, child_frame=other_frame
    )
    interface = CameraInterface(BaseCameraConfig(interface_type="test"))

    with pytest.raises(InvalidCameraObservation):
        interface.store_camera_observation(
            cas=CAS(),
            camera_model=RosCameraModelAdapter.from_camera_info(camera_info),
            camera_frame=camera_info.header.frame_id,
            timestamp_nanoseconds=123,
            modalities=(CameraModality.COLOR,),
            world_T_camera=pose,
        )
    assert runtime_world.get_semantic_annotations_by_type(Camera) == []


def test_static_pose_uses_observation_camera_without_reparenting_robot(
    runtime_world: World,
    camera_info: CameraInfo,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """
    A static numeric pose overrides the observation, not the robot chain.
    """
    reference, camera_body = _mounted_camera_bodies(runtime_world)
    camera_info.header.frame_id = "/color_optical_frame"
    cas = CAS()
    CameraInterface(BaseCameraConfig(interface_type="test")).store_camera_observation(
        cas=cas,
        camera_model=RosCameraModelAdapter.from_camera_info(camera_info),
        camera_frame=camera_info.header.frame_id,
        timestamp_nanoseconds=123,
        modalities=(CameraModality.COLOR,),
        world_T_camera=None,
    )
    descriptor = StaticCameraTransformAnnotator.Descriptor()
    descriptor.parameters.world_frame = "map"
    descriptor.parameters.world_T_camera = (
        HomogeneousTransformationMatrix.from_xyz_quaternion(
            pos_x=0.4,
            pos_y=0.2,
            pos_z=1.3,
            quat_x=0.0,
            quat_y=0.0,
            quat_z=0.0,
            quat_w=1.0,
        )
    )
    annotator = StaticCameraTransformAnnotator(descriptor=descriptor)
    monkeypatch.setattr(annotator, "get_cas", lambda: cas)
    original_parent = camera_body.parent_connection

    assert annotator.update() is Status.SUCCESS

    pose = cas.require_camera_observation().world_T_camera_or_raise()
    assert pose.reference_frame is reference
    assert pose.child_frame is camera_body
    assert camera_body.parent_connection is original_parent
    assert len(runtime_world.connections) == 2
    np.testing.assert_allclose(
        pose.to_np(), descriptor.parameters.world_T_camera.to_np()
    )


def test_static_pose_creates_isolated_world_reference(
    runtime_world: World,
    camera_info: CameraInfo,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """
    A static pose anchors a standalone camera without a robot model.
    """
    cas = CAS()
    CameraInterface(BaseCameraConfig(interface_type="test")).store_camera_observation(
        cas=cas,
        camera_model=RosCameraModelAdapter.from_camera_info(camera_info),
        camera_frame=camera_info.header.frame_id,
        timestamp_nanoseconds=123,
        modalities=(CameraModality.COLOR,),
        world_T_camera=None,
    )
    camera_root = cas.require_camera_observation().camera.root
    numeric_pose = HomogeneousTransformationMatrix.from_xyz_quaternion(
        pos_x=0.4,
        pos_y=0.2,
        pos_z=1.3,
        quat_x=0.0,
        quat_y=0.0,
        quat_z=0.0,
        quat_w=1.0,
    )
    descriptor = StaticCameraTransformAnnotator.Descriptor()
    descriptor.parameters.world_T_camera = numeric_pose
    annotator = StaticCameraTransformAnnotator(descriptor=descriptor)
    monkeypatch.setattr(annotator, "get_cas", lambda: cas)

    assert annotator.update() is Status.SUCCESS

    pose = cas.require_camera_observation().world_T_camera_or_raise()
    assert pose.child_frame is camera_root
    assert runtime_world.root is pose.reference_frame
    assert camera_root.parent_connection.parent is pose.reference_frame
    assert len(runtime_world.bodies) == 2
    np.testing.assert_allclose(
        camera_root.parent_connection.origin.to_np(), numeric_pose.to_np()
    )


def test_camera_observation_reuses_camera_rooted_in_stream_frame(
    runtime_world: World,
    camera_info: CameraInfo,
) -> None:
    """
    A live stream reuses the semantic camera mounted on its optical frame.
    """
    camera_body = Body(name=PrefixedName(name=camera_info.header.frame_id))
    native_model = PinholeCameraModel.from_field_of_view(
        resolution=CameraResolution(width=1280, height=960),
        field_of_view=FieldOfView(),
    )
    camera = Camera(
        name=PrefixedName(name="head_camera"),
        root=camera_body,
        camera_model=native_model,
        modalities=(CameraModality.COLOR,),
    )
    with runtime_world.modify_world():
        runtime_world.add_body(camera_body)
        runtime_world.add_semantic_annotation(camera)
    cas = CAS()
    world_T_camera = camera_body.global_transform
    interface = CameraInterface(BaseCameraConfig(interface_type="test"))

    interface.store_camera_observation(
        cas=cas,
        camera_model=RosCameraModelAdapter.from_camera_info(camera_info),
        camera_frame=camera_info.header.frame_id,
        timestamp_nanoseconds=123,
        modalities=(CameraModality.COLOR,),
        world_T_camera=world_T_camera,
    )

    assert cas.camera_observation.camera is camera
    assert cas.camera_observation.effective_camera_model.resolution == CameraResolution(
        width=camera_info.width,
        height=camera_info.height,
    )
    assert cas.camera_observation.timestamp_nanoseconds == 123
    assert cas.camera_observation.world_T_camera is world_T_camera


def test_camera_observation_creates_standalone_camera_without_world_pose(
    runtime_world: World,
    camera_info: CameraInfo,
) -> None:
    """
    A live stream can create its semantic camera without a robot or TF pose.
    """
    interface = CameraInterface(BaseCameraConfig(interface_type="test"))
    cas = CAS()

    interface.store_camera_observation(
        cas=cas,
        camera_model=RosCameraModelAdapter.from_camera_info(camera_info),
        camera_frame=camera_info.header.frame_id,
        timestamp_nanoseconds=456,
        modalities=(CameraModality.COLOR,),
        world_T_camera=None,
    )

    cameras = runtime_world.get_semantic_annotations_by_type(Camera)
    assert cameras == [cas.camera_observation.camera]
    assert cas.camera_observation.camera.root.name.name == camera_info.header.frame_id
    assert cas.camera_observation.world_T_camera is None
    assert cas.camera_observation.effective_camera_model.distortion.to_ordered_coefficients() == tuple(
        camera_info.d
    )


def test_camera_observation_connects_new_camera_in_rooted_world_without_pose(
    runtime_world: World,
    camera_info: CameraInfo,
) -> None:
    """
    An unlocalized camera preserves the world's single-root invariant.
    """
    world_root = Body(name=PrefixedName(name="world"))
    with runtime_world.modify_world():
        runtime_world.add_body(world_root)
    interface = CameraInterface(BaseCameraConfig(interface_type="test"))
    cas = CAS()

    interface.store_camera_observation(
        cas=cas,
        camera_model=RosCameraModelAdapter.from_camera_info(camera_info),
        camera_frame=camera_info.header.frame_id,
        timestamp_nanoseconds=456,
        modalities=(CameraModality.COLOR,),
        world_T_camera=None,
    )

    assert runtime_world.root is world_root
    assert cas.require_camera_observation().world_T_camera is None


# %% Live reader contracts


def test_live_reader_rejects_tf_source_for_a_different_optical_frame(
    runtime_world: World,
    camera_info: CameraInfo,
) -> None:
    """
    Reject a configured TF source that is not the image optical frame.
    """
    interface = object.__new__(KinectCameraInterface)
    interface.camera_config = SynchronizedImageConfiguration()
    interface._has_new_data = True
    interface.lookup_viewpoint = True
    interface.tf_from = "another_optical_frame"
    interface.color = np.zeros(
        (camera_info.height, camera_info.width, 3), dtype=np.uint8
    )
    interface.depth = np.zeros((camera_info.height, camera_info.width), dtype=np.uint16)
    interface.camera_info = camera_info
    interface.timestamp = Time(sec=10, nanosec=20)
    interface.lock = Lock()

    with pytest.raises(InvalidCameraObservation):
        interface.set_data(CAS())

    assert runtime_world.bodies == []


def test_live_reader_uses_robot_optical_body_for_tf_pose(
    runtime_world: World,
    camera_info: CameraInfo,
) -> None:
    """
    The live reader does not add a second robot camera mount.
    """
    reference, camera_body = _mounted_camera_bodies(runtime_world)
    camera_info.header.frame_id = "/color_optical_frame"
    original_parent = camera_body.parent_connection
    interface = object.__new__(KinectCameraInterface)
    interface.camera_config = SynchronizedImageConfiguration()
    interface._has_new_data = True
    interface.lookup_viewpoint = True
    interface.tf_from = "color_optical_frame"
    interface.tf_to = "map"
    interface._world_T_camera = HomogeneousTransformationMatrix.from_xyz_quaternion(
        pos_x=0.4, pos_y=0.2, pos_z=1.3
    )
    interface.color = np.zeros(
        (camera_info.height, camera_info.width, 3), dtype=np.uint8
    )
    interface.depth = np.zeros((camera_info.height, camera_info.width), dtype=np.uint16)
    interface.camera_info = camera_info
    interface.timestamp = Time(sec=10, nanosec=20)
    interface.lock = Lock()
    cas = CAS()

    interface.set_data(cas)

    observation = cas.require_camera_observation()
    assert observation.camera.root is camera_body
    assert observation.world_T_camera.reference_frame is reference
    assert observation.world_T_camera.child_frame is camera_body
    assert camera_body.parent_connection is original_parent
    assert len(runtime_world.connections) == 2


def test_live_reader_creates_isolated_camera_mount_for_tf_pose(
    runtime_world: World,
    camera_info: CameraInfo,
) -> None:
    """
    An isolated live camera gets a world reference and sampled mount.
    """
    interface = object.__new__(KinectCameraInterface)
    interface.camera_config = SynchronizedImageConfiguration()
    interface._has_new_data = True
    interface.lookup_viewpoint = True
    interface.tf_from = "color_optical_frame"
    interface.tf_to = "map"
    interface._world_T_camera = HomogeneousTransformationMatrix.from_xyz_quaternion(
        pos_x=0.4, pos_y=0.2, pos_z=1.3
    )
    interface.color = np.zeros(
        (camera_info.height, camera_info.width, 3), dtype=np.uint8
    )
    interface.depth = np.zeros((camera_info.height, camera_info.width), dtype=np.uint16)
    interface.camera_info = camera_info
    interface.timestamp = Time(sec=10, nanosec=20)
    interface.lock = Lock()
    cas = CAS()

    interface.set_data(cas)

    observation = cas.require_camera_observation()
    assert runtime_world.root is observation.world_T_camera.reference_frame
    assert observation.camera.root is observation.world_T_camera.child_frame
    assert observation.camera.root.parent_connection.parent is runtime_world.root
    assert len(runtime_world.bodies) == 2
    np.testing.assert_allclose(
        observation.camera.root.parent_connection.origin.to_np(),
        observation.world_T_camera.to_np(),
    )


def test_rgbd_reader_stores_camera_observation(
    runtime_world: World,
    camera_info: CameraInfo,
) -> None:
    """
    The live RGB-D reader publishes semantic metadata with each CAS frame.
    """
    interface = object.__new__(KinectCameraInterface)
    interface.camera_config = SynchronizedImageConfiguration()
    interface._has_new_data = True
    interface.lookup_viewpoint = False
    interface.color = np.zeros(
        (camera_info.height, camera_info.width, 3), dtype=np.uint8
    )
    interface.depth = np.zeros((camera_info.height, camera_info.width), dtype=np.uint16)
    interface.camera_info = camera_info
    interface.color_to_depth_ratio = None
    interface.timestamp = Time(sec=10, nanosec=20)
    interface.lock = Lock()
    cas = CAS()

    interface.set_data(cas)

    assert cas.camera_observation.camera.root.name.name == camera_info.header.frame_id
    assert cas.camera_observation.camera.modalities == (
        CameraModality.COLOR,
        CameraModality.DEPTH,
    )
    assert cas.camera_observation.effective_camera_model.resolution == CameraResolution(
        width=camera_info.width,
        height=camera_info.height,
    )
    assert cas.camera_observation.timestamp_nanoseconds == 10_000_000_020
    assert not any(isinstance(view, CameraInfo) for view in cas.views.values())


def test_rgb_only_reader_stores_camera_observation(
    runtime_world: World,
    camera_info: CameraInfo,
) -> None:
    """
    The live RGB-only reader publishes the same semantic metadata contract.
    """
    interface = object.__new__(ROSCameraWithoutDepthInterface)
    interface.camera_config = LiveColorCameraConfig()
    interface._has_new_data = True
    interface.lookup_viewpoint = False
    interface.color = np.zeros(
        (camera_info.height, camera_info.width, 3), dtype=np.uint8
    )
    interface.camera_info = camera_info
    interface.timestamp = Time(sec=30, nanosec=40)
    interface.lock = Lock()
    cas = CAS()

    interface.set_data(cas)

    assert cas.camera_observation.camera.root.name.name == camera_info.header.frame_id
    assert cas.camera_observation.camera.modalities == (CameraModality.COLOR,)
    assert cas.camera_observation.effective_camera_model.resolution == CameraResolution(
        width=camera_info.width,
        height=camera_info.height,
    )
    assert cas.camera_observation.timestamp_nanoseconds == 30_000_000_040
    assert not any(isinstance(view, CameraInfo) for view in cas.views.values())


def test_rgb_only_rotation_updates_effective_calibration(
    runtime_world: World,
    camera_info: CameraInfo,
) -> None:
    """
    A rotated frame publishes calibration matching the delivered image.
    """
    original_width = camera_info.width
    original_height = camera_info.height
    original_focal_length_x = camera_info.k[0]
    original_focal_length_y = camera_info.k[4]
    original_principal_point_x = camera_info.k[2]
    original_principal_point_y = camera_info.k[5]
    original_tangential_x = camera_info.d[2]
    original_tangential_y = camera_info.d[3]
    interface = object.__new__(ROSCameraWithoutDepthInterface)
    interface.camera_config = LiveColorCameraConfig(
        rotate_image=ImageRotation.COUNTERCLOCKWISE_90
    )
    interface._has_new_data = True
    interface.lookup_viewpoint = False
    interface.color = np.zeros(
        (camera_info.height, camera_info.width, 3), dtype=np.uint8
    )
    interface.camera_info = camera_info
    interface.timestamp = Time(sec=50, nanosec=60)
    interface.lock = Lock()
    cas = CAS()

    interface.set_data(cas)

    expected_resolution = CameraResolution(
        width=original_height,
        height=original_width,
    )
    expected_intrinsic_matrix = np.array(
        [
            [original_focal_length_y, 0.0, original_principal_point_y],
            [
                0.0,
                original_focal_length_x,
                original_width - 1.0 - original_principal_point_x,
            ],
            [0.0, 0.0, 1.0],
        ]
    )
    assert cas.color_image.shape[:2] == (
        expected_resolution.height,
        expected_resolution.width,
    )
    np.testing.assert_allclose(
        cas.camera_observation.effective_camera_model.intrinsic_matrix,
        expected_intrinsic_matrix,
    )
    assert (
        cas.camera_observation.effective_camera_model.resolution == expected_resolution
    )
    assert (
        float(cas.camera_observation.effective_camera_model.principal_point.y)
        == expected_intrinsic_matrix[1, 2]
    )
    distortion = cas.camera_observation.effective_camera_model.distortion
    assert isinstance(distortion, RationalPolynomialCameraDistortion)
    assert distortion.tangential_coefficient_1 == -original_tangential_y
    assert distortion.tangential_coefficient_2 == original_tangential_x
    assert not any(isinstance(view, CameraInfo) for view in cas.views.values())


def test_rgbd_high_resolution_crop_updates_effective_resolution(
    runtime_world: World,
    camera_info: CameraInfo,
) -> None:
    """
    The high-resolution crop publishes its delivered color-image dimensions.
    """
    camera_info.width = 1280
    camera_info.height = 1024
    camera_info.k = [900.0, 0.0, 640.0, 0.0, 910.0, 512.0, 0.0, 0.0, 1.0]
    interface = object.__new__(KinectCameraInterface)
    interface.camera_config = SynchronizedImageConfiguration(hi_res_mode=True)
    interface._has_new_data = True
    interface.lookup_viewpoint = False
    interface.color = np.zeros((1024, 1280, 3), dtype=np.uint8)
    interface.depth = np.zeros((480, 640), dtype=np.uint16)
    interface.camera_info = camera_info
    interface.color_to_depth_ratio = None
    interface.timestamp = Time(sec=70, nanosec=80)
    interface.lock = Lock()
    cas = CAS()

    interface.set_data(cas)

    expected_resolution = CameraResolution(width=1280, height=960)
    assert cas.color_image.shape[:2] == (
        expected_resolution.height,
        expected_resolution.width,
    )
    assert (
        cas.camera_observation.effective_camera_model.resolution == expected_resolution
    )
    assert (
        float(cas.camera_observation.effective_camera_model.principal_point.y)
        == camera_info.k[5]
    )
    assert not any(isinstance(view, CameraInfo) for view in cas.views.values())


def test_opencv_reader_stores_camera_observation(
    runtime_world: World,
    camera_info: CameraInfo,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """
    The physical OpenCV reader publishes the common camera metadata contract.
    """
    interface = object.__new__(OpenCVCameraWithoutDepthInterface)
    interface.camera_config = OpenCVCameraTestConfig(camera_info=camera_info)
    interface.video_capture = SeekableImageSequence()
    interface.device_driver_flag = 0
    interface.stream_type = 1
    interface._loop_counter = 0
    interface._backup_color = np.zeros(
        (camera_info.height, camera_info.width, 3), dtype=np.uint8
    )
    interface._has_new_data = True
    interface.rk_logger = logging.getLogger("robokudo-opencv-camera-test")
    cas = CAS()
    timestamp_nanoseconds = 123_456_789
    monkeypatch.setattr(
        "robokudo.io.camera_without_depth_interface.time.time_ns",
        lambda: timestamp_nanoseconds,
    )

    interface.set_data(cas)

    assert cas.camera_observation.camera.root.name.name == camera_info.header.frame_id
    assert cas.camera_observation.camera.modalities == (CameraModality.COLOR,)
    assert cas.camera_observation.effective_camera_model.resolution == CameraResolution(
        width=camera_info.width,
        height=camera_info.height,
    )
    assert cas.camera_observation.world_T_camera is None
    assert cas.camera_observation.timestamp_nanoseconds == timestamp_nanoseconds
    assert not any(isinstance(view, CameraInfo) for view in cas.views.values())


# %% Camera-root world binding


def test_tf_binding_attaches_existing_camera_root_to_new_reference(
    runtime_world: World,
) -> None:
    """
    An existing camera root receives one mount when its world reference is created.
    """
    camera_body = Body(name=PrefixedName(name="standalone_camera"))
    with runtime_world.modify_world():
        runtime_world.add_body(camera_body)
    numeric_pose = HomogeneousTransformationMatrix()

    pose = CameraInterface.bind_world_T_camera(
        world_frame="map",
        camera_frame=str(camera_body.name),
        world_T_camera=numeric_pose,
    )

    assert pose.child_frame is camera_body
    assert runtime_world.root is pose.reference_frame
    assert camera_body.parent_connection.parent is pose.reference_frame
    assert len(runtime_world.connections) == 1
    np.testing.assert_array_equal(pose.to_np(), numeric_pose.to_np())


# %% Controlled transform availability
@pytest.fixture
def transform_camera() -> ROSCameraInterface:
    """
    Provide an interface with a controlled TF buffer and no subscriptions.
    """
    camera = ROSCameraInterface.__new__(ROSCameraInterface)
    camera.lookup_viewpoint = True
    camera.tf_from = "camera_optical_frame"
    camera.tf_to = "map"
    camera.tf_buffer = Mock()
    camera._world_T_camera = None
    return camera


def test_unavailable_transform_returns_false(
    transform_camera: ROSCameraInterface,
) -> None:
    """
    An unavailable sample is reported without attempting its lookup.
    """
    transform_camera.tf_buffer.can_transform.return_value = False
    assert transform_camera.lookup_transform() is False
    transform_camera.tf_buffer.lookup_transform.assert_not_called()


def test_available_transform_updates_pose(transform_camera: ROSCameraInterface) -> None:
    """
    A sampled pose is cached as a homogeneous transformation matrix.
    """
    sample = TransformStamped()
    sample.transform.translation.x = 1.0
    sample.transform.translation.y = 2.0
    sample.transform.translation.z = 3.0
    sample.transform.rotation.w = 1.0
    transform_camera.tf_buffer.can_transform.return_value = True
    transform_camera.tf_buffer.lookup_transform.return_value = sample
    timestamp = ROSClockTime(nanoseconds=123)
    assert transform_camera.lookup_transform(timestamp) is True
    expected_pose = HomogeneousTransformationMatrix.from_xyz_quaternion(
        pos_x=sample.transform.translation.x,
        pos_y=sample.transform.translation.y,
        pos_z=sample.transform.translation.z,
        quat_x=sample.transform.rotation.x,
        quat_y=sample.transform.rotation.y,
        quat_z=sample.transform.rotation.z,
        quat_w=sample.transform.rotation.w,
    )
    np.testing.assert_array_equal(
        transform_camera._world_T_camera.to_np(), expected_pose.to_np()
    )
    transform_camera.tf_buffer.lookup_transform.assert_called_once_with(
        target_frame=transform_camera.tf_to,
        source_frame=transform_camera.tf_from,
        time=timestamp,
    )


def test_disabled_lookup_does_not_access_buffer(
    transform_camera: ROSCameraInterface,
) -> None:
    """
    A camera with pose lookup disabled requires no TF sample.
    """
    transform_camera.lookup_viewpoint = False
    assert transform_camera.lookup_transform() is True
    transform_camera.tf_buffer.can_transform.assert_not_called()
    transform_camera.tf_buffer.lookup_transform.assert_not_called()


def test_transform_lookup_errors_propagate(
    transform_camera: ROSCameraInterface,
) -> None:
    """
    A buffer failure after its availability check reaches the caller.
    """
    transform_camera.tf_buffer.can_transform.return_value = True
    transform_camera.tf_buffer.lookup_transform.side_effect = LookupException
    with pytest.raises(LookupException):
        transform_camera.lookup_transform()


# %% Isolated semantic world
@pytest.fixture
def runtime_world() -> Iterator[World]:
    """
    Install an isolated semantic world and restore the previous world afterwards.
    """
    previous_world = robokudo_world.world_instance()
    world = World()
    robokudo_world.set_world(world)
    yield world
    robokudo_world.set_world(previous_world)


# %% Missing sampled camera pose


def test_camera_pose_requires_a_sampled_transform(
    transform_camera: ROSCameraInterface,
) -> None:
    """A camera pose cannot be bound before its first successful TF lookup."""
    with pytest.raises(CameraPoseMissing):
        transform_camera.world_T_camera_from_tf(Time())
