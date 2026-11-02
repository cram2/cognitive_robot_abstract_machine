"""
Camera interface module for RoboKudo.

This module provides base classes and implementations for interfacing with
various camera types in RoboKudo. It supports:

* ROS camera interfaces (raw and compressed)
* Kinect-style RGB-D cameras
* Camera calibration handling
* Transform lookups
* Synchronized data acquisition
* Thread-safe operation

The module handles:

* RGB and depth image acquisition
* Camera calibration information
* Camera-to-world transforms
* Data synchronization
* Format conversions
"""

from __future__ import annotations

# %% Imports
import logging
import struct
from threading import Lock

import builtin_interfaces.msg
import cv2
import message_filters
import numpy as np
import rclpy
from message_filters import ApproximateTimeSynchronizer, Subscriber
from rclpy.duration import Duration
from rclpy.node import Node
from rclpy.time import Time
from sensor_msgs.msg import CompressedImage, CameraInfo, Image
from tf2_ros import Buffer
from typing_extensions import Any, TYPE_CHECKING

from krrood.ormatic.utils import classproperty
from robokudo.cas import CASViews, CAS
from robokudo.defs import PACKAGE_NAME
from robokudo.exceptions import (
    CameraDataMissing,
    CameraPoseMissing,
    InvalidCameraObservation,
)
from robokudo.io.exceptions import CameraAnnotationAmbiguous
from robokudo.io.camera_model_adapters import RosCameraModelAdapter
from robokudo.io.tf_listener_proxy import TFListenerProxy
from robokudo.types.camera import CameraObservation
from robokudo.utils.cv_bridge_workaround import CVBridgeWorkaround
from robokudo.utils.ros_frames import find_bodies_for_ros_frame, normalize_ros_frame_id
from robokudo.world import (
    init_world_entity_tracker_from_world,
    world_instance,
)
from semantic_digital_twin.adapters.ros.node_registry import ROSNodeRegistry
from semantic_digital_twin.datastructures.camera_model import (
    CameraModality,
    PinholeCameraModel,
)
from semantic_digital_twin.datastructures.prefixed_name import PrefixedName
from semantic_digital_twin.robots.robot_parts import Camera
from semantic_digital_twin.spatial_types import HomogeneousTransformationMatrix
from semantic_digital_twin.world import World
from semantic_digital_twin.world_description.connections import Connection6DoF
from semantic_digital_twin.world_description.world_entity import Body

if TYPE_CHECKING:
    import numpy.typing as npt

    from robokudo.types.cv import ColorToDepthRatio


# %% Camera definitions


class CameraInterface(object):
    """
    Base class for all camera interfaces in RoboKudo.

    This class defines the basic interface that all camera implementations must provide.
    It handles configuration and data availability tracking.
    """

    @classproperty
    def standalone_mount_prefix(cls) -> str:
        """
        Connection-name prefix reserved for camera bodies created by RoboKudo.
        """
        return "robokudo_camera_mount_"

    def __init__(self, camera_config: Any) -> None:
        """
        Initialize the camera interface.

        :param camera_config: Configuration for the camera
        """
        self._has_new_data: bool = False
        """
        Whether new data is available.
        """
        self.camera_config: Any = camera_config
        """
        Camera configuration object.
        """
        self.rk_logger: logging.Logger = logging.getLogger(PACKAGE_NAME)
        """
        RoboKudo logger instance.
        """

    def has_new_data(self) -> bool:
        """
        Check if new data is available.

        :return: True if new data is available, False otherwise
        """
        return self._has_new_data

    @staticmethod
    def bind_world_T_camera(
        world_frame: str,
        camera_frame: str,
        world_T_camera: HomogeneousTransformationMatrix,
    ) -> HomogeneousTransformationMatrix:
        """
        Bind a sampled pose without changing a robot camera's attachment.

        :param world_frame: Name of the pose reference frame.
        :param camera_frame: Name of the camera frame.
        :param world_T_camera: Sampled numeric camera pose.
        :return: Camera pose bound to runtime-world bodies.
        """
        world = world_instance()
        reference_name = normalize_ros_frame_id(world_frame)
        camera_name = normalize_ros_frame_id(camera_frame)
        if reference_name == camera_name:
            raise InvalidCameraObservation(
                reason="the camera and world reference frames are identical"
            )
        reference_matches = find_bodies_for_ros_frame(world, reference_name)
        camera_matches = find_bodies_for_ros_frame(world, camera_name)
        if len(reference_matches) > 1 or len(camera_matches) > 1:
            raise InvalidCameraObservation(reason="camera pose frame is ambiguous")

        world_body = reference_matches[0] if reference_matches else None
        camera_body = camera_matches[0] if camera_matches else None
        current_root = world.root
        if (
            world_body is None
            and current_root is not None
            and current_root is not camera_body
        ):
            raise InvalidCameraObservation(
                reason=f"world reference frame '{reference_name}' is absent"
            )

        mount = None
        if (
            world_body is None
            or camera_body is None
            or camera_body.parent_connection is None
        ):
            with world.modify_world():
                if world_body is None:
                    world_body = Body(name=PrefixedName(name=reference_name))
                    world.add_body(world_body)
                if camera_body is None:
                    camera_body = Body(name=PrefixedName(name=camera_name))
                    world.add_body(camera_body)
                mount = Connection6DoF.create_with_dofs(
                    parent=world_body,
                    child=camera_body,
                    world=world,
                    name=CameraInterface._standalone_mount_name(camera_body),
                )
                world.add_connection(mount)
        if (
            mount is None
            and camera_body.parent_connection is not None
            and camera_body.parent_connection.parent is world_body
            and camera_body.parent_connection.name
            == CameraInterface._standalone_mount_name(camera_body)
        ):
            mount = camera_body.parent_connection

        runtime_world_T_camera = HomogeneousTransformationMatrix(
            data=world_T_camera,
            reference_frame=world_body,
            child_frame=camera_body,
        )
        if mount is not None:
            with world.modify_world():
                mount.origin = runtime_world_T_camera
        return runtime_world_T_camera

    @staticmethod
    def _standalone_mount_name(camera_body: Body) -> PrefixedName:
        """
        Identify the adjustable mount created for one standalone camera.
        """
        return PrefixedName(
            name=f"{CameraInterface.standalone_mount_prefix}{camera_body.id}"
        )

    def static_world_T_camera_if_configured(
        self, camera_frame: str
    ) -> HomogeneousTransformationMatrix | None:
        """
        Return the configured static camera pose bound to the runtime world.

        :param camera_frame: Optical frame identifying the camera in the runtime world.
        :return: Camera pose in the configured world frame, or ``None`` if disabled.
        """
        if not self.camera_config.static_camera_transform_enabled:
            return None

        return self.bind_world_T_camera(
            world_frame=self.camera_config.static_world_frame,
            camera_frame=camera_frame,
            world_T_camera=self.camera_config.static_world_T_camera,
        )

    def store_camera_observation(
        self,
        cas: CAS,
        camera_model: PinholeCameraModel,
        camera_frame: str,
        timestamp_nanoseconds: int,
        modalities: tuple[CameraModality, ...],
        world_T_camera: HomogeneousTransformationMatrix | None,
    ) -> None:
        """
        Store semantic identity and effective calibration for a camera frame.

        :param cas: CAS receiving the observation.
        :param camera_model: Effective calibration of the delivered image.
        :param camera_frame: Optical frame associated with the calibration.
        :param timestamp_nanoseconds: Acquisition time in nanoseconds since the epoch.
        :param modalities: Kinds of image data delivered by the interface.
        :param world_T_camera: Sampled camera pose, if available.
        """
        camera_frame = normalize_ros_frame_id(camera_frame)
        camera = self._resolve_or_create_camera(
            camera_frame=camera_frame,
            camera_model=camera_model,
            modalities=modalities,
        )
        if world_T_camera is not None and world_T_camera.child_frame is None:
            world_T_camera.child_frame = camera.root
        observation = CameraObservation(
            camera=camera,
            effective_camera_model=camera_model,
            world_T_camera=world_T_camera,
            timestamp_nanoseconds=timestamp_nanoseconds,
        )
        if world_T_camera is not None and world_T_camera.child_frame is not camera.root:
            observation = observation.with_world_T_camera(
                HomogeneousTransformationMatrix(
                    data=world_T_camera,
                    reference_frame=world_T_camera.reference_frame,
                    child_frame=camera.root,
                )
            )
        self._register_camera(camera)
        cas.camera_observation = observation

    def _resolve_or_create_camera(
        self,
        camera_frame: str,
        camera_model: PinholeCameraModel,
        modalities: tuple[CameraModality, ...],
    ) -> Camera:
        """
        Resolve the stream camera or construct an unregistered standalone camera.

        :param camera_frame: Optical frame associated with the image stream.
        :param camera_model: First trustworthy calibration for a new camera.
        :param modalities: Kinds of image data produced by a new camera.
        :return: Camera annotation representing the physical image source.
        :raises CameraAnnotationAmbiguous: If several cameras use the stream frame.
        """
        camera_frame = normalize_ros_frame_id(camera_frame)
        runtime_world = world_instance()
        cameras = runtime_world.get_semantic_annotations_by_type(Camera)
        frame_cameras = [
            camera
            for camera in cameras
            if normalize_ros_frame_id(camera.root.name.name) == camera_frame
        ]
        if len(frame_cameras) == 1:
            return frame_cameras[0]
        if len(frame_cameras) > 1:
            raise CameraAnnotationAmbiguous(
                camera_names=tuple(str(camera.name) for camera in frame_cameras)
            )

        camera_bodies = find_bodies_for_ros_frame(runtime_world, camera_frame)
        if len(camera_bodies) > 1:
            raise InvalidCameraObservation(reason="camera frame is ambiguous")
        camera_body = (
            camera_bodies[0]
            if camera_bodies
            else Body(name=PrefixedName(name=camera_frame))
        )
        return Camera(
            name=PrefixedName(name=camera_frame),
            root=camera_body,
            camera_model=camera_model,
            modalities=modalities,
        )

    def _register_camera(self, camera: Camera) -> None:
        """
        Register a validated stream camera and connect a new optical root to the world.

        :param camera: Semantic camera whose observation has passed frame validation.
        """
        runtime_world = world_instance()
        if runtime_world.find_world_entity_with_id(camera.id) is camera:
            return
        existing_root = runtime_world.root
        camera_body = camera.root
        with runtime_world.modify_world():
            if camera_body._world is None:
                runtime_world.add_body(camera_body)
                if existing_root is not None:
                    # An unconstrained mount keeps the world connected without
                    # representing an observed camera pose.
                    runtime_world.add_connection(
                        Connection6DoF.create_with_dofs(
                            parent=existing_root,
                            child=camera_body,
                            world=runtime_world,
                            name=self._standalone_mount_name(camera_body),
                        )
                    )
            runtime_world.add_semantic_annotation(camera)
        init_world_entity_tracker_from_world(runtime_world)

    def set_data(self, cas: CAS) -> None:
        """
        This method is supposed to read in, convert (if needed) and put the data into
        the CAS. If you are running a CameraInterface which is getting data via callback
        methods, please make sure to keep callbacks light and do the main conversion
        work here! Callbacks should be short.

        :param cas: The CAS where the data should be placed in
        """
        raise NotImplementedError


class ROSCameraInterface(CameraInterface):
    """
    Base class for ROS-based camera interfaces.

    This class extends the base camera interface with ROS-specific functionality like
    transform lookups and camera intrinsics handling.
    """

    def __init__(self, camera_config: Any, node: Node | None = None) -> None:
        """
        Initialize the ROS camera interface.

        :param camera_config: Configuration for the ROS camera
        :param node: A ROS node for transform lookups
        """
        super().__init__(camera_config)

        self.node = node if node is not None else ROSNodeRegistry().get()
        """
        ROS node for communication with ROS.
        """
        if hasattr(self.camera_config, "lookup_viewpoint"):
            self.lookup_viewpoint: bool = self.camera_config.lookup_viewpoint

            self.tf_from: str = (
                normalize_ros_frame_id(camera_config.tf_from)
                if self.lookup_viewpoint
                else camera_config.tf_from
            )
            """
            Transform source frame.
            """
            self.tf_to: str = (
                normalize_ros_frame_id(camera_config.tf_to)
                if self.lookup_viewpoint
                else camera_config.tf_to
            )
            """
            Transform target frame.
            """
        else:
            self.lookup_viewpoint: bool = False
            """
            Whether to look up camera transforms.
            """
        self._world_T_camera: HomogeneousTransformationMatrix | None = None
        """
        Camera pose from the most recent successful TF lookup.
        """
        if self.lookup_viewpoint:
            self.tf_buffer: Buffer = TFListenerProxy().buffer_for_node(self.node)
            """
            TF buffer populated by the per-node transform listener.
            """

    def lookup_transform(self, timestamp: Time | None = None) -> bool:
        """
        Cache a camera pose when the requested TF sample is available.

        :param timestamp: Acquisition time, or the latest sample when omitted.
        :return: Whether lookup is disabled or a sample was available.
        """
        if not self.lookup_viewpoint:
            return True
        timestamp = timestamp if timestamp is not None else Time()
        if not self.tf_buffer.can_transform(
            target_frame=self.tf_to,
            source_frame=self.tf_from,
            time=timestamp,
            timeout=Duration(seconds=1.0 / 30.0),
        ):
            return False
        world_T_camera = self.tf_buffer.lookup_transform(
            target_frame=self.tf_to, source_frame=self.tf_from, time=timestamp
        )
        translation = world_T_camera.transform.translation
        rotation = world_T_camera.transform.rotation
        self._world_T_camera = HomogeneousTransformationMatrix.from_xyz_quaternion(
            pos_x=float(translation.x),
            pos_y=float(translation.y),
            pos_z=float(translation.z),
            quat_x=float(rotation.x),
            quat_y=float(rotation.y),
            quat_z=float(rotation.z),
            quat_w=float(rotation.w),
        )
        return True

    def world_T_camera_from_tf(
        self, timestamp: builtin_interfaces.msg.Time
    ) -> HomogeneousTransformationMatrix | None:
        """
        Return the sampled TF camera pose bound to the runtime world.

        :param timestamp: Acquisition time associated with the cached TF sample.
        :return: Bound camera pose, or ``None`` when TF lookup is disabled.
        """
        if not self.lookup_viewpoint:
            return None
        if self._world_T_camera is None:
            raise CameraPoseMissing()
        return self.bind_world_T_camera(
            world_frame=self.tf_to,
            camera_frame=self.tf_from,
            world_T_camera=self._world_T_camera,
        )

    def validate_camera_frame_for_tf(self, camera_frame: str) -> None:
        """
        Require the image optical frame to match the configured TF source.
        """
        if self.lookup_viewpoint and normalize_ros_frame_id(
            camera_frame
        ) != normalize_ros_frame_id(self.tf_from):
            raise InvalidCameraObservation(
                reason="image camera frame differs from the configured TF source"
            )


def depth_convert_workaround(msg: CompressedImage) -> npt.NDArray:
    """
    Convert compressed depth image to proper depth format.

    This is a workaround for handling compressed depth images in ROS.
    Source: https://answers.ros.org/question/249775/display-compresseddepth-image-python-cv2/

    :param msg: Compressed depth image message
    :return: Depth image as numpy array
    :raises Exception: If compression type is wrong or decoding fails
    """
    # 'msg' as type CompressedImage
    depth_fmt, compr_type = msg.format.split(";")
    # remove white space
    depth_fmt = depth_fmt.strip()
    compr_type = compr_type.strip()
    if "compressedDepth" not in compr_type:
        raise Exception(
            "Compression type is not 'compressedDepth'."
            "You probably subscribed to the wrong topic."
        )

    # remove header from raw data
    depth_header_size = 12
    raw_data = msg.data[depth_header_size:]

    depth_img_raw = cv2.imdecode(
        np.frombuffer(raw_data, np.uint8), cv2.IMREAD_UNCHANGED
    )
    # replaced np.fromstring with np.frombuffer because np.fromstring is deprecated in newer versions of numpy
    if depth_img_raw is None:
        # probably wrong header size
        raise Exception(
            "Could not decode compressed depth image."
            "You may need to change 'depth_header_size'!"
        )

    if depth_fmt == "16UC1":
        # write raw image data
        return depth_img_raw
    elif depth_fmt == "32FC1":
        raw_header = msg.data[:depth_header_size]
        # header: int, float, float
        [compfmt, depthQuantA, depthQuantB] = struct.unpack("iff", raw_header)
        depth_img_scaled = depthQuantA / (
            depth_img_raw.astype(np.float32) - depthQuantB
        )
        # filter max values
        depth_img_scaled[depth_img_raw == 0] = 0

        # depth_img_scaled provides distance in meters as f32
        # for storing it as png, we need to convert it to 16UC1 again (depth in mm)
        depth_img_mm = (depth_img_scaled * 1000).astype(np.uint16)
        return depth_img_mm
    else:
        raise Exception("Decoding of '" + depth_fmt + "' is not implemented!")


class KinectCameraInterface(ROSCameraInterface):
    """
    Interface for Kinect-style RGB-D cameras using ROS.

    This class implements a camera interface for RGB-D cameras that publish color and
    depth images through ROS topics. It supports both raw and compressed image formats.
    """

    def __init__(self, camera_config: Any) -> None:
        """
        Initialize the Kinect camera interface.

        Sets up ROS subscribers and synchronization for color, depth, and camera info
        topics.

        :param camera_config: Configuration for the Kinect camera
        """
        super().__init__(camera_config)

        self.color_subscriber: message_filters.Subscriber = Subscriber(
            self.node,
            CompressedImage if self.compressed_color_configured() else Image,
            camera_config.topic_color,
        )
        """
        Color image subscriber.
        """
        self.depth_subscriber: message_filters.Subscriber = Subscriber(
            self.node,
            CompressedImage if self.compressed_depth_configured() else Image,
            camera_config.topic_depth,
        )
        """
        Depth image subscriber.
        """
        self.camera_info_subscriber: message_filters.Subscriber = Subscriber(
            self.node, CameraInfo, camera_config.topic_camera_info
        )
        """
        Camera info subscriber.
        """
        ts = ApproximateTimeSynchronizer(
            [self.color_subscriber, self.depth_subscriber, self.camera_info_subscriber],
            queue_size=10,
            slop=0.4,
        )
        ts.registerCallback(self.callback)

        self.rk_logger.info("Subscribed to: ")
        self.rk_logger.info(f"  {camera_config.topic_color}")
        self.rk_logger.info(f"  {camera_config.topic_depth}")
        self.rk_logger.info(f"  {camera_config.topic_camera_info}")

        self.color: npt.NDArray | None = None
        """
        Latest color image.
        """
        self.depth: npt.NDArray | None = None
        """
        Latest depth image.
        """
        self.camera_info: CameraInfo | None = None
        """
        Latest camera info message.
        """
        self.color_to_depth_ratio: ColorToDepthRatio | None = None
        """
        Ratio between color and depth image sizes.
        """
        self.timestamp: builtin_interfaces.msg.Time | None = None
        """
        Latest message timestamp.
        """
        self.lock: Lock = Lock()
        """
        Thread synchronization lock.
        """
        self.bridge: CVBridgeWorkaround = CVBridgeWorkaround()
        """
        NumPy-compatible replacement for cv_bridge.
        """

    def compressed_depth_configured(self) -> bool:
        """
        Check if compressed depth images are configured.

        :return: True if compressed depth is configured, False otherwise
        """
        return (
            hasattr(self.camera_config, "depth_hints")
            and self.camera_config.depth_hints == "compressedDepth"
        )

    def compressed_color_configured(self) -> bool:
        """
        Check if compressed color images are configured.

        :return: True if compressed color is configured, False otherwise
        """
        return (
            hasattr(self.camera_config, "color_hints")
            and self.camera_config.color_hints == "compressed"
        )

    def get_node(self) -> rclpy.node.Node:
        return self.node

    def blackhole_callback(self, data: Any) -> None:
        """
        This callback is just a dummy to receive data coming from a workaround
        subscription to handle problems with the ApproximateTimeSynchronizer.

        :param data: Dummy data
        """
        pass

    def callback(
        self,
        color_data: Image | CompressedImage,
        depth_data: Image | CompressedImage | None = None,
        camera_info: CameraInfo | None = None,
    ) -> None:
        """
        Process synchronized camera data.

        This callback handles incoming color, depth, and camera info messages. It
        converts the data to OpenCV format and stores it for later use.

        TODO make this generic. handle the encoding and order properly. For standard and
        compressed images. this might also depend on the fix of image_transport_plugins
        being published as a package. Startpoint can be found at the bottom of this
        method.

        :param color_data: Color image message
        :param depth_data: Depth image message
        :param camera_info: Camera calibration message
        """
        self.lock.acquire()
        if self.rk_logger.isEnabledFor(logging.DEBUG):
            self.rk_logger.debug("Received data:")

            color_time = Time(
                seconds=color_data.header.stamp.sec,
                nanoseconds=color_data.header.stamp.nanosec,
            )

            if depth_data is not None:
                depth_time = Time(
                    seconds=depth_data.header.stamp.sec,
                    nanoseconds=depth_data.header.stamp.nanosec,
                )
                self.rk_logger.debug(
                    f"  Color time - Depth time: {(color_time - depth_time).nanoseconds / 1e9:.6f}"
                )

            if camera_info is not None:
                camera_info_time = Time(
                    seconds=camera_info.header.stamp.sec,
                    nanoseconds=camera_info.header.stamp.nanosec,
                )
                self.rk_logger.debug(
                    f"  Color time - Camera Info time: {(color_time - camera_info_time).nanoseconds / 1e9:.6f}"
                )

        if self.compressed_color_configured():
            color_arr = np.frombuffer(color_data.data, np.uint8)
            self.color = cv2.imdecode(color_arr, cv2.IMREAD_COLOR)
        else:
            self.color = self.bridge.imgmsg_to_cv2(color_data, "bgr8")

        self.timestamp = color_data.header.stamp

        if self.compressed_depth_configured():
            if depth_data is None:
                raise CameraDataMissing(
                    data_name="Depth data",
                    context="compressed depth conversion",
                )
            self.depth = depth_convert_workaround(depth_data)
        else:
            if depth_data is None:
                raise CameraDataMissing(
                    data_name="Depth data",
                    context="uncompressed depth conversion",
                )
            self.depth = self.bridge.imgmsg_to_cv2(depth_data, "32FC1")

        self.camera_info = camera_info

        # self.rk_logger.info("Callback processing done - Final steps")
        if not self.lookup_transform():
            self._has_new_data = False
            self.lock.release()
            return

        self._has_new_data = True
        self.lock.release()

    def set_data(self, cas: CAS) -> None:
        if not self.has_new_data():
            return

        self.lock.acquire()
        if self.camera_config.hi_res_mode:
            self.color = self.color[0:960, 0:1280]

        width = self.camera_info.width
        height = self.camera_info.height
        if self.camera_config.hi_res_mode:
            height = 960
        self.camera_info.width = width
        self.camera_info.height = height

        self.color_to_depth_ratio = self.camera_config.color_to_depth_ratio

        cas.set(CASViews.COLOR_IMAGE, self.color)
        cas.set(CASViews.DEPTH_IMAGE, self.depth)
        cas.set(CASViews.COLOR_TO_DEPTH_RATIO, self.color_to_depth_ratio)

        camera_frame = self.camera_info.header.frame_id or self.camera_config.tf_from
        self.validate_camera_frame_for_tf(camera_frame)
        world_T_camera = self.world_T_camera_from_tf(self.timestamp)
        self.store_camera_observation(
            cas=cas,
            camera_model=RosCameraModelAdapter.from_camera_info(self.camera_info),
            camera_frame=camera_frame,
            timestamp_nanoseconds=(
                self.timestamp.sec * 1_000_000_000 + self.timestamp.nanosec
            ),
            modalities=(CameraModality.COLOR, CameraModality.DEPTH),
            world_T_camera=world_T_camera,
        )

        self._has_new_data = False

        self.lock.release()
