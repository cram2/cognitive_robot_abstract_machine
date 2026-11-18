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
import math
import struct
from dataclasses import dataclass
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
from geometry_msgs.msg import TransformStamped
from sensor_msgs.msg import CompressedImage, CameraInfo, Image
from typing_extensions import Any, TYPE_CHECKING

from krrood.patterns.called_positionally import called_positionally
from krrood.patterns.classproperty import classproperty
from robokudo.cas import CASViews, CAS
from robokudo.defs import PACKAGE_NAME
from robokudo.descriptors.camera_configs.components import TfComponent
from robokudo.exceptions import (
    CameraDataMissing,
    CameraPoseMissing,
    InvalidCameraObservation,
    UnknownWorldFrameError,
)
from robokudo.io.exceptions import CameraAnnotationAmbiguous
from robokudo.io.camera_model_adapters import RosCameraModelAdapter
from robokudo.types.camera import CameraObservation
from robokudo.utils.cv_bridge_workaround import CVBridgeWorkaround
from robokudo.utils.ros_frames import find_bodies_for_ros_frame, normalize_ros_frame_id
from semantic_digital_twin.adapters.ros.node_registry import ROSNodeRegistry
from semantic_digital_twin.adapters.ros.ros2_to_semdt_converters import (
    TransformToSemDTConverter,
)
from semantic_digital_twin.adapters.ros.tfwrapper import TFWrapper
from semantic_digital_twin.adapters.ros.world_synchronizer import WorldSynchronizer
from semantic_digital_twin.datastructures.camera_model import (
    CameraModality,
    PinholeCameraModel,
)
from semantic_digital_twin.datastructures.prefixed_name import PrefixedName
from semantic_digital_twin.robots.robot_parts import Camera
from krrood_spatial_math.spatial_types import HomogeneousTransformationMatrix
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

    def adds_missing_camera(self, *, world: World) -> bool:
        """
        Whether this interface adds its camera to ``world`` when ``world`` lacks it.

        :param world: The world perception perceives into.
        :return: Whether a missing camera body and :class:`Camera` are added.
        """
        return True

    def bind_world_T_camera(
        self,
        *,
        world: World,
        world_frame: str,
        camera_frame: str,
        world_T_camera: HomogeneousTransformationMatrix,
    ) -> HomogeneousTransformationMatrix:
        """
        Bind a sampled camera pose to the bodies of ``world``.

        A camera another process attached, such as a robot's camera, keeps that
        attachment. Any other camera hangs below the world frame at the sampled pose,
        unless ``world`` lacks it and :meth:`adds_missing_camera` is false.

        :param world: The world perception perceives into.
        :param world_frame: Name of the pose reference frame.
        :param camera_frame: Name of the camera frame.
        :param world_T_camera: Sampled numeric camera pose.
        :return: Camera pose bound to the bodies of ``world``, without a child frame for
            a camera that has no body there.
        :raises InvalidCameraObservation: If the frames are identical or ambiguous.
        :raises UnknownWorldFrameError: If ``world`` has a root but no body for
            ``world_frame``.
        """
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

        camera_body = camera_matches[0] if camera_matches else None
        world_body = reference_matches[0] if reference_matches else None
        if (
            world_body is None
            and world.root is not None
            and world.root is not camera_body
        ):
            raise UnknownWorldFrameError(world_frame=reference_name)
        if camera_body is None and not self.adds_missing_camera(world=world):
            return HomogeneousTransformationMatrix(
                data=world_T_camera, reference_frame=world_body
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
                    name=self._standalone_mount_name(camera_body),
                )
                world.add_connection(mount)
        elif (
            self.is_mounted_standalone(camera_body=camera_body)
            and camera_body.parent_connection.parent is world_body
        ):
            mount = camera_body.parent_connection

        bound_world_T_camera = HomogeneousTransformationMatrix(
            data=world_T_camera,
            reference_frame=world_body,
            child_frame=camera_body,
        )
        if mount is not None:
            with world.modify_world():
                mount.origin = bound_world_T_camera
        return bound_world_T_camera

    @classmethod
    def is_mounted_standalone(cls, *, camera_body: Body) -> bool:
        """
        :return: Whether ``camera_body`` hangs on the adjustable mount perception
            created for it, rather than on an attachment another process made.
        """
        return (
            camera_body.parent_connection is not None
            and camera_body.parent_connection.name
            == cls._standalone_mount_name(camera_body)
        )

    @staticmethod
    def _standalone_mount_name(camera_body: Body) -> PrefixedName:
        """
        Identify the adjustable mount created for one standalone camera.
        """
        return PrefixedName(
            name=f"{CameraInterface.standalone_mount_prefix}{camera_body.id}"
        )

    def static_world_T_camera_if_configured(
        self, *, world: World, camera_frame: str
    ) -> HomogeneousTransformationMatrix | None:
        """
        Return the configured static camera pose bound to ``world``.

        :param world: The world perception perceives into.
        :param camera_frame: Optical frame identifying the camera in ``world``.
        :return: Camera pose in the configured world frame, or ``None`` if disabled.
        """
        if not self.camera_config.static_camera_transform_enabled:
            return None

        return self.bind_world_T_camera(
            world=world,
            world_frame=self.camera_config.static_world_frame,
            camera_frame=camera_frame,
            world_T_camera=self.camera_config.static_world_T_camera,
        )

    def store_camera_observation(
        self,
        *,
        cas: CAS,
        world: World,
        camera_model: PinholeCameraModel,
        camera_frame: str,
        timestamp_nanoseconds: int,
        modalities: tuple[CameraModality, ...],
        world_T_camera: HomogeneousTransformationMatrix | None,
    ) -> None:
        """
        Store semantic identity and effective calibration for a camera frame.

        :param cas: CAS receiving the observation.
        :param world: The world perception perceives into.
        :param camera_model: Effective calibration of the delivered image.
        :param camera_frame: Optical frame associated with the calibration.
        :param timestamp_nanoseconds: Acquisition time in nanoseconds since the epoch.
        :param modalities: Kinds of image data delivered by the interface.
        :param world_T_camera: Sampled camera pose, if available.
        """
        camera_frame = normalize_ros_frame_id(camera_frame)
        camera = self._resolve_or_create_camera(
            world=world,
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
        self._register_camera(world=world, camera=camera)
        cas.camera_observation = observation

    def _resolve_or_create_camera(
        self,
        *,
        world: World,
        camera_frame: str,
        camera_model: PinholeCameraModel,
        modalities: tuple[CameraModality, ...],
    ) -> Camera:
        """
        Resolve the camera annotating the body of the stream frame, or construct an
        unregistered camera for that body, or for a new standalone body if ``world``
        lacks one.

        :param world: The world perception perceives into.
        :param camera_frame: Optical frame associated with the image stream.
        :param camera_model: First trustworthy calibration for a new camera.
        :param modalities: Kinds of image data produced by a new camera.
        :return: Camera annotation representing the physical image source.
        :raises InvalidCameraObservation: If the stream frame names several bodies.
        :raises CameraAnnotationAmbiguous: If several cameras annotate the body of the
            stream frame.
        """
        camera_frame = normalize_ros_frame_id(camera_frame)
        camera_bodies = find_bodies_for_ros_frame(world, camera_frame)
        if len(camera_bodies) > 1:
            raise InvalidCameraObservation(reason="camera frame is ambiguous")
        camera_body = (
            camera_bodies[0]
            if camera_bodies
            else Body(name=PrefixedName(name=camera_frame))
        )
        body_cameras = [
            camera
            for camera in world.get_semantic_annotations_by_type(Camera)
            if camera.root is camera_body
        ]
        if len(body_cameras) > 1:
            raise CameraAnnotationAmbiguous(
                camera_names=tuple(str(camera.name) for camera in body_cameras)
            )
        if body_cameras:
            return body_cameras[0]
        return Camera(
            name=PrefixedName(name=camera_frame),
            root=camera_body,
            camera_model=camera_model,
            modalities=modalities,
        )

    def _register_camera(self, *, world: World, camera: Camera) -> None:
        """
        Register a validated stream camera and connect a new optical root to ``world``.

        Nothing is registered when :meth:`adds_missing_camera` is false.

        :param world: The world perception perceives into.
        :param camera: Semantic camera whose observation has passed frame validation.
        """
        if world.find_world_entity_with_id(camera.id) is camera:
            return
        if not self.adds_missing_camera(world=world):
            return
        existing_root = world.root
        camera_body = camera.root
        with world.modify_world():
            if camera_body._world is None:
                world.add_body(camera_body)
                if existing_root is not None:
                    # An unconstrained mount keeps the world connected without
                    # representing an observed camera pose.
                    world.add_connection(
                        Connection6DoF.create_with_dofs(
                            parent=existing_root,
                            child=camera_body,
                            world=world,
                            name=self._standalone_mount_name(camera_body),
                        )
                    )
            world.add_semantic_annotation(camera)

    def set_data(self, *, cas: CAS, world: World) -> None:
        """
        This method is supposed to read in, convert (if needed) and put the data into
        the CAS. If you are running a CameraInterface which is getting data via callback
        methods, please make sure to keep callbacks light and do the main conversion
        work here! Callbacks should be short.

        The camera is placed in ``world`` where it was when it recorded the data, if
        that is known.

        :param cas: The CAS where the data should be placed in
        :param world: The world perception perceives into
        """
        raise NotImplementedError


class ROSCameraInterface(CameraInterface):
    """
    Base class for ROS-based camera interfaces.

    This class extends the base camera interface with ROS-specific functionality like
    transform lookups and camera intrinsics handling.
    """

    def __init__(self, camera_config: TfComponent, node: Node | None = None) -> None:
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
        self.lookup_viewpoint: bool = camera_config.lookup_viewpoint
        """
        Whether the camera pose is looked up through TF.
        """
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
        self._world_T_camera: TransformStamped | None = None
        """
        Camera pose from the most recent successful TF lookup, stamped with the time TF
        provided it for.
        """
        self.color: npt.NDArray | None = None
        """
        Latest color image.
        """
        self.depth: npt.NDArray | None = None
        """
        Latest depth image; None for a camera without depth.
        """
        self.camera_info: CameraInfo | None = None
        """
        Latest camera info message.
        """
        self.timestamp: builtin_interfaces.msg.Time | None = None
        """
        Acquisition time of the latest frame.
        """
        self.lock: Lock = Lock()
        """
        Guards the latest frame against being replaced while it is read.
        """
        if self.lookup_viewpoint:
            self.tf_wrapper: TFWrapper = TFWrapper(node=self.node)
            """
            Looks the camera pose up in TF.
            """

    def has_new_data(self) -> bool:
        """
        Check :meth:`CameraInterface.has_new_data`, dropping a new frame for which TF
        provides no camera pose at its acquisition time as :meth:`lookup_transform`
        waits for it.
        """
        with self.lock:
            if not self._has_new_data:
                return False
            self._has_new_data = self.lookup_transform(
                timestamp=Time.from_msg(self.timestamp)
            )
            return self._has_new_data

    def store_frame(self, *, frame: CameraFrame) -> None:
        """
        Make ``frame`` the latest frame, or drop it while the latest frame is read.
        """
        # Skip the frame instead of waiting, so that the executor delivering it keeps
        # delivering TF while a frame is read.
        if not self.lock.acquire(blocking=False):
            return
        self.color = frame.color
        self.depth = frame.depth
        self.camera_info = frame.camera_info
        self.timestamp = frame.timestamp
        self._has_new_data = True
        self.lock.release()

    def adds_missing_camera(self, *, world: World) -> bool:
        """
        Unlike :meth:`CameraInterface.adds_missing_camera`, leave a camera located
        through TF to the processes ``world`` is synchronized with.
        """
        return not (
            self.lookup_viewpoint
            and WorldSynchronizer.all_callbacks_of_this_type_from_world(world)
        )

    def lookup_transform(self, *, timestamp: Time | None = None) -> bool:
        """
        Cache a camera pose when the requested TF sample is available, waiting for it at
        most the configured camera pose timeout.

        :param timestamp: Acquisition time, or the latest sample when omitted.
        :return: Whether lookup is disabled or a sample was available.
        """
        if not self.lookup_viewpoint:
            return True
        timestamp = timestamp if timestamp is not None else Time()
        if not self.tf_wrapper.wait_for_transform(
            target_frame=self.tf_to,
            source_frame=self.tf_from,
            time=timestamp,
            timeout=Duration(
                seconds=self.camera_config.camera_pose_timeout.total_seconds()
            ),
        ):
            return False
        self._world_T_camera = self.tf_wrapper.lookup_transform(
            target_frame=self.tf_to, source_frame=self.tf_from, time=timestamp
        )
        return True

    def world_T_camera_from_tf(
        self, *, world: World, timestamp: builtin_interfaces.msg.Time
    ) -> HomogeneousTransformationMatrix | None:
        """
        Return the camera pose TF provides at ``timestamp``, bound to ``world``.

        :param world: The world perception perceives into.
        :param timestamp: Acquisition time of the frame the camera pose is for.
        :return: Bound camera pose, or ``None`` when TF lookup is disabled.
        :raises CameraPoseMissing: If TF provides no camera pose at ``timestamp``.
        """
        if not self.lookup_viewpoint:
            return None
        if (
            self._world_T_camera is None
            or self._world_T_camera.header.stamp != timestamp
        ) and not self.lookup_transform(timestamp=Time.from_msg(timestamp)):
            raise CameraPoseMissing()
        world_T_camera = self.bind_world_T_camera(
            world=world,
            world_frame=self.tf_to,
            camera_frame=self.tf_from,
            world_T_camera=TransformToSemDTConverter.convert(
                data=self._world_T_camera.transform, world=world
            ),
        )
        self.warn_if_world_disagrees_with_tf(world=world, world_T_camera=world_T_camera)
        return world_T_camera

    def warn_if_world_disagrees_with_tf(
        self, *, world: World, world_T_camera: HomogeneousTransformationMatrix
    ) -> None:
        """
        Log a warning if the world currently places a camera another process attached
        farther from TF's latest camera pose than the configured tolerances allow.

        :param world: The world perception perceives into.
        :param world_T_camera: Camera pose bound to ``world``.
        """
        camera_body = world_T_camera.child_frame
        if (
            camera_body is None
            or camera_body.parent_connection is None
            or self.is_mounted_standalone(camera_body=camera_body)
        ):
            return
        latest_world_T_camera = TransformToSemDTConverter.convert(
            data=self.tf_wrapper.lookup_transform(
                target_frame=self.tf_to, source_frame=self.tf_from
            ).transform,
            world=world,
        )
        disagreement = HomogeneousTransformationMatrix(
            data=np.linalg.inv(latest_world_T_camera.to_np())
            @ np.linalg.inv(world_T_camera.reference_frame.global_transform.to_np())
            @ camera_body.global_transform.to_np()
        )
        distance = float(disagreement.position.norm())
        angle = float(disagreement.axis_angle.angle)
        if (
            distance <= self.camera_config.tolerated_camera_pose_distance
            and angle <= self.camera_config.tolerated_camera_pose_angle
        ):
            return
        self.rk_logger.warning(
            f"The world places camera '{self.tf_from}' {distance:.3f} m and "
            f"{math.degrees(angle):.1f} degrees away from where TF places it in "
            f"'{self.tf_to}'; perceived objects follow TF and may be misplaced "
            f"relative to the robot in the world."
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


# %% camera frame


@dataclass(kw_only=True)
class CameraFrame:
    """
    The decoded images of one frame a ROS camera delivered, with its calibration.
    """

    color: npt.NDArray
    """
    The color image, in BGR channel order.
    """

    depth: npt.NDArray | None = None
    """
    The depth image; None for a camera without depth.
    """

    camera_info: CameraInfo | None = None
    """
    The calibration of the camera; None if the camera delivered none.
    """

    timestamp: builtin_interfaces.msg.Time
    """
    Acquisition time of the color image.
    """

    @classmethod
    def from_messages(
        cls,
        *,
        color_data: Image | CompressedImage,
        depth_data: Image | CompressedImage | None = None,
        camera_info: CameraInfo | None = None,
    ) -> CameraFrame:
        """
        Decode the messages of one frame.

        :param color_data: The color image message.
        :param depth_data: The depth image message; None for a camera without depth.
        :param camera_info: The calibration message.
        :return: The frame the messages carry.
        """
        # TODO make this generic. handle the encoding and order properly. For standard
        # and compressed images. this might also depend on the fix of
        # image_transport_plugins being published as a package.
        bridge = CVBridgeWorkaround()
        color = (
            cv2.imdecode(np.frombuffer(color_data.data, np.uint8), cv2.IMREAD_COLOR)
            if isinstance(color_data, CompressedImage)
            else bridge.imgmsg_to_cv2(color_data, "bgr8")
        )
        if depth_data is None:
            depth = None
        elif isinstance(depth_data, CompressedImage):
            depth = depth_convert_workaround(depth_data)
        else:
            depth = bridge.imgmsg_to_cv2(depth_data, "32FC1")
        return cls(
            color=color,
            depth=depth,
            camera_info=camera_info,
            timestamp=color_data.header.stamp,
        )


# %% RGB-D camera


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

        self.color_to_depth_ratio: ColorToDepthRatio | None = None
        """
        Ratio between color and depth image sizes.
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

    @called_positionally
    def callback(
        self,
        color_data: Image | CompressedImage,
        depth_data: Image | CompressedImage | None = None,
        camera_info: CameraInfo | None = None,
    ) -> None:
        """
        Make the synchronized color, depth and camera info messages the latest frame.

        .. note:: A frame arriving while the previous frame is read is dropped.

        :param color_data: Color image message
        :param depth_data: Depth image message
        :param camera_info: Camera calibration message
        :raises CameraDataMissing: If the frame has no depth image.
        """
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

        if depth_data is None:
            raise CameraDataMissing(data_name="Depth data", context="an RGB-D frame")
        self.store_frame(
            frame=CameraFrame.from_messages(
                color_data=color_data, depth_data=depth_data, camera_info=camera_info
            )
        )

    def set_data(self, *, cas: CAS, world: World) -> None:
        with self.lock:
            if not self._has_new_data:
                return

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

            camera_frame = (
                self.camera_info.header.frame_id or self.camera_config.tf_from
            )
            self.validate_camera_frame_for_tf(camera_frame)
            world_T_camera = self.world_T_camera_from_tf(
                world=world, timestamp=self.timestamp
            )
            self.store_camera_observation(
                cas=cas,
                world=world,
                camera_model=RosCameraModelAdapter.from_camera_info(self.camera_info),
                camera_frame=camera_frame,
                timestamp_nanoseconds=(
                    self.timestamp.sec * 1_000_000_000 + self.timestamp.nanosec
                ),
                modalities=(CameraModality.COLOR, CameraModality.DEPTH),
                world_T_camera=world_T_camera,
            )

            self._has_new_data = False
