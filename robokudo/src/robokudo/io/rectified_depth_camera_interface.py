"""
Consume rectified stereo depth independently of its inference backend.
"""

from __future__ import annotations

# %% imports

from copy import deepcopy
from dataclasses import dataclass, field
from enum import IntEnum, StrEnum
from threading import Lock

import numpy as np
from message_filters import (
    ApproximateTimeSynchronizer,
    SimpleFilter,
    Subscriber,
    TimeSynchronizer,
)
from rclpy.qos import QoSProfile
from rclpy.time import Time
from sensor_msgs.msg import CameraInfo, Image
from typing_extensions import TYPE_CHECKING

from robokudo.cas import CASViews
from robokudo.io.camera_interface import ROSCameraInterface
from robokudo.exceptions import InvalidCameraObservation
from robokudo.io.camera_model_adapters import RosCameraModelAdapter
from semantic_digital_twin.datastructures.camera_model import (
    CameraDistortionModel,
    CameraModality,
)
from krrood.patterns.called_positionally import called_positionally
from robokudo.types.cv import ColorToDepthRatio
from robokudo.utils.cv_bridge_workaround import CVBridgeWorkaround

if TYPE_CHECKING:
    import numpy.typing as npt
    from rclpy.node import Node
    from std_msgs.msg import Header

    from robokudo.cas import CAS
    from robokudo.descriptors.camera_configs.config_stereo import (
        StereoCameraConfig,
    )


# %% image contracts


class DepthFrameMismatch(StrEnum):
    """
    Reasons a stereo acquisition cannot represent aligned pinhole depth.
    """

    ACQUISITION = "color and depth acquisition timestamps differ"
    OPTICAL_FRAME = "input messages do not share the configured optical frame"
    DIMENSIONS = "input dimensions differ or are empty"
    DEPTH_ENCODING = "depth must be float32 meters"
    IMAGE_REGION = "cropped or binned calibration is unsupported"
    FOCAL_LENGTH = "rectified focal lengths must be finite and positive"
    PROJECTION = "left projection is not a standard pinhole projection"


class CameraEncoding(StrEnum):
    """
    Image and calibration encodings at the RK camera boundary.
    """

    METRIC_DEPTH = "32FC1"
    COLOR = "bgr8"


class CameraUnit(IntEnum):
    """
    Conversions between camera transport and RK units.
    """

    MILLIMETERS_PER_METER = 1000


@dataclass(kw_only=True)
class ColorDepthImages:
    """
    A color image and depth inferred from that exact acquisition.
    """

    color: Image
    """
    Rectified left color image.
    """
    depth: Image
    """
    Aligned metric depth image.
    """

    @property
    def header(self) -> Header:
        """
        Acquisition header shared by both images.
        """
        return self.color.header


@dataclass(kw_only=True)
class RectifiedDepthFrame:
    """
    An exact image pair and its associated rectified calibration.
    """

    images: ColorDepthImages
    """
    Images sharing an acquisition timestamp.
    """
    calibration: CameraInfo
    """
    Calibration describing the rectified left image.
    """

    def validate(self, *, optical_frame: str) -> None:
        """
        Require full-size aligned images in a standard left pinhole frame.
        """
        color, depth = self.images.color, self.images.depth
        calibration = self.calibration
        if color.header.stamp != depth.header.stamp:
            raise InvalidCameraObservation(reason=DepthFrameMismatch.ACQUISITION)
        if any(
            message.header.frame_id != optical_frame
            for message in (color, depth, calibration)
        ):
            raise InvalidCameraObservation(reason=DepthFrameMismatch.OPTICAL_FRAME)
        if (
            color.width <= 0
            or color.height <= 0
            or any(
                message.width != color.width or message.height != color.height
                for message in (depth, calibration)
            )
        ):
            raise InvalidCameraObservation(reason=DepthFrameMismatch.DIMENSIONS)
        if depth.encoding != CameraEncoding.METRIC_DEPTH:
            raise InvalidCameraObservation(reason=DepthFrameMismatch.DEPTH_ENCODING)
        if (
            calibration.binning_x not in (0, 1)
            or calibration.binning_y not in (0, 1)
            or any(
                (
                    calibration.roi.x_offset,
                    calibration.roi.y_offset,
                    calibration.roi.width,
                    calibration.roi.height,
                    calibration.roi.do_rectify,
                )
            )
        ):
            raise InvalidCameraObservation(reason=DepthFrameMismatch.IMAGE_REGION)
        projection = np.asarray(calibration.p).reshape(3, 4)
        if (
            not np.isfinite(projection).all()
            or projection[0, 0] <= 0
            or projection[1, 1] <= 0
        ):
            raise InvalidCameraObservation(reason=DepthFrameMismatch.FOCAL_LENGTH)
        if not np.allclose(projection[2], [0, 0, 1, 0]) or not np.allclose(
            [projection[0, 1], projection[1, 0], projection[0, 3], projection[1, 3]], 0
        ):
            raise InvalidCameraObservation(reason=DepthFrameMismatch.PROJECTION)

    def millimeter_depth(
        self, *, bridge: CVBridgeWorkaround, max_depth: float
    ) -> npt.NDArray[np.float32]:
        """
        Return contiguous RK depth with invalid or distant values set to zero.
        """
        depth = np.empty(
            (self.images.depth.height, self.images.depth.width), dtype=np.float32
        )
        depth[:] = bridge.imgmsg_to_cv2(self.images.depth, CameraEncoding.METRIC_DEPTH)
        depth[~np.isfinite(depth) | (depth <= 0) | (depth > max_depth)] = 0
        depth *= CameraUnit.MILLIMETERS_PER_METER
        return depth

    def rectified_calibration(self) -> CameraInfo:
        """
        Return camera information describing the undistorted pixel grid.
        """
        calibration = deepcopy(self.calibration)
        calibration.k = (
            np.asarray(calibration.p).reshape(3, 4)[:, :3].reshape(-1).tolist()
        )
        calibration.d = [0.0] * len(calibration.d)
        calibration.r = np.eye(3).reshape(-1).tolist()
        calibration.distortion_model = CameraDistortionModel.PLUMB_BOB.value
        return calibration


# %% bounded frame delivery


@dataclass(kw_only=True)
class DepthCameraStatistics:
    """
    Counts of synchronized and discarded depth frames.
    """

    matched_frames: int = 0
    """
    Image pairs associated with calibration.
    """
    replaced_frames: int = 0
    """
    Pending frames superseded before consumption.
    """
    missing_transforms: int = 0
    """
    Frames discarded because acquisition-time TF was unavailable.
    """
    obsolete_frames: int = 0
    """
    Repeated or out-of-order acquisitions discarded.
    """


@dataclass(eq=False, kw_only=True)
class RectifiedDepthCameraInterface(ROSCameraInterface):
    """
    Deliver rectified images with aligned depth converted to millimeters.
    """

    camera_config: StereoCameraConfig
    """
    Topic, synchronization and transform configuration.
    """
    node: Node | None = None
    """
    ROS node; the registered RK node is used when omitted.
    """
    statistics: DepthCameraStatistics = field(
        default_factory=DepthCameraStatistics, init=False
    )
    """
    Frame acceptance and drop counts.
    """
    color_subscriber: Subscriber = field(init=False)
    """
    Raw rectified image subscription.
    """
    depth_subscriber: Subscriber = field(init=False)
    """
    Metric depth subscription.
    """
    camera_info_subscriber: Subscriber = field(init=False)
    """
    Per-frame calibration subscription.
    """
    image_synchronizer: TimeSynchronizer = field(init=False)
    """
    Exact acquisition matching for RGB and depth.
    """
    calibration_synchronizer: TimeSynchronizer = field(init=False)
    """
    Calibration association with exact image pairs.
    """
    image_pairs: SimpleFilter = field(default_factory=SimpleFilter, init=False)
    """
    Matched image pairs awaiting calibration.
    """
    pending_frame: RectifiedDepthFrame | None = field(default=None, init=False)
    """
    Latest frame awaiting pipeline consumption.
    """
    lock: Lock = field(default_factory=Lock, init=False)
    """
    Protects pending-frame replacement and consumption.
    """
    bridge: CVBridgeWorkaround = field(default_factory=CVBridgeWorkaround, init=False)
    """
    Converts ROS image payloads into arrays.
    """
    last_timestamp: int = field(default=-1, init=False)
    """
    Newest acquisition accepted for delivery, in nanoseconds.
    """

    def __post_init__(self) -> None:
        """
        Subscribe with bounded histories and acquisition-based matching.
        """
        super().__init__(self.camera_config, node=self.node)
        configuration = self.camera_config
        self.color_subscriber = Subscriber(
            self.node,
            Image,
            configuration.topic_color,
            qos_profile=QoSProfile(
                depth=configuration.history_depth,
                reliability=configuration.color_reliability,
            ),
        )
        self.depth_subscriber = Subscriber(
            self.node,
            Image,
            configuration.topic_depth,
            qos_profile=QoSProfile(
                depth=configuration.history_depth,
                reliability=configuration.depth_reliability,
            ),
        )
        self.camera_info_subscriber = Subscriber(
            self.node,
            CameraInfo,
            configuration.topic_camera_info,
            qos_profile=QoSProfile(
                depth=configuration.history_depth,
                reliability=configuration.calibration_reliability,
            ),
        )
        self.image_synchronizer = TimeSynchronizer(
            [self.color_subscriber, self.depth_subscriber],
            configuration.synchronization_queue_size,
        )
        self.image_synchronizer.registerCallback(self.receive_images)
        calibration_inputs = [self.image_pairs, self.camera_info_subscriber]
        if configuration.calibration_tolerance == 0:
            self.calibration_synchronizer = TimeSynchronizer(
                calibration_inputs, configuration.synchronization_queue_size
            )
        else:
            self.calibration_synchronizer = ApproximateTimeSynchronizer(
                calibration_inputs,
                configuration.synchronization_queue_size,
                configuration.calibration_tolerance,
            )
        self.calibration_synchronizer.registerCallback(self.receive_frame)

    @called_positionally
    def receive_images(self, color: Image, depth: Image) -> None:
        """
        Associate an exact image pair with its per-frame calibration.
        """
        self.image_pairs.signalMessage(ColorDepthImages(color=color, depth=depth))

    @called_positionally
    def receive_frame(self, images: ColorDepthImages, calibration: CameraInfo) -> None:
        """
        Retain the newest synchronized acquisition for pipeline consumption.
        """
        timestamp = Time.from_msg(images.header.stamp).nanoseconds
        with self.lock:
            self.statistics.matched_frames += 1
            if timestamp <= self.last_timestamp:
                self.statistics.obsolete_frames += 1
                return
            if self.pending_frame is not None:
                self.statistics.replaced_frames += 1
            self.pending_frame = RectifiedDepthFrame(
                images=images, calibration=calibration
            )
            self.last_timestamp = timestamp
            self._has_new_data = True

    @called_positionally
    def set_data(self, cas: CAS) -> bool:
        """
        Return ``False`` from :meth:`ROSCameraInterface.set_data` when a frame is dropped.
        """
        with self.lock:
            frame = self.pending_frame
            self.pending_frame = None
            self._has_new_data = False
        if frame is None:
            return False
        timestamp = Time.from_msg(frame.images.header.stamp)
        frame.validate(optical_frame=self.camera_config.tf_from)
        if (
            not self.camera_config.static_camera_transform_enabled
            and not self.lookup_transform(timestamp)
        ):
            self.statistics.missing_transforms += 1
            return False
        self.store_frame(cas=cas, frame=frame)
        return True

    def store_frame(self, *, cas: CAS, frame: RectifiedDepthFrame) -> None:
        """
        Populate RK image views and the acquisition-time camera pose.
        """
        calibration = frame.rectified_calibration()
        color = np.ascontiguousarray(
            self.bridge.imgmsg_to_cv2(frame.images.color, CameraEncoding.COLOR)
        )
        depth = frame.millimeter_depth(
            bridge=self.bridge, max_depth=self.camera_config.max_depth
        )
        cas.set(CASViews.COLOR_IMAGE, color)
        cas.set(CASViews.DEPTH_IMAGE, depth)
        cas.set(CASViews.COLOR_TO_DEPTH_RATIO, ColorToDepthRatio())
        camera_frame = frame.images.color.header.frame_id
        self.validate_camera_frame_for_tf(camera_frame)
        world_T_camera = self.static_world_T_camera_if_configured(camera_frame)
        if world_T_camera is None:
            world_T_camera = self.world_T_camera_from_tf(frame.images.header.stamp)
        self.store_camera_observation(
            cas=cas,
            camera_model=RosCameraModelAdapter.from_camera_info(calibration),
            camera_frame=camera_frame,
            timestamp_nanoseconds=Time.from_msg(frame.images.header.stamp).nanoseconds,
            modalities=(CameraModality.COLOR, CameraModality.DEPTH),
            world_T_camera=world_T_camera,
        )
