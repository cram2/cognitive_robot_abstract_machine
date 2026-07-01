"""A lightweight replacement for cv_bridge while NumPy 2 is in use.

This class exists because the currently available ``cv_bridge`` binary can be
compiled against NumPy 1.x and fails to import in environments that use NumPy 2.
When ``cv_bridge`` wheels/packages become available that are compiled against
NumPy 2, this module should be removed and users should switch back to
``cv_bridge.CvBridge``.

Only a subset of the original interface is implemented here:
* ``imgmsg_to_cv2``
* ``cv2_to_imgmsg``

The implementation intentionally focuses on encodings used by RoboKudo camera
pipelines and type conversion helpers.
"""

from __future__ import annotations

import re

import cv2
import numpy as np
from sensor_msgs.msg import Image
from typing_extensions import TYPE_CHECKING, Tuple

if TYPE_CHECKING:
    import numpy.typing as npt


class CVBridgeWorkaround:
    """Subset replacement for ``cv_bridge.CvBridge``.

    The API is intentionally close to cv_bridge for drop-in usage in existing
    code paths.
    """

    _GENERIC_ENCODING_RE = re.compile(r"^(8|16|32|64)([USFusf])C([1-4])$")

    _ENCODING_TO_DTYPE_CHANNELS = {
        "mono8": (np.uint8, 1),
        "mono16": (np.uint16, 1),
        "8uc1": (np.uint8, 1),
        "8sc1": (np.int8, 1),
        "8uc3": (np.uint8, 3),
        "8sc3": (np.int8, 3),
        "8uc4": (np.uint8, 4),
        "8sc4": (np.int8, 4),
        "16uc1": (np.uint16, 1),
        "16sc1": (np.int16, 1),
        "16uc3": (np.uint16, 3),
        "16sc3": (np.int16, 3),
        "16uc4": (np.uint16, 4),
        "16sc4": (np.int16, 4),
        "32sc1": (np.int32, 1),
        "32fc1": (np.float32, 1),
        "32fc3": (np.float32, 3),
        "32fc4": (np.float32, 4),
        "64fc1": (np.float64, 1),
        "64fc3": (np.float64, 3),
        "64fc4": (np.float64, 4),
        "rgb8": (np.uint8, 3),
        "bgr8": (np.uint8, 3),
        "rgba8": (np.uint8, 4),
        "bgra8": (np.uint8, 4),
    }

    def imgmsg_to_cv2(
        self, img_msg: Image, desired_encoding: str = "passthrough"
    ) -> npt.NDArray:
        """Convert ROS ``sensor_msgs/Image`` to a NumPy/OpenCV image."""
        image = self._decode_image_message(img_msg)
        source_encoding = img_msg.encoding.strip().lower()
        target_encoding = desired_encoding.strip().lower()

        if target_encoding in ("", "passthrough") or target_encoding == source_encoding:
            return image

        if target_encoding == "bgr8":
            if source_encoding == "bgr8":
                return image
            if source_encoding == "rgb8":
                return cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            if source_encoding == "bgra8":
                return cv2.cvtColor(image, cv2.COLOR_BGRA2BGR)
            if source_encoding == "rgba8":
                return cv2.cvtColor(image, cv2.COLOR_RGBA2BGR)
            if source_encoding in ("mono8", "8uc1"):
                return cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
            raise ValueError(
                f"Cannot convert ROS image encoding '{img_msg.encoding}' to '{desired_encoding}'"
            )

        if target_encoding == "32fc1":
            if image.ndim != 2:
                raise ValueError(
                    f"Cannot convert non-single-channel image '{img_msg.encoding}' to '{desired_encoding}'"
                )
            if image.dtype == np.float32:
                return image
            return image.astype(np.float32)

        raise ValueError(
            f"Unsupported desired encoding '{desired_encoding}' for ROS image conversion"
        )

    def cv2_to_imgmsg(
        self, cv_image: npt.NDArray, encoding: str = "passthrough"
    ) -> Image:
        """Convert a NumPy/OpenCV image to ROS ``sensor_msgs/Image``."""
        np_image = np.asarray(cv_image)
        if np_image.ndim not in (2, 3):
            raise ValueError(
                f"Expected 2D or 3D image array, got shape {np_image.shape} with ndim={np_image.ndim}"
            )

        if not np_image.flags.c_contiguous:
            np_image = np.ascontiguousarray(np_image)

        if not np_image.dtype.isnative:
            np_image = np_image.byteswap().view(np_image.dtype.newbyteorder("="))

        requested_encoding = encoding.strip()
        if requested_encoding.lower() in ("", "passthrough"):
            ros_encoding = self._infer_ros_encoding_from_cv_image(np_image)
        else:
            expected_dtype, expected_channels = self._encoding_to_dtype_channels(
                requested_encoding
            )
            actual_channels = 1 if np_image.ndim == 2 else int(np_image.shape[2])
            if np.dtype(expected_dtype) != np_image.dtype or actual_channels != int(
                expected_channels
            ):
                raise ValueError(
                    f"Image dtype/channels ({np_image.dtype}, {actual_channels}) do not match requested encoding '{encoding}'"
                )
            ros_encoding = requested_encoding

        ros_image = Image()
        ros_image.height = int(np_image.shape[0])
        ros_image.width = int(np_image.shape[1])
        ros_image.encoding = ros_encoding
        ros_image.is_bigendian = int(not np.little_endian)
        ros_image.step = int(np_image.strides[0])
        ros_image.data = np_image.tobytes()
        return ros_image

    @classmethod
    def _encoding_to_dtype_channels(cls, encoding: str) -> Tuple[np.dtype, int]:
        normalized = encoding.strip().lower()
        if normalized in cls._ENCODING_TO_DTYPE_CHANNELS:
            dtype, channels = cls._ENCODING_TO_DTYPE_CHANNELS[normalized]
            return np.dtype(dtype), int(channels)

        match = cls._GENERIC_ENCODING_RE.match(normalized)
        if match is None:
            raise ValueError(f"Unsupported ROS image encoding '{encoding}'")

        bits, kind, channels = match.groups()
        dtype_map = {
            ("8", "u"): np.uint8,
            ("8", "s"): np.int8,
            ("16", "u"): np.uint16,
            ("16", "s"): np.int16,
            ("32", "u"): np.uint32,
            ("32", "s"): np.int32,
            ("32", "f"): np.float32,
            ("64", "f"): np.float64,
        }
        dtype = dtype_map.get((bits, kind.lower()))
        if dtype is None:
            raise ValueError(f"Unsupported ROS image encoding '{encoding}'")
        return np.dtype(dtype), int(channels)

    @classmethod
    def _decode_image_message(cls, msg: Image) -> npt.NDArray:
        base_dtype, channels = cls._encoding_to_dtype_channels(msg.encoding)
        if msg.height <= 0 or msg.width <= 0:
            raise ValueError(
                f"Invalid ROS image shape height={msg.height}, width={msg.width}"
            )

        msg_dtype = base_dtype.newbyteorder(">" if msg.is_bigendian else "<")
        row_bytes = int(msg.step)
        pixel_row_bytes = int(msg.width) * channels * base_dtype.itemsize
        if row_bytes < pixel_row_bytes:
            raise ValueError(
                f"ROS image step ({row_bytes}) is smaller than pixel row bytes ({pixel_row_bytes})"
            )

        required_bytes = int(msg.height) * row_bytes
        if len(msg.data) < required_bytes:
            raise ValueError(
                f"ROS image payload too small: got {len(msg.data)} bytes, expected at least {required_bytes}"
            )

        raw = np.frombuffer(msg.data, dtype=np.uint8, count=required_bytes)
        raw = raw.reshape((int(msg.height), row_bytes))[:, :pixel_row_bytes]

        decoded = raw.view(msg_dtype)
        if channels == 1:
            decoded = decoded.reshape((int(msg.height), int(msg.width)))
        else:
            decoded = decoded.reshape((int(msg.height), int(msg.width), channels))

        if not decoded.dtype.isnative:
            decoded = decoded.byteswap().view(base_dtype)

        return decoded

    @classmethod
    def _infer_ros_encoding_from_cv_image(cls, cv_image: npt.NDArray) -> str:
        if cv_image.ndim == 2:
            channel_count = 1
        elif cv_image.ndim == 3:
            channel_count = int(cv_image.shape[2])
        else:
            raise ValueError(
                f"Expected 2D or 3D image array, got shape {cv_image.shape} with ndim={cv_image.ndim}"
            )

        dtype = cv_image.dtype
        if channel_count == 1:
            if dtype == np.uint8:
                return "mono8"
            if dtype == np.uint16:
                return "mono16"

        if dtype == np.uint8:
            if channel_count == 3:
                # OpenCV convention for 3-channel color images.
                return "bgr8"
            if channel_count == 4:
                return "bgra8"
            return f"8UC{channel_count}"
        if dtype == np.int8:
            return f"8SC{channel_count}"
        if dtype == np.uint16:
            return f"16UC{channel_count}"
        if dtype == np.int16:
            return f"16SC{channel_count}"
        if dtype == np.int32:
            return f"32SC{channel_count}"
        if dtype == np.float32:
            return f"32FC{channel_count}"
        if dtype == np.float64:
            return f"64FC{channel_count}"

        raise ValueError(
            f"Unsupported dtype for ROS image conversion: dtype={dtype}, channels={channel_count}"
        )
