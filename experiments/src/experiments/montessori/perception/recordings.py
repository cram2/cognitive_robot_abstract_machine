"""
Reading the camera back out of a recorded rosbag.

A recording holds what the node subscribes to, so the same frames can be run through the
same pipeline afterwards, on a desk with no robot in front of it. Where the recording
carries no transform tree of its own -- the scene bags record the camera alone -- the
camera's pose is read from a recording that does, which is sound on this setup only
because the camera is bolted to the table it looks at.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path

import numpy as np
import rosbag2_py
from rclpy.serialization import deserialize_message
from sensor_msgs.msg import CameraInfo, CompressedImage, Image
from tf2_msgs.msg import TFMessage
from typing_extensions import Dict, Iterator, List, Optional, Self

from experiments.montessori.perception.camera import (
    CameraIntrinsics,
    CameraTopic,
    ImageTransport,
    RgbdFrame,
    decode_compressed_color_image,
    decode_compressed_depth_image,
    decode_depth_image,
)
from experiments.montessori.perception.exceptions import NothingRecordedOnTopic
from semantic_digital_twin.spatial_types.spatial_types import (
    HomogeneousTransformationMatrix,
)

# %% what a recording carries

RAW_DEPTH_TOPIC = str(CameraTopic.DEPTH).removesuffix(
    f"/{ImageTransport.COMPRESSED_DEPTH}"
)
"""
The uncompressed depth stream, which recordings made before the node moved onto the
compressed transports carry instead of :attr:`CameraTopic.DEPTH`.
"""

REFERENCE_FRAME = "map"
"""
Frame the recordings root their transform tree in, and so the frame detections made on
them come out in.
"""

STORAGE_IDENTIFIER = "mcap"
"""
The storage format these recordings are written in.
"""


class TransformTopic(StrEnum):
    """
    The topics a robot publishes its transform tree on.
    """

    STATIC = "/tf_static"
    DYNAMIC = "/tf"


def open_bag(bag: Path, topics: List[str]) -> rosbag2_py.SequentialReader:
    """
    Open a recording, reading only the topics asked for.

    :param bag: Directory of the recording.
    :param topics: The topics to read; every other one is skipped.
    """
    reader = rosbag2_py.SequentialReader()
    reader.open(
        rosbag2_py.StorageOptions(uri=str(bag), storage_id=STORAGE_IDENTIFIER),
        rosbag2_py.ConverterOptions("", ""),
    )
    reader.set_filter(rosbag2_py.StorageFilter(topics=topics))
    return reader


# %% where the camera stood


@dataclass(frozen=True)
class RecordedTransformTree:
    """
    The fixed part of a transform tree, as one recording published it.

    Only the static edges are kept: on this setup the camera hangs off a pole bolted to
    the table, so every edge between the frame detections are reported in and the
    camera's own is fixed.
    """

    parent_of: Dict[str, str]
    """
    The frame each frame hangs off.
    """

    transform_to: Dict[str, np.ndarray]
    """
    Each frame's pose in its parent, as a 4x4 homogeneous transformation.
    """

    @classmethod
    def of_bag(cls, bag: Path) -> Self:
        """
        Read a recording's static transforms.

        :param bag: Directory of the recording.
        :raises NothingRecordedOnTopic: If the recording published no static transforms.
        """
        reader = open_bag(bag, [str(TransformTopic.STATIC)])
        parent_of: Dict[str, str] = {}
        transform_to: Dict[str, np.ndarray] = {}
        while reader.has_next():
            _, payload, _ = reader.read_next()
            for transform in deserialize_message(payload, TFMessage).transforms:
                parent_of[transform.child_frame_id] = transform.header.frame_id
                translation = transform.transform.translation
                rotation = transform.transform.rotation
                transform_to[transform.child_frame_id] = (
                    HomogeneousTransformationMatrix.from_xyz_quaternion(
                        translation.x,
                        translation.y,
                        translation.z,
                        rotation.x,
                        rotation.y,
                        rotation.z,
                        rotation.w,
                    ).to_np()
                )
        if not parent_of:
            raise NothingRecordedOnTopic(bag.name, str(TransformTopic.STATIC))
        return cls(parent_of=parent_of, transform_to=transform_to)

    def pose_of(self, frame: str, reference_frame: str) -> np.ndarray:
        """
        Compose the chain from a frame up to the one it is wanted in.

        :param frame: The frame whose pose is wanted.
        :param reference_frame: The frame to express it in.
        :return: The pose as a 4x4 homogeneous transformation.
        :raises KeyError: If the chain runs out before reaching the reference frame.
        """
        chain = []
        current = frame
        while current != reference_frame:
            chain.append(self.transform_to[current])
            current = self.parent_of[current]
        pose = np.eye(4)
        for transform in reversed(chain):
            pose = pose @ transform
        return pose


# %% one frame out of a recording


@dataclass(frozen=True)
class RecordedLook:
    """
    One colour image a recording holds, with the depth image published closest to it.
    """

    color_payload: bytes
    """
    The colour image as the camera compressed it.
    """

    color_format: str
    """
    The ``format`` field the camera published that payload under.
    """

    depth: np.ndarray
    """
    Depth in metres, zero where the sensor returned no reading.
    """

    def to_frame(
        self, intrinsics: CameraIntrinsics, reference_frame_T_camera: np.ndarray
    ) -> RgbdFrame:
        """
        Decode this look into the frame a pipeline runs on.

        :param intrinsics: The intrinsics both images were taken with.
        :param reference_frame_T_camera: Where the camera stood.
        :raises UndecodableCompressedImage: If the colour payload does not decode into
            pixels.
        """
        return RgbdFrame(
            color=decode_compressed_color_image(self.color_payload, self.color_format),
            depth=self.depth,
            intrinsics=intrinsics,
            reference_frame_T_camera=reference_frame_T_camera,
        )


@dataclass(frozen=True)
class RecordedCamera:
    """
    The camera one recording holds: how it saw, where it stood, and what it published.

    The looks are read in the order the colour images were published, each paired with
    the newest depth image before it, which is the pairing the live node makes.
    """

    bag: Path
    """
    Directory of the recording the frames are read from.
    """

    reference_frame: str
    """
    Frame the camera's pose, and so every detection, is expressed in.
    """

    camera_bag: Optional[Path] = None
    """
    Recording the camera's pose is read from, or None where this one carries transforms
    of its own.
    """

    @property
    def transform_bag(self) -> Path:
        """
        The recording the camera's pose is read from.
        """
        return self.camera_bag if self.camera_bag is not None else self.bag

    @property
    def camera_info(self) -> CameraInfo:
        """
        The calibration the camera published into this recording.

        :raises NothingRecordedOnTopic: If the recording carries no calibration.
        """
        reader = open_bag(self.bag, [str(CameraTopic.CAMERA_INFO)])
        if not reader.has_next():
            raise NothingRecordedOnTopic(self.bag.name, str(CameraTopic.CAMERA_INFO))
        _, payload, _ = reader.read_next()
        return deserialize_message(payload, CameraInfo)

    @property
    def intrinsics(self) -> CameraIntrinsics:
        """
        How this camera projects, as it reported it.
        """
        return CameraIntrinsics.from_camera_info_matrix(self.camera_info.k)

    @property
    def reference_frame_T_camera(self) -> np.ndarray:
        """
        Where the camera's optical frame stood, as a 4x4 homogeneous transformation.

        :raises NothingRecordedOnTopic: If no recording holds the transform tree.
        """
        return RecordedTransformTree.of_bag(self.transform_bag).pose_of(
            self.camera_info.header.frame_id, self.reference_frame
        )

    def looks(self) -> Iterator[RecordedLook]:
        """
        Every colour image the recording holds, paired with the depth taken before it.

        A colour image published before any depth image has nothing to pair with and is
        skipped, the same way the live node waits for both.
        """
        reader = open_bag(
            self.bag,
            [str(CameraTopic.COLOR), str(CameraTopic.DEPTH), RAW_DEPTH_TOPIC],
        )
        depth: Optional[np.ndarray] = None
        while reader.has_next():
            topic, payload, _ = reader.read_next()
            if topic == str(CameraTopic.COLOR):
                if depth is not None:
                    color = deserialize_message(payload, CompressedImage)
                    yield RecordedLook(
                        color_payload=bytes(color.data),
                        color_format=color.format,
                        depth=depth,
                    )
                continue
            depth = _decode_depth(topic, payload)

    @property
    def color_image_count(self) -> int:
        """
        How many colour images the recording holds, as its own metadata reports.
        """
        metadata = rosbag2_py.Info().read_metadata(str(self.bag), STORAGE_IDENTIFIER)
        return sum(
            topic.message_count
            for topic in metadata.topics_with_message_count
            if topic.topic_metadata.name == str(CameraTopic.COLOR)
        )

    def look_at(self, fraction: float) -> RecordedLook:
        """
        The look a given way through the recording, without holding the rest in memory.

        :param fraction: How far in to take it, from 0 at the first look to 1 at the
            last.
        :raises NothingRecordedOnTopic: If the recording holds no colour image that a
            depth image was published before.
        """
        wanted = max(int(self.color_image_count * fraction), 0)
        for index, look in enumerate(self.looks()):
            if index >= wanted:
                return look
        raise NothingRecordedOnTopic(self.bag.name, str(CameraTopic.COLOR))


def _decode_depth(topic: str, payload: bytes) -> np.ndarray:
    """
    Read a depth message off whichever transport carried it.

    :param topic: The topic it arrived on.
    :param payload: The serialized message.
    :return: Depth in metres, zero where the sensor returned no reading.
    """
    if topic == str(CameraTopic.DEPTH):
        message = deserialize_message(payload, CompressedImage)
        return decode_compressed_depth_image(message.data, message.format)
    message = deserialize_message(payload, Image)
    return decode_depth_image(
        message.data, message.height, message.width, message.step, message.encoding
    )
