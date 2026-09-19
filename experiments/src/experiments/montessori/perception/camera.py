"""
The camera data a detection runs on: one registered colour/depth pair, its pinhole
intrinsics, and where the camera stood when it was taken.
"""

from __future__ import annotations

import struct
from dataclasses import dataclass
from enum import StrEnum

import cv2
import numpy as np
from typing_extensions import Optional, Self

from experiments.montessori.perception.exceptions import (
    DepthAndColourNotRegistered,
    UndecodableCompressedImage,
    UnsupportedImageEncoding,
)
from semantic_digital_twin.spatial_types.math import inverse_frame

# %% encodings


class ImageEncoding(StrEnum):
    """
    The ``sensor_msgs/Image`` encodings this package reads.
    """

    RGB8 = "rgb8"
    BGR8 = "bgr8"
    DEPTH_IN_MILLIMETRES = "16UC1"
    DEPTH_IN_METRES = "32FC1"


MILLIMETRES_PER_METRE = 1000.0
"""
Divisor turning a :attr:`ImageEncoding.DEPTH_IN_MILLIMETRES` reading into metres.
"""


class ImageTransport(StrEnum):
    """
    The compressed streams ``image_transport`` offers alongside a camera's raw one.

    A camera advertises each of these as a sub-topic of its raw image topic, so
    ``/camera/color/image_raw`` also carries ``/camera/color/image_raw/compressed``.
    """

    COMPRESSED = "compressed"
    COMPRESSED_DEPTH = "compressedDepth"


FORMAT_FIELD_SEPARATOR = ";"
"""
Separates the source encoding from the codec in a ``CompressedImage``'s ``format``.
"""

PAYLOAD_ENCODING_MARKER = "compressed"
"""
Word a ``format`` field puts in front of the encoding its payload is stored in.
"""


class CameraTopic(StrEnum):
    """
    The camera streams this package reads.

    The depth stream must be the one the driver has registered onto colour: both carry
    the same frame, resolution and intrinsics, so a pixel names the same ray in each.

    Both images are read over a compressed transport rather than raw. A raw frame of
    this camera is several megabytes, which a wireless link cannot carry: the datagrams
    are fragmented and almost none of them arrive whole, so a node subscribed to the raw
    stream is served nothing at all.
    """

    COLOR = f"/camera/color/image_raw/{ImageTransport.COMPRESSED}"
    DEPTH = f"/camera/depth/image_raw/{ImageTransport.COMPRESSED_DEPTH}"
    CAMERA_INFO = "/camera/color/camera_info"


# %% intrinsics


@dataclass(frozen=True)
class CameraIntrinsics:
    """
    A pinhole camera's focal lengths and principal point, in pixels.
    """

    focal_length_x: float
    """
    Focal length along the image's x-axis.
    """

    focal_length_y: float
    """
    Focal length along the image's y-axis.
    """

    principal_point_x: float
    """
    X-coordinate of the point where the optical axis meets the image.
    """

    principal_point_y: float
    """
    Y-coordinate of the point where the optical axis meets the image.
    """

    @classmethod
    def from_camera_info_matrix(cls, camera_info_matrix: np.ndarray) -> Self:
        """
        Read the intrinsics out of a ``sensor_msgs/CameraInfo``'s own ``k``.

        :param camera_info_matrix: The 3x3 intrinsic matrix, in any shape that reshapes
            to ``(3, 3)``.
        """
        matrix = np.asarray(camera_info_matrix, dtype=float).reshape(3, 3)
        return cls(
            focal_length_x=float(matrix[0, 0]),
            focal_length_y=float(matrix[1, 1]),
            principal_point_x=float(matrix[0, 2]),
            principal_point_y=float(matrix[1, 2]),
        )

    def to_matrix(self) -> np.ndarray:
        """
        :return: These intrinsics as a 3x3 projection matrix.
        """
        return np.array(
            [
                [self.focal_length_x, 0.0, self.principal_point_x],
                [0.0, self.focal_length_y, self.principal_point_y],
                [0.0, 0.0, 1.0],
            ]
        )

    def deproject(self, pixels: np.ndarray, depths: np.ndarray) -> np.ndarray:
        """
        Turn pixels with known depth into points in the camera's optical frame, where x
        points right, y down, and z along the optical axis.

        :param pixels: Pixel coordinates as ``(n, 2)`` ``(x, y)`` pairs.
        :param depths: Depth of each pixel in metres, shape ``(n,)``.
        :return: The points, shape ``(n, 3)``.
        """
        pixels = np.asarray(pixels, dtype=float).reshape(-1, 2)
        depths = np.asarray(depths, dtype=float).reshape(-1)
        return np.stack(
            [
                (pixels[:, 0] - self.principal_point_x) * depths / self.focal_length_x,
                (pixels[:, 1] - self.principal_point_y) * depths / self.focal_length_y,
                depths,
            ],
            axis=1,
        )


# %% frames


@dataclass(frozen=True)
class RgbdFrame:
    """
    One colour image with the depth image registered onto it, plus the camera's pose in
    the frame detections are reported in.

    Registered means the two images share a resolution and a set of intrinsics, so pixel
    ``(x, y)`` names the same ray in both.
    """

    color: np.ndarray
    """
    The colour image in OpenCV's channel order, shape ``(height, width, 3)`` of
    ``uint8`` blue/green/red.
    """

    depth: np.ndarray
    """
    Depth in metres, shape ``(height, width)``; zero marks a pixel the sensor returned
    no reading for.
    """

    intrinsics: CameraIntrinsics
    """
    The intrinsics both images were taken with.
    """

    reference_frame_T_camera: np.ndarray
    """
    The camera's optical frame expressed in the frame detections are reported in, as a
    4x4 homogeneous transformation.
    """

    def __post_init__(self) -> None:
        if self.color.shape[:2] != self.depth.shape[:2]:
            raise DepthAndColourNotRegistered(
                self.color.shape[:2], self.depth.shape[:2]
            )

    @property
    def height(self) -> int:
        """
        Number of image rows.
        """
        return int(self.color.shape[0])

    @property
    def width(self) -> int:
        """
        Number of image columns.
        """
        return int(self.color.shape[1])

    def project(self, points: np.ndarray) -> np.ndarray:
        """
        Where points in the world fall in this frame's image.

        :param points: Points in the frame the camera's pose was given in, as ``(n, 3)``
            ``(x, y, z)`` in metres.
        :return: The pixels they land on, as ``(n, 2)`` ``(x, y)`` pairs.
        """
        points = np.asarray(points, dtype=float).reshape(-1, 3)
        camera_T_reference_frame = inverse_frame(self.reference_frame_T_camera)
        camera_points = (
            camera_T_reference_frame @ np.hstack([points, np.ones((len(points), 1))]).T
        )[:3]
        pixels = self.intrinsics.to_matrix() @ camera_points
        return (pixels[:2] / pixels[2]).T

    def depth_at(self, pixels: np.ndarray) -> np.ndarray:
        """
        Read the depth of a set of pixels, dropping the ones the sensor gave no reading
        for.

        :param pixels: Pixel coordinates as ``(n, 2)`` ``(x, y)`` pairs.
        :return: The depths in metres that were actually measured, shape ``(m,)`` with
            ``m <= n``.
        """
        pixels = np.asarray(pixels, dtype=int).reshape(-1, 2)
        inside = (
            (pixels[:, 0] >= 0)
            & (pixels[:, 0] < self.width)
            & (pixels[:, 1] >= 0)
            & (pixels[:, 1] < self.height)
        )
        pixels = pixels[inside]
        depths = self.depth[pixels[:, 1], pixels[:, 0]]
        return depths[depths > 0.0]


def decode_color_image(
    data: bytes, height: int, width: int, step: int, encoding: str
) -> np.ndarray:
    """
    Read a ``sensor_msgs/Image``'s bytes into an OpenCV-ordered colour image.

    :param data: The message's raw bytes.
    :param height: Number of rows.
    :param width: Number of columns.
    :param step: Row stride in bytes, which may exceed ``width * 3``.
    :param encoding: The message's own encoding.
    :return: The image, shape ``(height, width, 3)`` of ``uint8`` blue/green/red.
    :raises UnsupportedImageEncoding: If the encoding is neither ``rgb8`` nor ``bgr8``.
    """
    if encoding not in (ImageEncoding.RGB8, ImageEncoding.BGR8):
        raise UnsupportedImageEncoding(
            encoding, [ImageEncoding.RGB8, ImageEncoding.BGR8]
        )
    rows = np.frombuffer(data, dtype=np.uint8).reshape(height, step)
    image = rows[:, : width * 3].reshape(height, width, 3)
    if encoding == ImageEncoding.RGB8:
        return image[:, :, ::-1].copy()
    return image.copy()


def decode_depth_image(
    data: bytes, height: int, width: int, step: int, encoding: str
) -> np.ndarray:
    """
    Read a ``sensor_msgs/Image``'s bytes into a depth image in metres.

    :param data: The message's raw bytes.
    :param height: Number of rows.
    :param width: Number of columns.
    :param step: Row stride in bytes.
    :param encoding: The message's own encoding.
    :return: Depth in metres, shape ``(height, width)``, zero where unmeasured.
    :raises UnsupportedImageEncoding: If the encoding is neither ``16UC1`` nor
        ``32FC1``.
    """
    if encoding == ImageEncoding.DEPTH_IN_MILLIMETRES:
        rows = np.frombuffer(data, dtype=np.uint16).reshape(height, step // 2)
        return rows[:, :width].astype(np.float32) / MILLIMETRES_PER_METRE
    if encoding == ImageEncoding.DEPTH_IN_METRES:
        rows = np.frombuffer(data, dtype=np.float32).reshape(height, step // 4)
        return np.nan_to_num(rows[:, :width].astype(np.float32))
    raise UnsupportedImageEncoding(
        encoding,
        [ImageEncoding.DEPTH_IN_MILLIMETRES, ImageEncoding.DEPTH_IN_METRES],
    )


# %% transport-compressed messages


COMPRESSED_DEPTH_HEADER_LAYOUT = "iff"
"""
Layout of the header a ``compressedDepth`` payload begins with, as :mod:`struct` reads
it: the compression format, then the two quantization terms.
"""

COMPRESSED_DEPTH_HEADER_SIZE = struct.calcsize(COMPRESSED_DEPTH_HEADER_LAYOUT)
"""
Number of bytes :data:`COMPRESSED_DEPTH_HEADER_LAYOUT` occupies.
"""


@dataclass(frozen=True)
class CompressedImageFormat:
    """
    A ``sensor_msgs/CompressedImage``'s ``format`` field, split into the parts that say
    how to read its payload.

    ``image_transport`` writes colour as ``rgb8; jpeg compressed bgr8`` and depth as
    ``16UC1; compressedDepth png``: the encoding the camera produced comes first, and
    what the codec did with it follows.
    """

    source_encoding: str
    """
    The encoding the camera published the image in before it was compressed.
    """

    payload_encoding: Optional[str]
    """
    The encoding the payload's pixels are stored in, or None where the codec names no
    encoding of its own, as ``compressedDepth`` does not.
    """

    @classmethod
    def from_format_field(cls, format_field: str) -> Self:
        """
        Read a message's own ``format`` field.

        :param format_field: The field as the publisher wrote it.
        """
        source_encoding, _, codec = format_field.partition(FORMAT_FIELD_SEPARATOR)
        words = codec.split()
        marks_payload = PAYLOAD_ENCODING_MARKER in words and words.index(
            PAYLOAD_ENCODING_MARKER
        ) + 1 < len(words)
        return cls(
            source_encoding=source_encoding.strip(),
            payload_encoding=(
                words[words.index(PAYLOAD_ENCODING_MARKER) + 1]
                if marks_payload
                else None
            ),
        )


@dataclass(frozen=True)
class DepthQuantization:
    """
    The header ``compressedDepth`` writes in front of its PNG payload.

    A :attr:`ImageEncoding.DEPTH_IN_METRES` image cannot be stored in a 16-bit PNG as it
    stands, so the codec maps each distance onto an integer and records here what it
    would take to undo that.
    """

    compression_format: int
    """
    Which of the codec's own compression formats produced the payload.
    """

    quantization_a: float
    """
    Numerator of the mapping from a stored integer back to a distance.
    """

    quantization_b: float
    """
    Offset subtracted from a stored integer before it is divided into
    :attr:`quantization_a`.
    """

    @classmethod
    def from_header_bytes(cls, header: bytes) -> Self:
        """
        Read the header off the front of a ``compressedDepth`` payload.

        :param header: The header's own :data:`COMPRESSED_DEPTH_HEADER_SIZE` bytes.
        """
        return cls(*struct.unpack(COMPRESSED_DEPTH_HEADER_LAYOUT, header))

    def to_header_bytes(self) -> bytes:
        """
        :return: This header as ``compressedDepth`` writes it.
        """
        return struct.pack(
            COMPRESSED_DEPTH_HEADER_LAYOUT,
            self.compression_format,
            self.quantization_a,
            self.quantization_b,
        )

    def to_metres(self, quantized: np.ndarray) -> np.ndarray:
        """
        Undo the mapping the codec applied to a
        :attr:`ImageEncoding.DEPTH_IN_METRES` image.

        :param quantized: The integers the payload stored.
        :return: Depth in metres, zero where the sensor returned no reading.
        """
        metres = np.zeros(quantized.shape, dtype=np.float32)
        measured = quantized > 0
        metres[measured] = self.quantization_a / (
            quantized[measured].astype(np.float32) - self.quantization_b
        )
        return metres


def decode_compressed_color_image(data: bytes, image_format: str) -> np.ndarray:
    """
    Read a compressed ``sensor_msgs/CompressedImage`` into an OpenCV-ordered colour
    image.

    :param data: The message's payload.
    :param image_format: The message's own ``format`` field.
    :return: The image, shape ``(height, width, 3)`` of ``uint8`` blue/green/red.
    :raises UndecodableCompressedImage: If the payload does not decode into pixels.
    """
    image = cv2.imdecode(np.frombuffer(data, dtype=np.uint8), cv2.IMREAD_COLOR)
    if image is None:
        raise UndecodableCompressedImage(image_format, len(data))
    if CompressedImageFormat.from_format_field(image_format).payload_encoding == (
        ImageEncoding.RGB8
    ):
        return image[:, :, ::-1].copy()
    return image


def decode_compressed_depth_image(data: bytes, image_format: str) -> np.ndarray:
    """
    Read a ``compressedDepth`` ``sensor_msgs/CompressedImage`` into a depth image in
    metres.

    :param data: The message's payload, header included.
    :param image_format: The message's own ``format`` field.
    :return: Depth in metres, shape ``(height, width)``, zero where unmeasured.
    :raises UndecodableCompressedImage: If the payload does not decode into pixels.
    :raises UnsupportedImageEncoding: If the camera published the depth in an encoding
        this package cannot read.
    """
    source_encoding = CompressedImageFormat.from_format_field(
        image_format
    ).source_encoding
    if source_encoding not in (
        ImageEncoding.DEPTH_IN_MILLIMETRES,
        ImageEncoding.DEPTH_IN_METRES,
    ):
        raise UnsupportedImageEncoding(
            source_encoding,
            [ImageEncoding.DEPTH_IN_MILLIMETRES, ImageEncoding.DEPTH_IN_METRES],
        )
    quantized = cv2.imdecode(
        np.frombuffer(data[COMPRESSED_DEPTH_HEADER_SIZE:], dtype=np.uint8),
        cv2.IMREAD_UNCHANGED,
    )
    if quantized is None:
        raise UndecodableCompressedImage(image_format, len(data))
    if source_encoding == ImageEncoding.DEPTH_IN_MILLIMETRES:
        return quantized.astype(np.float32) / MILLIMETRES_PER_METRE
    return DepthQuantization.from_header_bytes(
        data[:COMPRESSED_DEPTH_HEADER_SIZE]
    ).to_metres(quantized)
