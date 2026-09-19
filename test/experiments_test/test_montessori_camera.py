"""
Tests for reading the camera's colour and depth messages, plain and transport-
compressed.
"""

from __future__ import annotations

import cv2
import numpy as np
import pytest

from experiments.montessori.perception.camera import (
    CameraIntrinsics,
    CompressedImageFormat,
    DepthQuantization,
    ImageEncoding,
    RgbdFrame,
    decode_color_image,
    decode_compressed_color_image,
    decode_compressed_depth_image,
    decode_depth_image,
)
from experiments.montessori.perception.exceptions import (
    DepthAndColourNotRegistered,
    UndecodableCompressedImage,
    UnsupportedImageEncoding,
)

# %% reading camera messages


def test_colour_image_is_read_into_opencv_channel_order():
    height, width = 2, 3
    red_pixel = bytes([255, 0, 0]) * (height * width)

    image = decode_color_image(red_pixel, height, width, width * 3, ImageEncoding.RGB8)

    assert image.shape == (height, width, 3)
    assert image[0, 0].tolist() == [0, 0, 255]


def test_colour_image_honours_a_row_stride_wider_than_the_image():
    height, width, step = 2, 2, 8
    data = bytes([1, 2, 3, 4, 5, 6, 0, 0] + [7, 8, 9, 10, 11, 12, 0, 0])

    image = decode_color_image(data, height, width, step, ImageEncoding.BGR8)

    assert image[1, 1].tolist() == [10, 11, 12]


def test_millimetre_depth_is_read_as_metres():
    data = np.array([[1500, 0]], dtype=np.uint16).tobytes()

    depth = decode_depth_image(data, 1, 2, 4, ImageEncoding.DEPTH_IN_MILLIMETRES)

    assert depth[0, 0] == pytest.approx(1.5)
    assert depth[0, 1] == pytest.approx(0.0)


def test_an_unknown_encoding_is_refused():
    with pytest.raises(UnsupportedImageEncoding):
        decode_color_image(b"", 0, 0, 0, "mono8")


def test_a_frame_whose_images_are_not_registered_is_refused():
    intrinsics = CameraIntrinsics(1.0, 1.0, 0.0, 0.0)

    with pytest.raises(DepthAndColourNotRegistered):
        RgbdFrame(
            color=np.zeros((4, 4, 3), dtype=np.uint8),
            depth=np.zeros((2, 2), dtype=np.float32),
            intrinsics=intrinsics,
            reference_frame_T_camera=np.eye(4),
        )


# %% reading transport-compressed camera messages

COLOR_FORMAT_FIELD = "rgb8; jpeg compressed bgr8"
"""
The ``format`` the camera stamps on its compressed colour stream.
"""

DEPTH_FORMAT_FIELD = "16UC1; compressedDepth png"
"""
The ``format`` the camera stamps on its compressed depth stream.
"""


def encode_compressed_depth(
    quantized: np.ndarray, quantization: DepthQuantization
) -> bytes:
    """
    Build the payload ``compressedDepth`` publishes: its header, then a PNG.

    :param quantized: The image as ``compressedDepth`` stores it.
    :param quantization: The header to put in front of the PNG.
    :return: The bytes a ``sensor_msgs/CompressedImage`` would carry.
    """
    return quantization.to_header_bytes() + cv2.imencode(".png", quantized)[1].tobytes()


def test_a_colour_format_field_names_the_encoding_its_payload_is_stored_in():
    parsed = CompressedImageFormat.from_format_field(COLOR_FORMAT_FIELD)

    assert parsed.source_encoding == ImageEncoding.RGB8
    assert parsed.payload_encoding == ImageEncoding.BGR8


def test_a_depth_format_field_names_no_payload_encoding():
    parsed = CompressedImageFormat.from_format_field(DEPTH_FORMAT_FIELD)

    assert parsed.source_encoding == ImageEncoding.DEPTH_IN_MILLIMETRES
    assert parsed.payload_encoding is None


def test_a_compressed_colour_image_is_read_into_opencv_channel_order():
    blue_pixel = np.full((2, 3, 3), (255, 0, 0), dtype=np.uint8)
    payload = cv2.imencode(".png", blue_pixel)[1].tobytes()

    image = decode_compressed_color_image(payload, "bgr8; png compressed bgr8")

    assert image.shape == (2, 3, 3)
    assert image[0, 0].tolist() == [255, 0, 0]


def test_a_compressed_colour_payload_stored_in_rgb_order_is_swapped():
    payload_pixel = np.full((1, 1, 3), (255, 0, 0), dtype=np.uint8)
    payload = cv2.imencode(".png", payload_pixel)[1].tobytes()

    image = decode_compressed_color_image(payload, "rgb8; png compressed rgb8")

    assert image[0, 0].tolist() == [0, 0, 255]


def test_compressed_millimetre_depth_is_read_as_metres():
    millimetres = np.array([[1500, 0]], dtype=np.uint16)
    payload = encode_compressed_depth(millimetres, DepthQuantization(0, 0.0, 0.0))

    depth = decode_compressed_depth_image(payload, DEPTH_FORMAT_FIELD)

    assert depth[0, 0] == pytest.approx(1.5)
    assert depth[0, 1] == pytest.approx(0.0)


def test_compressed_metre_depth_is_read_back_through_its_own_quantisation():
    quantization = DepthQuantization(0, 1000.0, 100.0)
    metres = 2.5
    quantized = np.array(
        [[quantization.quantization_a / metres + quantization.quantization_b, 0]],
        dtype=np.uint16,
    )
    payload = encode_compressed_depth(quantized, quantization)

    depth = decode_compressed_depth_image(payload, "32FC1; compressedDepth png")

    assert depth[0, 0] == pytest.approx(metres, abs=1e-3)
    assert depth[0, 1] == pytest.approx(0.0)


def test_a_compressed_image_whose_payload_is_not_an_image_is_refused():
    with pytest.raises(UndecodableCompressedImage):
        decode_compressed_color_image(b"not an image", COLOR_FORMAT_FIELD)


def test_a_compressed_depth_image_in_an_unknown_encoding_is_refused():
    payload = encode_compressed_depth(
        np.zeros((1, 1), dtype=np.uint16), DepthQuantization(0, 0.0, 0.0)
    )

    with pytest.raises(UnsupportedImageEncoding):
        decode_compressed_depth_image(payload, "mono8; compressedDepth png")
