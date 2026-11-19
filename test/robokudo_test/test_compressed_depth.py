"""
Compressed depth decoding preserves samples and reports invalid input.
"""

from __future__ import annotations

# %% Imports

import struct

import cv2
import numpy as np
import pytest
from sensor_msgs.msg import CompressedImage

from robokudo.datastructures import ROSImageEncoding
from robokudo.io.camera_interface import depth_convert_workaround
from robokudo.io.exceptions import (
    CompressedDepthDecodingFailed,
    UnsupportedDepthCompression,
    UnsupportedDepthEncoding,
)

# %% Compressed acquisitions


@pytest.fixture
def compressed_depth_message() -> CompressedImage:
    """
    Provide a PNG depth payload with a compressed-depth transport header.
    """
    _, encoded_depth = cv2.imencode(".png", np.array([[2, 4]], dtype=np.uint16))
    return CompressedImage(
        format=f"{ROSImageEncoding.UINT16_SINGLE_CHANNEL}; compressedDepth png",
        data=struct.pack("iff", 0, 1.0, 0.0) + encoded_depth.tobytes(),
    )


def test_compressed_depth_preserves_unsigned_samples(
    compressed_depth_message: CompressedImage,
) -> None:
    """
    Unsigned depth samples remain in their original millimeter representation.
    """
    depth = depth_convert_workaround(compressed_depth_message)

    assert depth.dtype == np.uint16
    np.testing.assert_array_equal(depth, [[2, 4]])


def test_compressed_depth_converts_quantized_meters(
    compressed_depth_message: CompressedImage,
) -> None:
    """
    Quantized floating-point depth is reconstructed as unsigned millimeters.
    """
    compressed_depth_message.format = (
        f"{ROSImageEncoding.FLOAT32_SINGLE_CHANNEL}; compressedDepth png"
    )

    depth = depth_convert_workaround(compressed_depth_message)

    assert depth.dtype == np.uint16
    np.testing.assert_array_equal(depth, [[500, 250]])


def test_compressed_depth_rejects_other_compression(
    compressed_depth_message: CompressedImage,
) -> None:
    """
    A different transport cannot be interpreted as compressed depth.
    """
    compressed_depth_message.format = f"{ROSImageEncoding.UINT16_SINGLE_CHANNEL}; jpeg"

    with pytest.raises(UnsupportedDepthCompression):
        depth_convert_workaround(compressed_depth_message)


def test_compressed_depth_rejects_undecodable_payload(
    compressed_depth_message: CompressedImage,
) -> None:
    """
    An invalid PNG payload raises the decoder-specific exception.
    """
    compressed_depth_message.data = struct.pack("iff", 0, 1.0, 0.0) + b"invalid PNG"

    with pytest.raises(CompressedDepthDecodingFailed):
        depth_convert_workaround(compressed_depth_message)


def test_compressed_depth_rejects_unsupported_encoding(
    compressed_depth_message: CompressedImage,
) -> None:
    """
    A decodable image still requires a supported depth representation.
    """
    compressed_depth_message.format = f"{ROSImageEncoding.BGR8}; compressedDepth png"

    with pytest.raises(UnsupportedDepthEncoding):
        depth_convert_workaround(compressed_depth_message)
