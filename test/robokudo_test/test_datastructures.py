"""
Shared ROS image encodings preserve image conversion contracts.
"""

from __future__ import annotations

# %% Imports

import numpy as np
import pytest

from robokudo.datastructures.ros_image_encoding import ROSImageEncoding
from robokudo.utils.cv_bridge_workaround import CVBridgeWorkaround

# %% ROS image encodings


@pytest.mark.parametrize(
    "encoding, image",
    [
        (ROSImageEncoding.BGR8, np.zeros((2, 3, 3), dtype=np.uint8)),
        (
            ROSImageEncoding.UINT16_SINGLE_CHANNEL,
            np.full((2, 3), 1500, dtype=np.uint16),
        ),
        (
            ROSImageEncoding.FLOAT32_SINGLE_CHANNEL,
            np.full((2, 3), 1.5, dtype=np.float32),
        ),
    ],
)
def test_shared_encoding_preserves_ros_image(
    encoding: ROSImageEncoding, image: np.ndarray
) -> None:
    """
    ROS messages and converted arrays retain the selected encoding and pixels.
    """
    bridge = CVBridgeWorkaround()
    message = bridge.cv2_to_imgmsg(image, encoding=encoding)

    assert message.encoding == encoding.value
    np.testing.assert_array_equal(
        bridge.imgmsg_to_cv2(message, desired_encoding=encoding), image
    )
