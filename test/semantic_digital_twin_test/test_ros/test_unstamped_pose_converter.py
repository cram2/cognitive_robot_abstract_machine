"""
Unstamped pose conversion needs no frame lookup.
"""

# %% imports

import pytest

from semantic_digital_twin.adapters.ros.ros2_to_semdt_converters import (
    PoseToSemDTConverter,
)
from semantic_digital_twin.adapters.ros.semdt_to_ros2_converters import (
    PoseToRos2Converter,
)
from semantic_digital_twin.spatial_types import Pose

# %% standalone conversion


def test_unstamped_pose_round_trip_without_world() -> None:
    """
    A pose without a header round-trips without creating a digital twin.
    """
    expected = Pose.from_xyz_rpy(0.6, -0.2, 0.9, roll=0.3, pitch=-0.7, yaw=1.1)
    message = PoseToRos2Converter.convert(expected)

    result = PoseToSemDTConverter.convert(message)

    assert result.to_np() == pytest.approx(expected.to_np())
    assert result.reference_frame is None
