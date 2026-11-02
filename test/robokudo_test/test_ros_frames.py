"""
ROS frame identifier normalization and validation.
"""

from __future__ import annotations

# %% Imports
import pytest

from robokudo.exceptions import InvalidROSFrameId
from robokudo.utils.ros_frames import find_bodies_for_ros_frame, normalize_ros_frame_id
from semantic_digital_twin.world import World

# %% Invalid identifiers


@pytest.mark.parametrize("frame_id", ["", "/", "///"])
def test_normalization_rejects_empty_ros_identifier(frame_id: str) -> None:
    """
    Invalid ROS identifiers raise a frame-specific exception with the supplied value.
    """
    with pytest.raises(InvalidROSFrameId) as raised:
        normalize_ros_frame_id(frame_id)
    assert raised.value.frame_id == frame_id


def test_body_lookup_rejects_empty_ros_identifier() -> None:
    """
    Body lookup preserves the frame-specific normalization error.
    """
    with pytest.raises(InvalidROSFrameId):
        find_bodies_for_ros_frame(World(), "")
