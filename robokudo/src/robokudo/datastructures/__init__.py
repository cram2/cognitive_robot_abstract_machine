"""
Shared data definitions for RoboKudo sensor interfaces.
"""

from __future__ import annotations

# %% Imports

from enum import StrEnum

# %% ROS image encodings


class ROSImageEncoding(StrEnum):
    """
    ROS image layouts independent of the physical units of their samples.
    """

    BGR8 = "bgr8"
    """
    Three unsigned 8-bit channels in blue, green, red order.
    """

    UINT16_SINGLE_CHANNEL = "16UC1"
    """
    One unsigned 16-bit integer channel per pixel.
    """

    FLOAT32_SINGLE_CHANNEL = "32FC1"
    """
    One 32-bit floating-point channel per pixel.
    """
