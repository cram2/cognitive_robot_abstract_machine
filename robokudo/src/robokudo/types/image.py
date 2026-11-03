"""
Image orientation choices shared by camera configurations and interfaces.
"""

from __future__ import annotations

# %% Imports
from enum import StrEnum


# %% Image orientation
class ImageRotation(StrEnum):
    """
    Rotations applied to an image and its camera calibration.
    """

    COUNTERCLOCKWISE_90 = "90_ccw"
    """
    A quarter turn counterclockwise.
    """

    CLOCKWISE_90 = "90_cw"
    """
    A quarter turn clockwise.
    """

    HALF_TURN = "180"
    """
    A half turn.
    """
