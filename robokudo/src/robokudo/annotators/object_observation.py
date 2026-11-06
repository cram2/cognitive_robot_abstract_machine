"""
An object marked by a mask in a color and depth frame.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import cached_property

import cv2
import numpy
from semantic_digital_twin.datastructures.camera_resolution import CameraResolution
from typing_extensions import TYPE_CHECKING

from robokudo.annotators.exceptions import ImageResolutionMismatch
from robokudo.types.cv import Point2D, Rect

if TYPE_CHECKING:
    import numpy.typing as numpy_typing
    from semantic_digital_twin.datastructures.camera_model import PinholeCameraModel


# %% Observations


@dataclass
class ColorAndDepthFrame:
    """
    A color and a depth image of the same resolution, taken by one camera.
    """

    color_image: numpy_typing.NDArray[numpy.uint8]
    """
    Color image in RGB channel order.
    """

    depth_image: numpy_typing.NDArray[numpy.float32]
    """
    Depth image in meters; pixels without a measurement are zero.
    """

    camera_model: PinholeCameraModel
    """
    Calibration of the camera at the resolution of the images.
    """

    def __post_init__(self) -> None:
        """
        :raises ImageResolutionMismatch: If the images differ in resolution.
        """
        color_height, color_width = self.color_image.shape[:2]
        depth_height, depth_width = self.depth_image.shape[:2]
        if (color_height, color_width) != (depth_height, depth_width):
            raise ImageResolutionMismatch(
                color_image_resolution=CameraResolution(
                    width=color_width, height=color_height
                ),
                depth_image_resolution=CameraResolution(
                    width=depth_width, height=depth_height
                ),
            )


@dataclass
class ObjectObservation:
    """
    One object in a frame, marked by a mask.
    """

    frame: ColorAndDepthFrame
    """
    The frame the object is seen in.
    """

    mask: numpy_typing.NDArray[numpy.bool_]
    """
    Pixels of :attr:`frame` that show the object.
    """

    @cached_property
    def mask_region(self) -> Rect:
        """
        The smallest rectangle containing every pixel of :attr:`mask`.
        """
        left, top, width, height = cv2.boundingRect(self.mask.astype(numpy.uint8))
        return Rect(pos=Point2D(x=left, y=top), width=width, height=height)
