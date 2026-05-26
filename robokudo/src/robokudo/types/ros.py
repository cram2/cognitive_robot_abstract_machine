"""ROS message type definitions.

This module provides Python class representations of common ROS message types.
These classes are used to maintain compatibility with ROS message structures
while working in pure Python environments.

The module includes:

* Standard ROS header
* Camera-related message types
* Region of interest definitions
"""

from dataclasses import dataclass, field

from typing_extensions import List

from robokudo.types.core import Type


@dataclass
class Header(Type):
    """ROS message header type.

    Standard ROS message header containing sequence number, timestamp,
    and coordinate frame information.
    """

    seq: int = 0
    """
    Sequence number
    """

    frame_id: str = str("")
    """
    Coordinate frame identifier
    """

    stamp: float = 0.0
    """
    Time stamp in seconds
    """


@dataclass
class RegionOfInterest(Type):
    """ROS region of interest message type.

    Defines a rectangular region within an image.
    """

    x_offset: int = 0
    """
    X coordinate of top-left corner
    """

    y_offset: int = 0
    """
    Y coordinate of top-left corner
    """

    height: int = 0
    """
    Height of region in pixels
    """

    width: int = 0
    """
    Width of region in pixels
    """

    do_rectify: bool = False
    """
    Whether to rectify the region
    """


@dataclass
class CameraInfo(Type):
    """ROS camera calibration and metadata message type.

    Contains camera calibration data and image metadata including:
    * Image dimensions
    * Distortion model and parameters
    * Camera matrices (K, R, P)
    * ROI and binning information
    """

    header: Header = field(default_factory=Header)
    """
    Message header
    """

    height: int = 0
    """
    Image height in pixels
    """

    width: int = 0
    """
    Image width in pixels
    """

    distortion_model: str = str("")
    """
    Name of distortion model
    """

    D: List[float] = field(default_factory=list)  # list of floats
    """
    Distortion parameters
    """

    K: List[float] = field(default_factory=list)  # 9-dim list of floats
    """
    Intrinsic camera matrix (3x3)
    """

    R: List[float] = field(default_factory=list)  # 9-dim list of floats
    """
    Rotation matrix (3x3)
    """

    P: List[float] = field(default_factory=list)  # 12-dim list of floats
    """
    Projection matrix (3x4)
    """

    binning_x: int = 0
    """
    Horizontal binning factor
    """

    binning_y: int = 0
    """
    Vertical binning factor
    """

    roi: RegionOfInterest = field(default_factory=RegionOfInterest)
    """
    Region of interest
    """
