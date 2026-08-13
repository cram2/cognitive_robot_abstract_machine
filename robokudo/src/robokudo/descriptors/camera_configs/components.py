from dataclasses import dataclass

from typing_extensions import Tuple


@dataclass(kw_only=True)
class DepthComponent:
    """Camera component with configuration for depth input data."""

    topic_depth: str
    """ROS depth topic name. Must be set by camera configs using this component."""

    depthOffset: int = 0
    """Depth offset."""

    depth_hints: str = "compressedDepth"
    """Datatype hint for the depth topic. For example: 'compressedDepth' or 'raw'."""


@dataclass(kw_only=True)
class ColorComponent:
    """Camera component with configuration for color input data."""

    topic_cam_info: str
    """ROS camera info topic name. Must be set by camera configs using this component."""

    topic_color: str
    """ROS color topic name. Must be set by camera configs using this component."""

    filterBlurredImages: bool = True
    """Flag to enable/disable filtering of blurred images"""

    color_hints: str = "compressed"
    """Datatype hint for the color topic. For example: 'compressed' or 'raw'."""


@dataclass(kw_only=True)
class TfComponent:
    """Camera component with configuration for transform handling."""

    tf_from: str
    """Frame ID of the camera's optical frame. Must be set by camera configs using this component."""

    tf_to: str = "map"
    """World frame ID to transform the camera's optical frame to."""

    lookup_viewpoint: bool = True
    """Whether to lookup the viewpoint between the camera and the world."""


@dataclass(kw_only=True)
class StableViewpointComponent:
    """Camera component with configuration for viewpoint stabilization."""

    only_stable_viewpoints: bool = True
    """Flag to use only stable viewpoints"""

    max_viewpoint_distance: float = 0.01
    """Maximum allowed distance for viewpoint changes"""

    max_viewpoint_rotation: float = 1.0
    """Maximum allowed rotation for viewpoint changes"""


@dataclass(kw_only=True)
class WorldDescriptorComponent:
    """Camera component with configuration for world descriptors."""

    world_descriptor: str = "world_iai_kitchen20"
    """Filename of the world descriptor configuration"""


@dataclass(kw_only=True)
class RGBDComponent(DepthComponent, ColorComponent):
    """Camera component with configuration for RGB-D input data."""

    color2depth_ratio: Tuple[float, float] = (1.0, 1.0)
    """
    If the resolution of the depth image differs from the color image,
    we need to define the factor for (x, y).
    Example: (0.5,0.5) for a 640x480 depth image compared to a
    1280x960 rgb image Otherwise, just put (1,1) here
    """

    hi_res_mode: bool = False
    """Setting this to true will apply some workarounds to match the depth data to RGB on the Kinect."""
