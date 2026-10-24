"""
Color and depth conventions for aligned ray-traced camera images.
"""

from __future__ import annotations

# %% Imports
from dataclasses import dataclass
from enum import StrEnum
import numpy as np
from typing_extensions import TYPE_CHECKING

if TYPE_CHECKING:
    from semantic_digital_twin.datastructures.camera_model import PinholeCameraModel
    from semantic_digital_twin.world_description.world_entity import Body


# %% Rendering conventions
class ColorRenderingMode(StrEnum):
    """
    Select how the ray tracer produces the color image.
    """

    SEMANTIC = "semantic"
    """
    Color visible bodies with their semantic geometry colors.
    """

    TEXTURED = "textured"
    """
    Render surface textures and materials.
    """


class DepthMode(StrEnum):
    """
    Distance convention represented by a rendered depth image.
    """

    RAY_DISTANCE = "ray_distance"
    """
    Distance from the camera along each viewing ray.
    """
    OPTICAL_AXIS = "optical_axis"
    """
    Distance projected onto the camera's optical axis.
    """


# %% Aligned depth and segmentation
@dataclass
class SegmentationDepthFrame:
    """
    Segmentation and meter depths produced by one camera intersection query.
    """

    segmentation: np.ndarray
    """
    World-body indices with negative values for pixels without a hit.
    """
    depth: np.ndarray
    """
    Depths in meters with negative values for pixels without a valid depth.
    """


# %% Aligned RGB-D products
@dataclass
class RenderedRGBDFrame:
    """
    Contain aligned width-first images with their origin at the top-left pixel.
    """

    color_bgr: np.ndarray
    """
    Color image encoded in BGR channel order.
    """

    depth_in_millimeters: np.ndarray
    """
    Depth in the selected convention, encoded in unsigned millimeters.
    """

    segmentation: np.ndarray
    """
    Body-index segmentation image aligned with the color and depth images.
    """

    object_color_map: dict[str, str]
    """
    Mapping from semantic RGB color keys to object names.
    """

    camera_model: PinholeCameraModel
    """
    Effective projection model used to render the frame.
    """


# %% Camera rays
@dataclass
class CameraRays:
    """
    World-space viewing rays and the image pixels they represent.
    """

    origin_points: np.ndarray
    """
    World-space starting point for each viewing ray.
    """
    direction_vectors: np.ndarray
    """
    World-space viewing direction of each ray.
    """
    pixel_coordinates: np.ndarray
    """
    Width-first image coordinates with their origin at the top-left pixel.
    """


@dataclass
class RayIntersections:
    """
    Surface intersections and their source rays and world bodies.
    """

    hit_points: np.ndarray
    """
    World-space points where rays intersect world geometry.
    """
    ray_indices: np.ndarray
    """
    Source ray index for each intersection.
    """
    bodies: list[Body]
    """
    World body intersected at each hit point.
    """


# %% Color image products
@dataclass
class RenderedColorFrame:
    """
    BGR image values and the associated semantic color names.
    """

    color_image: np.ndarray
    """
    Width-first color image in BGR channel order.
    """
    object_color_map: dict[str, str]
    """
    Object names indexed by semantic RGB color keys.
    """
