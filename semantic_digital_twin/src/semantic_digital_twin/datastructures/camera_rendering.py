"""
Color and depth conventions for aligned ray-traced camera images.
"""

from __future__ import annotations

# %% Imports
from abc import ABC, abstractmethod
from dataclasses import dataclass
import io
import math
import numpy as np
import trimesh
from PIL import Image

from semantic_digital_twin.world_description.geometry import Color
from semantic_digital_twin.spatial_computations.exceptions import (
    UnsupportedCameraProjection,
)
from typing_extensions import TYPE_CHECKING

if TYPE_CHECKING:
    import numpy.typing as npt

    from semantic_digital_twin.datastructures.camera_model import PinholeCameraModel
    from semantic_digital_twin.spatial_types import HomogeneousTransformationMatrix
    from semantic_digital_twin.world_description.world_entity import Body
    from semantic_digital_twin.world import World


# %% Body color assignments
@dataclass
class BodyColorAssignment:
    """
    Associate a geometry color with the visible bodies sharing it.
    """

    color: Color
    """
    Collision geometry color assigned to the bodies.
    """

    bodies: list[Body]
    """
    Visible bodies sharing the color.
    """


# %% Color rendering conventions
@dataclass(frozen=True)
class ColorRenderingMode(ABC):
    """
    Produce aligned BGR images and visible bodies grouped by color.
    """

    @abstractmethod
    def validate_camera_model(self, camera_model: PinholeCameraModel) -> None:
        """
        Reject projection models that this color renderer cannot represent.

        :param camera_model: Effective projection model of the rendered frame.
        :raises UnsupportedCameraProjection: If the projection cannot be represented.
        """
        ...

    @abstractmethod
    def render(
        self,
        segmentation: npt.NDArray[np.int32],
        camera_model: PinholeCameraModel,
        scene: trimesh.Scene,
        world: World,
    ) -> RenderedColorFrame:
        """
        Render width-first BGR colors aligned with body-index segmentation.

        :param segmentation: Visible body indices, with negative values for background.
        :param camera_model: Validated effective projection model.
        :param scene: World geometry with the rendering camera already positioned.
        :param world: World supplying the bodies identified by the segmentation.
        :return: BGR colors and visible bodies grouped by geometry color.
        """
        ...


@dataclass(frozen=True)
class BodyColorRenderingMode(ColorRenderingMode):
    """
    Color each visible body uniformly with its first collision geometry color.
    """

    def validate_camera_model(self, camera_model: PinholeCameraModel) -> None:
        pass

    def render(
        self,
        segmentation: npt.NDArray[np.int32],
        camera_model: PinholeCameraModel,
        scene: trimesh.Scene,
        world: World,
    ) -> RenderedColorFrame:
        color_rgb = np.zeros((*segmentation.shape, 3), dtype=np.uint8)
        body_colors: dict[Color, BodyColorAssignment] = {}
        for body_index in np.unique(segmentation[segmentation >= 0]):
            body = world.kinematic_structure[int(body_index)]
            body_rgb = self.get_body_color(body)
            color_rgb[segmentation == body_index] = body_rgb
            color = body.collision[0].color
            body_colors.setdefault(
                color, BodyColorAssignment(color=color, bodies=[])
            ).bodies.append(body)
        return RenderedColorFrame(
            color_image=color_rgb[:, :, ::-1].copy(),
            body_colors=list(body_colors.values()),
        )

    @staticmethod
    def get_body_color(body: Body) -> npt.NDArray[np.uint8]:
        """
        Return the first collision geometry's unsigned-byte RGB color.

        :param body: Body with collision geometry supplying the color.
        :return: Unsigned-byte RGB channels for the selected body color.
        """
        color = body.collision[0].color
        channel_maximum = np.iinfo(np.uint8).max
        return np.clip(
            np.round(np.asarray(color.to_rgb()) * channel_maximum), 0, channel_maximum
        ).astype(np.uint8)


@dataclass(frozen=True)
class TexturedColorRenderingMode(ColorRenderingMode):
    """
    Render surface textures and materials with a centered, equal-focal-length camera.
    """

    def validate_camera_model(self, camera_model: PinholeCameraModel) -> None:
        resolution = camera_model.resolution
        if not (
            math.isclose(float(camera_model.principal_point.x), resolution.width / 2)
            and math.isclose(
                float(camera_model.principal_point.y), resolution.height / 2
            )
            and math.isclose(camera_model.focal_length_x, camera_model.focal_length_y)
        ):
            raise UnsupportedCameraProjection(
                reason="textured rendering requires a centered camera with equal focal lengths"
            )

    def render(
        self,
        segmentation: npt.NDArray[np.int32],
        camera_model: PinholeCameraModel,
        scene: trimesh.Scene,
        world: World,
    ) -> RenderedColorFrame:
        resolution = camera_model.resolution
        image_bytes = scene.save_image(
            resolution=(resolution.width, resolution.height), visible=False
        )
        image = np.array(Image.open(io.BytesIO(image_bytes)).convert("RGB"))
        return RenderedColorFrame(
            color_image=image.transpose(1, 0, 2)[:, :, ::-1].copy(),
            body_colors=[],
        )


# %% Depth conventions
@dataclass(frozen=True)
class DepthMode(ABC):
    """
    Compute one meter depth per surface hit, using negative values for invalid depths.
    """

    @abstractmethod
    def hit_depth(
        self,
        hit_points: npt.NDArray[np.float64],
        ray_origins: npt.NDArray[np.float64],
        ray_directions: npt.NDArray[np.float64],
        world_T_camera: HomogeneousTransformationMatrix,
    ) -> npt.NDArray[np.float64]:
        """
        Return depths aligned with the input hits.

        :param hit_points: World-space surface points, one row per hit.
        :param ray_origins: World-space ray origins aligned with the hits.
        :param ray_directions: World-space unit ray directions aligned with the hits.
        :param world_T_camera: Pose of the rendering camera, looking along its negative z axis.
        :return: Meter depths, with negative values for invalid depths.
        """
        ...


@dataclass(frozen=True)
class OpticalAxisDepthMode(DepthMode):
    """
    Measure depth along the rendering camera's negative z axis.
    """

    def hit_depth(
        self,
        hit_points: npt.NDArray[np.float64],
        ray_origins: npt.NDArray[np.float64],
        ray_directions: npt.NDArray[np.float64],
        world_T_camera: HomogeneousTransformationMatrix,
    ) -> npt.NDArray[np.float64]:
        camera_T_world = world_T_camera.inverse().to_np()
        world_P_hits = np.concatenate(
            [hit_points, np.ones((len(hit_points), 1))], axis=1
        )
        camera_P_hits = (camera_T_world @ world_P_hits.T).T
        depths = -camera_P_hits[:, 2]
        return np.where(depths > 0.0, depths, -1.0)


@dataclass(frozen=True)
class RayDistanceDepthMode(DepthMode):
    """
    Measure depth from each ray origin along its viewing direction.
    """

    def hit_depth(
        self,
        hit_points: npt.NDArray[np.float64],
        ray_origins: npt.NDArray[np.float64],
        ray_directions: npt.NDArray[np.float64],
        world_T_camera: HomogeneousTransformationMatrix,
    ) -> npt.NDArray[np.float64]:
        return trimesh.util.diagonal_dot(hit_points - ray_origins, ray_directions)


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

    body_colors: list[BodyColorAssignment]
    """
    Visible bodies grouped by their collision geometry color.
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
    BGR image values and the associated bodies grouped by color.
    """

    color_image: np.ndarray
    """
    Width-first color image in BGR channel order.
    """
    body_colors: list[BodyColorAssignment]
    """
    Visible bodies grouped by their collision geometry color.
    """
