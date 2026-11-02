"""
Rendering a mesh from cameras around it, keeping for each pixel the face it shows.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import trimesh
from numpy.typing import NDArray

from semantic_digital_twin.datastructures.camera_resolution import CameraResolution
from semantic_digital_twin.spatial_types import (
    HomogeneousTransformationMatrix,
    Point3,
    RotationMatrix,
    Vector3,
)


@dataclass
class ViewDirection:
    """
    Where a camera looks at a mesh from, as seen from the mesh's center: the reference
    plane is the mesh's horizontal plane, the reference direction its x-axis and the
    zenith its z-axis.
    """

    azimuth: float
    """
    The angle from the reference direction to the camera projected onto the reference
    plane, counterclockwise as seen from the zenith, in radians.
    """

    elevation: float
    """
    The angle from the reference plane up to the camera, in radians.
    """

    @staticmethod
    def around(perspectives: int, elevation: float) -> list[ViewDirection]:
        """
        :param perspectives: How many directions.
        :param elevation: The elevation of each direction, in radians.
        :return: Directions evenly spaced in azimuth all around, the first along the
            reference direction.
        """
        return [
            ViewDirection(azimuth, elevation)
            for azimuth in np.linspace(0.0, 2 * np.pi, perspectives, endpoint=False)
        ]

    @property
    def towards_camera(self) -> NDArray[np.float64]:
        """
        :return: The unit vector from the mesh's center towards the camera.
        """
        return np.array(
            [
                np.cos(self.elevation) * np.cos(self.azimuth),
                np.cos(self.elevation) * np.sin(self.azimuth),
                np.sin(self.elevation),
            ]
        )

    def camera_pose(
        self, target: NDArray[np.float64], distance: float
    ) -> HomogeneousTransformationMatrix:
        """
        :param target: The point the camera looks at.
        :param distance: How far the camera is from it.
        :return: The pose of the camera in the mesh's frame, in the convention of
            :class:`~semantic_digital_twin.robots.robot_parts.Camera` and
            :meth:`~semantic_digital_twin.spatial_computations.raytracer.RayTracer.create_camera_rays`:
            its x-axis is the direction the camera looks, its negative y-axis points to the
            top and its negative z-axis to the right of the image, so that the zenith is
            up in the image.
        """
        towards_camera = self.towards_camera
        zenith = np.array([0.0, 0.0, 1.0])
        image_up = zenith - (zenith @ towards_camera) * towards_camera
        return HomogeneousTransformationMatrix.from_point_rotation_matrix(
            point=Point3(*(target + towards_camera * distance)),
            rotation_matrix=RotationMatrix.from_vectors(
                x=Vector3(*-towards_camera),
                y=Vector3(*-image_up / np.linalg.norm(image_up)),
            ),
        )


@dataclass
class RenderedViews:
    """
    Gray images of a mesh, each with the face that each of its pixels shows.
    """

    images: NDArray[np.uint8]
    """
    The images, as an array of views, rows and columns.
    """

    faces: NDArray[np.int64]
    """
    For each pixel of each image, the index of the face it shows; -1 where it shows the
    background.
    """

    @property
    def silhouettes(self) -> NDArray[np.bool_]:
        """
        :return: For each pixel of each image, whether it shows the mesh.
        """
        return self.faces >= 0


@dataclass
class CameraRing:
    """
    Cameras around a mesh that render it as gray images, shaded by how directly each
    face looks at the camera.
    """

    directions: list[ViewDirection] = field(
        default_factory=lambda: ViewDirection.around(8, np.radians(25.0))
        + ViewDirection.around(3, np.radians(65.0))
    )
    """
    The directions the mesh is looked at from.

    By default eight all around, 25 degrees from above, and three steeper ones, 65
    degrees from above, so that handles on any side and flat objects are seen; in the
    handle finder benchmark these eleven views were a good compromise between accuracy
    and speed.
    """

    resolution: CameraResolution = field(
        default_factory=lambda: CameraResolution(width=448, height=448)
    )
    """
    The size of each image, in pixels.
    """

    margin: float = 1.05
    """
    How much larger than the mesh the square each camera sees is.
    """

    darkest_shade: int = 40
    """
    The brightness of a face seen edge-on.
    """

    brightest_shade: int = 240
    """
    The brightness of a face seen head-on.
    """

    background: int = 255
    """
    The brightness of the background.
    """

    def render(self, mesh: trimesh.Trimesh) -> RenderedViews:
        """
        :param mesh: The mesh, its z-axis pointing up.
        :return: One orthographic view of the mesh from each of :attr:`directions`.
        """
        rendered = [self._render_view(mesh, direction) for direction in self.directions]
        return RenderedViews(
            images=np.concatenate([view.images for view in rendered]),
            faces=np.concatenate([view.faces for view in rendered]),
        )

    def _render_view(
        self, mesh: trimesh.Trimesh, direction: ViewDirection
    ) -> RenderedViews:
        """
        :param mesh: The mesh.
        :param direction: Where the camera looks from.
        :return: The single orthographic view from there.
        """
        diagonal = float(np.linalg.norm(mesh.extents))
        pose = direction.camera_pose(mesh.bounds.mean(axis=0), diagonal).to_np()
        forward, eye = pose[:3, 0], pose[:3, 3]
        up, right = -pose[:3, 1], -pose[:3, 2]
        half_width = diagonal / 2 * self.margin
        width, height = self.resolution.width, self.resolution.height
        half_height = half_width * height / width
        across, upwards = np.meshgrid(
            ((np.arange(width) + 0.5) / width * 2 - 1) * half_width,
            -((np.arange(height) + 0.5) / height * 2 - 1) * half_height,
        )
        origins = eye + across[..., None] * right + upwards[..., None] * up
        triangles, rays = mesh.ray.intersects_id(
            origins.reshape(-1, 3),
            np.broadcast_to(forward, (width * height, 3)),
            multiple_hits=False,
        )
        faces = np.full(width * height, -1, dtype=np.int64)
        faces[rays] = triangles
        image = np.full(width * height, float(self.background))
        facing = np.abs(mesh.face_normals[triangles] @ forward)
        image[rays] = (
            self.darkest_shade + (self.brightest_shade - self.darkest_shade) * facing
        )
        shape_of_views = (1, height, width)
        return RenderedViews(
            images=image.reshape(shape_of_views).astype(np.uint8),
            faces=faces.reshape(shape_of_views),
        )
