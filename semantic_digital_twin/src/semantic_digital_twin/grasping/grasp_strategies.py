"""
Where on its surface an object of a kind of shape may be grasped, reusable by every
annotation of that shape.
"""

from __future__ import annotations

import math
from abc import ABC, abstractmethod
from dataclasses import dataclass, field

import numpy as np
import trimesh
from typing_extensions import TYPE_CHECKING

from krrood.bounds import Bounds
from krrood.entity_query_language.factories import ConditionType, and_, or_
from krrood.entity_query_language.query.match import Match
from semantic_digital_twin.exceptions import NoGraspGeometry
from semantic_digital_twin.grasping.surface_grasp import SurfaceGrasp

if TYPE_CHECKING:
    from semantic_digital_twin.semantic_annotations.mixins import HasGraspCandidates

# %% the strategy


@dataclass
class GraspStrategy(ABC):
    """
    Where on its surface an object may be grasped, as conditions on a
    :class:`~semantic_digital_twin.grasping.surface_grasp.SurfaceGrasp`.
    """

    depth: Bounds[float] = field(default_factory=lambda: Bounds(0.0, 0.5))
    """
    The depths to close the fingers at, as fractions of :meth:`depth_unit`, the upper
    one excluded.
    """

    yaw: Bounds[float] = field(default_factory=lambda: Bounds(0.0, 2 * math.pi))
    """
    The yaws to approach from, in radians off each of :meth:`central_yaws`, the upper
    one excluded.
    """

    pitch: Bounds[float] = field(default_factory=lambda: Bounds(0.0, 1.2))
    """
    How far the approach may tilt off vertical, in radians, the upper bound excluded.
    """

    roll: Bounds[float] = field(
        default_factory=lambda: Bounds(-math.pi / 4, math.pi / 4)
    )
    """
    The rolls to close the fingers at, in radians, the upper one excluded.
    """

    def conditions(
        self, grasp: Match[SurfaceGrasp], graspable: HasGraspCandidates
    ) -> list[ConditionType]:
        """
        :param grasp: The grasp of a statement.
        :param graspable: The object grasped.
        :return: The conditions every grasp of this strategy must satisfy.
        :raises NoGraspGeometry: If the object has no shape to measure.
        """
        surface = graspable.grasp_surface()
        heights = self.grasped_heights(surface)
        depth_unit = self.depth_unit(graspable, surface)
        return [
            grasp.height >= heights.lower,
            grasp.height < heights.upper,
            grasp.depth >= self.depth.lower * depth_unit,
            grasp.depth < self.depth.upper * depth_unit,
            or_(
                *(
                    and_(
                        grasp.yaw >= central_yaw + self.yaw.lower,
                        grasp.yaw < central_yaw + self.yaw.upper,
                    )
                    for central_yaw in self.central_yaws(surface)
                )
            ),
            grasp.pitch >= self.pitch.lower,
            grasp.pitch < self.pitch.upper,
            grasp.roll >= self.roll.lower,
            grasp.roll < self.roll.upper,
        ]

    @abstractmethod
    def grasped_heights(self, surface: trimesh.Trimesh) -> Bounds[float]:
        """
        :param surface: The object's grasp surface, in its root body's frame.
        :return: The heights the object may be grasped at, as fractions of the grasp
            surface's bounding box from its bottom to its top, the upper one excluded.
        """

    @abstractmethod
    def depth_unit(
        self, graspable: HasGraspCandidates, surface: trimesh.Trimesh
    ) -> float:
        """
        :param graspable: The object grasped.
        :param surface: The object's grasp surface, in its root body's frame.
        :return: The length the depths are fractions of, in meters.
        :raises NoGraspGeometry: If the object has no shape to measure it on.
        """

    def central_yaws(self, surface: trimesh.Trimesh) -> list[float]:
        """
        :param surface: The object's grasp surface, in its root body's frame.
        :return: The yaws :attr:`yaw` is measured from, in radians.
        """
        return [0.0]


@dataclass
class GraspBetweenHeights(GraspStrategy, ABC):
    """
    A strategy that grasps an object between two fixed heights.
    """

    height: Bounds[float] = field(default_factory=lambda: Bounds(0.05, 0.95))
    """
    The heights to grasp at, as fractions of the grasp surface's bounding box from its
    bottom to its top, the upper one excluded.
    """

    def grasped_heights(self, surface: trimesh.Trimesh) -> Bounds[float]:
        return self.height


# %% grasping a solid object


@dataclass
class GraspFromAnySide(GraspBetweenHeights):
    """
    Grasp a solid object from any side, closing on it up to half its narrower horizontal
    extent deep.
    """

    def depth_unit(
        self, graspable: HasGraspCandidates, surface: trimesh.Trimesh
    ) -> float:
        """
        :return: The narrower horizontal extent of the grasp surface.
        """
        return float(min(surface.extents[:2]))


@dataclass
class GraspABar(GraspFromAnySide):
    """
    Grasp a thin, elongated part, such as a handle, from any side, closing on it up to
    half its thickness deep.
    """

    height: Bounds[float] = field(default_factory=lambda: Bounds(0.1, 0.9))

    def depth_unit(
        self, graspable: HasGraspCandidates, surface: trimesh.Trimesh
    ) -> float:
        """
        :return: The thinnest extent of the grasp surface.
        """
        return float(min(surface.extents))


@dataclass
class GraspAcrossTheLength(GraspBetweenHeights):
    """
    Grasp a flat, elongated object lying on its side, such as cutlery, seen along its
    length from either end and closing across it.
    """

    height: Bounds[float] = field(default_factory=lambda: Bounds(0.2, 0.8))

    depth: Bounds[float] = field(default_factory=lambda: Bounds(0.2, 0.6))

    yaw: Bounds[float] = field(
        default_factory=lambda: Bounds(-math.pi / 16, math.pi / 16)
    )

    pitch: Bounds[float] = field(default_factory=lambda: Bounds(0.0, 0.6))

    roll: Bounds[float] = field(
        default_factory=lambda: Bounds(3 * math.pi / 8, 5 * math.pi / 8)
    )

    def depth_unit(
        self, graspable: HasGraspCandidates, surface: trimesh.Trimesh
    ) -> float:
        """
        :return: The length of the object, its longer horizontal extent.
        """
        return float(max(surface.extents[:2]))

    def central_yaws(self, surface: trimesh.Trimesh) -> list[float]:
        """
        :return: The yaws the object is seen from along its length, from either end.
        """
        length_yaw = 0.0 if surface.extents[0] >= surface.extents[1] else math.pi / 2
        return [length_yaw, length_yaw + math.pi]


# %% pinching a thin wall


@dataclass
class PinchAWall(GraspStrategy, ABC):
    """
    Pinch the thin upright wall of an object, such as an open container, from above,
    closing across the wall.
    """

    depth: Bounds[float] = field(default_factory=lambda: Bounds(0.25, 0.75))

    pitch: Bounds[float] = field(default_factory=lambda: Bounds(0.0, 0.5))

    roll: Bounds[float] = field(
        default_factory=lambda: Bounds(-math.pi / 8, math.pi / 8)
    )

    number_of_directions: int = 12
    """
    In how many directions around the object the wall's thickness is measured.
    """

    def depth_unit(
        self, graspable: HasGraspCandidates, surface: trimesh.Trimesh
    ) -> float:
        """
        Cast rays outward from the middle of the grasp surface, halfway between the
        grasped heights, and take the distance between the first two surfaces each ray
        passes, the inside and the outside of the wall.

        :return: The median thickness of the wall over the directions that hit it.
        :raises NoGraspGeometry: If no direction hits a wall.
        """
        yaws = np.linspace(0, 2 * np.pi, self.number_of_directions, endpoint=False)
        directions = np.column_stack([np.cos(yaws), np.sin(yaws), np.zeros(len(yaws))])
        lowest, highest = surface.bounds
        middle = (lowest + highest) / 2
        heights = self.grasped_heights(surface)
        middle[2] = lowest[2] + (heights.lower + heights.upper) / 2 * (
            highest[2] - lowest[2]
        )
        # Embree reports only the inner wall here, so the exact intersector is used.
        intersector = trimesh.ray.ray_triangle.RayMeshIntersector(surface)
        locations, ray_indices, _ = intersector.intersects_location(
            ray_origins=np.tile(middle, (len(yaws), 1)), ray_directions=directions
        )
        thicknesses = []
        for index in range(len(yaws)):
            distances = np.sort(
                np.linalg.norm(
                    locations[ray_indices == index][:, :2] - middle[:2], axis=1
                )
            )
            if len(distances) > 1:
                thicknesses.append(float(distances[1] - distances[0]))
        if not thicknesses:
            raise NoGraspGeometry(graspable)
        return float(np.median(thicknesses))


@dataclass
class PinchARim(PinchAWall, GraspBetweenHeights):
    """
    Pinch a rim, the upper edge of an open container's wall, between two heights of the
    rim itself.
    """

    height: Bounds[float] = field(default_factory=lambda: Bounds(0.25, 0.75))


@dataclass
class PinchTheTopOfAWall(PinchAWall):
    """
    Pinch the top of an open container's wall, as a rim would be pinched, when the rim
    is not a part of its own.
    """

    rim_depth: float = 0.02
    """
    How far below the container's highest point its rim reaches, in meters.
    """

    rim_fraction: Bounds[float] = field(default_factory=lambda: Bounds(0.25, 0.75))
    """
    How far below the highest point to pinch, as fractions of :attr:`rim_depth`.
    """

    def grasped_heights(self, surface: trimesh.Trimesh) -> Bounds[float]:
        """
        :return: The heights between the two fractions of the top :attr:`rim_depth`.
        """
        rim_depth_fraction = self.rim_depth / float(surface.extents[2])
        return Bounds(
            1 - self.rim_fraction.upper * rim_depth_fraction,
            1 - self.rim_fraction.lower * rim_depth_fraction,
        )
