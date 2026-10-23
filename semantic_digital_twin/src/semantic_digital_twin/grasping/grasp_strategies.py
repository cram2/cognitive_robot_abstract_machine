"""
Where on its surface an object of a kind of shape may be grasped, reusable by every
annotation of that shape.
"""

from __future__ import annotations

import math
from abc import ABC, abstractmethod
from dataclasses import dataclass

import numpy as np
import trimesh
from random_events.interval import Bound, SimpleInterval
from typing_extensions import TYPE_CHECKING

from krrood.entity_query_language.factories import ConditionType, and_, or_
from krrood.entity_query_language.query.match import Match
from krrood.ormatic.utils import classproperty
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

    minimal_depth: float = 0.0
    """
    The least depth to close the fingers at, as a fraction of :meth:`depth_unit`.
    """

    maximal_depth: float = 0.5
    """
    The depth to close the fingers short of, as a fraction of :meth:`depth_unit`.
    """

    maximal_pitch: float = 1.2
    """
    How far the approach may tilt off vertical, in radians.
    """

    maximal_roll: float = math.pi / 4
    """
    How far the fingers may turn off :attr:`central_roll`, either way, in radians.
    """

    @classproperty
    def central_roll(cls) -> float:
        """
        :return: The roll the fingers close at when they are not turned, in radians.
        """
        return 0.0

    def conditions(
        self, grasp: Match[SurfaceGrasp], graspable: HasGraspCandidates
    ) -> list[ConditionType]:
        """
        :param grasp: The grasp of a statement.
        :param graspable: The object grasped.
        :return: The conditions that the grasp lies where this strategy allows.
        :raises NoGraspGeometry: If the object has no shape to measure.
        """
        surface = graspable.grasp_surface()
        heights = self.grasped_heights(surface)
        depth_unit = self.depth_unit(graspable, surface)
        return [
            grasp.height >= heights.lower,
            grasp.height < heights.upper,
            grasp.depth >= self.minimal_depth * depth_unit,
            grasp.depth < self.maximal_depth * depth_unit,
            *self.yaw_conditions(grasp, surface),
            grasp.pitch >= 0.0,
            grasp.pitch < self.maximal_pitch,
            grasp.roll >= self.central_roll - self.maximal_roll,
            grasp.roll < self.central_roll + self.maximal_roll,
        ]

    @abstractmethod
    def grasped_heights(self, surface: trimesh.Trimesh) -> SimpleInterval:
        """
        :param surface: The object's grasp surface, in its root body's frame.
        :return: The heights the object may be grasped at, as fractions of the grasp
            surface's bounding box from its bottom to its top.
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

    def yaw_conditions(
        self, grasp: Match[SurfaceGrasp], surface: trimesh.Trimesh
    ) -> list[ConditionType]:
        """
        :param grasp: The grasp of a statement.
        :param surface: The object's grasp surface, in its root body's frame.
        :return: The conditions on the yaw the object may be grasped from; by default
            every yaw.
        """
        return [grasp.yaw >= 0.0, grasp.yaw < 2 * math.pi]


@dataclass
class GraspBetweenHeights(GraspStrategy, ABC):
    """
    A strategy that grasps an object between two fixed heights.
    """

    minimal_height: float = 0.05
    """
    The minimal height to grasp at, as a fraction of the grasp surface's bounding box
    from its bottom to its top.
    """

    maximal_height: float = 0.95
    """
    The height to grasp below, as such a fraction.
    """

    def grasped_heights(self, surface: trimesh.Trimesh) -> SimpleInterval:
        return SimpleInterval.from_data(
            self.minimal_height, self.maximal_height, Bound.CLOSED, Bound.OPEN
        )


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

    minimal_height: float = 0.1

    maximal_height: float = 0.9

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

    minimal_height: float = 0.2

    maximal_height: float = 0.8

    minimal_depth: float = 0.2

    maximal_depth: float = 0.6

    maximal_pitch: float = 0.6

    maximal_roll: float = math.pi / 8

    yaw_tolerance: float = math.pi / 16
    """
    How far off its length the object may be seen from, either way, in radians.
    """

    @classproperty
    def central_roll(cls) -> float:
        """
        :return: A quarter turn, closing across the length.
        """
        return math.pi / 2

    def depth_unit(
        self, graspable: HasGraspCandidates, surface: trimesh.Trimesh
    ) -> float:
        """
        :return: The length of the object, its longer horizontal extent.
        """
        return float(max(surface.extents[:2]))

    def yaw_conditions(
        self, grasp: Match[SurfaceGrasp], surface: trimesh.Trimesh
    ) -> list[ConditionType]:
        """
        :return: The conditions that the object is seen from either end of its length.
        """
        length_yaw = 0.0 if surface.extents[0] >= surface.extents[1] else math.pi / 2
        return [
            or_(
                *(
                    and_(
                        grasp.yaw >= end_yaw - self.yaw_tolerance,
                        grasp.yaw < end_yaw + self.yaw_tolerance,
                    )
                    for end_yaw in (length_yaw, length_yaw + math.pi)
                )
            )
        ]


# %% pinching a thin wall


@dataclass
class PinchAWall(GraspStrategy, ABC):
    """
    Pinch the thin upright wall of an object, such as an open container, from above,
    closing across the wall.
    """

    minimal_depth: float = 0.25

    maximal_depth: float = 0.75

    maximal_pitch: float = 0.5

    maximal_roll: float = math.pi / 8

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
        middle[2] = lowest[2] + self.grasped_heights(surface).center() * (
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

    minimal_height: float = 0.25

    maximal_height: float = 0.75


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

    minimal_rim_fraction: float = 0.25
    """
    How far below the highest point to pinch at least, as a fraction of
    :attr:`rim_depth`.
    """

    maximal_rim_fraction: float = 0.75
    """
    How far below the highest point to pinch at most, as such a fraction.
    """

    def grasped_heights(self, surface: trimesh.Trimesh) -> SimpleInterval:
        """
        :return: The heights between the two fractions of the top :attr:`rim_depth`.
        """
        rim_fraction = self.rim_depth / float(surface.extents[2])
        return SimpleInterval.from_data(
            1 - self.maximal_rim_fraction * rim_fraction,
            1 - self.minimal_rim_fraction * rim_fraction,
            Bound.CLOSED,
            Bound.OPEN,
        )
