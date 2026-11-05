from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from typing_extensions import TYPE_CHECKING

from krrood_spatial_math.spatial_types import (
    HomogeneousTransformationMatrix,
    Point3,
    RotationMatrix,
    Vector3,
)
from semantic_digital_twin.exceptions import GraspNotOnObjectError
from semantic_digital_twin.grasping.grasp_candidates import GraspCandidate
from semantic_digital_twin.grasping.object_relative_grasp import ObjectRelativeGrasp

if TYPE_CHECKING:
    from semantic_digital_twin.semantic_annotations.mixins import IsGraspable

# %% grasps on an object's surface


@dataclass
class SurfaceGrasp(ObjectRelativeGrasp):
    """
    A grasp placed on the surface of an object around its root body's z-axis.

    With ``root_V_outward`` the horizontal direction at :attr:`yaw`, ``root_P_surface``
    is the first point of the grasp surface that a horizontal ray hits when it comes
    from ``root_V_outward`` at :attr:`height`. The gripper closes at
    ``root_P_grasp = root_P_surface - depth * root_V_outward``. The x-axis of
    ``root_T_grasp`` is the direction of approach, straight down tilted by :attr:`pitch`
    towards ``-root_V_outward``; its y-axis is the closing direction, ``root_V_outward``
    made perpendicular to the approach and turned by :attr:`roll` about it.
    """

    yaw: float
    """
    The yaw of ``root_V_outward`` about the root body's z-axis, in radians.
    """

    height: float
    """
    The height of ``root_P_surface``, as a fraction of the grasp surface's bounding box
    from its bottom to its top.
    """

    depth: float
    """
    How far behind ``root_P_surface`` the gripper closes, in meters.
    """

    pitch: float
    """
    How far the direction of approach is tilted from straight down, in radians.
    """

    roll: float
    """
    How far the closing direction is turned about the direction of approach, in radians.
    """

    def grasp_candidate(self, graspable: IsGraspable) -> GraspCandidate:
        root_V_outward = self._outward_direction()
        root_P_surface = self._surface_point(graspable, root_V_outward)
        if root_P_surface is None:
            raise GraspNotOnObjectError(grasp=self, graspable=graspable)
        root_V_approach = Vector3.NEGATIVE_Z() * math.cos(
            self.pitch
        ) - root_V_outward * math.sin(self.pitch)
        root_V_across = root_V_outward - root_V_approach * root_V_outward.dot(
            root_V_approach
        )
        root_V_across.scale(1)
        root_V_closing = root_V_across * math.cos(self.roll) + root_V_approach.cross(
            root_V_across
        ) * math.sin(self.roll)
        root_T_grasp = HomogeneousTransformationMatrix.from_point_rotation_matrix(
            point=root_P_surface - root_V_outward * self.depth,
            rotation_matrix=RotationMatrix.from_vectors(
                x=root_V_approach, y=root_V_closing
            ),
            reference_frame=graspable.root,
        )
        return GraspCandidate(graspable, root_T_grasp.pose)

    def reaches_surface_of(self, graspable: IsGraspable) -> bool:
        """
        :return: Whether the object's grasp surface reaches ``root_P_surface``.
        """
        return self._surface_point(graspable, self._outward_direction()) is not None

    def _outward_direction(self) -> Vector3:
        """
        :return: ``root_V_outward``.
        """
        return Vector3(math.cos(self.yaw), math.sin(self.yaw), 0.0)

    def _surface_point(
        self, graspable: IsGraspable, root_V_outward: Vector3
    ) -> Point3 | None:
        """
        :param graspable: The object whose grasp surface the ray is cast against.
        :param root_V_outward: The horizontal direction the ray comes from.
        :return: ``root_P_surface``; ``None`` if the ray hits nothing.
        """
        mesh = graspable.grasp_surface()
        outward = root_V_outward.to_np()[:3]
        lowest, highest = mesh.bounds
        origin = (lowest + highest) / 2 + outward * float(
            np.linalg.norm(highest - lowest)
        )
        origin[2] = lowest[2] + self.height * (highest[2] - lowest[2])
        locations, _, _ = mesh.ray.intersects_location(
            ray_origins=origin[None, :], ray_directions=-outward[None, :]
        )
        if len(locations) == 0:
            return None
        return Point3(
            *locations[int(np.argmin(np.linalg.norm(locations - origin, axis=1)))]
        )
