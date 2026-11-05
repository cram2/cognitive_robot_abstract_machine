"""
Grasps of elongated objects, such as handles and cutlery, around the axis they run
along.
"""

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
    from semantic_digital_twin.semantic_annotations.mixins import HasGraspCandidates

# %% grasps around an object's axis


@dataclass
class AxialGrasp(ObjectRelativeGrasp):
    """
    A grasp of an elongated object with the gripper's tool centre point on the axis the
    object runs along, closing across it.

    The axis runs through the centre of the grasp surface's bounding box along its
    longest extent, as ``root_V_axis``, from ``root_P_start`` at its lower end. The
    gripper closes at ``root_P_grasp = root_P_start + position * length * root_V_axis``.
    The x-axis of ``root_T_grasp`` is the direction of approach, ``root_V_reference``
    turned by :attr:`angle` about ``root_V_axis``; its y-axis is the closing direction,
    ``root_V_axis`` crossed with the approach. ``root_V_reference`` points straight down,
    or along the root body's negative y-axis when the axis is vertical.
    """

    position: float
    """
    Where along the axis the gripper closes, as a fraction of the axis from its lower
    end to its upper end.
    """

    angle: float
    """
    How far the approach is turned about ``root_V_axis`` from ``root_V_reference``, in
    radians.
    """

    def reaches_surface_of(self, graspable: HasGraspCandidates) -> bool:
        """
        :return: Whether :attr:`position` lies between the ends of the axis.
        """
        return 0.0 <= self.position <= 1.0

    def grasp_candidate(self, graspable: HasGraspCandidates) -> GraspCandidate:
        if not self.reaches_surface_of(graspable):
            raise GraspNotOnObjectError(grasp=self, graspable=graspable)
        lowest, highest = graspable.grasp_surface().bounds
        axis_index = int(np.argmax(highest - lowest))
        root_V_axis = Vector3.from_iterable(np.eye(3)[axis_index])
        start = (lowest + highest) / 2
        start[axis_index] = lowest[axis_index]
        root_P_start = Point3.from_iterable(start)
        length = float(highest[axis_index] - lowest[axis_index])
        root_V_reference = (
            Vector3.NEGATIVE_Y() if axis_index == 2 else Vector3.NEGATIVE_Z()
        )
        root_V_approach = root_V_reference * math.cos(self.angle) + root_V_axis.cross(
            root_V_reference
        ) * math.sin(self.angle)
        root_T_grasp = HomogeneousTransformationMatrix.from_point_rotation_matrix(
            point=root_P_start + root_V_axis * (self.position * length),
            rotation_matrix=RotationMatrix.from_vectors(
                x=root_V_approach, y=root_V_axis.cross(root_V_approach)
            ),
            reference_frame=graspable.root,
        )
        return GraspCandidate(graspable, root_T_grasp.pose)
