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
    RotationMatrix,
    Vector3,
)
from semantic_digital_twin.exceptions import GraspNotOnObjectError
from semantic_digital_twin.grasping.grasp_candidates import GraspCandidate
from semantic_digital_twin.grasping.object_relative_grasp import ObjectRelativeGrasp

if TYPE_CHECKING:
    from semantic_digital_twin.semantic_annotations.mixins import IsGraspable

# %% grasps around an object's axis


@dataclass(kw_only=True)
class AxialGrasp(ObjectRelativeGrasp):
    """
    A grasp of an elongated object with the gripper's tool centre point on the object's
    :meth:`~IsGraspable.grasp_axis`, closing across it.

    The gripper approaches across the axis, turned by :attr:`angle` about it from the
    root body's negative z-axis, or from its negative y-axis when the axis runs mostly
    along z.
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

    def reaches_surface_of(self, *, graspable: IsGraspable) -> bool:
        """
        :return: Whether :attr:`position` lies between the ends of the axis.
        """
        return 0.0 <= self.position <= 1.0

    def grasp_candidate(self, *, graspable: IsGraspable) -> GraspCandidate:
        if not self.reaches_surface_of(graspable=graspable):
            raise GraspNotOnObjectError(grasp=self, graspable=graspable)
        axis = graspable.grasp_axis()
        root_V_axis = axis.direction
        main_axis_index = int(np.argmax(np.abs(root_V_axis.to_np()[:3])))
        root_V_reference = (
            Vector3.NEGATIVE_Y() if main_axis_index == 2 else Vector3.NEGATIVE_Z()
        )
        root_V_reference = root_V_reference - root_V_axis * root_V_reference.dot(
            root_V_axis
        )
        root_V_reference.scale(1)
        root_V_approach = root_V_reference * math.cos(self.angle) + root_V_axis.cross(
            root_V_reference
        ) * math.sin(self.angle)
        root_T_grasp = HomogeneousTransformationMatrix.from_point_rotation_matrix(
            point=axis.start + (axis.end - axis.start) * self.position,
            rotation_matrix=RotationMatrix.from_vectors(
                x=root_V_approach, y=root_V_axis.cross(root_V_approach)
            ),
            reference_frame=graspable.root,
        )
        return GraspCandidate(graspable=graspable, grasp_pose=root_T_grasp.pose)
