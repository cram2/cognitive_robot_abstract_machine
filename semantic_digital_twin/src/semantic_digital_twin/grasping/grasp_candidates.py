"""
The grasps an object offers, independent of the robot performing them.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from typing_extensions import List

from semantic_digital_twin.exceptions import (
    MissingReferenceFrameError,
    ReferenceFrameMismatchError,
)
from semantic_digital_twin.semantic_annotations.mixins import HasRootBody
from semantic_digital_twin.spatial_types import Point3, Vector3
from semantic_digital_twin.spatial_types.spatial_types import Pose, RotationMatrix

# %% grasp candidates


@dataclass(eq=False)
class GraspCandidate:
    """
    A grasp an object offers, together with the object offering it.

    A grasp frame has its x-axis pointing the way the gripper travels toward the object,
    its y-axis along the axis the fingers close along, and its z-axis completing the
    frame. Every end effector states the same two axes in its own tool frame, as
    :attr:`~semantic_digital_twin.robots.robot_parts.EndEffector.approach_axis` and
    :attr:`~semantic_digital_twin.robots.robot_parts.EndEffector.closing_axis`, which is
    how a grasp stays independent of the robot performing it.
    """

    graspable: HasGraspCandidates
    """
    The annotation of the object offering this grasp.
    """

    grasp_pose: Pose
    """
    The grasp frame relative to :attr:`graspable`'s root body, so that it stays correct
    when the object moves.
    """

    def __post_init__(self):
        if self.grasp_pose.reference_frame is None:
            raise MissingReferenceFrameError(self.grasp_pose)
        if self.grasp_pose.reference_frame is not self.graspable.root:
            raise ReferenceFrameMismatchError(
                expected_frame=self.graspable.root,
                actual_frame=self.grasp_pose.reference_frame,
                context="grasp pose",
            )

    @classmethod
    def from_body_origin(cls, graspable: HasGraspCandidates) -> GraspCandidate:
        """
        The grasp that takes an object at the origin of its own body.

        :param graspable: The annotation of the object to be grasped.
        :return: A grasp at that object's origin.
        """
        return cls(graspable, Pose(reference_frame=graspable.root))

    @property
    def world_T_grasp(self) -> Pose:
        """
        :return: The grasp frame in the world frame, where the object is now.
        """
        world = self.graspable.root._world
        return world.transform(self.grasp_pose, world.root)

    def moved_to(self, reference_T_object: Pose) -> Pose:
        """
        Transform this grasp candidate to where it would be, once the object is placed.

        :param reference_T_object: The pose the object is going to have.
        :return:``reference_T_grasp``, the grasp in the same frame that pose is in.
        """
        return reference_T_object.to_homogeneous_matrix() @ self.grasp_pose


@dataclass(eq=False)
class HasGraspCandidates(HasRootBody):
    """
    A mixin class for semantic annotations that can say where they may be grasped.

    Only an annotation rooted in a body can be grasped at all, since a region carries no
    collision geometry for fingers to close on.
    """

    grasp_candidate_count: int = field(default=12, kw_only=True)
    """
    How many grasp candidates :meth:`grasp_candidates` generates.
    """

    def grasp_candidates(self) -> List[GraspCandidate]:
        """
        The grasps this annotation offers, in no particular order.

        The default grasps the object at its own origin, from evenly spaced directions
        around its z-axis. Annotations whose geometry admits a better grip override
        this.
        """
        return [
            GraspCandidate(
                self,
                Pose(
                    orientation=RotationMatrix.from_rpy(yaw=yaw).to_quaternion(),
                    reference_frame=self.root,
                ),
            )
            for yaw in np.linspace(
                0, 2 * np.pi, self.grasp_candidate_count, endpoint=False
            )
        ]


# %% grasp geometry helpers


@dataclass
class RimWallSection:
    """
    Where a bowl's wall runs at one point of its rim, in the bowl's own frame.
    """

    center: Point3
    """
    The middle of the wall, halfway between its inner and its outer surface.
    """

    outward: Vector3
    """
    The direction from the bowl's axis to the wall.
    """
