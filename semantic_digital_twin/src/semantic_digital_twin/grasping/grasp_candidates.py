"""
The grasps an object offers, independent of the robot performing them.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from typing_extensions import TYPE_CHECKING

from semantic_digital_twin.exceptions import (
    MissingReferenceFrameError,
    ObjectHasNoWidthWithoutCollision,
    ReferenceFrameMismatchError,
)
from semantic_digital_twin.spatial_types.spatial_types import Pose

if TYPE_CHECKING:
    from semantic_digital_twin.semantic_annotations.mixins import HasGraspCandidates

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
    def width_along_closing_axis(self) -> float:
        """
        How wide the object stands along the axis the fingers close along, in metres.

        Read as the width of the object's bounding box seen along that axis, which is
        the object's own width wherever the grasp lies square to the box.

        :raises ObjectHasNoWidthWithoutCollision: If the object carries no collision
            geometry.
        """
        body = self.graspable.root
        if not body.has_collision():
            raise ObjectHasNoWidthWithoutCollision(body.name)
        bounding_box = body.collision.as_bounding_box_collection_in_frame(
            body
        ).bounding_box()
        extents = np.array(
            [
                interval.upper - interval.lower
                for interval in (
                    bounding_box.x_interval,
                    bounding_box.y_interval,
                    bounding_box.z_interval,
                )
            ]
        )
        closing_direction = self.grasp_pose.to_np()[:3, 1]
        return float(np.abs(closing_direction) @ extents)

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
        return reference_T_object.homogeneous_matrix @ self.grasp_pose
