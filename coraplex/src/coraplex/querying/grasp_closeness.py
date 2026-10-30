from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from krrood.entity_query_language.predicate import RenderedFields, SymbolicFunction
from krrood.entity_query_language.verbalization.fragments.base import (
    VerbalizationFragment,
)
from krrood.entity_query_language.verbalization.vocabulary.parts_of_speech import (
    FunctionVerbalizationTemplates,
)
from semantic_digital_twin.grasping.grasp_candidates import GraspCandidate
from semantic_digital_twin.semantic_annotations.mixins import HasGraspCandidates
from semantic_digital_twin.spatial_types.spatial_types import Pose

# %% how close a grasp is to where the robot stands


@dataclass(frozen=True, order=True)
class GraspCloseness:
    """
    How close a grasp is to where the robot stands: a grasp nearer by horizontal
    distance is closer, and of two equally near grasps the one approached more directly
    from the standing position is.
    """

    horizontal_distance: float
    """
    The horizontal distance from the standing position to the grasp, in meters.
    """

    approach_angle: float
    """
    The angle between the direction the grasp is approached along and the direction from
    the standing position to it, in radians.
    """

    @classmethod
    def from_grasp(
        cls,
        grasp: GraspCandidate,
        world_T_object: NDArray[np.float64],
        world_P_standing: NDArray[np.float64],
    ) -> GraspCloseness:
        """
        :param grasp: A grasp on the object.
        :param world_T_object: The object's root frame in the world frame.
        :param world_P_standing: Where the robot stands, as a homogeneous point in the
            world frame.
        :return: How close `grasp` is to the standing position.
        """
        world_T_grasp = world_T_object @ grasp.grasp_pose.to_np()
        world_V_standing_to_grasp = world_T_grasp[:, 3] - world_P_standing
        horizontal_distance = float(np.linalg.norm(world_V_standing_to_grasp[:2]))
        distance = np.linalg.norm(world_V_standing_to_grasp)
        if distance == 0:
            # Standing exactly at the grasp leaves no direction to approach it from.
            return cls(horizontal_distance, 0.0)
        cosine = world_T_grasp[:, 0] @ world_V_standing_to_grasp / distance
        return cls(horizontal_distance, float(np.arccos(np.clip(cosine, -1.0, 1.0))))


# %% the grasps of an object by their closeness to a standing position


@dataclass(eq=False)
class GraspsByCloseness(SymbolicFunction):
    """
    The grasps an object offers, ordered by their :class:`GraspCloseness` to a standing
    position, the closest first, and at most :attr:`number_of_grasps` of them.
    """

    graspable: HasGraspCandidates
    """
    The object offering the grasps.
    """

    standing_position: Pose
    """
    Where the robot stands while taking one of the grasps, read when the grasps are
    ranked, so a pose relative to the robot is where the robot stands then.
    """

    number_of_grasps: int | None = 3
    """
    How many of the closest grasps are kept, all of them if None.
    """

    def __call__(self) -> list[GraspCandidate]:
        world = self.graspable.root._world
        world_P_standing = world.transform(self.standing_position, world.root).to_np()[
            :, 3
        ]
        world_T_object = self.graspable.root.global_transform.to_np()
        grasps_by_closeness = sorted(
            self.graspable.grasp_candidates(),
            key=lambda grasp: GraspCloseness.from_grasp(
                grasp, world_T_object, world_P_standing
            ),
        )
        return grasps_by_closeness[: self.number_of_grasps]

    @classmethod
    def _verbalization_fragment_(cls, fields: RenderedFields) -> VerbalizationFragment:
        """
        :param fields: The rendered fragment for each field.
        :return: The noun phrase *"the grasps by closeness of <graspable> and <standing
            position>"*.
        """
        return FunctionVerbalizationTemplates.possessive(
            cls, fields["graspable"], fields["standing_position"]
        )
