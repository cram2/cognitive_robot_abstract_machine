from __future__ import annotations

from dataclasses import dataclass

from typing_extensions import Optional

from coraplex.datastructures.enums import Arms
from semantic_digital_twin.spatial_types.spatial_types import (
    Pose,
    Point3,
    Quaternion,
)
from semantic_digital_twin.world_description.world_entity import (
    KinematicStructureEntity,
)


@dataclass(eq=False, init=False)
class GraspPose(Pose):
    """
    A pose from which a grasp can be performed, along with the arm performing it.
    """

    arm: Optional[Arms] = None
    """
    Arm corresponding to the grasp pose.
    """

    def __init__(
        self,
        position: Optional[Point3] = None,
        orientation: Optional[Quaternion] = None,
        reference_frame: Optional[KinematicStructureEntity] = None,
        arm: Optional[Arms] = None,
    ):
        super().__init__(position, orientation, reference_frame)
        self.arm = arm

    @classmethod
    def from_pose(cls, pose: Pose, arm: Arms) -> GraspPose:
        return cls(
            position=pose.to_position(),
            orientation=pose.to_quaternion(),
            reference_frame=pose.reference_frame,
            arm=arm,
        )
