from __future__ import annotations

from dataclasses import dataclass

from giskardpy.motion_statechart.graph_node import MotionStatechartNode
from giskardpy.motion_statechart.weights import DefaultWeights
from giskardpy.motion_statechart.tasks.cartesian_tasks import (
    CartesianOrientation,
    CartesianPose,
    CartesianPositionStraight,
)


@dataclass(eq=False, repr=False, kw_only=True)
class CartesianPoseStraight(CartesianPose):
    """
    Differs from
    :class:`~giskardpy.motion_statechart.tasks.cartesian_tasks.CartesianPose` in moving
    the tip link along a straight line towards the goal position.
    """

    weight: float = DefaultWeights.WEIGHT_ABOVE_COLLISION_AVOIDANCE
    """
    Above collision avoidance by default.
    """

    def create_tasks(self) -> list[MotionStatechartNode]:
        """
        :return: New position and orientation tasks reaching :attr:`goal`, the position
            along a straight line.
        """
        return [
            CartesianPositionStraight(
                name=f"{self.name}/position",
                root_link=self.root_link,
                tip_link=self.tip_link,
                goal=self.goal.position,
                reference_velocity=self.reference_linear_velocity,
                threshold=self.position_threshold,
                weight=self.weight,
                binding_policy=self.binding_policy,
            ),
            CartesianOrientation(
                name=f"{self.name}/orientation",
                root_link=self.root_link,
                tip_link=self.tip_link,
                goal=self.goal.rotation_matrix,
                reference_velocity=self.reference_angular_velocity,
                threshold=self.orientation_threshold,
                weight=self.weight,
                binding_policy=self.binding_policy,
            ),
        ]
