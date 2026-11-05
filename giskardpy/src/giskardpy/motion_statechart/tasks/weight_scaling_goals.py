from __future__ import annotations

from dataclasses import dataclass

import krrood.symbolic_math.symbolic_math as sm
from giskardpy.motion_statechart.graph_node import ConvergingTask, MotionNodeArtifacts
from giskardpy.motion_statechart.mixins import (
    HasGoalParameter,
    HasRootLinkParameter,
    HasTipLinkParameter,
)


@dataclass(eq=False, repr=False, kw_only=True)
class MaxManipulability(
    ConvergingTask, HasRootLinkParameter, HasTipLinkParameter, HasGoalParameter[float]
):
    """
    This goal maximizes the manipulability of the kinematic chain between root_link and
    tip_link.

    This chain should only include rotational joint and no linear joints i.e. torso lift
    joints or odometry joints.
    """

    goal: float = 0.5
    """
    Manipulability value the goal drives the measure towards.
    """

    def build_artifacts(self) -> MotionNodeArtifacts:
        """
        Build a constraint that drives the manipulability measure towards
        :attr:`goal`.

        :return: The artifacts of this task, whose error is how far the manipulability measure is from its target value.
        """
        artifacts = MotionNodeArtifacts()
        root_P_tip = self.context.world.compose_forward_kinematics_expression(
            self.root_link, self.tip_link
        ).position[:3]

        joint_symbols = root_P_tip.free_variables()
        position_expression = sm.vstack([root_P_tip])
        jacobian = position_expression.jacobian(joint_symbols)
        jacobian_gram = jacobian.dot(jacobian.T)
        manipulability = sm.sqrt(jacobian_gram.det())

        artifacts.geometry.add_position_constraint(
            reference_velocity=1,
            expression_goal=self.goal,
            quadratic_weight=1,
            expression_current=manipulability,
            name=self.name,
        )

        artifacts.error = sm.abs(self.goal - manipulability)
        return artifacts
