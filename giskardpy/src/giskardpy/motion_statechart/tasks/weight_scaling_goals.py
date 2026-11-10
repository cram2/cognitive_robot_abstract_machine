from __future__ import annotations

from dataclasses import dataclass

import krrood.symbolic_math.symbolic_math as sm
from giskardpy.motion_statechart.graph_node import ConvergingTask, MotionNodeArtifacts
from giskardpy.motion_statechart.mixins import HasKinematicChainParameters


@dataclass(eq=False, repr=False, kw_only=True)
class MaxManipulability(ConvergingTask, HasKinematicChainParameters):
    """
    This goal maximizes the manipulability of the kinematic chain between root_link and
    tip_link.

    This chain should only include rotational joint and no linear joints i.e. torso lift
    joints or odometry joints.
    """

    goal_manipulability: float = 0.5
    """
    The manipulability measure of the chain this task drives towards.
    """

    def build_artifacts(self) -> MotionNodeArtifacts:
        """
        Build a constraint that drives the manipulability measure towards
        :attr:`goal_manipulability`.

        :return: The artifacts of this task, whose error is how far the manipulability measure is from its target value.
        """
        artifacts = MotionNodeArtifacts()
        root_P_tip = self.context.world.compose_forward_kinematics_expression(
            self.root_link, self.tip_link
        ).position[:3]

        joint_symbols = root_P_tip.free_variables()
        position_expression = sm.vstack([root_P_tip])
        jacobian = position_expression.jacobian(variables=joint_symbols)
        jacobian_gram = jacobian.dot(other=jacobian.T)
        manipulability = sm.sqrt(jacobian_gram.det())

        artifacts.geometry.add_position_constraint(
            cruise_velocity=1,
            velocity_scale=1,
            expression_goal=self.goal_manipulability,
            quadratic_weight=1,
            expression_current=manipulability,
            name=self.name,
        )

        artifacts.error = sm.abs(self.goal_manipulability - manipulability)
        return artifacts
