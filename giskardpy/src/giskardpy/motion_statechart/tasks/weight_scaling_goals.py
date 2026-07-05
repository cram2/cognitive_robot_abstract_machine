from __future__ import annotations

from dataclasses import dataclass, field

import krrood.symbolic_math.symbolic_math as sm
from giskardpy.motion_statechart.context import MotionStatechartContext
from giskardpy.motion_statechart.graph_node import Task, NodeArtifacts
from semantic_digital_twin.world_description.world_entity import Body


@dataclass
class MaxManipulability(Task):
    """
    This goal maximizes the manipulability of the kinematic chain between root_link and tip_link.
    This chain should only include rotational joint and no linear joints i.e. torso lift joints or odometry joints.
    """

    root_link: Body = field(kw_only=True)
    """
    The root of the kinematic chain whose manipulability is maximized.
    """
    tip_link: Body = field(kw_only=True)
    """
    The tip of the kinematic chain whose manipulability is maximized.
    """
    manipulability_threshold: float = field(default=0.5, kw_only=True)
    """
    Manipulability value the goal drives the measure towards; also defines the observation threshold.
    """

    def build(self, context: MotionStatechartContext) -> NodeArtifacts:
        artifacts = NodeArtifacts()
        root_P_tip = context.world.compose_forward_kinematics_expression(
            self.root_link, self.tip_link
        ).to_position()[:3]

        joint_symbols = root_P_tip.free_variables()
        position_expression = sm.vstack([root_P_tip])
        jacobian = position_expression.jacobian(joint_symbols)
        jacobian_gram = jacobian.dot(jacobian.T)
        manipulability = sm.sqrt(jacobian_gram.det())

        artifacts.geometry.add_position_constraint(
            reference_velocity=1,
            expression_goal=self.manipulability_threshold,
            quadratic_weight=1,
            expression_current=manipulability,
            name=self.name,
        )

        artifacts.observation = (
            sm.abs(self.manipulability_threshold - manipulability) <= 0.01
        )
        return artifacts
