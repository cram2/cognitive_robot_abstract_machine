from collections import defaultdict
from dataclasses import dataclass
from typing import List

import krrood.symbolic_math.symbolic_math as sm
from giskardpy.motion_statechart.graph_node import Task
from semantic_digital_twin.spatial_types import Point3
from semantic_digital_twin.spatial_types.derivatives import Derivatives
from semantic_digital_twin.world_description.world_entity import Body


@dataclass
class BaseArmWeightScaling(Task):
    """
    This goals adds weight scaling constraints with the distance between a tip_link and its goal Position as a
    scaling expression. The larger the scaling expression the more is the base movement used toa achieve
    all other constraints instead of arm movements. When the expression decreases this relation changes to favor
    arm movements instead of base movements.
    """

    root_link: Body
    tip_link: Body
    tip_goal: Point3
    arm_joints: List[str]
    base_joints: List[str]
    gain: float = 100000

    def __post_init__(self):
        root_P_tip = context.world._forward_kinematic_manager.compose_expression(
            self.root_link, self.tip_link
        ).to_position()
        root_P_goal = context.world.transform(
            target_frame=self.root_link, spatial_object=self.tip_goal
        )
        scaling_exp = root_P_goal - root_P_tip

        list_gains = []
        for t in range(context.qp_controller.config.prediction_horizon):
            gains = defaultdict(dict)
            arm_v = None
            for name in self.arm_joints:
                vs = context.world.get_connection_by_name(name).active_dofs
                for v in vs:
                    v_gain = self.gain * (scaling_exp / v.upper_limits.velocity).norm()
                    arm_v = v
                    gains[Derivatives.velocity][v] = v_gain
                    gains[Derivatives.acceleration][v] = v_gain
                    gains[Derivatives.jerk][v] = v_gain
            base_v = None
            for name in self.base_joints:
                vs = context.world.get_connection_by_name(name).active_dofs
                for v in vs:
                    v_gain = (
                        self.gain
                        / 100
                        * sm.Scalar(1).safe_division(
                            (scaling_exp / v.upper_limits.velocity).norm()
                        )
                    )
                    base_v = v
                    gains[Derivatives.velocity][v] = v_gain
                    gains[Derivatives.acceleration][v] = v_gain
                    gains[Derivatives.jerk][v] = v_gain
            list_gains.append(gains)

        context.add_debug_expression(
            "base_scaling",
            self.gain
            * sm.Scalar(1).safe_division(
                (scaling_exp / base_v.upper_limits.velocity).norm()
            ),
        )
        context.add_debug_expression(
            "arm_scaling",
            self.gain * (scaling_exp / arm_v.upper_limits.velocity).norm(),
        )
        context.add_debug_expression("norm", scaling_exp.norm())
        context.add_debug_expression("division", 1 / scaling_exp.norm())
        self.add_quadratic_weight_gain("baseToArmScaling", gains=list_gains)


@dataclass
class MaxManipulability(Task):
    """
    This goal maximizes the manipulability of the kinematic chain between root_link and tip_link.
    This chain should only include rotational joint and no linear joints i.e. torso lift joints or odometry joints.
    """

    root_link: Body
    tip_link: Body
    gain: float = 5
    m_threshold: float = 0.15

    def __post_init__(self):
        root_P_tip = context.world._forward_kinematic_manager.compose_expression(
            self.root_link, self.tip_link
        ).to_position()[:3]

        symbols = context.free_variables()
        e = sm.vstack([root_P_tip])
        J = e.jacobian(symbols)
        JJT = J.dot(J.T)
        m = sm.sqrt(JJT.det())

        self.add_position_constraint(
            reference_velocity=1,
            expr_goal=self.m_threshold,
            weight=0.1,
            expr_current=m,
            name=self.name,
        )

        context.add_debug_expression(
            f"mIndex {self.tip_link.name.name}", m, derivatives_to_plot=[0, 1]
        )
        context.add_debug_expression(
            f"mIndex {self.tip_link.name.name} threshold",
            self.m_threshold,
            derivatives_to_plot=[0, 1],
        )
        self.observation_expression = sm.abs(self.m_threshold - m) <= 0.01
