"""
Giskardpy Tasks for the pouring domain (D_pour) of the BMP framework.

Both the fill-level ODE (Torricelli) and the tilt goal are expressed as
symbolic QP constraints in ``build()``. ``on_tick()`` is a pure observation
check; it performs no integration and no QP manipulation.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import krrood.symbolic_math.symbolic_math as sm
from giskardpy.motion_statechart.context import MotionStatechartContext
from giskardpy.motion_statechart.data_types import (
    DefaultWeights,
    ObservationStateValues,
)
from giskardpy.motion_statechart.graph_node import NodeArtifacts, Task
from semantic_digital_twin.physics.pouring_equations import (
    PouringEquation,
    tilt_expression_from_fk,
)
from semantic_digital_twin.world_description.connections import LiquidConnection
from semantic_digital_twin.world_description.world_entity import Body


@dataclass(eq=False, repr=False)
class PouringTask(Task):
    """
    Motion Statechart task for controlling the tilt and fill level of a container.
    """

    fill_equation: PouringEquation
    """Pouring ODE coupling tilt to the fill-level DOF."""

    fill_connection: LiquidConnection
    """Virtual DOF whose position encodes fill level in [0, 1]."""

    root_link: Body = field(kw_only=True)
    """Root of the kinematic chain used to derive the cup tilt expression."""

    tip_link: Body = field(kw_only=True)
    """Tip of the kinematic chain (the cup body)."""

    goal_value: float
    """Target fill level to achieve."""

    tolerance: float
    """Acceptance band around goal_value."""

    reference_velocity: float = field(default=0.05, kw_only=True)
    """Desired rate of decrease for the normalized fill level."""

    weight: float = field(default=DefaultWeights.WEIGHT_ABOVE_CA, kw_only=True)
    """QP constraint weight for the tilt-driving gradient."""

    def build(self, context: MotionStatechartContext) -> NodeArtifacts:
        """
        Creates the constraints for the fill level and the tilt angle.

        :param context: The build context.
        :return: The generated task artifacts.
        """
        artifacts = NodeArtifacts()
        fill_sym = self.fill_connection.dof.variables.position

        root_T_tip = context.world.compose_forward_kinematics_expression(
            self.root_link, self.tip_link
        )
        tilt_expr = tilt_expression_from_fk(root_T_tip)
        self.fill_vel_ode = self.fill_equation.symbolic_velocity(tilt_expr, fill_sym)

        artifacts.constraints.add_equality_constraint(
            name=f"{self.fill_connection.name}",
            equality_bound=sm.Scalar(self.goal_value) - fill_sym,
            quadratic_weight=self.weight,
            task_expression=fill_sym
            + self.fill_vel_ode,  # This is a linear approximation of fill sym as a function of fill and tilt.
            reference_velocity=self.reference_velocity,
        )
        return artifacts

    def on_tick(
        self, context: MotionStatechartContext
    ) -> Optional[ObservationStateValues]:
        """
        Checks if the goal fill level has been reached and that the outflow is zero.

        :param context: The runtime context.
        :return: The observation state.
        """
        fill = float(self.fill_connection.position)
        outflow = float(self.fill_vel_ode.evaluate()[0])
        if fill <= self.goal_value + self.tolerance and outflow >= 0.0:
            return ObservationStateValues.TRUE
        return None
