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
    """Target fill level to achieve in terms of percentage."""

    fill_level_tolerance: float
    """tolerance threshold around goal_value."""

    outflow_tolerance: float = field(default=0.001, kw_only=True)
    """tolerance threshold around zero for final outflow rate."""

    reference_velocity: float = field(default=0.05, kw_only=True)
    """Desired rate of decrease for the normalized fill level."""

    weight: float = field(default=DefaultWeights.WEIGHT_ABOVE_CA, kw_only=True)
    """QP constraint weight for the tilt-driving gradient."""

    def build(self, context: MotionStatechartContext) -> NodeArtifacts:
        """
        Creates the equality constraint that drives the arm to tilt the cup until the goal fill level is reached.

        The fill-level DOF is physics-driven (passive in the QP), so the constraint cannot command fill velocity
        directly. Instead it drives arm joint velocities via the gradient of :attr:`fill_vel_ode` with respect to
        the tilt angle.

        The constraint is formulated as::

            d(fill_vel_ode)/dt = (goal - fill) - fill_vel_ode

        whose steady state is ``fill_vel_ode = goal - fill``, meaning outflow is proportional to the remaining
        error and reaches zero exactly when fill equals the goal. The ``-fill_vel_ode`` term on the right-hand side
        acts as proportional damping: when the cup is already pouring fast enough (or has overshot the goal), the
        QP immediately drives the arm to reduce tilt rather than waiting for the fill error to accumulate.

        Without this damping term the equality bound would be ``goal - fill``, which evaluates to zero at the goal
        and therefore commands the QP to hold outflow *constant* rather than stop it, causing sustained overshoot.

        :param context: The build context.
        :return: The generated task artifacts.
        """
        artifacts = NodeArtifacts()
        fill_sym = self.fill_connection.dof.variables.position

        root_T_tip = context.world.compose_forward_kinematics_expression(
            self.root_link, self.tip_link
        )
        # TODO the tilt expression only works properly when the root link is the world.root
        self.tilt_expr = tilt_expression_from_fk(root_T_tip)
        self.fill_vel_ode = self.fill_equation.symbolic_velocity(
            self.tilt_expr, fill_sym
        )

        artifacts.constraints.add_equality_constraint(
            name=f"{self.fill_connection.name}",
            equality_bound=(sm.Scalar(self.goal_value) - fill_sym) - self.fill_vel_ode,
            quadratic_weight=self.weight,
            task_expression=self.fill_vel_ode,
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
        if (
            fill <= self.goal_value + self.fill_level_tolerance
            and -self.outflow_tolerance < outflow < self.outflow_tolerance
        ):
            return ObservationStateValues.TRUE
        return None
