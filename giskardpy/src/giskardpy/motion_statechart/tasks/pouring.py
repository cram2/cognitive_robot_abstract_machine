from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

from giskardpy.motion_statechart.context import MotionStatechartContext
from giskardpy.motion_statechart.data_types import (
    DefaultWeights,
    ObservationStateValues,
)
from giskardpy.motion_statechart.graph_node import NodeArtifacts, Task
from semantic_digital_twin.physics.equations.pouring_equations import (
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
        Creates the terminal fill-level prediction constraint for linearized MPC pouring.

        The pouring ODE ``ḣ = f(α, h)`` is linearized at the current operating point and
        the discrete-time recursion is unrolled analytically over the control horizon.  The
        resulting single QP row drives the MPC-predicted fill level at the end of the horizon
        toward :attr:`goal_value`.

        Because the terminal constraint couples earlier velocity decisions to a larger share of
        the predicted fill change (via geometric-series weights), the optimizer sees that
        continued high tilt will overshoot and proactively starts tilting back before the fill
        level reaches the goal — the key advantage over a purely reactive formulation.

        :param context: The build context.
        :return: The generated task artifacts.
        """
        artifacts = NodeArtifacts()
        self.fill_connection = context.world.get_connection(
            self.fill_connection.parent, self.fill_connection.child
        )
        fill_sym = self.fill_connection.dof.variables.position

        root_T_tip = context.world.compose_forward_kinematics_expression(
            self.root_link, self.tip_link
        )
        # TODO the tilt expression only works properly when the root link is the world.root
        self.tilt_expr = tilt_expression_from_fk(root_T_tip)
        self.fill_vel_ode = self.fill_equation.symbolic_velocity(
            self.tilt_expr, fill_sym
        )

        artifacts.constraints.add_fill_prediction_constraint(
            name=f"{self.fill_connection.name}",
            tilt_expression=self.tilt_expr,
            fill_sym=fill_sym,
            fill_vel_ode=self.fill_vel_ode,
            fill_equation=self.fill_equation,
            goal_value=self.goal_value,
            quadratic_weight=self.weight,
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
        print(f"Fill level: {fill}, Outflow: {outflow}")
        if (
            fill <= self.goal_value + self.fill_level_tolerance
            and -self.outflow_tolerance < outflow < self.outflow_tolerance
        ):
            return ObservationStateValues.TRUE
        return None
