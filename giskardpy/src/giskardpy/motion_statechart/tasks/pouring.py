"""
Giskardpy Task for the pouring domain (D_pour) of the BMP framework.

Both the fill-level ODE (Torricelli) and the tilt goal are expressed as
symbolic QP constraints in ``build()``. ``on_tick()`` is a pure observation
check; it performs no integration and no QP manipulation.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Optional

import krrood.symbolic_math.symbolic_math as sm
from giskardpy.motion_statechart.context import MotionStatechartContext
from giskardpy.motion_statechart.data_types import (
    DefaultWeights,
    ObservationStateValues,
)
from giskardpy.motion_statechart.graph_node import NodeArtifacts, Task
from semantic_digital_twin.reasoning.body_motion_problem.pouring.torricelli import (
    TorricelliEquation,
)


@dataclass(eq=False, repr=False)
class PouringTask(Task):
    """
    MSC Task that controls tilt and fill level via two symbolic QP constraints.

    ``build()`` registers:

    1. A velocity equality constraint on the fill joint encoding the Torricelli ODE:
       ``d(fill)/dt = -k * max(0, sin(θ − θ_threshold)) * √fill``.
    2. A position equality constraint on the tilt joint whose target is a
       continuous function of the fill level: tilt ramps linearly from ``theta_max``
       (high fill) to ``0`` (fill ≤ goal).  ``ramp_margin`` controls how many
       fill-level units above the goal the ramp-down begins.

    ``on_tick()`` only reads state and returns the terminal observation; it does not
    integrate the ODE or manipulate QP float variables.
    """

    fill_equation: TorricelliEquation
    """Torricelli ODE coupling the tilt joint to the fill-level DOF."""

    goal_value: float
    """Target fill level to achieve."""

    tolerance: float
    """Acceptance band around goal_value."""

    theta_max: float = field(default=math.pi / 3, kw_only=True)
    """Maximum tilt angle in radians."""

    ramp_margin: float = field(default=0.15, kw_only=True)
    """
    Fill-level units above goal_value at which tilt ramp-down begins.
    Larger values start the ramp earlier, reducing overshoot at the cost of slower pour.
    """

    weight: float = field(default=DefaultWeights.WEIGHT_BELOW_CA, kw_only=True)
    """QP constraint weight for the tilt joint."""

    max_velocity: float = field(default=1.0, kw_only=True)
    """Maximum velocity for the tilt joint in rad/s."""

    def build(self, context: MotionStatechartContext) -> NodeArtifacts:
        """
        Register the fill ODE velocity constraint and the symbolic tilt goal constraint.

        :param context: MSC build context providing world state and the QP config.
        :return: NodeArtifacts with one velocity constraint (fill) and one equality
                 constraint (tilt).
        """
        artifacts = NodeArtifacts()
        tilt_connection = self.fill_equation.tilt_connection
        fill_connection = self.fill_equation.fill_connection

        current_tilt = tilt_connection.dof.variables.position
        fill_sym = fill_connection.dof.variables.position

        # Constraint 1: fill velocity follows the Torricelli ODE.
        fill_vel_goal = self.fill_equation.symbolic_velocity(fill_sym, current_tilt)
        artifacts.constraints.add_velocity_eq_constraint(
            velocity_goal=fill_vel_goal,
            quadratic_weight=DefaultWeights.WEIGHT_ABOVE_CA,
            task_expression=fill_sym,
            velocity_limit=self.fill_equation.k,
            name=str(fill_connection.name),
        )

        # Constraint 2: tilt goal is a continuous function of fill level.
        # tilt_fraction ∈ [0, 1]: reaches 1 when fill is ramp_margin above goal,
        # drops to 0 when fill ≤ goal.
        tilt_fraction = sm.limit(
            (fill_sym - self.goal_value) / self.ramp_margin,
            sm.Scalar(0.0),
            sm.Scalar(1.0),
        )
        tilt_goal = tilt_fraction * self.theta_max

        velocity_upper = tilt_connection.dof.limits.upper.velocity
        velocity_lower = tilt_connection.dof.limits.lower.velocity
        if velocity_lower is not None and velocity_upper is not None:
            velocity = sm.limit(self.max_velocity, velocity_lower, velocity_upper)
        else:
            velocity = self.max_velocity

        if not tilt_connection.dof.has_position_limits():
            tilt_error = sm.shortest_angular_distance(current_tilt, tilt_goal)
        else:
            tilt_error = tilt_goal - current_tilt

        artifacts.constraints.add_equality_constraint(
            name=str(tilt_connection.name),
            reference_velocity=velocity,
            equality_bound=tilt_error,
            quadratic_weight=self.weight,
            task_expression=current_tilt,
        )
        return artifacts

    def on_tick(
        self, context: MotionStatechartContext
    ) -> Optional[ObservationStateValues]:
        """
        Return TRUE when fill reaches the goal tolerance band, else None.

        The QP fill-velocity constraint ensures outflow stops once the tilt returns
        below ``theta_threshold``, so checking only the fill level is sufficient.

        :param context: MSC runtime context (unused; state is read directly from connections).
        :return: ``TRUE`` when fill ≤ goal + tolerance, else ``None``.
        """
        fill = float(self.fill_equation.fill_connection.position)
        if fill <= self.goal_value + self.tolerance:
            return ObservationStateValues.TRUE
        return None
