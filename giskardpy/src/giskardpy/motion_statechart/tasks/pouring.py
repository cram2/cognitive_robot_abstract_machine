"""
Giskardpy Task for the pouring domain (D_pour) of the BMP framework.

Integrates the Torricelli fluid-dynamics ODE inside the MSC control loop so that
fill-level state and tilt-joint control are handled by the statechart itself, with
no external tick-loop logic required.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Optional

import krrood.symbolic_math.symbolic_math as sm
from krrood.symbolic_math.symbolic_math import FloatVariable
from giskardpy.motion_statechart.context import MotionStatechartContext
from giskardpy.motion_statechart.data_types import (
    DefaultWeights,
    ObservationStateValues,
)
from giskardpy.motion_statechart.graph_node import NodeArtifacts, Task
from semantic_digital_twin.world_description.connections import (
    PrismaticConnection,
    RevoluteConnection,
)


@dataclass(eq=False, repr=False)
class PouringTask(Task):
    """
    MSC Task that controls tilt to achieve a fill-level goal via Torricelli ODE.

    At each tick the task:

    1. Reads the current tilt angle from the world state.
    2. Integrates ``d(fill)/dt = -k * max(0, sin(θ − θ_threshold)) * √fill`` one step
       and writes the new fill level to ``fill_connection``.
    3. Uses a predictive lookahead to decide when to start the ramp-down: if simulating
       ``n_ramp`` decreasing-tilt steps from the current angle predicts a final fill
       within tolerance of the goal, the ramp-down target (tilt = 0) is activated.
    4. Returns ``TRUE`` observation when fill ≤ goal + tolerance **and** tilt ≤ theta_threshold,
       confirming that the cup is upright and no more liquid can flow.

    The tilt joint is driven by the QP via an equality constraint whose bound is the
    dynamic ``_tilt_target`` float variable, updated each tick in ``on_tick``.
    """

    tilt_connection: RevoluteConnection
    """Revolute DOF representing the container tilt angle (the actuated joint)."""

    fill_connection: PrismaticConnection
    """Virtual DOF whose position encodes the current fill level in [0, 1]."""

    goal_value: float
    """Target fill level to achieve."""

    tolerance: float
    """Acceptance band around goal_value."""

    theta_max: float = field(default=math.pi / 3, kw_only=True)
    """Maximum tilt angle in radians."""

    theta_threshold: float = field(default=0.3, kw_only=True)
    """Tilt onset: below this angle sin(θ − threshold) ≤ 0 and outflow stops."""

    n_ramp: int = field(default=20, kw_only=True)
    """Steps used in the predictive lookahead to estimate the ramp-down trajectory."""

    k: float = field(default=1.0, kw_only=True)
    """Torricelli outflow rate constant."""

    dt: float = field(default=0.1, kw_only=True)
    """ODE integration timestep in seconds."""

    weight: float = field(default=DefaultWeights.WEIGHT_BELOW_CA, kw_only=True)
    """QP constraint weight for the tilt joint."""

    max_velocity: float = field(default=1.0, kw_only=True)
    """Maximum velocity for the tilt joint in rad/s."""

    _tilt_target: FloatVariable = field(init=False, repr=False)
    """Dynamic float variable used as the QP equality-constraint target for tilt."""

    _ramp_down_started: bool = field(default=False, init=False, repr=False)
    """True once the predictive lookahead has triggered the ramp-down phase."""

    _tilt_step_per_tick: float = field(default=0.0, init=False, repr=False)
    """Actual tilt change per QP tick, calibrated from velocity limit and control_dt."""

    def build(self, context: MotionStatechartContext) -> NodeArtifacts:
        """
        Register the dynamic tilt-target variable and create the QP tilt constraint.

        :param context: MSC build context providing world state and float variable registry.
        :return: NodeArtifacts with one equality constraint on the tilt joint.
        """
        artifacts = NodeArtifacts()

        self._tilt_target = FloatVariable(name=f"{self.name}/tilt_target")
        context.float_variable_data.register_expression(self._tilt_target)
        context.float_variable_data.set_value(self._tilt_target, self.theta_max)

        vel_limit = self.tilt_connection.dof.limits.upper.velocity
        effective_vel = vel_limit if vel_limit is not None else self.max_velocity
        self._tilt_step_per_tick = (
            effective_vel * context.qp_controller_config.control_dt
        )

        current_tilt = self.tilt_connection.dof.variables.position
        velocity_upper = self.tilt_connection.dof.limits.upper.velocity
        velocity_lower = self.tilt_connection.dof.limits.lower.velocity
        if velocity_lower is not None and velocity_upper is not None:
            velocity = sm.limit(self.max_velocity, velocity_lower, velocity_upper)
        else:
            velocity = self.max_velocity

        if not self.tilt_connection.dof.has_position_limits():
            error = sm.shortest_angular_distance(current_tilt, self._tilt_target)
        else:
            error = self._tilt_target - current_tilt

        artifacts.constraints.add_equality_constraint(
            name=str(self.tilt_connection.name),
            reference_velocity=velocity,
            equality_bound=error,
            quadratic_weight=self.weight,
            task_expression=current_tilt,
        )

        return artifacts

    def on_tick(
        self, context: MotionStatechartContext
    ) -> Optional[ObservationStateValues]:
        """
        Integrate ODE, run predictive lookahead, update tilt target, report observation.

        Called before the QP is solved each tick, so the updated ``_tilt_target``
        is visible to ``compute_command``.

        :param context: MSC runtime context.
        :return: ``TRUE`` when fill is achieved and tilt is below threshold, else ``None``.
        """
        theta = float(self.tilt_connection.position)

        fill = float(self.fill_connection.position)
        sin_term = max(0.0, math.sin(theta - self.theta_threshold))
        d_fill = -self.k * sin_term * math.sqrt(max(0.0, fill))
        new_fill = max(0.0, fill + d_fill * self.dt)
        context.world.set_positions_1DOF_connection({self.fill_connection: new_fill})

        if not self._ramp_down_started:
            if self._lookahead_lands_at_goal(theta, new_fill):
                self._ramp_down_started = True

        if self._ramp_down_started:
            context.float_variable_data.set_value(self._tilt_target, 0.0)
        else:
            context.float_variable_data.set_value(self._tilt_target, self.theta_max)

        fill_achieved = new_fill <= self.goal_value + self.tolerance
        if fill_achieved and theta <= self.theta_threshold:
            return ObservationStateValues.TRUE
        return None

    def _lookahead_lands_at_goal(self, from_theta: float, current_fill: float) -> bool:
        """
        Return True if simulating the ramp-down from ``from_theta`` to ``theta_threshold``
        predicts a final fill within tolerance of the goal.

        Uses QP-calibrated step size (``_tilt_step_per_tick``) to match the actual
        rate at which the tilt joint decreases during ramp-down.

        :param from_theta: Current tilt angle from which the ramp-down would start.
        :param current_fill: Fill level at the start of the simulated ramp-down.
        """
        if from_theta <= self.theta_threshold:
            return False
        n_steps = max(
            1, round((from_theta - self.theta_threshold) / self._tilt_step_per_tick)
        )
        simulated_fill = current_fill
        for i in range(n_steps - 1, -1, -1):
            theta_step = (
                self.theta_threshold + (from_theta - self.theta_threshold) * i / n_steps
            )
            sin_term = max(0.0, math.sin(theta_step - self.theta_threshold))
            d_fill = -self.k * sin_term * math.sqrt(max(0.0, simulated_fill))
            simulated_fill = max(0.0, simulated_fill + d_fill * self.dt)
        return abs(simulated_fill - self.goal_value) <= self.tolerance
