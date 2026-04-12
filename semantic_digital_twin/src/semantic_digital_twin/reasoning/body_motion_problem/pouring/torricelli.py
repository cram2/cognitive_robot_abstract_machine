"""
Torricelli fluid-dynamics ODE for the pouring domain.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import krrood.symbolic_math.symbolic_math as sm
from krrood.symbolic_math.symbolic_math import Scalar

from semantic_digital_twin.physics.differential_equation import DifferentialEquation
from semantic_digital_twin.world_description.connections import (
    PrismaticConnection,
    RevoluteConnection,
)


@dataclass
class TorricelliEquation(DifferentialEquation):
    """
    Torricelli outflow ODE: d(fill)/dt = -k * max(0, sin(θ − θ_threshold)) * √fill.

    Reads the current tilt angle from :attr:`tilt_connection` at each step, so
    advancing the world and calling :meth:`step` is all that is required.

    :param tilt_connection: Revolute DOF providing the current tilt angle θ.
    :param fill_connection: Prismatic DOF whose position encodes fill level in [0, 1].
    :param k: Outflow rate constant.
    :param theta_threshold: Tilt onset angle; below this value no liquid flows.
    """

    tilt_connection: RevoluteConnection
    fill_connection: PrismaticConnection
    k: float = field(default=1.0)
    theta_threshold: float = field(default=0.3)

    def step(self, state: float, dt: float) -> float:
        """
        Euler-integrate one step using the current tilt angle from the world.

        :param state: Current fill level in [0, 1].
        :param dt: Integration timestep in seconds.
        :return: New fill level after one Euler step, clamped to [0, 1].
        """
        theta = float(self.tilt_connection.position)
        d_fill = self.compute_derivative(state, theta)
        return max(0.0, state + d_fill * dt)

    def compute_derivative(self, state: float, theta: float) -> float:
        """
        Compute d(fill)/dt for an explicit tilt angle.

        :param state: Current fill level.
        :param theta: Tilt angle in radians.
        :return: Rate of change of fill level.
        """
        sin_term = max(0.0, math.sin(theta - self.theta_threshold))
        return -self.k * sin_term * math.sqrt(max(0.0, state))

    def symbolic_velocity(self, fill_sym: Scalar, tilt_sym: Scalar) -> Scalar:
        """
        Symbolic Torricelli velocity d(fill)/dt as a CasADi expression.

        :param fill_sym: Symbolic fill-level position DOF variable.
        :param tilt_sym: Symbolic tilt-angle position DOF variable.
        :return: Symbolic desired fill velocity.
        """
        sin_term = sm.max(sm.Scalar(0.0), sm.sin(tilt_sym - self.theta_threshold))
        return -self.k * sin_term * sm.sqrt(sm.max(sm.Scalar(0.0), fill_sym))

    def simulate_rampdown(
        self,
        state: float,
        dt: float,
        from_theta: float,
        to_theta: float,
        n_steps: int,
    ) -> float:
        """
        Simulate fill level as tilt linearly ramps from ``from_theta`` to ``to_theta``.

        Used for predictive lookahead to decide when to begin the tilt ramp-down.

        :param state: Fill level at the start of the simulated ramp.
        :param dt: Integration timestep per step.
        :param from_theta: Starting tilt angle for the simulated ramp.
        :param to_theta: Ending tilt angle for the simulated ramp.
        :param n_steps: Number of steps in the simulation.
        :return: Predicted fill level at the end of the ramp.
        """
        for i in range(n_steps - 1, -1, -1):
            theta_step = to_theta + (from_theta - to_theta) * i / n_steps
            d_fill = self.compute_derivative(state, theta_step)
            state = max(0.0, state + d_fill * dt)
        return state
