"""
Pouring-domain differential equations.
"""

from __future__ import annotations

import math
from abc import abstractmethod
from dataclasses import dataclass, field

import krrood.symbolic_math.symbolic_math as sm
from krrood.symbolic_math.symbolic_math import Scalar

from semantic_digital_twin.physics.differential_equation import DifferentialEquation
from semantic_digital_twin.semantic_annotations.mixins import ContainerGeometry
from semantic_digital_twin.world_description.connections import (
    PrismaticConnection,
    RevoluteConnection,
)


@dataclass
class PouringEquation(DifferentialEquation):
    """
    Abstract ODE for pouring-domain fill-level dynamics.

    Owns the tilt and fill connections plus the outflow rate constant ``k``.
    Concrete subclasses implement :meth:`compute_derivative` and
    :meth:`symbolic_velocity`; :meth:`step` and :meth:`simulate_rampdown` are
    provided here using those two primitives.

    :param tilt_connection: Revolute DOF providing the current tilt angle θ.
    :param fill_connection: Prismatic DOF whose position encodes fill level in [0, 1].
    :param k: Outflow rate constant.
    """

    tilt_connection: RevoluteConnection
    fill_connection: PrismaticConnection
    k: float = field(default=1.0, kw_only=True)

    @abstractmethod
    def compute_derivative(self, state: float, theta: float) -> float:
        """
        Compute d(fill_normalized)/dt for an explicit tilt angle.

        :param state: Current fill level in [0, 1].
        :param theta: Tilt angle in radians.
        :return: Rate of change of fill level.
        """

    @abstractmethod
    def symbolic_velocity(self, fill_sym: Scalar, tilt_sym: Scalar) -> Scalar:
        """
        Symbolic d(fill_normalized)/dt as a CasADi expression.

        :param fill_sym: Symbolic fill-level position DOF variable.
        :param tilt_sym: Symbolic tilt-angle position DOF variable.
        :return: Symbolic desired fill velocity.
        """

    def symbolic_tilt_floor(self, fill_sym: Scalar) -> Scalar:
        """
        Symbolic minimum tilt angle at which flow begins for the given fill level.

        Used by :class:`PouringTask` as the lower end of the tilt ramp-down so that
        the ramp always maintains a tilt sufficient for flow until the goal is reached.
        The default returns zero, which is appropriate for equations where flow depends
        only on a fixed threshold rather than fill-level geometry.

        :param fill_sym: Symbolic fill-level position DOF variable.
        :return: Symbolic tilt floor angle in radians.
        """
        return sm.Scalar(0.0)

    def step(self, state: float, dt: float) -> float:
        """
        Euler-integrate one step using the current tilt angle from the world.

        :param state: Current fill level in [0, 1].
        :param dt: Integration timestep in seconds.
        :return: New fill level after one Euler step, clamped to [0, 1].
        """
        theta = float(self.tilt_connection.position)
        return max(0.0, state + self.compute_derivative(state, theta) * dt)

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

        :param state: Fill level at the start of the simulated ramp.
        :param dt: Integration timestep per step.
        :param from_theta: Starting tilt angle for the simulated ramp.
        :param to_theta: Ending tilt angle for the simulated ramp.
        :param n_steps: Number of steps in the simulation.
        :return: Predicted fill level at the end of the ramp.
        """
        for i in range(n_steps - 1, -1, -1):
            theta_step = to_theta + (from_theta - to_theta) * i / n_steps
            state = max(0.0, state + self.compute_derivative(state, theta_step) * dt)
        return state


@dataclass
class ArticulatedPouringEquation(PouringEquation):
    """
    Pouring ODE derived from the 2-D rectangular-cup model.

    Computes the effective discharge gap from actual cup dimensions (height ``A``,
    half-width ``r``) and the current tilt angle::

        L(h)    = √((A − h)² + r²)
        φ(h)    = atan2(A − h, r)
        d(α, h) = max(0, L(h) · sin(α − φ(h)))
        ḣ       = −k · d(α, h)

    The fill level is stored in the world as a normalized value in ``[0, 1]``; the
    derivative is converted via ``d(fill_norm)/dt = ḣ / A``.

    :param container_geometry: Physical dimensions of the container.
    """

    container_geometry: ContainerGeometry

    def _lever_arm(self, fill_normalized: float) -> float:
        h = fill_normalized * self.container_geometry.height
        A, r = self.container_geometry.height, self.container_geometry.half_width
        return math.sqrt((A - h) ** 2 + r**2)

    def _tilt_offset(self, fill_normalized: float) -> float:
        h = fill_normalized * self.container_geometry.height
        A, r = self.container_geometry.height, self.container_geometry.half_width
        return math.atan2(A - h, r)

    def compute_derivative(self, state: float, theta: float) -> float:
        """
        Compute d(fill_normalized)/dt using the geometric discharge gap.

        :param state: Current fill level in [0, 1].
        :param theta: Tilt angle in radians.
        :return: Rate of change of normalized fill level.
        """
        gap = max(
            0.0, self._lever_arm(state) * math.sin(theta - self._tilt_offset(state))
        )
        return -self.k * gap / self.container_geometry.height

    def symbolic_tilt_floor(self, fill_sym: Scalar) -> Scalar:
        """
        Returns the geometric tilt offset φ(fill) — the minimum tilt for flow.

        :param fill_sym: Symbolic fill-level position DOF variable.
        :return: Symbolic φ(fill) in radians.
        """
        A = self.container_geometry.height
        r = self.container_geometry.half_width
        return sm.atan2(A - fill_sym * A, r)

    def symbolic_velocity(self, fill_sym: Scalar, tilt_sym: Scalar) -> Scalar:
        """
        Symbolic d(fill_normalized)/dt as a CasADi expression.

        :param fill_sym: Symbolic fill-level position DOF variable.
        :param tilt_sym: Symbolic tilt-angle position DOF variable.
        :return: Symbolic desired fill velocity.
        """
        A = self.container_geometry.height
        r = self.container_geometry.half_width
        h_sym = fill_sym * A
        L_sym = sm.sqrt((A - h_sym) ** 2 + r**2)
        phi_sym = sm.atan2(A - h_sym, r)
        gap_sym = sm.max(sm.Scalar(0.0), L_sym * sm.sin(tilt_sym - phi_sym))
        return -self.k * gap_sym / A
