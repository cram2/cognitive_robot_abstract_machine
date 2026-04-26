"""
Pouring-domain physics classes: differential equations, fill-level mixin, and inflow equation.

Provides the SDT-native building blocks for pouring simulation that carry no
giskardpy dependency. The giskardpy-dependent physics models (PouringMSCModel,
CoupledPouringMSCModel) live in pycram.body_motion_problem.pouring.physics.
"""

from __future__ import annotations

from abc import abstractmethod
from dataclasses import dataclass, field

import krrood.symbolic_math.symbolic_math as sm
from krrood.symbolic_math.symbolic_math import Scalar

from semantic_digital_twin.physics.differential_equation import DifferentialEquation
from semantic_digital_twin.world_description.connections import (
    PrismaticConnection,
    RevoluteConnection,
)
from semantic_digital_twin.world_description.geometry import ContainerGeometry


@dataclass
class PouringEquation(DifferentialEquation):
    """
    Abstract ODE for pouring-domain fill-level dynamics.

    Owns the tilt and fill connections plus the outflow rate constant ``k``.
    Concrete subclasses implement :meth:`symbolic_velocity` and optionally
    override :meth:`symbolic_tilt_floor`.

    :param tilt_connection: Revolute DOF providing the current tilt angle θ.
    :param fill_connection: Prismatic DOF whose position encodes fill level in [0, 1].
    :param k: Outflow rate constant.
    """

    tilt_connection: RevoluteConnection
    fill_connection: PrismaticConnection
    k: float = field(default=1.0, kw_only=True)

    @abstractmethod
    def symbolic_velocity(self) -> Scalar:
        """
        Symbolic d(fill_normalized)/dt as a CasADi expression.

        :return: Symbolic desired fill velocity.
        """

    def symbolic_tilt_floor(self, fill_sym: Scalar) -> Scalar:
        """
        Symbolic minimum tilt angle at which flow begins for the given fill level.

        :param fill_sym: Symbolic fill-level position DOF variable.
        :return: Symbolic tilt floor angle in radians.
        """
        return sm.Scalar(0.0)


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

    :param container_geometry: Physical dimensions of the container.
    """

    container_geometry: ContainerGeometry

    def symbolic_tilt_floor(self, fill_sym: Scalar) -> Scalar:
        """
        Returns the geometric tilt offset φ(fill) — the minimum tilt for flow.

        :param fill_sym: Symbolic fill-level position DOF variable.
        :return: Symbolic φ(fill) in radians.
        """
        A = self.container_geometry.height
        r = self.container_geometry.half_width
        return sm.atan2(A - fill_sym * A, r)

    def symbolic_velocity(self) -> Scalar:
        """
        :return: Symbolic d(fill_normalized)/dt as a CasADi expression.
        """
        A = self.container_geometry.height
        r = self.container_geometry.half_width
        h_sym = self.fill_connection.dof.variables.position * A
        L_sym = sm.sqrt((A - h_sym) ** 2 + r**2)
        phi_sym = sm.atan2(A - h_sym, r)
        gap_sym = sm.max(
            sm.Scalar(0.0),
            L_sym * sm.sin(self.tilt_connection.dof.variables.position - phi_sym),
        )
        return -self.k * gap_sym / A


@dataclass
class InflowEquation(DifferentialEquation):
    """
    Fill-level ODE for a container receiving liquid.

    Converts an inflow volume rate to a normalised fill velocity
    for this container using its own cross-sectional geometry.

    :param container_geometry: Physical dimensions of the receiving container.
    :param inflow: The symbolic inflow volume rate.
    """

    container_geometry: ContainerGeometry
    inflow: Scalar = field(default_factory=lambda: sm.Scalar(0.0))

    def symbolic_velocity(self) -> Scalar:
        """
        :return: Normalised fill velocity from inflow.
        """
        receiver_volume = (
            self.container_geometry.half_width * self.container_geometry.height
        )
        return self.inflow / receiver_volume
