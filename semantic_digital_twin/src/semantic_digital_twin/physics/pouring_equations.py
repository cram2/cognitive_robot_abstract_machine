"""
Pouring-domain physics classes: differential equations, fill-level mixin, and inflow equation.

Provides the SDT-native building blocks for pouring simulation that carry no
giskardpy dependency. The giskardpy-dependent physics model (PouringMSCModel) live in pycram.body_motion_problem.pouring.physics.
"""

from __future__ import annotations

from abc import abstractmethod
from dataclasses import dataclass, field

import krrood.symbolic_math.symbolic_math as sm
from krrood.symbolic_math.symbolic_math import Scalar

from semantic_digital_twin.physics.differential_equation import DifferentialEquation
from semantic_digital_twin.world_description.geometry import ContainerGeometry
from semantic_digital_twin.spatial_types import HomogeneousTransformationMatrix, Vector3


@dataclass
class PouringEquation(DifferentialEquation):
    """
    Abstract ODE for pouring-domain fill-level dynamics.

    Owns the outflow rate constant ``k``.
    Concrete subclasses implement :meth:`symbolic_velocity`.

    :param k: Outflow rate constant.
    """

    k: float = field(default=1.0, kw_only=True)

    @abstractmethod
    def symbolic_velocity(
        self, tilt_expression: Scalar, fill_expression: Scalar
    ) -> Scalar:
        """
        Symbolic d(fill_normalized)/dt as a CasADi expression.

        :param tilt_expression: Symbolic tilt angle θ in radians.
        :param fill_expression: Symbolic fill level in [0, 1].
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

    def symbolic_velocity(
        self, tilt_expression: Scalar, fill_expression: Scalar
    ) -> Scalar:
        """
        :param tilt_expression: Symbolic tilt angle θ in radians.
        :param fill_expression: Symbolic fill level in [0, 1].
        :return: Symbolic d(fill_normalized)/dt as a CasADi expression.
        """
        A = self.container_geometry.height
        r = self.container_geometry.half_width
        h_sym = fill_expression * A
        L_sym = sm.sqrt((A - h_sym) ** 2 + r**2)
        phi_sym = sm.atan2(A - h_sym, r)
        gap_sym = sm.max(
            sm.Scalar(0.0),
            L_sym * sm.sin(tilt_expression - phi_sym),
        )
        return -self.k * gap_sym / A


def tilt_expression_from_fk(root_T_cup: HomogeneousTransformationMatrix) -> Scalar:
    """
    Symbolic tilt angle of a cup about the vertical axis given its FK transform.

    Uses the z-component of the cup's local up axis in the root frame:
    θ = acos(R_zz).

    :param root_T_cup: Symbolic FK expression from root to cup frame.
    :return: Symbolic tilt angle in radians ∈ [-π, π].
    """
    cup_z_in_root = root_T_cup.to_rotation_matrix() @ Vector3(0, 0, 1)
    return sm.acos(sm.limit(cup_z_in_root.z, -3.14, 3.14))


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
