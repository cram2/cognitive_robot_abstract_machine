from __future__ import annotations

from abc import abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, Protocol

import krrood.symbolic_math.symbolic_math as sm
from krrood.adapters.json_serializer import SubclassJSONSerializer
from krrood.symbolic_math.symbolic_math import FloatVariable, Scalar
from typing_extensions import Self, Tuple

from semantic_digital_twin.physics.equations.differential_equation import (
    DifferentialEquation,
)
from semantic_digital_twin.spatial_types import HomogeneousTransformationMatrix, Vector3

DEFAULT_POUR_EXIT_SPEED: float = 0.05
"""Default horizontal speed of liquid leaving a fully tilted cup, in metres per second."""

STANDARD_GRAVITY: float = 9.81
"""Gravitational acceleration used for the pouring projectile, in metres per second squared."""

MINIMUM_DROP_HEIGHT: float = 0.01
"""Lower bound on the source-to-receiver drop used in the projectile flight time, in metres.

Keeps the flight-time square root away from zero so its gradient stays bounded when the source
rim approaches the receiver opening plane.
"""


class FillContext(Protocol):
    """
    Kinematic context a fill-level ODE is evaluated in.

    Exposes the symbolic quantities a fill equation may depend on. A :class:`LiquidConnection`
    satisfies this protocol directly and is the context used in production.
    """

    tilt_expression: Scalar
    """Symbolic tilt angle of the container about the vertical, in radians."""

    fill_position: Scalar
    """Symbolic normalized fill level in ``[0, 1]`` (the fill DOF position)."""


@dataclass
class SymbolicFillContext:
    """
    Standalone :class:`FillContext` for callers that have no connection (tests, autodiff, and tasks
    that derive the tilt from their own kinematic chain).
    """

    tilt_expression: Scalar
    """Symbolic tilt angle of the container about the vertical, in radians."""

    fill_position: Scalar
    """Symbolic normalized fill level in ``[0, 1]``."""


@dataclass
class FillEquation(DifferentialEquation):
    """
    Abstract first-order ODE for a container's normalized fill level.

    Subclasses produce the symbolic fill velocity from the :class:`FillContext` they are evaluated
    in, giving outflow (pouring) and inflow equations one substitutable interface.
    """

    @abstractmethod
    def symbolic_velocity(self, context: FillContext) -> Scalar:
        """
        Symbolic ``d(fill_normalized)/dt`` evaluated in ``context``.

        :param context: Kinematic context providing the tilt and fill symbols.
        :return: Symbolic fill velocity.
        """


@dataclass
class PouringEquation(SubclassJSONSerializer, FillEquation):
    """
    Abstract ODE for pouring-domain fill-level dynamics.

    Owns the outflow rate constant.
    Concrete subclasses implement :meth:`symbolic_velocity`.
    """

    outflow_rate_constant: float = field(default=1.0, kw_only=True)
    """Proportionality constant scaling the discharge gap to the normalized drain rate."""

    def symbolic_ode_jacobians(
        self, tilt_expression: Scalar, fill_expression: Scalar
    ) -> Tuple[Scalar, Scalar]:
        """
        Partial derivatives of the fill velocity ODE w.r.t. tilt and fill level.

        Uses CasADi autodiff on fresh symbolic variables, then substitutes the actual
        expressions. Both derivatives are computed in a single call to avoid evaluating
        :meth:`symbolic_velocity` twice.

        :param tilt_expression: Symbolic tilt angle α at the current operating point.
        :param fill_expression: Symbolic fill level h at the current operating point.
        :return: ``(∂f/∂α, ∂f/∂h)`` evaluated at ``(tilt_expression, fill_expression)``.
        """
        alpha_var = FloatVariable("_ode_alpha")
        h_var = FloatVariable("_ode_h")
        f = self.symbolic_velocity(SymbolicFillContext(alpha_var, h_var))
        df_dalpha = f.jacobian([alpha_var])[0, 0].substitute(
            [alpha_var, h_var], [tilt_expression, fill_expression]
        )
        df_dh = f.jacobian([h_var])[0, 0].substitute(
            [alpha_var, h_var], [tilt_expression, fill_expression]
        )
        return df_dalpha, df_dh


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
    """

    container_height: float
    """Inner height ``A`` of the rectangular container, in metres."""

    container_width: float
    """Inner width of the rectangular container (twice the half-width ``r``), in metres."""

    def to_json(self) -> Dict[str, Any]:
        result = super().to_json()
        result["container_height"] = self.container_height
        result["container_width"] = self.container_width
        result["outflow_rate_constant"] = self.outflow_rate_constant
        return result

    @classmethod
    def _from_json(cls, data: Dict[str, Any], **kwargs) -> Self:
        return cls(
            container_height=data["container_height"],
            container_width=data["container_width"],
            outflow_rate_constant=data["outflow_rate_constant"],
        )

    def symbolic_velocity(self, context: FillContext) -> Scalar:
        """
        :param context: Kinematic context providing the tilt and fill symbols.
        :return: Symbolic d(fill_normalized)/dt as a CasADi expression.
        """
        A = self.container_height
        r = self.container_width / 2
        h_sym = context.fill_position * A
        L_sym = sm.sqrt((A - h_sym) ** 2 + r**2)
        phi_sym = sm.atan2(A - h_sym, r)
        gap_sym = sm.max(
            sm.Scalar(0.0),
            L_sym * sm.sin(context.tilt_expression - phi_sym),
        )
        return -self.outflow_rate_constant * gap_sym / A

    @property
    def cross_section_volume(self) -> float:
        """The 2-D rectangular-cup volume used to convert normalised fill rates to volume rates."""
        return self.container_width / 2 * self.container_height


@dataclass
class GatedArticulatedPouringEquation(ArticulatedPouringEquation):
    """
    Articulated pouring ODE whose tilt-driven outflow is modulated by a differentiable gate.

    Liquid leaves the container only while the gate is open — i.e. while the liquid's projectile
    would land in the target it pours into — so the controlled pour is volume-conserving with the
    target's gated inflow and produces no spill.
    """

    gate: Scalar = field(default_factory=lambda: sm.Scalar(1.0))
    """Symbolic transfer gate in ``[0, 1]``; ``1`` when the rim is positioned over the target."""

    def symbolic_velocity(self, context: FillContext) -> Scalar:
        """
        :param context: Kinematic context providing the tilt and fill symbols.
        :return: Gated d(fill_normalized)/dt; zero while the gate is closed.
        """
        return self.gate * super().symbolic_velocity(context)


def tilt_expression_from_fk(root_T_cup: HomogeneousTransformationMatrix) -> Scalar:
    """
    Symbolic tilt angle of a cup about the vertical axis given its FK transform.

    Uses the z-component of the cup's local up axis in the root frame:
    θ = acos(R_zz).

    :param root_T_cup: Symbolic FK expression from root to cup frame.
    :return: Symbolic tilt angle in radians.
    """
    root_V_cup_z = root_T_cup.to_rotation_matrix() @ Vector3.Z()
    return sm.safe_acos(root_V_cup_z.z)


@dataclass
class InflowEquation(FillEquation):
    """
    Fill-level ODE for a container receiving liquid.

    Converts an inflow volume rate to a normalised fill velocity
    for this container using its own cross-sectional geometry.

    :param inflow: The symbolic inflow volume rate.
    """

    container_height: float
    """Height of the receiving container in metres."""

    container_width: float
    """Width of the receiving container in metres."""

    inflow: Scalar = field(default_factory=lambda: sm.Scalar(0.0))
    """The symbolic inflow volume rate entering this container."""

    @property
    def cross_section_volume(self) -> float:
        """The 2-D rectangular-cup volume used to normalise the inflow volume rate."""
        return self.container_width / 2 * self.container_height

    def symbolic_velocity(self, context: FillContext) -> Scalar:
        """
        :param context: Kinematic context; unused, as the inflow rate is already bound.

        ..note:: The receiver's own fill level is not yet read; accepting the shared
            :class:`FillContext` keeps the interface uniform and leaves a hook for future
            overflow gating.

        :return: Normalised fill velocity from inflow.
        """
        return self.inflow / self.cross_section_volume


@dataclass
class GatedInflowEquation(InflowEquation):
    """
    Inflow ODE whose volume rate is modulated by a differentiable geometric gate.

    Models cup-to-cup transfer: :attr:`inflow` carries the source cup's outflow *volume* rate
    and :attr:`gate` scales it to zero unless the liquid's projectile lands in this receiver's
    opening, so liquid only enters while it would physically land in the receiver.
    """

    gate: Scalar = field(default_factory=lambda: sm.Scalar(1.0))
    """Symbolic transfer gate in ``[0, 1]``; ``1`` when the pour's projectile lands in this receiver."""

    source_tilt_expression: Scalar = field(default_factory=lambda: sm.Scalar(0.0))
    """Symbolic tilt angle of the source cup whose outflow feeds this inflow."""

    exit_speed: float = field(default=DEFAULT_POUR_EXIT_SPEED)
    """Horizontal speed of the liquid leaving the fully tilted source, in metres per second.

    Stored so the no-spill positioning task reconstructs the same projectile landing point the
    gate uses.
    """

    def symbolic_velocity(self, context: FillContext) -> Scalar:
        """
        :param context: Kinematic context; forwarded to the base inflow conversion.
        :return: Gated normalised fill velocity; zero while the gate is closed.
        """
        return self.gate * super().symbolic_velocity(context)
