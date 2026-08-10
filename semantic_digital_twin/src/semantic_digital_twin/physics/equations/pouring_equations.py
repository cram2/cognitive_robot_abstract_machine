from __future__ import annotations

from abc import abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, Protocol

import krrood.symbolic_math.symbolic_math as sm
from krrood.adapters.json_serializer import SubclassJSONSerializer
from krrood.symbolic_math.symbolic_math import FloatVariable, Scalar
from typing_extensions import Self, Tuple

from semantic_digital_twin.exceptions import NonPositiveContainerGeometryError
from semantic_digital_twin.physics.equations.differential_equation import (
    DifferentialEquation,
)
from semantic_digital_twin.spatial_types import HomogeneousTransformationMatrix, Vector3

DEFAULT_POUR_EXIT_SPEED: float = 0.2
"""
Default horizontal speed of liquid leaving a fully tilted cup, in metres per second.
"""

DEFAULT_DISCHARGE_COEFFICIENT: float = 0.3
"""
Default discharge coefficient scaling the Torricelli exit speed of a rim pour.

Lumps the losses that make a rim pour slower than ideal orifice efflux (only a thin film
crosses the lip, plus contraction and viscosity), so it is well below an orifice's
``0.6``-``1.0``. Tune it per cup and liquid to match the observed pour range.
"""

STANDARD_GRAVITY: float = 9.81
"""
Gravitational acceleration used for the pouring projectile, in metres per second
squared.
"""

DEFAULT_GATE_SHARPNESS: float = 80.0
"""
Default logistic steepness of the geometric transfer gates.

Shared by the live coupling construction and the serializable coupling descriptor so a
coupling rebuilt from a default descriptor reproduces the same gate that was originally
built.
"""

MINIMUM_POUR_HEAD: float = 0.01
"""
Lower bound on the pour head used in the Torricelli exit speed, in metres.

Keeps the square-root's gradient bounded as the head above the lip approaches zero,
mirroring :data:`MINIMUM_DROP_HEIGHT` for the projectile flight time.
"""

MINIMUM_DROP_HEIGHT: float = 0.01
"""
Lower bound on the source-to-receiver drop used in the projectile flight time, in
metres.

Keeps the flight-time square root away from zero so its gradient stays bounded when the
source rim approaches the receiver opening plane.
"""


class FillContext(Protocol):
    """
    Kinematic context a fill-level ODE is evaluated in.

    Exposes the symbolic quantities a fill equation may depend on. A
    :class:`LiquidConnection` satisfies this protocol directly and is the context used
    in production.
    """

    tilt_expression: Scalar
    """
    Symbolic tilt angle of the container about the vertical, in radians.
    """

    fill_position: Scalar
    """
    Symbolic normalized fill level in ``[0, 1]`` (the fill DOF position).
    """


@dataclass
class SymbolicFillContext:
    """
    Standalone :class:`FillContext` for callers that have no connection (tests,
    autodiff, and tasks that derive the tilt from their own kinematic chain).
    """

    tilt_expression: Scalar
    """
    Symbolic tilt angle of the container about the vertical, in radians.
    """

    fill_position: Scalar
    """
    Symbolic normalized fill level in ``[0, 1]``.
    """


@dataclass
class RectangularContainerGeometry:
    """
    2-D rectangular-cup geometry shared by the pouring-domain fill equations.

    The 2-D cup model works with the half-width throughout: the lip sits at the top corner of
    the rectangle, one half-width away from the vertical centre axis the container tilts about.

    ..note:: This mixin carries no persistence of its own; the equations inheriting it map its
        fields within their own single-rooted DAO hierarchy.
    """

    container_height: float
    """
    Inner height of the rectangular container, in metres.
    """

    container_width: float
    """
    Inner width of the rectangular container (twice the half-width), in metres.
    """

    def __post_init__(self) -> None:
        if self.container_height <= 0.0 or self.container_width <= 0.0:
            raise NonPositiveContainerGeometryError(
                container_height=self.container_height,
                container_width=self.container_width,
            )

    @property
    def half_cross_section_area(self) -> float:
        """
        Area of half the rectangular cross-section, ``(width / 2) * height``, in square
        metres.

        Converts between normalized fill rates and volume rates. Following the half-
        width convention of the 2-D cup model, it spans from the centre axis to the
        pouring lip; the drain and the inflow both normalize with it, so a coupled
        transfer stays volume-consistent.
        """
        return self.container_width / 2 * self.container_height


@dataclass
class FillEquation(DifferentialEquation):
    """
    Abstract first-order ODE for a container's normalized fill level.

    Subclasses produce the symbolic fill velocity from the :class:`FillContext` they are
    evaluated in, giving outflow (pouring) and inflow equations one substitutable
    interface.
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

    Owns the outflow rate constant. Concrete subclasses implement
    :meth:`symbolic_velocity`.
    """

    outflow_rate_constant: float = field(default=1.0, kw_only=True)
    """
    Proportionality constant scaling the discharge gap to the normalized drain rate.
    """

    def ungated(self) -> PouringEquation:
        """
        The serializable, gate-free counterpart of this equation.

        The symbolic gate is bound to the world it was built in and cannot cross a
        process boundary; serialization therefore stores the ungated drain and the
        receiving world re-gates it when the transfer coupling is rebuilt there. Ungated
        equations return themselves.

        :return: This equation without any transfer gate.
        """
        return self

    def symbolic_ode_jacobians(
        self, tilt_expression: Scalar, fill_expression: Scalar
    ) -> Tuple[Scalar, Scalar]:
        """
        Partial derivatives of the fill velocity ODE w.r.t.

        tilt and fill level.         Uses CasADi autodiff on fresh symbolic variables,
        then substitutes the actual         expressions. Both derivatives are computed
        in a single call to avoid evaluating         :meth:`symbolic_velocity` twice.

        :param tilt_expression: Symbolic tilt angle α at the current operating point.
        :param fill_expression: Symbolic fill level h at the current operating point.
        :return:``(∂f/∂α, ∂f/∂h)`` evaluated at ``(tilt_expression, fill_expression)``.
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
class ArticulatedPouringEquation(RectangularContainerGeometry, PouringEquation):
    """
    Pouring ODE derived from the 2-D rectangular-cup model.

    Computes the effective discharge gap from the actual cup dimensions (height ``A``,
    half-width ``r``) and the current tilt angle::

        L(h)    = √((A − h)² + r²)
        φ(h)    = atan2(A − h, r)
        d(α, h) = max(0, L(h) · sin(α − φ(h)))
        ḣ       = −k · d(α, h)
    """

    discharge_coefficient: float = field(
        default=DEFAULT_DISCHARGE_COEFFICIENT, kw_only=True
    )
    """
    Dimensionless coefficient scaling the Torricelli exit speed to a realistic rim pour.
    """

    def to_json(self) -> Dict[str, Any]:
        result = super().to_json()
        result["container_height"] = self.container_height
        result["container_width"] = self.container_width
        result["outflow_rate_constant"] = self.outflow_rate_constant
        result["discharge_coefficient"] = self.discharge_coefficient
        return result

    @classmethod
    def _constructor_arguments_from_json(
        cls, data: Dict[str, Any], **kwargs
    ) -> Dict[str, Any]:
        """
        Extract this class's constructor arguments from a JSON dict.

        Subclasses extend the returned dict with their own arguments so deserialization
        composes along the inheritance chain.

        :param data: The JSON dict.
        :param kwargs: Additional deserialization context.
        :return: Keyword arguments for the constructor.
        """
        return {
            "container_height": data["container_height"],
            "container_width": data["container_width"],
            "outflow_rate_constant": data["outflow_rate_constant"],
            "discharge_coefficient": data.get(
                "discharge_coefficient", DEFAULT_DISCHARGE_COEFFICIENT
            ),
        }

    @classmethod
    def _from_json(cls, data: Dict[str, Any], **kwargs) -> Self:
        return cls(**cls._constructor_arguments_from_json(data, **kwargs))

    def with_gate(self, gate: Scalar) -> GatedArticulatedPouringEquation:
        """
        The gated counterpart of this equation, draining only while ``gate`` is open.

        Subclasses override this so coupling a source to a receiver preserves the head
        model (analytic or learned) instead of always rebuilding an analytic drain.

        :param gate: The shared transfer gate in ``[0, 1]``.
        :return: A gated equation with this equation's parameters.
        """
        return GatedArticulatedPouringEquation(
            container_height=self.container_height,
            container_width=self.container_width,
            outflow_rate_constant=self.outflow_rate_constant,
            discharge_coefficient=self.discharge_coefficient,
            gate=gate,
        )

    def head_above_lip(self, context: FillContext) -> Scalar:
        """
        Height of the liquid surface above the pouring lip, in metres.

        Positive only while the tilt lifts the liquid past the lip, so it is zero
        whenever the container is not spilling.  This is the head that drives the pour.

        :param context: Kinematic context providing the tilt and fill symbols.
        :return: Symbolic head above the lip.
        """
        height = self.container_height
        half_width = self.container_width / 2
        liquid_height = context.fill_position * height
        lip_distance = sm.sqrt((height - liquid_height) ** 2 + half_width**2)
        lip_angle = sm.atan2(height - liquid_height, half_width)
        return sm.max(
            sm.Scalar(0.0),
            lip_distance * sm.sin(context.tilt_expression - lip_angle),
        )

    def exit_velocity(self, context: FillContext) -> Scalar:
        """
        Horizontal exit speed of the pour stream, in metres per second.

        Applies Torricelli's law to the head above the lip and scales it by the discharge
        coefficient: ``C_d * sqrt(2 g h_head)``.  The head is floored at :data:`MINIMUM_POUR_HEAD`
        so the speed stays finite and its gradient bounded as the head approaches zero.

        :param context: Kinematic context providing the tilt and fill symbols.
        :return: Symbolic exit speed.
        """
        head = sm.max(sm.Scalar(MINIMUM_POUR_HEAD), self.head_above_lip(context))
        return self.discharge_coefficient * sm.sqrt(2 * STANDARD_GRAVITY * head)

    def symbolic_velocity(self, context: FillContext) -> Scalar:
        """
        :param context: Kinematic context providing the tilt and fill symbols.
        :return: Symbolic d(fill_normalized)/dt as a CasADi expression.
        """
        return (
            -self.outflow_rate_constant
            * self.head_above_lip(context)
            / self.container_height
        )


@dataclass
class GatedArticulatedPouringEquation(ArticulatedPouringEquation):
    """
    Articulated pouring ODE whose tilt-driven outflow is modulated by a differentiable
    gate.

    Liquid leaves the container only while the gate is open — i.e. while the liquid's
    projectile would land in the target it pours into — so the controlled pour is
    volume-conserving with the target's gated inflow and produces no spill.
    """

    gate: Scalar = field(default_factory=lambda: sm.Scalar(1.0))
    """
    Symbolic transfer gate in ``[0, 1]``; ``1`` when the rim is positioned over the
    target.
    """

    def symbolic_velocity(self, context: FillContext) -> Scalar:
        """
        :param context: Kinematic context providing the tilt and fill symbols.
        :return: Gated d(fill_normalized)/dt; zero while the gate is closed.
        """
        return self.gate * super().symbolic_velocity(context)

    def ungated(self) -> ArticulatedPouringEquation:
        """
        :return: The analytic drain with this equation's parameters, without the symbolic gate.
        """
        return ArticulatedPouringEquation(
            container_height=self.container_height,
            container_width=self.container_width,
            outflow_rate_constant=self.outflow_rate_constant,
            discharge_coefficient=self.discharge_coefficient,
        )


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
class InflowEquation(RectangularContainerGeometry, FillEquation):
    """
    Fill-level ODE for a container receiving liquid.

    Converts an inflow volume rate to a normalised fill velocity for this container
    using its own cross-sectional geometry.
    """

    inflow: Scalar = field(default_factory=lambda: sm.Scalar(0.0))
    """
    The symbolic inflow volume rate entering this container.
    """

    def symbolic_velocity(self, context: FillContext) -> Scalar:
        """
        :param context: Kinematic context; unused, as the inflow rate is already bound.

        ..note:: The receiver's own fill level is not yet read; accepting the shared
            :class:`FillContext` keeps the interface uniform and leaves a hook for future
            overflow gating.

        :return: Normalised fill velocity from inflow.
        """
        return self.inflow / self.half_cross_section_area


@dataclass
class GatedInflowEquation(InflowEquation):
    """
    Inflow ODE whose volume rate is modulated by a differentiable geometric gate.

    Models cup-to-cup transfer: :attr:`inflow` carries the source cup's outflow *volume* rate
    and :attr:`gate` scales it to zero unless the liquid's projectile lands in this receiver's
    opening, so liquid only enters while it would physically land in the receiver.
    """

    gate: Scalar = field(default_factory=lambda: sm.Scalar(1.0))
    """
    Symbolic transfer gate in ``[0, 1]``; ``1`` when the pour's projectile lands in this
    receiver.
    """

    source_tilt_expression: Scalar = field(default_factory=lambda: sm.Scalar(0.0))
    """
    Symbolic tilt angle of the source cup whose outflow feeds this inflow.
    """

    exit_speed: float = field(default=DEFAULT_POUR_EXIT_SPEED)
    """
    Nominal horizontal speed of the liquid leaving the fully tilted source, in metres
    per second.

    Both the gate construction and the no-spill positioning task prefer the source's
    live Torricelli exit speed; this nominal value is the shared fallback they use when
    the source exposes no live outflow model, so both still derive the same projectile
    landing point.
    """

    def symbolic_velocity(self, context: FillContext) -> Scalar:
        """
        :param context: Kinematic context; forwarded to the base inflow conversion.
        :return: Gated normalised fill velocity; zero while the gate is closed.
        """
        return self.gate * super().symbolic_velocity(context)
