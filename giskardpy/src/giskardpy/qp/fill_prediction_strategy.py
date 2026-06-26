"""
Terminal fill-level prediction constraint and enforcement strategy for linearized MPC pouring.

The :class:`FillPredictionStrategy` builds a single equality constraint row that drives the
MPC-predicted fill level at the end of the control horizon to a target value.  The prediction
is obtained by linearizing the pouring ODE at the current operating point and unrolling the
discrete-time recursion analytically, so the terminal fill level is a linear function of the
joint velocity decision variables — compatible with the existing PIQP quadratic solver.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import krrood.symbolic_math.symbolic_math as sm
from krrood.symbolic_math.symbolic_math import Scalar, Vector

from giskardpy.qp.constraint import GiskardEqualityConstraint, LargeNumber
from giskardpy.qp.dof_limits import DirectLimits
from giskardpy.qp.enforcement_strategy import IntegralStrategy, normalize_slack_weight


def _geometric_series(base: Scalar, n: int) -> Scalar:
    """
    Computes ``1 + base + base² + … + base^n`` symbolically.

    :param base: Symbolic base value λ.
    :param n: Highest power (inclusive).
    :return: Symbolic geometric series sum.
    """
    power = sm.Scalar(1.0)
    total = sm.Scalar(0.0)
    for _ in range(n + 1):
        total = total + power
        power = power * base
    return total


def _compute_power(base: Scalar, n: int) -> Scalar:
    """
    Computes ``base^n`` symbolically via repeated multiplication.

    :param base: Symbolic base value.
    :param n: Non-negative integer exponent.
    :return: Symbolic ``base^n``.
    """
    result = sm.Scalar(1.0)
    for _ in range(n):
        result = result * base
    return result


@dataclass
class GiskardFillPredictionConstraint(GiskardEqualityConstraint):
    """
    Equality constraint carrying the ODE parameters that :class:`FillPredictionStrategy`
    needs to build the terminal fill-level prediction bound.

    ``expression`` is set to ``fill_vel_ode`` so :class:`IntegralStrategy` inherits a
    reactive Jacobian ``∂f/∂q = a·J_α`` that stays well-defined near the pouring singularity.
    The strategy replaces the plain ``c.bound`` with ``(goal − h_N^free) − fill_vel_ode``,
    combining a proactive MPC prediction with the reactive outflow term that prevents runaway
    tilt when ``df_dalpha ≈ 0``.
    """

    tilt_expression: Scalar = field(kw_only=True)
    """Symbolic tilt angle α(q) derived from forward kinematics."""

    fill_sym: Scalar = field(kw_only=True)
    """Symbolic current fill level h₀ (passive DOF position variable)."""

    fill_vel_ode: Scalar = field(kw_only=True)
    """Symbolic ODE value f(α₀, h₀) at the current operating point."""

    df_dh: Scalar = field(kw_only=True)
    """∂f/∂h evaluated at the current operating point."""

    goal_value: float = field(kw_only=True)
    """Target fill level."""


@dataclass
class FillPredictionStrategy(IntegralStrategy):
    """
    Enforcement strategy for the terminal fill-level prediction constraint.

    Inherits :meth:`IntegralStrategy.create_matrix` so the constraint matrix is
    ``∂(fill_vel_ode)/∂q · dt`` replicated across the control horizon.  This reactive
    Jacobian has the same singularity behaviour as the old reactive constraint: when
    ``df_dalpha → 0`` the matrix row vanishes, the QP drives velocities to zero via
    regularization, and the arm holds its tilt angle while the fill level self-corrects.

    The equality bound combines a proactive MPC prediction with the reactive outflow term:

    .. math::

        b = (h^{\\mathrm{goal}} - h_M^{\\mathrm{free}}) - f_0

    where :math:`h_M^{\\mathrm{free}}` is the predicted terminal fill level under zero
    velocity, and :math:`f_0 = \\dot{h}` is the current ODE outflow.  The ``-f_0`` term
    acts as reactive anti-windup: when outflow is large (``f_0 ≪ 0``), the bound becomes
    strongly positive and drives tilt-back even if ``h_M^{\\mathrm{free}}`` looks acceptable.
    """

    def create_equality_bounds(self) -> Vector:
        """
        Computes the capped equality bound ``(goal − h_M^free) − fill_vel_ode``.

        :math:`h_M^{\\mathrm{free}}` is the predicted fill level at the end of the control
        horizon under zero velocity:

        .. math::

            h_M^{\\mathrm{free}} = \\lambda^M h_0 + G_{M-1} \\, \\Delta t \\, (f_0 - b h_0)
        """
        self._require_constraint_type(GiskardEqualityConstraint)
        [c] = self.constraints
        dt = self.qp_controller_config.model_predictive_control_time_step
        m = self.qp_controller_config.control_horizon

        lambda_ = sm.Scalar(1.0) + sm.Scalar(dt) * c.df_dh
        lambda_m = _compute_power(lambda_, m)
        g_m_minus_1 = _geometric_series(lambda_, m - 1)

        h_free = lambda_m * c.fill_sym + g_m_minus_1 * sm.Scalar(dt) * (
            c.fill_vel_ode - c.df_dh * c.fill_sym
        )
        bound = sm.Scalar(c.goal_value) - h_free - c.fill_vel_ode
        capped = self.capped_bound(bound, dt, c.normalization_factor, m)
        return sm.Vector([capped])

    def create_slack_variables(self) -> DirectLimits:
        """
        Creates one normalized slack variable for the single terminal fill constraint.
        """
        [c] = self.constraints
        return DirectLimits(
            lower_bounds=Vector([-LargeNumber]),
            upper_bounds=Vector([LargeNumber]),
            quadratic_weights=Vector(
                [
                    normalize_slack_weight(
                        c.quadratic_weight,
                        c.normalization_factor,
                        self.qp_controller_config.control_horizon,
                    )
                ]
            ),
            linear_weights=Vector([sm.Scalar(0.0)]),
            names=[c.name],
        )
