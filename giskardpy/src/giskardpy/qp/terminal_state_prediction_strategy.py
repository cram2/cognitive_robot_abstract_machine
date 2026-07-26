"""
Terminal-state prediction constraint and enforcement strategy for linearized MPC.

The :class:`TerminalStatePredictionStrategy` builds a single equality constraint row that drives
the MPC-predicted value of a scalar state at the end of the control horizon to a target value.
The prediction is obtained by linearizing the state's first-order ODE at the current operating
point and unrolling the discrete-time recursion analytically, so the terminal state is a linear
function of the joint velocity decision variables — compatible with the existing PIQP quadratic
solver.

The mechanism is domain-agnostic: any scalar state governed by ``ẋ = f(x, q)`` whose terminal
value should reach a goal (e.g. a container fill level driven by a tilt or a valve angle) can use
it by supplying the symbolic state velocity and the passive state variable.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from functools import cached_property

import krrood.symbolic_math.symbolic_math as sm
from krrood.symbolic_math.symbolic_math import Matrix, Scalar, Vector

from giskardpy.qp.constraint import (
    GiskardEqualityConstraint,
    GiskardInequalityConstraint,
    LargeNumber,
)
from giskardpy.qp.dof_limits import DirectLimits
from giskardpy.qp.enforcement_strategy import IntegralStrategy, normalize_slack_weight


def _geometric_series(base: Scalar, n: int) -> Scalar:
    """
    Computes ``1 + base + base² + … + base^n`` symbolically.

    :param base: Symbolic base value λ.
    :param n: Highest power (inclusive). A negative value yields an empty sum of zero.
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


def horizon_normalized_weights(
    weights: list[Scalar], control_horizon: int
) -> list[Scalar]:
    """
    Rescales lookahead weights to unit average over the horizon for QP conditioning.

    The raw terminal-state sensitivity carries an extra ``dt`` per block, making the matrix
    coefficients far smaller than the proven reactive integral and ill-conditioning the QP.
    Rescaling so the weights sum to the control horizon preserves their relative lookahead
    emphasis while keeping the row at the calibrated reactive scale.  This is a solver-conditioning
    concern, not a property of the linearized dynamics, so it lives with the strategy rather than
    the model.

    :param weights: Geometric lookahead weights from
        :meth:`LinearizedScalarStateModel.lookahead_weights`.
    :param control_horizon: Number of velocity decision steps the weight sum is rescaled to.
    :return: The rescaled weights, summing to ``control_horizon``.
    """
    total = sm.Scalar(0.0)
    for weight in weights:
        total = total + weight
    horizon = sm.Scalar(float(control_horizon))
    return [weight * horizon / total for weight in weights]


@dataclass
class LinearizedScalarStateModel:
    """
    Discrete-time linearization of a first-order scalar ODE about the current operating point.

    Encodes the recursion ``x_{k+1} = λ·x_k + c + dt·a·δu_k`` with ``λ = 1 + dt·(∂f/∂x)`` and
    ``c = dt·(f₀ − (∂f/∂x)·x₀)``.  Solving it over the control horizon splits the terminal state
    into a control-independent free response and a control contribution whose per-step control
    deviations are weighted by a geometric series.
    """

    state_value: Scalar
    """Current state value x₀ (passive DOF position)."""

    state_velocity: Scalar
    """Current ODE rate f₀ = f(x₀, q₀)."""

    state_sensitivity: Scalar
    """Partial derivative ∂f/∂x at the operating point."""

    time_step: float
    """MPC discretization step dt in seconds."""

    control_horizon: int
    """Number of velocity decision steps M over which commands are applied."""

    @property
    def decay(self) -> Scalar:
        """Linearized state decay factor ``λ = 1 + dt·(∂f/∂x)``."""
        return sm.Scalar(1.0) + sm.Scalar(self.time_step) * self.state_sensitivity

    def free_response(self) -> Scalar:
        """
        Predicted terminal state if the control is held constant (zero joint velocity).

        This is the autonomous evolution of the linearized system, not a frozen-state assumption:
        the state keeps evolving at the held control.
        """
        decay_to_horizon = _compute_power(self.decay, self.control_horizon)
        series = _geometric_series(self.decay, self.control_horizon - 1)
        return decay_to_horizon * self.state_value + series * sm.Scalar(
            self.time_step
        ) * (self.state_velocity - self.state_sensitivity * self.state_value)

    def lookahead_weights(self) -> list[Scalar]:
        """
        Geometric lookahead weight ``G_{M-2-i}`` of each velocity block.

        Block ``i`` (earliest first) raises the control for the remaining ``M-1-i`` state steps,
        so the terminal state is more sensitive to early decisions; the final block has weight
        zero because no state step follows it.
        """
        return [
            _geometric_series(self.decay, self.control_horizon - 2 - block)
            for block in range(self.control_horizon)
        ]


@dataclass
class TerminalStatePredictionConstraint(GiskardEqualityConstraint):
    """
    Equality constraint carrying the operating point that
    :class:`TerminalStatePredictionStrategy` linearizes into a terminal-state prediction.

    The inherited ``expression`` holds the state rate ``f(x₀, q₀)``: its free variables register the
    joint variables with the QP, and its jacobian w.r.t. them gives the per-step control sensitivity
    the strategy weights across the horizon.  The strategy derives ``∂f/∂x`` and the predicted
    terminal bound from this expression and :attr:`state_variable`, so no further linearization
    fields are stored here.
    """

    state_variable: Scalar = field(kw_only=True)
    """Symbolic current state x₀ (passive DOF position variable); the rate is differentiated w.r.t. it."""

    goal_value: float = field(kw_only=True)
    """Target state value at the end of the horizon."""

    bound: Scalar = field(default_factory=lambda: sm.Scalar(0.0), kw_only=True)
    """Unused inherited equality bound; the strategy computes the terminal bound ``goal − x_free`` instead."""


@dataclass
class TerminalStatePredictionStrategy(IntegralStrategy):
    """
    Enforcement strategy for the terminal-state prediction constraint.

    The constraint couples each velocity decision to the predicted terminal state with the
    relative emphasis derived from the linearized recursion: the state-rate jacobian
    ``∂f/∂q·dt`` is scaled per horizon block by :func:`horizon_normalized_weights`, so an earlier
    velocity — which keeps the control applied for more of the remaining horizon — affects the
    terminal state more than a later one, and the final block has no effect.  The weights are
    normalized to unit average so the row stays at the well-conditioned scale of the plain reactive
    integral.

    The bound ``goal − x_free`` supplies the proactive terminal prediction error.  Where the
    control sensitivity ``∂f/∂q → 0`` the whole row vanishes, the QP regularizes velocities to
    zero, and the state self-corrects.
    """

    @cached_property
    def _state_model(self) -> LinearizedScalarStateModel:
        """
        Linearizes the single constraint's ODE at the current operating point, built once per solve.

        The state sensitivity ``∂f/∂x`` is differentiated here from the constraint's state-rate
        expression w.r.t. its own state variable, keeping all jacobian computation in one layer.
        """
        [constraint] = self.constraints
        state_sensitivity = constraint.expression.jacobian([constraint.state_variable])[
            0, 0
        ]
        return LinearizedScalarStateModel(
            state_value=constraint.state_variable,
            state_velocity=constraint.expression,
            state_sensitivity=state_sensitivity,
            time_step=self.qp_controller_config.model_predictive_control_time_step,
            control_horizon=self.qp_controller_config.control_horizon,
        )

    def create_matrix(self) -> Matrix:
        """
        Builds the constraint row by scaling the state-rate jacobian per horizon block with the
        geometric velocity weights, padding the jerk columns with zeros.
        """
        [constraint] = self.constraints
        time_step = self.qp_controller_config.model_predictive_control_time_step
        jacobian = (
            sm.Vector([constraint.expression]).jacobian(self.position_variables)
            * time_step
        )
        weights = horizon_normalized_weights(
            self._state_model.lookahead_weights(),
            self.qp_controller_config.control_horizon,
        )
        blocks = [jacobian * weight for weight in weights]
        return sm.hstack(
            blocks + [sm.Matrix.zeros(jacobian.shape[0], self.number_of_jerk_columns)]
        )

    def create_equality_bounds(self) -> Vector:
        """
        Computes the capped equality bound ``goal − x_free``.

        ``x_free`` is the predicted terminal state under zero joint velocity, so the bound is the
        terminal prediction error the QP drives to zero.

        .. note:: This relies on a prediction horizon long enough for ``x_free`` to span the
            terminal overshoot; at very short horizons the prediction is too myopic to ease off in
            time and the state overshoots the goal.
        """
        self._require_constraint_type(GiskardEqualityConstraint)
        [constraint] = self.constraints
        bound = sm.Scalar(constraint.goal_value) - self._state_model.free_response()
        capped = self.capped_bound(
            bound,
            self.qp_controller_config.model_predictive_control_time_step,
            constraint.normalization_factor,
            self.qp_controller_config.control_horizon,
        )
        return sm.Vector([capped])

    def create_slack_variables(self) -> DirectLimits:
        """
        Creates one normalized slack variable for the single terminal-state constraint.
        """
        [constraint] = self.constraints
        return DirectLimits(
            lower_bounds=Vector([-LargeNumber]),
            upper_bounds=Vector([LargeNumber]),
            quadratic_weights=Vector(
                [
                    normalize_slack_weight(
                        constraint.quadratic_weight,
                        constraint.normalization_factor,
                        self.qp_controller_config.control_horizon,
                    )
                ]
            ),
            linear_weights=Vector([sm.Scalar(0.0)]),
            names=[constraint.name],
        )


@dataclass
class TerminalSpillPredictionConstraint(GiskardInequalityConstraint):
    """
    Inequality constraint capping the MPC-predicted quantity accumulated over the control horizon.

    The inherited ``expression`` holds an instantaneous non-negative rate ``ṡ(q₀)`` (e.g. a spill
    volume rate): its jacobian w.r.t. the joint variables gives the per-step control sensitivity the
    strategy weights across the horizon, and its horizon integral is the predicted accumulation the
    strategy bounds from above.  The accumulator is taken to start at zero every solve, so — unlike
    :class:`TerminalStatePredictionConstraint` — no world state variable is needed.

    .. note:: The prediction assumes the rate's non-control dependencies (here, an externally moved
        target) are held constant over the horizon; anticipating a moving target requires feeding
        its velocity into the free response.
    """

    spill_tolerance: float = field(default=0.0, kw_only=True)
    """Upper bound on the accumulation predicted over the control horizon."""

    lower_bound: Scalar = field(
        default_factory=lambda: sm.Scalar(-LargeNumber), kw_only=True
    )
    """Unused inherited lower bound; the cap is one-sided, so the strategy leaves it open."""

    upper_bound: Scalar = field(default_factory=lambda: sm.Scalar(0.0), kw_only=True)
    """Unused inherited upper bound; the strategy computes ``spill_tolerance − predicted`` instead."""


@dataclass
class TerminalSpillPredictionStrategy(IntegralStrategy):
    """
    Enforcement strategy for the terminal-accumulation cap (:class:`TerminalSpillPredictionConstraint`).

    Reuses the linearized recursion of :class:`TerminalStatePredictionStrategy` with a zero initial
    value and zero self-sensitivity, so the predicted terminal accumulation reduces to the horizon
    integral of the current rate, ``H·dt·ṡ``, and the control matrix is the rate jacobian weighted
    per horizon block by the same geometric lookahead.  The bound is applied as a one-sided upper
    inequality (``predicted ≤ spill_tolerance``) rather than an equality, so the cap only pushes back
    when the predicted accumulation exceeds the tolerance and stays inactive otherwise.
    """

    @cached_property
    def _state_model(self) -> LinearizedScalarStateModel:
        """
        Linearizes the single constraint's rate as a zero-initialized pure integrator.

        The self-sensitivity is zero because the accumulation does not feed back into its own rate,
        so the decay is unity and the free response is the plain horizon integral of the rate.
        """
        [constraint] = self.constraints
        return LinearizedScalarStateModel(
            state_value=sm.Scalar(0.0),
            state_velocity=constraint.expression,
            state_sensitivity=sm.Scalar(0.0),
            time_step=self.qp_controller_config.model_predictive_control_time_step,
            control_horizon=self.qp_controller_config.control_horizon,
        )

    def create_matrix(self) -> Matrix:
        """
        Builds the constraint row by weighting the rate jacobian per horizon block, matching the
        terminal-state prediction's proactive lookahead so early velocity decisions carry more of
        the predicted accumulation.
        """
        [constraint] = self.constraints
        time_step = self.qp_controller_config.model_predictive_control_time_step
        jacobian = (
            sm.Vector([constraint.expression]).jacobian(self.position_variables)
            * time_step
        )
        weights = horizon_normalized_weights(
            self._state_model.lookahead_weights(),
            self.qp_controller_config.control_horizon,
        )
        blocks = [jacobian * weight for weight in weights]
        return sm.hstack(
            blocks + [sm.Matrix.zeros(jacobian.shape[0], self.number_of_jerk_columns)]
        )

    def create_upper_bounds(self) -> Vector:
        """
        Caps the control contribution so the predicted terminal accumulation stays at or below the
        tolerance: ``spill_tolerance − free_response``.
        """
        self._require_constraint_type(GiskardInequalityConstraint)
        [constraint] = self.constraints
        bound = (
            sm.Scalar(constraint.spill_tolerance) - self._state_model.free_response()
        )
        return sm.Vector(
            [
                self.capped_bound(
                    bound,
                    self.qp_controller_config.model_predictive_control_time_step,
                    constraint.normalization_factor,
                    self.qp_controller_config.control_horizon,
                )
            ]
        )

    def create_lower_bounds(self) -> Vector:
        """
        Leaves the accumulation unconstrained from below, so reducing it is always feasible.

        The bound is left fully open (not capped) because this is a one-sided upper cap: capping the
        lower side would collapse it into a tight equality whenever the predicted spill saturates the
        upper cap, over-constraining the QP.
        """
        self._require_constraint_type(GiskardInequalityConstraint)
        return sm.Vector([sm.Scalar(-LargeNumber)])
