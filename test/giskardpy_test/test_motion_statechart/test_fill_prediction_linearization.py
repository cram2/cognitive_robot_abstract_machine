from __future__ import annotations

import pytest
from typing_extensions import Callable

import krrood.symbolic_math.symbolic_math as sm
from giskardpy.qp.terminal_state_prediction_strategy import (
    LinearizedScalarStateModel,
    horizon_normalized_weights,
)
from semantic_digital_twin.physics.equations.pouring_equations import (
    ArticulatedPouringEquation,
    SymbolicFillContext,
)

_TIME_STEP = 0.05
_CONTROL_HORIZON = 5

ModelFactory = Callable[[float], LinearizedScalarStateModel]
"""
Builds a linearized model at the flowing operating point for a given state velocity.
"""


def _make_flowing_setup() -> tuple[ArticulatedPouringEquation, float, float]:
    """
    Builds a pouring equation and an operating point at which liquid is actively
    flowing.

    The tilt angle is chosen well above the geometric spill threshold so the outflow gap
    is inside the smooth region of ``max(0, ...)`` and both ODE partials are non-zero.
    """
    equation = ArticulatedPouringEquation(
        container_height=0.2, container_width=0.08, outflow_rate_constant=1.0
    )
    tilt_angle = 1.3
    fill_level = 0.8
    return equation, tilt_angle, fill_level


def _ode_value(
    equation: ArticulatedPouringEquation, tilt_angle: float, fill_level: float
) -> float:
    """
    Evaluates the nonlinear fill velocity at a concrete operating point.
    """
    return equation.symbolic_velocity(
        SymbolicFillContext(sm.Scalar(tilt_angle), sm.Scalar(fill_level))
    ).evaluate()[0]


def _ode_partials(
    equation: ArticulatedPouringEquation, tilt_angle: float, fill_level: float
) -> tuple[float, float]:
    """
    Evaluates ``(df/dtilt, df/dfill)`` at a concrete operating point.
    """
    df_dtilt, df_dfill = equation.symbolic_ode_jacobians(
        sm.Scalar(tilt_angle), sm.Scalar(fill_level)
    )
    return df_dtilt.evaluate()[0], df_dfill.evaluate()[0]


def _nonlinear_rollout(
    equation: ArticulatedPouringEquation,
    tilt_angle: float,
    fill_level: float,
    tilt_velocity: float,
) -> float:
    """
    Brute-force forward-Euler rollout of the true nonlinear pouring ODE.
    """
    fill = fill_level
    tilt = tilt_angle
    for _ in range(_CONTROL_HORIZON):
        fill += _TIME_STEP * _ode_value(equation, tilt, fill)
        tilt += tilt_velocity * _TIME_STEP
    return fill


def _expected_lookahead_weights(decay: float, control_horizon: int) -> list[float]:
    """
    Computes the geometric lookahead weight of every velocity block numerically.

    Block ``i`` carries weight ``sum_{k=0}^{M-2-i} decay^k``; the final block has weight
    zero because no state step follows it.
    """
    return [
        sum(decay**power for power in range(control_horizon - 1 - block))
        for block in range(control_horizon)
    ]


@pytest.fixture
def flowing_model_factory() -> ModelFactory:
    """
    Factory for :class:`LinearizedScalarStateModel` at the flowing operating point.

    The fill sensitivity is derived from the pouring equation; only the state velocity
    varies between tests.
    """
    equation, tilt_angle, fill_level = _make_flowing_setup()
    _, fill_sensitivity = _ode_partials(equation, tilt_angle, fill_level)

    def _build(state_velocity: float) -> LinearizedScalarStateModel:
        return LinearizedScalarStateModel(
            state_value=sm.Scalar(fill_level),
            state_velocity=sm.Scalar(state_velocity),
            state_sensitivity=sm.Scalar(fill_sensitivity),
            time_step=_TIME_STEP,
            control_horizon=_CONTROL_HORIZON,
        )

    return _build


class TestLinearizedScalarStateModel:
    """
    Validates the linearized fill-prediction math used by the QP constraint.
    """

    def test_decay_is_one_plus_time_step_times_sensitivity(
        self, flowing_model_factory: ModelFactory
    ) -> None:
        """
        The decay factor must equal ``1 + dt * df/dfill`` at the operating point.
        """
        equation, tilt_angle, fill_level = _make_flowing_setup()
        _, fill_sensitivity = _ode_partials(equation, tilt_angle, fill_level)
        model = flowing_model_factory(0.0)

        assert model.decay.evaluate()[0] == pytest.approx(
            1.0 + _TIME_STEP * fill_sensitivity
        )

    def test_free_response_matches_held_tilt_rollout(
        self, flowing_model_factory: ModelFactory
    ) -> None:
        """
        The free response must equal a nonlinear rollout in which the tilt is held
        constant, confirming it is the zero-control prediction rather than a frozen-fill
        assumption.
        """
        equation, tilt_angle, fill_level = _make_flowing_setup()
        outflow_rate = _ode_value(equation, tilt_angle, fill_level)
        model = flowing_model_factory(outflow_rate)

        predicted = model.free_response().evaluate()[0]
        held_tilt = _nonlinear_rollout(equation, tilt_angle, fill_level, 0.0)
        assert predicted == pytest.approx(held_tilt, abs=1e-3)

    def test_lookahead_weights_equal_geometric_series_of_decay(
        self, flowing_model_factory: ModelFactory
    ) -> None:
        """
        Every velocity block must carry exactly the geometric series of the decay over
        its remaining state steps; earlier blocks therefore carry strictly larger weight
        and the final block carries zero because no fill step follows it.
        """
        model = flowing_model_factory(0.0)
        decay = model.decay.evaluate()[0]

        weights = [weight.evaluate()[0] for weight in model.lookahead_weights()]
        assert weights == pytest.approx(
            _expected_lookahead_weights(decay, _CONTROL_HORIZON)
        )
        assert weights[-1] == pytest.approx(0.0)
        assert all(earlier > later for earlier, later in zip(weights, weights[1:]))

    def test_normalized_weights_preserve_horizon_scale(
        self, flowing_model_factory: ModelFactory
    ) -> None:
        """
        Normalizing the lookahead weights must keep their decreasing shape while summing
        to the control horizon, so the matrix stays at the calibrated reactive scale.
        """
        model = flowing_model_factory(0.0)

        weights = [
            weight.evaluate()[0]
            for weight in horizon_normalized_weights(
                model.lookahead_weights(), _CONTROL_HORIZON
            )
        ]
        assert sum(weights) == pytest.approx(_CONTROL_HORIZON)
        assert weights[-1] == pytest.approx(0.0)
        assert weights[0] > weights[-2]

    def test_single_step_horizon_predicts_one_euler_step(self) -> None:
        """
        With a single-step control horizon the free response is one Euler step of the
        ODE and the only velocity block carries zero weight, since no state step can
        follow it.
        """
        fill_level = 0.8
        outflow_rate = -0.2
        fill_sensitivity = -0.5
        model = LinearizedScalarStateModel(
            state_value=sm.Scalar(fill_level),
            state_velocity=sm.Scalar(outflow_rate),
            state_sensitivity=sm.Scalar(fill_sensitivity),
            time_step=_TIME_STEP,
            control_horizon=1,
        )

        predicted = model.free_response().evaluate()[0]
        assert predicted == pytest.approx(fill_level + _TIME_STEP * outflow_rate)
        weights = [weight.evaluate()[0] for weight in model.lookahead_weights()]
        assert weights == pytest.approx([0.0])

    def test_negative_sensitivity_free_response_matches_linear_recursion(self) -> None:
        """
        With a negative state sensitivity the free response must follow the contracting
        linearized recursion ``x_{k+1} = x_k + dt * (f0 + a * (x_k - x0))``.
        """
        fill_level = 0.8
        outflow_rate = -0.2
        fill_sensitivity = -2.0
        model = LinearizedScalarStateModel(
            state_value=sm.Scalar(fill_level),
            state_velocity=sm.Scalar(outflow_rate),
            state_sensitivity=sm.Scalar(fill_sensitivity),
            time_step=_TIME_STEP,
            control_horizon=_CONTROL_HORIZON,
        )
        assert model.decay.evaluate()[0] == pytest.approx(
            1.0 + _TIME_STEP * fill_sensitivity
        )

        expected_fill = fill_level
        for _ in range(_CONTROL_HORIZON):
            expected_fill += _TIME_STEP * (
                outflow_rate + fill_sensitivity * (expected_fill - fill_level)
            )
        assert model.free_response().evaluate()[0] == pytest.approx(expected_fill)


class TestIncreasingFillLinearization:
    """
    Validates the linearization for a container that is filling (inflow goal) rather
    than draining.

    For a pure inflow the fill velocity does not depend on the receiver's own fill
    level, so the fill sensitivity is zero and the linearized model reduces to a well-
    conditioned integrator.
    """

    def test_free_response_is_a_pure_integrator_when_filling(self) -> None:
        """
        With zero fill sensitivity the free response is the fill plus the projected
        inflow.
        """
        fill_level = 0.0
        inflow_rate = 0.1
        model = LinearizedScalarStateModel(
            state_value=sm.Scalar(fill_level),
            state_velocity=sm.Scalar(inflow_rate),
            state_sensitivity=sm.Scalar(0.0),
            time_step=_TIME_STEP,
            control_horizon=_CONTROL_HORIZON,
        )
        predicted = model.free_response().evaluate()[0]
        assert predicted == pytest.approx(
            fill_level + _CONTROL_HORIZON * _TIME_STEP * inflow_rate
        )
        assert predicted > fill_level
