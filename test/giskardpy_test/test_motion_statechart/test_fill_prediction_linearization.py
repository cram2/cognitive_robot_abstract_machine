from __future__ import annotations

import pytest

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


def _make_flowing_setup() -> tuple[ArticulatedPouringEquation, float, float]:
    """
    Builds a pouring equation and an operating point at which liquid is actively flowing.

    The tilt angle is chosen well above the geometric spill threshold so the outflow gap is
    inside the smooth region of ``max(0, ...)`` and both ODE partials are non-zero.
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
    """Evaluates the nonlinear fill velocity at a concrete operating point."""
    return equation.symbolic_velocity(
        SymbolicFillContext(sm.Scalar(tilt_angle), sm.Scalar(fill_level))
    ).evaluate()[0]


def _ode_partials(
    equation: ArticulatedPouringEquation, tilt_angle: float, fill_level: float
) -> tuple[float, float]:
    """Evaluates ``(df/dtilt, df/dfill)`` at a concrete operating point."""
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
    """Brute-force forward-Euler rollout of the true nonlinear pouring ODE."""
    fill = fill_level
    tilt = tilt_angle
    for _ in range(_CONTROL_HORIZON):
        fill += _TIME_STEP * _ode_value(equation, tilt, fill)
        tilt += tilt_velocity * _TIME_STEP
    return fill


class TestLinearizedScalarStateModel:
    """Validates the linearized fill-prediction math used by the QP constraint."""

    def test_free_response_matches_held_tilt_rollout(self):
        """
        The free response must equal a nonlinear rollout in which the tilt is held constant,
        confirming it is the zero-control prediction rather than a frozen-fill assumption.
        """
        equation, tilt_angle, fill_level = _make_flowing_setup()
        outflow_rate = _ode_value(equation, tilt_angle, fill_level)
        _, fill_sensitivity = _ode_partials(equation, tilt_angle, fill_level)

        model = LinearizedScalarStateModel(
            state_value=sm.Scalar(fill_level),
            state_velocity=sm.Scalar(outflow_rate),
            state_sensitivity=sm.Scalar(fill_sensitivity),
            time_step=_TIME_STEP,
            control_horizon=_CONTROL_HORIZON,
        )

        predicted = model.free_response().evaluate()[0]
        held_tilt = _nonlinear_rollout(equation, tilt_angle, fill_level, 0.0)
        assert predicted == pytest.approx(held_tilt, abs=1e-3)

    def test_lookahead_weights_decrease_and_vanish_at_horizon_end(self):
        """
        Earlier velocity decisions must carry strictly larger lookahead weight, and the final
        block must carry zero weight because no fill step follows it.
        """
        equation, tilt_angle, fill_level = _make_flowing_setup()
        _, fill_sensitivity = _ode_partials(equation, tilt_angle, fill_level)

        model = LinearizedScalarStateModel(
            state_value=sm.Scalar(fill_level),
            state_velocity=sm.Scalar(0.0),
            state_sensitivity=sm.Scalar(fill_sensitivity),
            time_step=_TIME_STEP,
            control_horizon=_CONTROL_HORIZON,
        )

        weights = [weight.evaluate()[0] for weight in model.lookahead_weights()]
        assert weights[-1] == pytest.approx(0.0)
        assert all(earlier > later for earlier, later in zip(weights, weights[1:-1]))

    def test_normalized_weights_preserve_horizon_scale(self):
        """
        Normalizing the lookahead weights must keep their decreasing shape while summing to the
        control horizon, so the matrix stays at the calibrated reactive scale.
        """
        equation, tilt_angle, fill_level = _make_flowing_setup()
        _, fill_sensitivity = _ode_partials(equation, tilt_angle, fill_level)

        model = LinearizedScalarStateModel(
            state_value=sm.Scalar(fill_level),
            state_velocity=sm.Scalar(0.0),
            state_sensitivity=sm.Scalar(fill_sensitivity),
            time_step=_TIME_STEP,
            control_horizon=_CONTROL_HORIZON,
        )

        weights = [
            weight.evaluate()[0]
            for weight in horizon_normalized_weights(
                model.lookahead_weights(), _CONTROL_HORIZON
            )
        ]
        assert sum(weights) == pytest.approx(_CONTROL_HORIZON)
        assert weights[-1] == pytest.approx(0.0)
        assert weights[0] > weights[-2]


class TestIncreasingFillLinearization:
    """
    Validates the linearization for a container that is filling (inflow goal) rather than draining.

    For a pure inflow the fill velocity does not depend on the receiver's own fill level, so the
    fill sensitivity is zero and the linearized model reduces to a well-conditioned integrator.
    """

    def test_free_response_is_a_pure_integrator_when_filling(self):
        """With zero fill sensitivity the free response is the fill plus the projected inflow."""
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
