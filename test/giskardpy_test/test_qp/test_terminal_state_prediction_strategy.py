from __future__ import annotations

import math

import pytest

import krrood.symbolic_math.symbolic_math as sm
from giskardpy.qp.constraint import GiskardEqualityConstraint
from giskardpy.qp.exceptions import (
    ConstraintTypeMismatchError,
    MultipleTerminalStateConstraintsError,
)
from giskardpy.qp.qp_controller_config import QPControllerConfig
from giskardpy.qp.terminal_state_prediction_strategy import (
    TerminalStatePredictionConstraint,
    TerminalStatePredictionStrategy,
    horizon_normalized_weights,
)


def _terminal_constraint(name: str) -> TerminalStatePredictionConstraint:
    return TerminalStatePredictionConstraint(
        name=name,
        expression=sm.Scalar(0.0),
        quadratic_weight=1.0,
        normalization_factor=1.0,
        enforcement_strategy=TerminalStatePredictionStrategy,
        state_variable=sm.Scalar(0.0),
        goal_value=0.5,
    )


def _strategy(constraints: list) -> TerminalStatePredictionStrategy:
    return TerminalStatePredictionStrategy(
        degrees_of_freedom=[],
        constraints=constraints,
        qp_controller_config=QPControllerConfig.create_with_simulation_defaults(),
    )


class TestTerminalStateConstraintValidation:
    def test_two_terminal_constraints_raise_dedicated_error(self):
        """Grouping two terminal-state constraints into one block must fail loudly, not with a bare ValueError."""
        strategy = _strategy(
            [_terminal_constraint("fill_goal"), _terminal_constraint("second_goal")]
        )
        with pytest.raises(MultipleTerminalStateConstraintsError):
            strategy.create_equality_bounds()

    def test_plain_equality_constraint_raises_type_mismatch(self):
        """A non-terminal-state equality constraint must be rejected with the dedicated mismatch error."""
        plain_constraint = GiskardEqualityConstraint(
            name="plain",
            expression=sm.Scalar(0.0),
            quadratic_weight=1.0,
            normalization_factor=1.0,
            enforcement_strategy=TerminalStatePredictionStrategy,
            bound=sm.Scalar(0.0),
        )
        strategy = _strategy([plain_constraint])
        with pytest.raises(ConstraintTypeMismatchError):
            strategy.create_equality_bounds()


class TestHorizonNormalizedWeightGuard:
    def test_zero_weight_sum_does_not_produce_nan(self):
        """A weight set that cancels to zero must fall back to the raw weights instead of dividing by zero."""
        weights = [sm.Scalar(1.0), sm.Scalar(-1.0)]
        normalized = horizon_normalized_weights(weights, control_horizon=2)
        values = [weight.evaluate()[0] for weight in normalized]
        assert all(math.isfinite(value) for value in values)
        assert values == [1.0, -1.0]
