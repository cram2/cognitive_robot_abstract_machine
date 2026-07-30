from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import pytest

import krrood.symbolic_math.symbolic_math as sm
from giskardpy.qp.constraint import GiskardEqualityConstraint, LargeNumber
from giskardpy.qp.enforcement_strategy import normalize_slack_weight
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
from semantic_digital_twin.datastructures.prefixed_name import PrefixedName
from semantic_digital_twin.spatial_types import Vector3
from semantic_digital_twin.world import World
from semantic_digital_twin.world_description.connections import PrismaticConnection
from semantic_digital_twin.world_description.degree_of_freedom import DegreeOfFreedom
from semantic_digital_twin.world_description.world_entity import Body

# %% operating point constants

_JOINT_POSITION = 0.2
"""
Position of the controlled joint at the operating point.
"""

_STATE_POSITION = 0.5
"""
Position of the passive state DOF at the operating point.
"""

_JOINT_SENSITIVITY = 0.1
"""
Constant ``df/dq`` of the linear state-rate expression.
"""

_STATE_SENSITIVITY = -0.5
"""
Constant ``df/dx`` of the linear state-rate expression.
"""

_GOAL_VALUE = 0.6
"""
Terminal state goal within the reachable cap.
"""

# %% fixtures


@dataclass
class LinearStateRateSetup:
    """
    A world with one controlled joint and one passive state DOF, plus a strategy whose
    single constraint carries the linear state rate ``f = df/dq * q + df/dx * x``.
    """

    world: World
    """
    The world owning both degrees of freedom.
    """

    joint_degree_of_freedom: DegreeOfFreedom
    """
    The controlled joint DOF the strategy optimizes.
    """

    state_degree_of_freedom: DegreeOfFreedom
    """
    The passive state DOF whose terminal value is constrained.
    """

    strategy: TerminalStatePredictionStrategy
    """
    The strategy under test, built over the single terminal constraint.
    """

    config: QPControllerConfig
    """
    The controller configuration the strategy was built with.
    """


def _world_with_joint_and_state() -> tuple[World, DegreeOfFreedom, DegreeOfFreedom]:
    """
    Builds a world with a controlled joint DOF and a passive state DOF.
    """
    world = World()
    with world.modify_world():
        map_body = Body(name=PrefixedName("map"))
        joint_body = Body(name=PrefixedName("joint_body"))
        state_body = Body(name=PrefixedName("state_body"))
        world.add_body(map_body)
        world.add_body(joint_body)
        world.add_body(state_body)
        joint_connection = PrismaticConnection.create_with_dofs(
            world=world, parent=map_body, child=joint_body, axis=Vector3.X()
        )
        world.add_connection(joint_connection)
        state_connection = PrismaticConnection.create_with_dofs(
            world=world, parent=map_body, child=state_body, axis=Vector3.X()
        )
        world.add_connection(state_connection)
    world.set_positions_1DOF_connection(
        {joint_connection: _JOINT_POSITION, state_connection: _STATE_POSITION}
    )
    return world, joint_connection.dof, state_connection.dof


def _terminal_constraint(
    name: str,
    joint_degree_of_freedom: DegreeOfFreedom,
    state_degree_of_freedom: DegreeOfFreedom,
    goal_value: float = _GOAL_VALUE,
) -> TerminalStatePredictionConstraint:
    """
    Builds a terminal constraint over the linear state rate of the two DOFs.
    """
    state_velocity = joint_degree_of_freedom.variables.position * sm.Scalar(
        _JOINT_SENSITIVITY
    ) + state_degree_of_freedom.variables.position * sm.Scalar(_STATE_SENSITIVITY)
    return TerminalStatePredictionConstraint(
        name=name,
        expression=state_velocity,
        quadratic_weight=1.0,
        normalization_factor=1.0,
        enforcement_strategy=TerminalStatePredictionStrategy,
        state_variable=state_degree_of_freedom.variables.position,
        goal_value=goal_value,
    )


def _strategy(
    constraints: list[GiskardEqualityConstraint],
    joint_degree_of_freedom: DegreeOfFreedom,
) -> TerminalStatePredictionStrategy:
    """
    Builds the strategy under test over the given constraints and controlled DOF.
    """
    return TerminalStatePredictionStrategy(
        degrees_of_freedom=[joint_degree_of_freedom],
        constraints=constraints,
        qp_controller_config=QPControllerConfig.create_with_simulation_defaults(),
    )


@pytest.fixture
def linear_state_rate_setup() -> LinearStateRateSetup:
    """
    Single-constraint strategy over a real joint and state DOF.
    """
    world, joint_degree_of_freedom, state_degree_of_freedom = (
        _world_with_joint_and_state()
    )
    constraint = _terminal_constraint(
        "fill_goal", joint_degree_of_freedom, state_degree_of_freedom
    )
    strategy = _strategy([constraint], joint_degree_of_freedom)
    return LinearStateRateSetup(
        world=world,
        joint_degree_of_freedom=joint_degree_of_freedom,
        state_degree_of_freedom=state_degree_of_freedom,
        strategy=strategy,
        config=strategy.qp_controller_config,
    )


# %% happy path


class TestTerminalStatePredictionRow:
    """
    Validates the single QP row the strategy builds from a linear state rate.
    """

    def test_state_model_differentiates_rate_by_state_variable(
        self, linear_state_rate_setup: LinearStateRateSetup
    ) -> None:
        """
        The linearization must extract ``df/dx`` from the rate expression's jacobian.
        """
        model = linear_state_rate_setup.strategy._state_model

        assert model.state_sensitivity.evaluate()[0] == pytest.approx(
            _STATE_SENSITIVITY
        )
        assert model.state_value.evaluate()[0] == pytest.approx(_STATE_POSITION)
        assert model.state_velocity.evaluate()[0] == pytest.approx(
            _JOINT_SENSITIVITY * _JOINT_POSITION + _STATE_SENSITIVITY * _STATE_POSITION
        )

    def test_matrix_scales_jacobian_per_block_with_normalized_weights(
        self, linear_state_rate_setup: LinearStateRateSetup
    ) -> None:
        """
        The row must hold ``dt * df/dq`` scaled by the normalized lookahead weight of
        each velocity block, followed by zero-padded jerk columns.
        """
        strategy = linear_state_rate_setup.strategy
        config = linear_state_rate_setup.config

        matrix = np.array(strategy.create_matrix().evaluate()).flatten()

        expected_weights = [
            weight.evaluate()[0]
            for weight in horizon_normalized_weights(
                strategy._state_model.lookahead_weights(), config.control_horizon
            )
        ]
        time_step = config.model_predictive_control_time_step
        expected_velocity_blocks = [
            time_step * _JOINT_SENSITIVITY * weight for weight in expected_weights
        ]
        assert matrix.shape == (config.control_horizon + config.prediction_horizon,)
        assert matrix[: config.control_horizon] == pytest.approx(
            expected_velocity_blocks
        )
        assert matrix[config.control_horizon :] == pytest.approx(
            [0.0] * config.prediction_horizon
        )

    def test_equality_bound_is_goal_minus_free_response(
        self, linear_state_rate_setup: LinearStateRateSetup
    ) -> None:
        """
        The bound must equal the terminal prediction error under zero joint velocity.
        """
        strategy = linear_state_rate_setup.strategy

        bound = strategy.create_equality_bounds().evaluate()[0]

        free_response = strategy._state_model.free_response().evaluate()[0]
        assert bound == pytest.approx(_GOAL_VALUE - free_response)

    def test_equality_bound_is_capped_to_reachable_change(
        self, linear_state_rate_setup: LinearStateRateSetup
    ) -> None:
        """
        A goal far beyond the horizon's reach must be capped, not passed through.
        """
        config = linear_state_rate_setup.config
        constraint = _terminal_constraint(
            "far_goal",
            linear_state_rate_setup.joint_degree_of_freedom,
            linear_state_rate_setup.state_degree_of_freedom,
            goal_value=5.0,
        )
        strategy = _strategy(
            [constraint], linear_state_rate_setup.joint_degree_of_freedom
        )

        bound = strategy.create_equality_bounds().evaluate()[0]

        reachable_change = (
            constraint.normalization_factor
            * config.model_predictive_control_time_step
            * config.control_horizon
        )
        assert bound == pytest.approx(reachable_change)

    def test_slack_variable_is_single_and_normalized(
        self, linear_state_rate_setup: LinearStateRateSetup
    ) -> None:
        """
        The strategy must add exactly one unbounded slack with the normalized weight.
        """
        strategy = linear_state_rate_setup.strategy
        config = linear_state_rate_setup.config
        constraint = strategy.constraints[0]

        slack = strategy.create_slack_variables()

        assert slack.names == ["fill_goal"]
        assert slack.lower_bounds.evaluate().flatten().tolist() == [-LargeNumber]
        assert slack.upper_bounds.evaluate().flatten().tolist() == [LargeNumber]
        expected_weight = normalize_slack_weight(
            sm.Scalar(constraint.quadratic_weight),
            constraint.normalization_factor,
            config.control_horizon,
        ).evaluate()[0]
        assert slack.quadratic_weights.evaluate()[0] == pytest.approx(expected_weight)
        assert slack.linear_weights.evaluate()[0] == pytest.approx(0.0)


# %% validation


class TestTerminalStateConstraintValidation:
    """
    Validates rejection of constraint sets the single-row strategy cannot represent.
    """

    def test_two_terminal_constraints_raise_dedicated_error(self) -> None:
        """
        Grouping two terminal-state constraints into one block must fail loudly, naming
        both.
        """
        _world, joint_degree_of_freedom, state_degree_of_freedom = (
            _world_with_joint_and_state()
        )
        strategy = _strategy(
            [
                _terminal_constraint(
                    "fill_goal", joint_degree_of_freedom, state_degree_of_freedom
                ),
                _terminal_constraint(
                    "second_goal", joint_degree_of_freedom, state_degree_of_freedom
                ),
            ],
            joint_degree_of_freedom,
        )

        with pytest.raises(MultipleTerminalStateConstraintsError) as error_info:
            strategy.create_equality_bounds()

        assert error_info.value.constraint_names == ["fill_goal", "second_goal"]

    def test_plain_equality_constraint_raises_type_mismatch(self) -> None:
        """
        A non-terminal-state equality constraint must be rejected, reporting the types.
        """
        _world, joint_degree_of_freedom, _state_degree_of_freedom = (
            _world_with_joint_and_state()
        )
        plain_constraint = GiskardEqualityConstraint(
            name="plain",
            expression=sm.Scalar(0.0),
            quadratic_weight=1.0,
            normalization_factor=1.0,
            enforcement_strategy=TerminalStatePredictionStrategy,
            bound=sm.Scalar(0.0),
        )
        strategy = _strategy([plain_constraint], joint_degree_of_freedom)

        with pytest.raises(ConstraintTypeMismatchError) as error_info:
            strategy.create_equality_bounds()

        assert error_info.value.expected_type is TerminalStatePredictionConstraint
        assert error_info.value.actual_type is GiskardEqualityConstraint
        assert error_info.value.constraint_name == "plain"


# %% weight normalization guard


class TestHorizonNormalizedWeightGuard:
    def test_zero_weight_sum_does_not_produce_nan(self) -> None:
        """
        A weight set that cancels to zero must fall back to the raw weights instead of
        dividing by zero.
        """
        weights = [sm.Scalar(1.0), sm.Scalar(-1.0)]
        normalized = horizon_normalized_weights(weights, control_horizon=2)
        values = [weight.evaluate()[0] for weight in normalized]
        assert all(math.isfinite(value) for value in values)
        assert values == [1.0, -1.0]

    def test_near_cancelling_weight_sum_keeps_raw_weights(self) -> None:
        """
        A sum below the magnitude epsilon must also fall back instead of exploding.
        """
        weights = [sm.Scalar(1.0), sm.Scalar(-1.0 + 1e-12)]
        normalized = horizon_normalized_weights(weights, control_horizon=2)
        values = [weight.evaluate()[0] for weight in normalized]
        assert all(math.isfinite(value) for value in values)
        assert values == pytest.approx([1.0, -1.0 + 1e-12])
