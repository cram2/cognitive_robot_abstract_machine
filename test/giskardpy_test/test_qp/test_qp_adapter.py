from collections import defaultdict

import numpy as np

import pytest

from giskardpy.qp.dof_limits import DirectLimits, QuadraticProgramDegreeOfFreedomLimits
from giskardpy.qp.enforcement_strategy import (
    SystemDynamicsStrategy,
    IntegralStrategy,
    VelocityStrategy,
    ExpressionEnforcementStrategy,
)
from giskardpy.qp.exceptions import (
    MismatchedLimitLengthsError,
    ConstraintTypeMismatchError,
    NoFactoryForQPDataTypeError,
)
from giskardpy.qp.constraint_collection import ConstraintCollection
from giskardpy.qp.qp_controller_config import QPControllerConfig
from giskardpy.qp.qp_data import (
    QPData,
    QPDataExplicit,
    QPDataTwoSidedInequality,
)
from giskardpy.qp.qp_data_factories import (
    QPDataFactory,
    QPDataExplicitFactory,
    QPDataTwoSidedInequalityFactory,
)
from giskardpy.qp.qp_data_symbolic import QPDataSymbolic
from giskardpy.qp.qp_debugger import QuadraticProgramDebugger
from giskardpy.qp.solvers.qp_solver_piqp import QPSolverPIQP
from krrood.symbolic_math.symbolic_math import (
    FloatVariable,
    Vector,
)
from semantic_digital_twin.spatial_types.derivatives import Derivatives
from semantic_digital_twin.world import World


def _world_state_matrix(world: World) -> np.ndarray:
    """Public-API equivalent of the raw world-state buffer expected by ``factory.evaluate``."""
    state = world.state
    return np.vstack(
        [state.positions, state.velocities, state.accelerations, state.jerks]
    )


def test_direct_limits_rejects_mismatched_lengths():
    with pytest.raises(MismatchedLimitLengthsError):
        DirectLimits(
            lower_bounds=Vector([-1.0, -1.0]),
            upper_bounds=Vector([1.0, 1.0]),
            quadratic_weights=Vector([0.1, 0.1]),
            linear_weights=Vector([0.0, 0.0]),
            names=["only_one_name"],
        )


def test_direct_limits_empty():
    empty = DirectLimits.empty()
    assert empty.lower_bounds.shape[0] == 0
    assert empty.upper_bounds.shape[0] == 0
    assert empty.quadratic_weights.shape[0] == 0
    assert empty.linear_weights.shape[0] == 0
    assert empty.names == []


def test_DofLimits(prismatic_bot):
    target_frequency = 20
    prediction_horizon = 10
    expected_jerk_limit = 1 / target_frequency
    limits = QuadraticProgramDegreeOfFreedomLimits.create(
        prismatic_bot.active_degrees_of_freedom,
        qp_controller_config=QPControllerConfig(
            target_frequency=target_frequency, prediction_horizon=prediction_horizon
        ),
    )
    assert np.allclose(
        limits.lower_bounds.evaluate(),
        np.array([-1.0] * 8 + [-expected_jerk_limit] * 10),
        rtol=1.0e-4,
    )
    assert np.allclose(
        limits.upper_bounds.evaluate(),
        np.array([1.0] * 8 + [expected_jerk_limit] * 10),
        rtol=1.0e-4,
    )
    assert np.allclose(
        limits.quadratic_weights.evaluate(),
        np.array(
            [
                0.001,
                0.002285714285714286,
                0.0035714285714285718,
                0.004857142857142858,
                0.0061428571428571435,
                0.007428571428571429,
                0.008714285714285716,
                0.01,
            ]
            + [0.0] * 10
        ),
    )


def test_DofLimits_two_joints(prismatic_bot2):
    target_frequency = 20
    prediction_horizon = 10
    expected_jerk_limit1 = 1 / target_frequency
    expected_jerk_limit2 = 1 / (target_frequency * 2)
    limits = QuadraticProgramDegreeOfFreedomLimits.create(
        prismatic_bot2.active_degrees_of_freedom,
        qp_controller_config=QPControllerConfig(
            target_frequency=target_frequency, prediction_horizon=prediction_horizon
        ),
    )
    expected_limits = np.array(
        [1.0, 0.5] * 8 + [expected_jerk_limit1, expected_jerk_limit2] * 10
    )
    assert np.allclose(
        limits.lower_bounds.evaluate(),
        -expected_limits,
        rtol=1.0e-4,
    )
    assert np.allclose(
        limits.upper_bounds.evaluate(),
        expected_limits,
        rtol=1.0e-4,
    )
    normal_weights = np.array(
        [
            0.001,
            0.002285714285714286,
            0.0035714285714285718,
            0.004857142857142858,
            0.0061428571428571435,
            0.007428571428571429,
            0.008714285714285716,
            0.01,
        ]
    )
    velocity_weights = np.array(
        list(zip(normal_weights, normal_weights / (0.5**2)))
    ).flatten()
    expected_weights = np.concatenate((velocity_weights, [0.0] * 20))
    assert np.allclose(
        limits.quadratic_weights.evaluate(),
        expected_weights,
    )


def test_mpc_model(prismatic_bot2):
    target_frequency = 20
    prediction_horizon = 10
    number_of_variables = len(prismatic_bot2.active_degrees_of_freedom)
    mpc_model = SystemDynamicsStrategy(
        degrees_of_freedom=prismatic_bot2.active_degrees_of_freedom,
        constraints=[],
        qp_controller_config=QPControllerConfig(
            target_frequency=target_frequency, prediction_horizon=prediction_horizon
        ),
    )
    bounds = mpc_model.create_equality_bounds()
    names = mpc_model.create_names()
    assert len(bounds[0].free_variables()) == 2
    assert len(bounds[1].free_variables()) == 2
    assert len(bounds[2].free_variables()) == 1
    assert len(bounds[3].free_variables()) == 1
    assert len(bounds[4:].free_variables()) == 0
    assert len(names) == len(bounds)

    variables = []
    variable_dicts = defaultdict(lambda: defaultdict(dict))
    for derivative in [Derivatives.velocity, Derivatives.jerk]:
        for k in range(prediction_horizon):
            if derivative == Derivatives.velocity and k >= prediction_horizon - 2:
                continue
            for name in range(number_of_variables):
                variables.append(FloatVariable(f"{derivative.name}_{name}_k{k}"))
                variable_dicts[derivative][name][k] = variables[-1]

    x = Vector(variables)
    constraints = mpc_model.create_matrix() @ x

    # constraints on first two idx
    for variable_id in range(number_of_variables):
        constraint_index = variable_id
        v_k0 = variable_dicts[Derivatives.velocity][variable_id][0]
        v_k1 = variable_dicts[Derivatives.velocity][variable_id][1]
        j_k0 = variable_dicts[Derivatives.jerk][variable_id][0]
        j_k1 = variable_dicts[Derivatives.jerk][variable_id][1]
        assert constraints[constraint_index] == j_k0 - v_k0
        assert (
            constraints[constraint_index + number_of_variables]
            == -v_k1 + 2 * v_k0 + j_k1
        )

    # constraints on middle idx
    constraint_index = 4
    for k in range(2, prediction_horizon - 2):
        for variable_id in range(number_of_variables):
            v_k = variable_dicts[Derivatives.velocity][variable_id][k]
            v_k_minus_1 = variable_dicts[Derivatives.velocity][variable_id][k - 1]
            v_k_minus_2 = variable_dicts[Derivatives.velocity][variable_id][k - 2]
            j_k = variable_dicts[Derivatives.jerk][variable_id][k]
            assert (
                constraints[constraint_index]
                == -v_k_minus_2 + 2 * v_k_minus_1 - v_k + j_k
            )
            constraint_index += 1

    # constraints on last two idx
    for variable_id in range(number_of_variables):
        v_k_minus_1 = variable_dicts[Derivatives.velocity][variable_id][
            prediction_horizon - 3
        ]
        v_k_minus_2 = variable_dicts[Derivatives.velocity][variable_id][
            prediction_horizon - 4
        ]
        j_k = variable_dicts[Derivatives.jerk][variable_id][prediction_horizon - 2]
        j_k1 = variable_dicts[Derivatives.jerk][variable_id][prediction_horizon - 1]
        assert constraints[constraint_index] == 2 * v_k_minus_1 - v_k_minus_2 + j_k
        assert (
            constraints[constraint_index + number_of_variables] == -v_k_minus_1 + j_k1
        )
        constraint_index += 1

    assert mpc_model.create_slack_matrix().shape == (
        number_of_variables * prediction_horizon,
        0,
    )


def test_integral_strategy_with_equality_constraints(prismatic_bot2):
    target_frequency = 20
    prediction_horizon = 10
    constraints = ConstraintCollection()
    dof1 = prismatic_bot2.active_degrees_of_freedom[0]
    constraints.add_equality_constraint(
        task_expression=dof1.variables.position,
        equality_bound=1,
        quadratic_weight=1,
        reference_velocity=1,
    )
    eq_constraint_model = IntegralStrategy(
        degrees_of_freedom=prismatic_bot2.active_degrees_of_freedom,
        constraints=constraints.equality_constraints,
        qp_controller_config=QPControllerConfig(
            target_frequency=target_frequency, prediction_horizon=prediction_horizon
        ),
    )
    assert np.allclose(
        sum(eq_constraint_model.create_matrix()[0, :].to_np()),
        (1 / target_frequency) * (prediction_horizon - 2),
    )
    bounds = eq_constraint_model.create_equality_bounds()
    slack_variables = eq_constraint_model.create_slack_variables()
    assert bounds[0] == 1 * (1 / target_frequency) * (prediction_horizon - 2)
    assert slack_variables.quadratic_weights[0] != 0
    assert slack_variables.linear_weights[0] == 0
    assert slack_variables.lower_bounds < 0
    assert slack_variables.upper_bounds > 0
    assert eq_constraint_model.create_slack_matrix()[0, 0] == (1 / target_frequency)


def test_integral_strategy_with_inequality_constraints(prismatic_bot2):
    target_frequency = 20
    prediction_horizon = 10
    constraints = ConstraintCollection()
    dof1 = prismatic_bot2.active_degrees_of_freedom[0]
    constraints.add_inequality_constraint(
        task_expression=dof1.variables.position,
        lower_error=0,
        upper_error=1,
        quadratic_weight=1,
        reference_velocity=1,
    )
    ineq_constraint_model = IntegralStrategy(
        degrees_of_freedom=prismatic_bot2.active_degrees_of_freedom,
        constraints=constraints.inequality_constraints,
        qp_controller_config=QPControllerConfig(
            target_frequency=target_frequency, prediction_horizon=prediction_horizon
        ),
    )
    assert np.allclose(
        sum(ineq_constraint_model.create_matrix()[0, :].to_np()),
        (1 / target_frequency) * (prediction_horizon - 2),
    )

    lower_bounds = ineq_constraint_model.create_lower_bounds()
    upper_bounds = ineq_constraint_model.create_upper_bounds()
    slack_variables = ineq_constraint_model.create_slack_variables()

    assert lower_bounds[0] == 0

    assert upper_bounds[0] == 1 * (1 / target_frequency) * (prediction_horizon - 2)
    assert slack_variables.quadratic_weights[0] != 0
    assert slack_variables.linear_weights[0] == 0
    assert slack_variables.lower_bounds < 0
    assert slack_variables.upper_bounds > 0
    assert ineq_constraint_model.create_slack_matrix()[0, 0] == (1 / target_frequency)


def test_velocity_strategy_builds_inequality_blocks(prismatic_bot2):
    target_frequency = 20
    prediction_horizon = 10
    time_step = 1 / target_frequency
    control_horizon = prediction_horizon - 2
    constraints = ConstraintCollection()
    dof1 = prismatic_bot2.active_degrees_of_freedom[0]
    constraints.add_velocity_constraint(
        task_expression=dof1.variables.position,
        lower_velocity_limit=-0.5,
        upper_velocity_limit=0.5,
        quadratic_weight=1,
        velocity_limit=0.5,
        name="velocity_constraint",
    )
    velocity_strategy = VelocityStrategy(
        degrees_of_freedom=prismatic_bot2.active_degrees_of_freedom,
        constraints=constraints.inequality_constraints,
        qp_controller_config=QPControllerConfig(
            target_frequency=target_frequency, prediction_horizon=prediction_horizon
        ),
    )
    assert velocity_strategy.create_matrix().shape[0] == control_horizon
    slack_variables = velocity_strategy.create_slack_variables()
    assert len(slack_variables.names) == control_horizon
    lower_bounds = velocity_strategy.create_lower_bounds()
    upper_bounds = velocity_strategy.create_upper_bounds()
    assert len(lower_bounds) == control_horizon
    assert np.allclose(lower_bounds.to_np(), -0.5 * time_step)
    assert np.allclose(upper_bounds.to_np(), 0.5 * time_step)


def test_inequality_bounds_on_equality_strategy_raise(prismatic_bot2):
    constraints = ConstraintCollection()
    dof1 = prismatic_bot2.active_degrees_of_freedom[0]
    constraints.add_equality_constraint(
        task_expression=dof1.variables.position,
        equality_bound=1,
        quadratic_weight=1,
        reference_velocity=1,
    )
    strategy = IntegralStrategy(
        degrees_of_freedom=prismatic_bot2.active_degrees_of_freedom,
        constraints=constraints.equality_constraints,
        qp_controller_config=QPControllerConfig(
            target_frequency=20, prediction_horizon=10
        ),
    )
    with pytest.raises(ConstraintTypeMismatchError):
        strategy.create_lower_bounds()
    with pytest.raises(ConstraintTypeMismatchError):
        strategy.create_upper_bounds()


def test_system_dynamics_strategy_is_not_an_expression_strategy(prismatic_bot2):
    strategy = SystemDynamicsStrategy(
        degrees_of_freedom=prismatic_bot2.active_degrees_of_freedom,
        constraints=[],
        qp_controller_config=QPControllerConfig(
            target_frequency=20, prediction_horizon=10
        ),
    )
    assert not isinstance(strategy, ExpressionEnforcementStrategy)
    assert isinstance(
        IntegralStrategy(
            degrees_of_freedom=prismatic_bot2.active_degrees_of_freedom,
            constraints=[],
            qp_controller_config=QPControllerConfig(
                target_frequency=20, prediction_horizon=10
            ),
        ),
        ExpressionEnforcementStrategy,
    )


def test_qp_data_symbolic(prismatic_bot2):
    constraints = ConstraintCollection()
    dof1 = prismatic_bot2.active_degrees_of_freedom[0]
    dof2 = prismatic_bot2.active_degrees_of_freedom[1]
    constraints.add_equality_constraint(
        task_expression=dof1.variables.position,
        equality_bound=1,
        quadratic_weight=1,
        reference_velocity=1,
        name="position_constraint",
    )
    constraints.add_equality_constraint(
        task_expression=dof2.variables.position,
        equality_bound=1,
        quadratic_weight=0,
        reference_velocity=1,
        name="0 weight constraint",
    )
    constraints.add_inequality_constraint(
        task_expression=dof2.variables.position,
        lower_error=0.1,
        upper_error=1,
        quadratic_weight=1,
        reference_velocity=1,
        name="ineq constraint",
    )
    qp_data_symbolic = QPDataSymbolic(
        degrees_of_freedom=prismatic_bot2.active_degrees_of_freedom,
        constraint_collection=constraints,
        qp_controller_config=QPControllerConfig(
            target_frequency=20, prediction_horizon=10
        ),
    )
    adapter = QPDataExplicitFactory(qp_data_symbolic)
    adapter.compile(
        world_state_symbols=prismatic_bot2.state.get_variables(),
        life_cycle_symbols=[],
        float_variables=[],
    )
    qp_data = adapter.evaluate(
        world_state=_world_state_matrix(prismatic_bot2),
        life_cycle_state=np.array([]),
        float_variables=np.array([]),
    )
    qp_data_filtered = qp_data.apply_filters()
    solution = QPSolverPIQP().solver_call(qp_data_filtered)
    debugger = QuadraticProgramDebugger(
        qp_data_symbolic=qp_data_symbolic, current_solution=solution
    )
    assert len(debugger.inequality_constraints) == 1
    assert len(debugger.equality_constraints) == 22
    assert "ineq constraint" in debugger.inequality_constraints.index
    assert "position_constraint" in debugger.equality_constraints.index
    assert "bounds" in debugger.equality_constraints.columns


def _build_qp_data_symbolic(prismatic_bot2) -> QPDataSymbolic:
    constraints = ConstraintCollection()
    dof1 = prismatic_bot2.active_degrees_of_freedom[0]
    constraints.add_equality_constraint(
        task_expression=dof1.variables.position,
        equality_bound=1,
        quadratic_weight=1,
        reference_velocity=1,
        name="position_constraint",
    )
    constraints.add_inequality_constraint(
        task_expression=dof1.variables.position,
        lower_error=0.1,
        upper_error=1,
        quadratic_weight=1,
        reference_velocity=1,
        name="ineq constraint",
    )
    return QPDataSymbolic(
        degrees_of_freedom=prismatic_bot2.active_degrees_of_freedom,
        constraint_collection=constraints,
        qp_controller_config=QPControllerConfig(
            target_frequency=20, prediction_horizon=10
        ),
    )


def test_two_sided_factory_evaluate_returns_fresh_object(prismatic_bot2):
    qp_data_symbolic = _build_qp_data_symbolic(prismatic_bot2)
    factory = QPDataTwoSidedInequalityFactory(qp_data_symbolic)
    factory.compile(
        world_state_symbols=prismatic_bot2.state.get_variables(),
        life_cycle_symbols=[],
        float_variables=[],
    )
    qp_data = factory.evaluate(
        world_state=_world_state_matrix(prismatic_bot2),
        life_cycle_state=np.array([]),
        float_variables=np.array([]),
    )
    second_qp_data = factory.evaluate(
        world_state=_world_state_matrix(prismatic_bot2),
        life_cycle_state=np.array([]),
        float_variables=np.array([]),
    )
    assert isinstance(qp_data, QPDataTwoSidedInequality)
    assert qp_data is not second_qp_data


def test_get_factory_for_unregistered_type_raises():
    class UnregisteredQPData(QPData):
        def apply_filters(self):
            return self

    with pytest.raises(NoFactoryForQPDataTypeError):
        QPDataFactory.get_factory_from_qp_data_type(UnregisteredQPData)


def test_qp_data_type_is_a_classmethod():
    assert QPSolverPIQP.qp_data_type() is QPDataExplicit
    assert QPDataExplicitFactory.qp_data_type() is QPDataExplicit


def test_active_degrees_of_freedom_are_deduplicated_and_ordered(prismatic_bot2):
    active = prismatic_bot2.active_degrees_of_freedom
    connection_dofs = [
        dof
        for connection in prismatic_bot2.connections
        for dof in connection.active_dofs
    ]

    assert len(active) == len(set(active))
    assert active == list(dict.fromkeys(connection_dofs))


def test_constraint_collection_groups_constraints_by_enforcement_strategy(
    prismatic_bot2,
):
    dof = prismatic_bot2.active_degrees_of_freedom[0]
    constraints = ConstraintCollection()
    constraints.add_equality_constraint(
        task_expression=dof.variables.position,
        equality_bound=1,
        quadratic_weight=1,
        reference_velocity=1,
        name="eq",
    )
    constraints.add_velocity_constraint(
        lower_velocity_limit=-1,
        upper_velocity_limit=1,
        quadratic_weight=1,
        task_expression=dof.variables.position,
        velocity_limit=1,
        name="vel",
    )

    assert set(constraints.get_equality_constraint_blocks()) == {IntegralStrategy}
    assert set(constraints.get_inequality_constraint_blocks()) == {VelocityStrategy}
    assert QPDataTwoSidedInequalityFactory.qp_data_type() is QPDataTwoSidedInequality
