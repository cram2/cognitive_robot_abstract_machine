"""
Direct unit tests for the focused units extracted from
:class:`giskardpy.qp.dof_limits.DegreeOfFreedomLimitProfiler`.
"""

import numpy as np
import pytest

import krrood.symbolic_math.symbolic_math as sm
from giskardpy.executor import Executor
from giskardpy.motion_statechart.context import MotionStatechartContext
from giskardpy.motion_statechart.motion_statechart import MotionStatechart
from giskardpy.motion_statechart.tasks.joint_tasks import JointPositionList, JointState
from giskardpy.qp.dof_limits import (
    BoundDirection,
    DegreeOfFreedomLimitProfiler,
    DegreeOfFreedomDecisionVariables,
    VelocityBoundProfiles,
)
from giskardpy.qp.exceptions import DegreeOfFreedomBrakingExceedsHorizonError
from giskardpy.qp.jerk_limited_braking import JerkLimitedBraking
from giskardpy.qp.pos_in_vel_limits import BrakingProfile
from giskardpy.qp.qp_controller_config import (
    NUMBER_OF_RESTING_STEPS,
    QPControllerConfig,
)
from semantic_digital_twin.spatial_types.derivatives import DerivativeMap, Derivatives
from semantic_digital_twin.world import World
from semantic_digital_twin.world_description.degree_of_freedom import (
    DegreeOfFreedom,
    DegreeOfFreedomLimits,
)

TARGET_FREQUENCY = 20
PREDICTION_HORIZON = 10
VELOCITY_LIMIT = 1.0
POSITION_LIMIT = 1.0


def _default_config() -> QPControllerConfig:
    return QPControllerConfig(
        target_frequency=TARGET_FREQUENCY, prediction_horizon=PREDICTION_HORIZON
    )


def _profiler() -> DegreeOfFreedomLimitProfiler:
    return DegreeOfFreedomLimitProfiler(_default_config())


def _single_dof(world: World) -> DegreeOfFreedom:
    return world.active_degrees_of_freedom[0]


def _truth_value(value: object) -> bool:
    """
    Evaluates the truth of a symbolic scalar, tolerating the native ``bool`` that
    symbolic operations fold to when all of their inputs are constants.
    """
    if isinstance(value, sm.Scalar):
        return bool(value.evaluate()[0])
    return bool(value)


def _directional_bounds_at(
    profiler: DegreeOfFreedomLimitProfiler, world: World, position: float
) -> tuple[np.ndarray, np.ndarray]:
    """
    Evaluates the lower and upper directional velocity bounds for the single degree of freedom of
    ``world`` placed at ``position``.
    """
    world.controlled_connections[0].position = position
    degree_of_freedom = _single_dof(world)
    limits = profiler.resolve_limits(degree_of_freedom)
    lower_limits, upper_limits = limits.lower, limits.upper
    time_step = profiler.time_step
    position_range = upper_limits.position - lower_limits.position
    velocity_limit = (
        min(upper_limits.velocity * time_step, position_range / 2) / time_step
    )
    braking_profile = BrakingProfile.fastest(
        initial_velocity=velocity_limit,
        acceleration_limit=upper_limits.acceleration,
        jerk_limit=upper_limits.jerk,
        time_step=time_step,
        prediction_horizon=profiler.prediction_horizon,
    )
    lower_bound = profiler._directional_velocity_bound(
        braking_profile=braking_profile,
        position_error=lower_limits.position - degree_of_freedom.variables.position,
        jerk_limit=upper_limits.jerk,
        velocity_limit=velocity_limit,
        direction=BoundDirection.LOWER,
    )
    upper_bound = profiler._directional_velocity_bound(
        braking_profile=braking_profile,
        position_error=upper_limits.position - degree_of_freedom.variables.position,
        jerk_limit=upper_limits.jerk,
        velocity_limit=velocity_limit,
        direction=BoundDirection.UPPER,
    )
    return lower_bound.evaluate(), upper_bound.evaluate()


def _horizon_bounds_at(
    profiler: DegreeOfFreedomLimitProfiler,
    world: World,
    distance_to_lower_limit: float,
    velocity: float,
) -> DegreeOfFreedomLimits:
    """
    Evaluates the horizon bounds of the single degree of freedom of ``world`` while it
    moves with ``velocity`` and zero acceleration at ``distance_to_lower_limit`` above
    its lower position limit.
    """
    degree_of_freedom = _single_dof(world)
    limits = profiler.resolve_limits(degree_of_freedom)
    state = world.state[degree_of_freedom.id]
    state.position = limits.lower.position + distance_to_lower_limit
    state.velocity = velocity
    state.acceleration = 0.0
    world.notify_state_change()
    horizon_limits = profiler.compute_horizon_bounds(
        degree_of_freedom_symbols=degree_of_freedom.variables, limits=limits
    )
    return DegreeOfFreedomLimits(
        lower=DerivativeMap(
            velocity=horizon_limits.lower.velocity.evaluate(),
            jerk=horizon_limits.lower.jerk.evaluate(),
        ),
        upper=DerivativeMap(
            velocity=horizon_limits.upper.velocity.evaluate(),
            jerk=horizon_limits.upper.jerk.evaluate(),
        ),
    )


def test_resolve_limits_with_position_limits(prismatic_bot):
    limits = _profiler().resolve_limits(_single_dof(prismatic_bot))

    assert limits.upper.position == POSITION_LIMIT
    assert limits.lower.position == -POSITION_LIMIT
    assert limits.upper.acceleration == np.inf
    assert limits.upper.jerk > 0


def test_resolve_limits_without_position_limits(prismatic_world_no_position_limits):
    limits = _profiler().resolve_limits(_single_dof(prismatic_world_no_position_limits))

    assert limits.lower.position is None
    assert limits.upper.position is None


# %% jerk limit from the braking time


@pytest.mark.parametrize("target_frequency", [20, 50, 100])
def test_derived_jerk_limit_follows_the_braking_time_at_every_control_frequency(
    prismatic_bot, target_frequency
):
    config = QPControllerConfig(target_frequency=target_frequency)
    degree_of_freedom = _single_dof(prismatic_bot)

    limits = DegreeOfFreedomLimitProfiler(config).resolve_limits(degree_of_freedom)

    expected_jerk_limit = (
        4 * degree_of_freedom.limits.upper.velocity / config.braking_time**2
    )
    assert limits.upper.jerk == pytest.approx(expected_jerk_limit)


def test_declared_jerk_limit_is_kept(prismatic_bot_with_jerk_limit):
    degree_of_freedom = _single_dof(prismatic_bot_with_jerk_limit)

    limits = _profiler().resolve_limits(degree_of_freedom)

    assert limits.upper.jerk == degree_of_freedom.limits.upper.jerk


def test_braking_that_does_not_fit_the_horizon_raises(
    prismatic_world_with_low_jerk_limit,
):
    """
    A degree of freedom that cannot brake from its velocity limit to rest before the
    resting steps of the horizon is rejected, whether or not it has position limits.
    """
    config = _default_config()
    degree_of_freedom = _single_dof(prismatic_world_with_low_jerk_limit)
    braking = JerkLimitedBraking(
        velocity_limit=degree_of_freedom.limits.upper.velocity,
        jerk_limit=degree_of_freedom.limits.upper.jerk,
        time_step=config.control_dt,
    )

    with pytest.raises(DegreeOfFreedomBrakingExceedsHorizonError) as error:
        DegreeOfFreedomDecisionVariables(
            degrees_of_freedom=prismatic_world_with_low_jerk_limit.active_degrees_of_freedom,
            qp_controller_config=config,
        ).direct_limits()

    assert error.value.minimum_prediction_horizon == (
        braking.number_of_steps + NUMBER_OF_RESTING_STEPS
    )


# %% acceleration limits

ACCELERATION_LIMIT_TOLERANCE = 1e-3
"""
Relative amount by which a solved acceleration may exceed its limit, covering the
tolerance of the QP solver.
"""


def test_declared_acceleration_limit_caps_the_jerk_limit(
    prismatic_world_with_acceleration_limit,
):
    """
    With only a jerk limit J in the QP, a velocity swing of at most 2 v reaches an
    acceleration of at most sqrt(2 v J), so J = A² / (2 v) keeps it within A.
    """
    degree_of_freedom = _single_dof(prismatic_world_with_acceleration_limit)
    velocity_limit = degree_of_freedom.limits.upper.velocity
    acceleration_limit = degree_of_freedom.limits.upper.acceleration

    limits = _profiler().resolve_limits(degree_of_freedom)

    assert limits.upper.jerk == pytest.approx(
        acceleration_limit**2 / (2 * velocity_limit)
    )


def _peak_acceleration_moving_to(
    world: World, goal: float, initial_velocity: float
) -> float:
    """
    Drives the single joint of ``world`` from 0, moving at ``initial_velocity``, to
    ``goal`` and returns the largest acceleration magnitude it reaches.

    The prediction horizon is the shortest one the joint's acceleration-capped braking
    fits into.

    :param world: World with a single prismatic joint that has an acceleration limit.
    :param goal: Position the joint is driven to.
    :param initial_velocity: Velocity the joint starts moving at.
    :return: Largest acceleration magnitude the joint reaches.
    """
    connection = world.controlled_connections[0]
    degree_of_freedom = connection.dof
    velocity_limit = degree_of_freedom.limits.upper.velocity
    acceleration_limit = degree_of_freedom.limits.upper.acceleration
    braking = JerkLimitedBraking(
        velocity_limit=velocity_limit,
        jerk_limit=acceleration_limit**2 / (2 * velocity_limit),
        time_step=1 / TARGET_FREQUENCY,
    )
    config = QPControllerConfig(
        target_frequency=TARGET_FREQUENCY,
        prediction_horizon=braking.number_of_steps + NUMBER_OF_RESTING_STEPS,
    )
    connection.position = 0.0
    world.state[degree_of_freedom.id].velocity = initial_velocity
    statechart = MotionStatechart()
    statechart.add_node(
        JointPositionList(goal_state=JointState.from_mapping({connection: goal}))
    )
    executor = Executor(
        MotionStatechartContext(world=world, qp_controller_config=config)
    )
    executor.compile(motion_statechart=statechart)
    accelerations = []
    for _ in range(6 * TARGET_FREQUENCY):
        executor.tick()
        accelerations.append(world.state[degree_of_freedom.id].acceleration)
    return float(np.max(np.abs(accelerations)))


@pytest.mark.parametrize(
    "initial_velocity_factor", [0.0, -1.0], ids=["from_rest", "reversing"]
)
def test_joint_goal_respects_the_declared_acceleration_limit(
    prismatic_world_with_acceleration_limit, initial_velocity_factor
):
    """
    Reversing at full speed is the largest velocity swing, and the case the jerk cap is
    sized for.
    """
    degree_of_freedom = _single_dof(prismatic_world_with_acceleration_limit)
    acceleration_limit = degree_of_freedom.limits.upper.acceleration

    peak_acceleration = _peak_acceleration_moving_to(
        prismatic_world_with_acceleration_limit,
        goal=1.5,
        initial_velocity=initial_velocity_factor
        * degree_of_freedom.limits.upper.velocity,
    )

    assert peak_acceleration <= acceleration_limit * (1 + ACCELERATION_LIMIT_TOLERANCE)


def test_unconstrained_velocity_bounds_are_flat():
    bounds = VelocityBoundProfiles.unconstrained(VELOCITY_LIMIT, PREDICTION_HORIZON)

    assert np.allclose(bounds.upper_bound.evaluate(), VELOCITY_LIMIT)
    assert np.allclose(bounds.lower_bound.evaluate(), -VELOCITY_LIMIT)
    assert np.allclose(bounds.goal_profile.evaluate(), 0.0)
    assert bounds.skip_first.evaluate()[0] == 0.0


def test_directional_velocity_bounds_mirror_at_center(prismatic_bot):
    lower_bound, upper_bound = _directional_bounds_at(_profiler(), prismatic_bot, 0.0)

    assert np.allclose(lower_bound, -upper_bound, atol=1e-6)


def test_directional_velocity_bound_reduced_near_upper_limit(prismatic_bot):
    lower_bound, upper_bound = _directional_bounds_at(_profiler(), prismatic_bot, 0.9)

    assert (
        upper_bound[0] < 0.9
    ), "Upper bound must brake when approaching the upper limit"
    assert np.isclose(
        lower_bound[0], -VELOCITY_LIMIT, atol=1e-3
    ), "Lower bound is unconstrained moving away from the upper limit"


def test_relax_jerk_on_initial_steps_relaxes_only_when_needed():
    profiler = _profiler()
    jerk_limit = 2.0
    violated = sm.Vector([10.0, 20.0, 30.0, 40.0, 50.0])

    relaxed_profile = sm.Vector([jerk_limit] * 5)
    profiler._relax_jerk_on_initial_steps(
        jerk_profile=relaxed_profile,
        projected_jerk_profile_violated=violated,
        needs_relaxed_jerk_limits=sm.Scalar.const_true(),
        jerk_limit=jerk_limit,
    )
    relaxed = relaxed_profile.evaluate()
    assert np.allclose(relaxed[:3], [10.0, 20.0, 30.0])
    assert np.allclose(relaxed[3:], jerk_limit)

    unchanged_profile = sm.Vector([jerk_limit] * 5)
    profiler._relax_jerk_on_initial_steps(
        jerk_profile=unchanged_profile,
        projected_jerk_profile_violated=violated,
        needs_relaxed_jerk_limits=sm.Scalar.const_false(),
        jerk_limit=jerk_limit,
    )
    assert np.allclose(unchanged_profile.evaluate(), jerk_limit)


def test_detect_velocity_bound_violation():
    number_of_steps = 5
    bounds = VelocityBoundProfiles.unconstrained(VELOCITY_LIMIT, number_of_steps)
    epsilon = 1e-5

    inside = sm.Vector([0.0] * number_of_steps)
    assert not _truth_value(bounds.is_violated_by(inside, epsilon))

    exceeds_upper = sm.Vector([2 * VELOCITY_LIMIT] + [0.0] * (number_of_steps - 1))
    assert _truth_value(bounds.is_violated_by(exceeds_upper, epsilon))

    nonzero_terminal = sm.Vector([0.0] * (number_of_steps - 1) + [VELOCITY_LIMIT / 2])
    assert _truth_value(bounds.is_violated_by(nonzero_terminal, epsilon))


def test_compute_horizon_bounds_flat_at_center(prismatic_bot):
    profiler = _profiler()
    prismatic_bot.controlled_connections[0].position = 0.0
    degree_of_freedom = _single_dof(prismatic_bot)

    horizon_limits = profiler.compute_horizon_bounds(
        degree_of_freedom_symbols=degree_of_freedom.variables,
        limits=profiler.resolve_limits(degree_of_freedom),
    )

    assert np.allclose(
        horizon_limits.upper.velocity.evaluate(), VELOCITY_LIMIT, atol=1e-3
    )
    assert np.allclose(
        horizon_limits.lower.velocity.evaluate(), -VELOCITY_LIMIT, atol=1e-3
    )


# %% braking margin


def test_final_braking_step_uses_exactly_the_jerk_limit(prismatic_bot):
    """
    A degree of freedom that rides the last braking level into a position limit can stop
    with the full jerk limit, so the first step's velocity is not confined to a window
    as narrow as a solver's tolerance.
    """
    profiler = DegreeOfFreedomLimitProfiler(
        QPControllerConfig.create_with_simulation_defaults()
    )
    time_step = profiler.time_step
    upper_limits = profiler.resolve_limits(_single_dof(prismatic_bot)).upper
    jerk_step = upper_limits.jerk * time_step**2
    approaching = _horizon_bounds_at(
        profiler, prismatic_bot, distance_to_lower_limit=0.01, velocity=-jerk_step
    )
    final_braking_velocity = approaching.lower.velocity[0]
    stopping = _horizon_bounds_at(
        profiler,
        prismatic_bot,
        distance_to_lower_limit=time_step * -final_braking_velocity / 2,
        velocity=final_braking_velocity,
    )

    reachable_velocity = final_braking_velocity + stopping.upper.jerk[0]

    assert reachable_velocity >= stopping.lower.velocity[0]


# %% objective weights


def test_degree_of_freedom_with_jerk_limit_has_no_jerk_cost_by_default(
    prismatic_bot_with_jerk_limit,
):
    """
    A degree of freedom that states its own jerk limit keeps the default of costing
    nothing for jerk, like one whose jerk limit is derived from the braking time.
    """
    config = _default_config()

    limits = DegreeOfFreedomDecisionVariables(
        degrees_of_freedom=prismatic_bot_with_jerk_limit.active_degrees_of_freedom,
        qp_controller_config=config,
    ).direct_limits()

    jerk_weights = limits.quadratic_weights.to_np()[config.control_horizon :]
    assert np.array_equal(jerk_weights, np.zeros(config.prediction_horizon))


JERK_WEIGHT = 1.0
"""
Objective weight given to jerk in the weight normalization tests.
"""


def _first_jerk_weight(world: World) -> float:
    """
    Returns the objective weight of the first jerk decision variable of the single
    degree of freedom of ``world``, with its jerk weighted by :data:`JERK_WEIGHT`.

    :param world: World with a single degree of freedom.
    :return: Objective weight of the first jerk decision variable.
    """
    config = _default_config()
    degree_of_freedom = _single_dof(world)
    config.set_dof_weight(degree_of_freedom.name, Derivatives.jerk, JERK_WEIGHT)
    limits = DegreeOfFreedomDecisionVariables(
        degrees_of_freedom=world.active_degrees_of_freedom,
        qp_controller_config=config,
    ).direct_limits()
    return float(limits.quadratic_weights.to_np()[config.control_horizon])


def _normalized_first_jerk_weight(jerk_limit: float) -> float:
    """
    The weight of the first jerk decision variable, which holds jerk times the squared
    time step, normalized by that variable's own bound.

    :param jerk_limit: Jerk limit of the degree of freedom.
    :return: Normalized weight of the first jerk decision variable.
    """
    config = _default_config()
    jerk_decision_variable_bound = jerk_limit * config.control_dt**2
    return (
        config.horizon_weight_gain_scalar
        * JERK_WEIGHT
        / jerk_decision_variable_bound**2
    )


def test_jerk_weight_of_a_degree_of_freedom_without_jerk_limit_is_kept(
    prismatic_bot,
):
    """
    A jerk weight also applies to a degree of freedom whose jerk limit is derived from
    the braking time, instead of being dropped for lack of a declared jerk limit.
    """
    degree_of_freedom = _single_dof(prismatic_bot)
    derived_jerk_limit = (
        4
        * degree_of_freedom.limits.upper.velocity
        / _default_config().braking_time ** 2
    )

    assert _first_jerk_weight(prismatic_bot) == pytest.approx(
        _normalized_first_jerk_weight(derived_jerk_limit)
    )


def test_jerk_weight_is_normalized_by_the_bound_of_its_decision_variable(
    prismatic_bot_with_jerk_limit,
):
    degree_of_freedom = _single_dof(prismatic_bot_with_jerk_limit)

    assert _first_jerk_weight(prismatic_bot_with_jerk_limit) == pytest.approx(
        _normalized_first_jerk_weight(degree_of_freedom.limits.upper.jerk)
    )
