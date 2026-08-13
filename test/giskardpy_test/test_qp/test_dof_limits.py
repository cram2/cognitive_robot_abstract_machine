"""
Direct unit tests for the focused units extracted from
:class:`giskardpy.qp.dof_limits.DegreeOfFreedomLimitProfiler`.
"""

import numpy as np
import pytest

import krrood.symbolic_math.symbolic_math as sm
from giskardpy.qp.dof_limits import (
    BoundDirection,
    DegreeOfFreedomLimitProfiler,
    VelocityBoundProfiles,
)
from giskardpy.qp.qp_controller_config import QPControllerConfig
from semantic_digital_twin.world import World
from semantic_digital_twin.world_description.degree_of_freedom import DegreeOfFreedom

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
    Evaluates the truth of a symbolic scalar, tolerating the native ``bool`` that symbolic
    operations fold to when all of their inputs are constants.
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
    lower_limits, upper_limits = profiler._resolve_limits(degree_of_freedom)
    config = profiler.qp_controller_config
    time_step = config.model_predictive_control_time_step
    position_range = upper_limits.position - lower_limits.position
    velocity_limit = (
        min(upper_limits.velocity * time_step, position_range / 2) / time_step
    )
    mpc_velocity_profile, mpc_acceleration_profile = profiler._nominal_velocity_profile(
        initial_velocity=velocity_limit,
        acceleration_limit=upper_limits.acceleration,
        jerk_limit=upper_limits.jerk,
        time_step=time_step,
        prediction_horizon=config.prediction_horizon,
        solver_class=config.qp_solver_class,
    )
    lower_bound = profiler._directional_velocity_bound(
        velocity_profile=mpc_velocity_profile,
        acceleration_profile=mpc_acceleration_profile,
        position_error=lower_limits.position - degree_of_freedom.variables.position,
        jerk_limit=upper_limits.jerk,
        velocity_limit=velocity_limit,
        time_step=time_step,
        direction=BoundDirection.LOWER,
    )
    upper_bound = profiler._directional_velocity_bound(
        velocity_profile=mpc_velocity_profile,
        acceleration_profile=mpc_acceleration_profile,
        position_error=upper_limits.position - degree_of_freedom.variables.position,
        jerk_limit=upper_limits.jerk,
        velocity_limit=velocity_limit,
        time_step=time_step,
        direction=BoundDirection.UPPER,
    )
    return lower_bound.evaluate(), upper_bound.evaluate()


def test_resolve_limits_with_position_limits(prismatic_bot):
    lower_limits, upper_limits = _profiler()._resolve_limits(_single_dof(prismatic_bot))

    assert upper_limits.position == POSITION_LIMIT
    assert lower_limits.position == -POSITION_LIMIT
    assert lower_limits.acceleration == -np.inf
    assert upper_limits.acceleration == np.inf
    assert upper_limits.jerk > 0
    assert lower_limits.jerk == pytest.approx(-upper_limits.jerk)


def test_resolve_limits_without_position_limits(prismatic_world_no_position_limits):
    lower_limits, upper_limits = _profiler()._resolve_limits(
        _single_dof(prismatic_world_no_position_limits)
    )

    assert lower_limits.position is None
    assert upper_limits.position is None


def test_unconstrained_velocity_bounds_are_flat():
    bounds = _profiler()._unconstrained_velocity_bounds(
        VELOCITY_LIMIT, PREDICTION_HORIZON
    )

    assert isinstance(bounds, VelocityBoundProfiles)
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
    profiler = _profiler()
    lower_bound = sm.Vector([-1.0] * 5)
    upper_bound = sm.Vector([1.0] * 5)
    epsilon = 1e-5

    inside = sm.Vector([0.0] * 5)
    assert not _truth_value(
        profiler._detect_velocity_bound_violation(
            inside, lower_bound, upper_bound, epsilon
        )
    )

    exceeds_upper = sm.Vector([2.0, 0.0, 0.0, 0.0, 0.0])
    assert _truth_value(
        profiler._detect_velocity_bound_violation(
            exceeds_upper, lower_bound, upper_bound, epsilon
        )
    )

    nonzero_terminal = sm.Vector([0.0, 0.0, 0.0, 0.0, 0.5])
    assert _truth_value(
        profiler._detect_velocity_bound_violation(
            nonzero_terminal, lower_bound, upper_bound, epsilon
        )
    )


def test_compute_horizon_bounds_flat_at_center(prismatic_bot):
    profiler = _profiler()
    prismatic_bot.controlled_connections[0].position = 0.0
    degree_of_freedom = _single_dof(prismatic_bot)
    lower_limits, upper_limits = profiler._resolve_limits(degree_of_freedom)
    config = profiler.qp_controller_config

    horizon_limits = profiler.compute_horizon_bounds(
        degree_of_freedom_symbols=degree_of_freedom.variables,
        lower_limits=lower_limits,
        upper_limits=upper_limits,
        solver_class=config.qp_solver_class,
        time_step=config.model_predictive_control_time_step,
        prediction_horizon=config.prediction_horizon,
    )

    assert np.allclose(
        horizon_limits.upper.velocity.evaluate(), VELOCITY_LIMIT, atol=1e-3
    )
    assert np.allclose(
        horizon_limits.lower.velocity.evaluate(), -VELOCITY_LIMIT, atol=1e-3
    )
