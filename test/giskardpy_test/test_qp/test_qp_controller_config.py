import numpy as np
import pytest

from giskardpy.executor import Executor
from giskardpy.motion_statechart.context import MotionStatechartContext
from giskardpy.motion_statechart.motion_statechart import MotionStatechart
from giskardpy.motion_statechart.tasks.joint_tasks import JointPositionList, JointState
from giskardpy.qp.exceptions import BrakingTimeExceedsHorizonError
from giskardpy.qp.jerk_limited_braking import JerkLimitedBraking
from giskardpy.qp.qp_controller_config import QPControllerConfig
from semantic_digital_twin.spatial_types.derivatives import Derivatives
from semantic_digital_twin.world import World

CONTROL_FREQUENCIES = [20, 50, 100]

# %% step counts derived from the highest optimized derivative


@pytest.mark.parametrize(
    "max_derivative, number_of_resting_steps",
    [(Derivatives.jerk, 2), (Derivatives.acceleration, 1)],
)
def test_resting_steps_follow_the_highest_optimized_derivative(
    max_derivative, number_of_resting_steps
):
    config = QPControllerConfig(target_frequency=20, max_derivative=max_derivative)

    assert config.number_of_resting_steps == number_of_resting_steps


@pytest.mark.parametrize(
    "max_derivative, minimum_prediction_horizon",
    [(Derivatives.jerk, 4), (Derivatives.acceleration, 3)],
)
def test_minimum_prediction_horizon_follows_the_highest_optimized_derivative(
    max_derivative, minimum_prediction_horizon
):
    config = QPControllerConfig(target_frequency=20, max_derivative=max_derivative)

    assert config.minimum_prediction_horizon == minimum_prediction_horizon


# %% prediction horizon


@pytest.mark.parametrize("target_frequency", CONTROL_FREQUENCIES)
def test_derived_prediction_horizon_covers_braking_and_resting_steps(
    target_frequency,
):
    config = QPControllerConfig(target_frequency=target_frequency)

    assert config.prediction_horizon == (
        config.number_of_braking_steps + config.number_of_resting_steps
    )


def test_default_configuration_keeps_the_former_default_horizon():
    """
    The default braking time reproduces the braking of the former default, a prediction
    horizon of 7 at 20 Hz.
    """
    former_default_prediction_horizon = 7

    config = QPControllerConfig(target_frequency=20)

    assert config.prediction_horizon == former_default_prediction_horizon


def test_derived_prediction_horizon_is_at_least_the_minimum():
    target_frequency = 50
    braking_time_of_one_step = 2 / target_frequency

    config = QPControllerConfig(
        target_frequency=target_frequency, braking_time=braking_time_of_one_step
    )

    assert (
        config.number_of_braking_steps + config.number_of_resting_steps
        < config.minimum_prediction_horizon
    )
    assert config.prediction_horizon == config.minimum_prediction_horizon


def test_explicit_prediction_horizon_longer_than_the_braking_is_kept():
    longer_prediction_horizon = (
        QPControllerConfig(target_frequency=20).prediction_horizon + 1
    )

    config = QPControllerConfig(
        target_frequency=20, prediction_horizon=longer_prediction_horizon
    )

    assert config.prediction_horizon == longer_prediction_horizon


def test_explicit_prediction_horizon_too_short_for_the_braking_raises():
    target_frequency = 100
    braking_time = 0.3
    number_of_resting_steps = QPControllerConfig(
        target_frequency=target_frequency, braking_time=braking_time
    ).number_of_resting_steps
    minimum_prediction_horizon = (
        JerkLimitedBraking.number_of_steps_for_braking_time(
            braking_time=braking_time, time_step=1 / target_frequency
        )
        + number_of_resting_steps
    )

    with pytest.raises(BrakingTimeExceedsHorizonError) as error:
        QPControllerConfig(
            target_frequency=target_frequency,
            braking_time=braking_time,
            prediction_horizon=minimum_prediction_horizon - 1,
        )

    assert error.value.minimum_prediction_horizon == minimum_prediction_horizon


# %% behaviour independent of the control frequency


def _peak_acceleration_of_a_joint_goal(
    world: World, config: QPControllerConfig
) -> float:
    """
    Drives the single joint of ``world`` from 0 towards a distant goal for three seconds
    and returns the largest acceleration magnitude it reaches.

    :param world: World with a single prismatic joint.
    :param config: Controller configuration the joint goal runs with.
    :return: Largest acceleration magnitude the joint reaches.
    """
    connection = world.controlled_connections[0]
    connection.position = 0.0
    statechart = MotionStatechart()
    statechart.add_node(
        JointPositionList(goal_state=JointState.from_mapping({connection: 1.5}))
    )
    executor = Executor(
        MotionStatechartContext(world=world, qp_controller_config=config)
    )
    executor.compile(motion_statechart=statechart)
    accelerations = []
    for _ in range(int(3 * config.target_frequency)):
        executor.tick()
        accelerations.append(world.state[connection.dof.id].acceleration)
    executor.set_velocity_acceleration_jerk_to_zero()
    return float(np.max(np.abs(accelerations)))


COARSEST_GRID_TOLERANCE = 0.15
"""
Relative shortfall of the peak acceleration below its continuous value that the coarsest
tested grid, five braking steps at 20 Hz, causes.
"""


@pytest.mark.parametrize("target_frequency", CONTROL_FREQUENCIES)
def test_peak_acceleration_follows_the_braking_time_at_every_control_frequency(
    prismatic_world_no_position_limits, target_frequency
):
    """
    A jerk-limited braking from the velocity limit v to rest in time T peaks at an
    acceleration of 2 v / T, so the same braking time gives the same peak acceleration
    whatever the control frequency, where a jerk limit derived from the horizon made it
    grow with the frequency.
    """
    connection = prismatic_world_no_position_limits.controlled_connections[0]
    config = QPControllerConfig(target_frequency=target_frequency)

    peak_acceleration = _peak_acceleration_of_a_joint_goal(
        prismatic_world_no_position_limits, config
    )

    assert peak_acceleration == pytest.approx(
        2 * connection.dof.limits.upper.velocity / config.braking_time,
        rel=COARSEST_GRID_TOLERANCE,
    )
