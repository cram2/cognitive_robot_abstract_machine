from __future__ import annotations

from dataclasses import dataclass
from datetime import timedelta

import numpy as np
import pytest

from giskardpy.motion_statechart.graph_node import EndMotion
from cramph.statechart import Statechart
from giskardpy.motion_statechart.tasks.joint_tasks import JointPositionList, JointState
from giskardpy.qp.dof_limits import (
    DegreeOfFreedomDecisionVariables,
    DegreeOfFreedomLimitProfiler,
    DirectLimits,
)
from giskardpy.qp.jerk_limited_braking import JerkLimitedBraking
from giskardpy.qp.qp_controller_config import QPControllerConfig
from semantic_digital_twin.world import World
from semantic_digital_twin.world_description.connections import ActiveConnection
from semantic_digital_twin.robots.pr2 import PR2Joint
from giskardpy.motion_control import MotionControl
from cramph.context import StatechartContext
from cramph.executor import StatechartExecutor

TARGET_FREQUENCY = 20
PREDICTION_HORIZON = 10
DT = 1 / TARGET_FREQUENCY
NUMBER_OF_VELOCITY_STEPS = PREDICTION_HORIZON - 2
EXPECTED_JERK_LIMIT = DT
POSITION_UPPER_LIMIT = 1.0
VELOCITY_LIMIT = 1.0


def test_joint_goal_inside_limits_reached(pr2_world_state_reset):
    connection = pr2_world_state_reset.get_connection_by_name(PR2Joint.HEAD_PAN)
    dof = connection.dof
    lower = dof.limits.lower.position
    upper = dof.limits.upper.position
    goal = 1.0

    executor = StatechartExecutor(
        context=StatechartContext(world=pr2_world_state_reset),
        extensions=[MotionControl()],
    )
    statechart = Statechart(context=executor.context)
    joint_goal = JointPositionList(
        goal_state=JointState.from_mapping({connection: goal})
    )
    statechart.add_node(joint_goal)
    end = EndMotion()
    statechart.add_node(end)
    end.start_condition = joint_goal.observes_true

    executor.compile(statechart=statechart)
    executor.tick_until_end()

    assert np.isclose(connection.position, goal, atol=0.01)
    assert lower <= connection.position <= upper


def test_joint_goal_clamped_to_upper_limit(pr2_world_state_reset):
    connection = pr2_world_state_reset.get_connection_by_name(PR2Joint.HEAD_PAN)
    dof = connection.dof
    upper = dof.limits.upper.position
    goal_beyond_limit = upper + 2.0

    executor = StatechartExecutor(
        context=StatechartContext(world=pr2_world_state_reset),
        extensions=[MotionControl()],
    )
    statechart = Statechart(context=executor.context)
    joint_goal = JointPositionList(
        goal_state=JointState.from_mapping({connection: goal_beyond_limit})
    )
    statechart.add_node(joint_goal)
    end = EndMotion()
    statechart.add_node(end)
    end.start_condition = joint_goal.observes_true

    executor.compile(statechart=statechart)
    executor.tick_until_end()

    assert np.isclose(connection.position, upper, atol=0.01)


def test_joint_goal_clamped_to_lower_limit(pr2_world_state_reset):
    connection = pr2_world_state_reset.get_connection_by_name(PR2Joint.HEAD_PAN)
    dof = connection.dof
    lower = dof.limits.lower.position
    goal_beyond_limit = lower - 2.0

    executor = StatechartExecutor(
        context=StatechartContext(world=pr2_world_state_reset),
        extensions=[MotionControl()],
    )
    statechart = Statechart(context=executor.context)
    joint_goal = JointPositionList(
        goal_state=JointState.from_mapping({connection: goal_beyond_limit})
    )
    statechart.add_node(joint_goal)
    end = EndMotion()
    statechart.add_node(end)
    end.start_condition = joint_goal.observes_true

    executor.compile(statechart=statechart)
    executor.tick_until_end()

    assert np.isclose(connection.position, lower, atol=0.01)


def test_joint_above_upper_limit_recovers(pr2_world_state_reset):
    connection = pr2_world_state_reset.get_connection_by_name(PR2Joint.HEAD_PAN)
    dof = connection.dof
    lower = dof.limits.lower.position
    upper = dof.limits.upper.position

    connection.position = upper + 0.5
    goal = 1.0

    executor = StatechartExecutor(
        context=StatechartContext(world=pr2_world_state_reset),
        extensions=[MotionControl()],
    )
    statechart = Statechart(context=executor.context)
    joint_goal = JointPositionList(
        goal_state=JointState.from_mapping({connection: goal})
    )
    statechart.add_node(joint_goal)
    end = EndMotion()
    statechart.add_node(end)
    end.start_condition = joint_goal.observes_true

    executor.compile(statechart=statechart)
    executor.tick_until_end()

    assert np.isclose(connection.position, goal, atol=0.01)
    assert lower <= connection.position <= upper + 0.01


def test_joint_below_lower_limit_recovers(pr2_world_state_reset):
    connection = pr2_world_state_reset.get_connection_by_name(PR2Joint.HEAD_PAN)
    dof = connection.dof
    lower = dof.limits.lower.position
    upper = dof.limits.upper.position

    connection.position = lower - 0.5
    goal = -1.0

    executor = StatechartExecutor(
        context=StatechartContext(world=pr2_world_state_reset),
        extensions=[MotionControl()],
    )
    statechart = Statechart(context=executor.context)
    joint_goal = JointPositionList(
        goal_state=JointState.from_mapping({connection: goal})
    )
    statechart.add_node(joint_goal)
    end = EndMotion()
    statechart.add_node(end)
    end.start_condition = joint_goal.observes_true

    executor.compile(statechart=statechart)
    executor.tick_until_end()

    assert np.isclose(connection.position, goal, atol=0.01)
    assert lower - 0.01 <= connection.position <= upper


def test_multiple_joints_outside_limits_recover(pr2_world_state_reset):
    head_pan = pr2_world_state_reset.get_connection_by_name(PR2Joint.HEAD_PAN)
    head_tilt = pr2_world_state_reset.get_connection_by_name(PR2Joint.HEAD_TILT)
    upper_arm_roll = pr2_world_state_reset.get_connection_by_name(
        PR2Joint.RIGHT_UPPER_ARM_ROLL
    )

    head_pan.position = head_pan.dof.limits.upper.position + 0.5
    head_tilt.position = head_tilt.dof.limits.upper.position + 0.3
    upper_arm_roll.position = upper_arm_roll.dof.limits.lower.position - 0.5

    goals = {
        head_pan: 0.0,
        head_tilt: 0.5,
        upper_arm_roll: 0.0,
    }

    executor = StatechartExecutor(
        context=StatechartContext(world=pr2_world_state_reset),
        extensions=[MotionControl()],
    )
    statechart = Statechart(context=executor.context)
    joint_goal = JointPositionList(goal_state=JointState.from_mapping(goals))
    statechart.add_node(joint_goal)
    end = EndMotion()
    statechart.add_node(end)
    end.start_condition = joint_goal.observes_true

    executor.compile(statechart=statechart)
    executor.tick_until_end()

    for conn, goal in goals.items():
        lower = conn.dof.limits.lower.position
        upper = conn.dof.limits.upper.position
        assert np.isclose(conn.position, goal, atol=0.01), f"{conn.name} not at goal"
        assert lower - 0.01 <= conn.position <= upper + 0.01, (
            f"{conn.name} outside limits"
        )


def _default_config() -> QPControllerConfig:
    return QPControllerConfig(
        target_frequency=TARGET_FREQUENCY, prediction_horizon=PREDICTION_HORIZON
    )


def _compute_limits(world: World) -> DirectLimits:
    return DegreeOfFreedomDecisionVariables(
        degrees_of_freedom=world.active_degrees_of_freedom,
        qp_controller_config=_default_config(),
    ).direct_limits()


def _connection(world: World) -> ActiveConnection:
    return world.controlled_connections[0]


def test_dof_limits_joint_in_center(prismatic_bot):
    _connection(prismatic_bot).position = 0.0

    limits = _compute_limits(prismatic_bot)
    upper_vel = limits.upper_bounds.evaluate()[:NUMBER_OF_VELOCITY_STEPS]
    lower_vel = limits.lower_bounds.evaluate()[:NUMBER_OF_VELOCITY_STEPS]

    assert np.allclose(upper_vel, VELOCITY_LIMIT, atol=1e-3)
    assert np.allclose(lower_vel, -VELOCITY_LIMIT, atol=1e-3)


def test_dof_limits_joint_near_upper_limit(prismatic_bot):
    _connection(prismatic_bot).position = 0.9

    limits = _compute_limits(prismatic_bot)
    upper_vel = limits.upper_bounds.evaluate()[:NUMBER_OF_VELOCITY_STEPS]
    lower_vel = limits.lower_bounds.evaluate()[:NUMBER_OF_VELOCITY_STEPS]

    assert upper_vel[0] < 0.9, (
        "Upper velocity at t=0 must be reduced near the upper position limit"
    )
    assert np.isclose(lower_vel[0], -VELOCITY_LIMIT, atol=1e-3), (
        "Lower velocity is unconstrained going away from the limit"
    )


def test_dof_limits_joint_near_lower_limit(prismatic_bot):
    _connection(prismatic_bot).position = -0.9

    limits = _compute_limits(prismatic_bot)
    upper_vel = limits.upper_bounds.evaluate()[:NUMBER_OF_VELOCITY_STEPS]
    lower_vel = limits.lower_bounds.evaluate()[:NUMBER_OF_VELOCITY_STEPS]

    assert lower_vel[0] > -0.9, (
        "Lower velocity at t=0 must be reduced (less negative) near the lower position limit"
    )
    assert np.isclose(upper_vel[0], VELOCITY_LIMIT, atol=1e-3), (
        "Upper velocity is unconstrained going away from the limit"
    )


def test_dof_limits_joint_above_upper_limit(prismatic_bot):
    _connection(prismatic_bot).position = 1.2

    limits = _compute_limits(prismatic_bot)
    upper_vel = limits.upper_bounds.evaluate()[:NUMBER_OF_VELOCITY_STEPS]

    assert upper_vel[0] <= 0.0, (
        "Upper velocity must be non-positive when joint is above the upper limit"
    )


def test_dof_limits_joint_below_lower_limit(prismatic_bot):
    _connection(prismatic_bot).position = -1.2

    limits = _compute_limits(prismatic_bot)
    lower_vel = limits.lower_bounds.evaluate()[:NUMBER_OF_VELOCITY_STEPS]

    assert lower_vel[0] >= 0.0, (
        "Lower velocity must be non-negative when joint is below the lower limit"
    )


def test_dof_limits_integrating_upper_velocity_bounds_stays_within_position_limit(
    prismatic_bot,
):
    initial_position = 0.5
    connection = _connection(prismatic_bot)
    connection.position = initial_position
    connection.velocity = 0.9

    limits = _compute_limits(prismatic_bot)
    upper_vel_bounds = limits.upper_bounds.evaluate()[:NUMBER_OF_VELOCITY_STEPS]

    final_position = initial_position + DT * np.sum(upper_vel_bounds)
    assert final_position <= POSITION_UPPER_LIMIT + 1e-3, (
        "Integrating the upper velocity bounds must not overshoot the upper position limit"
    )


def test_dof_limits_jerk_is_relaxed_when_braking_is_insufficient(prismatic_bot):
    connection = _connection(prismatic_bot)
    connection.position = 0.9
    connection.velocity = 0.9

    limits = _compute_limits(prismatic_bot)
    jerk_upper_at_t0 = limits.upper_bounds.evaluate()[NUMBER_OF_VELOCITY_STEPS]

    assert jerk_upper_at_t0 > EXPECTED_JERK_LIMIT, (
        "Jerk limit at t=0 must be relaxed when normal braking cannot prevent a position limit violation"
    )


def test_dof_limits_no_position_limits(prismatic_world_no_position_limits):
    limits = _compute_limits(prismatic_world_no_position_limits)
    upper_vel = limits.upper_bounds.evaluate()[:NUMBER_OF_VELOCITY_STEPS]
    lower_vel = limits.lower_bounds.evaluate()[:NUMBER_OF_VELOCITY_STEPS]

    assert np.allclose(upper_vel, VELOCITY_LIMIT, atol=1e-3), (
        "Without position limits, upper velocity bounds must be flat"
    )
    assert np.allclose(lower_vel, -VELOCITY_LIMIT, atol=1e-3), (
        "Without position limits, lower velocity bounds must be flat"
    )


# %% reaching a position limit

LIMIT_REACHING_TOLERANCE = 1e-3

STOPPED_SPEED = VELOCITY_LIMIT / 20
"""
Speed below which a joint driven to a position limit counts as stopped: far below its
velocity limit, yet above the speed at which a joint that stopped short creeps on.
"""


def test_joint_goal_at_upper_limit_is_reached(prismatic_bot):
    """
    A joint goal placed exactly on a position limit is reached, instead of the joint
    stopping short of the limit.
    """
    connection = _connection(prismatic_bot)
    upper = connection.dof.limits.upper.position

    executor = StatechartExecutor(
        context=StatechartContext(world=prismatic_bot),
        extensions=[MotionControl(qp_controller_config=_default_config())],
    )
    statechart = Statechart(context=executor.context)
    joint_goal = JointPositionList(
        goal_state=JointState.from_mapping({connection: upper}),
        threshold=LIMIT_REACHING_TOLERANCE,
    )
    statechart.add_node(joint_goal)
    end = EndMotion()
    statechart.add_node(end)
    end.start_condition = joint_goal.observes_true

    executor.compile(statechart=statechart)
    executor.tick_until_end()

    assert np.isclose(connection.position, upper, atol=LIMIT_REACHING_TOLERANCE)


def test_joint_driven_to_upper_limit_comes_to_rest_only_on_it(prismatic_bot):
    """
    A joint driven at full speed to a position limit brakes onto the limit, instead of
    coming to rest short of it and creeping the rest of the way.
    """
    upper = _connection(prismatic_bot).dof.limits.upper.position

    recording = JointMotionRecording.from_joint_goal(
        world=prismatic_bot,
        config=QPControllerConfig(target_frequency=TARGET_FREQUENCY),
        start_position=0.0,
        start_velocity=0.0,
        goal=upper,
        duration=timedelta(seconds=2),
    )

    speeds = np.abs(recording.velocities)
    has_moved = np.maximum.accumulate(speeds > STOPPED_SPEED)
    positions_once_stopped = recording.positions[has_moved & (speeds < STOPPED_SPEED)]
    assert positions_once_stopped.size > 0
    assert np.allclose(positions_once_stopped, upper, atol=LIMIT_REACHING_TOLERANCE)


# %% staying within a position limit

POSITION_LIMIT_TOLERANCE = 1e-6
"""
Distance past a position limit that the solver's tolerance on the velocity bounds can
add up to, far below a real overshoot.
"""

JERK_LIMIT_TOLERANCE = 1e-3
"""
Relative tolerance on the jerk limit, for the solver's tolerance on its bounds.
"""

CONTROL_FREQUENCIES_AND_BRAKING_TIMES = [
    pytest.param(20, timedelta(seconds=0.3), id="20Hz_braking_0.3s"),
    pytest.param(20, timedelta(seconds=1.0), id="20Hz_braking_1s"),
    pytest.param(100, timedelta(seconds=0.1), id="100Hz_braking_0.1s"),
    pytest.param(100, timedelta(seconds=0.3), id="100Hz_braking_0.3s"),
]


@dataclass(kw_only=True)
class JointMotionRecording:
    """
    Position and commanded velocity of the only joint of a world at the start of a joint
    goal and after every control cycle of it.
    """

    positions: np.ndarray
    """
    Position of the joint at the start and after every control cycle.
    """

    velocities: np.ndarray
    """
    Commanded velocity of the joint at the start and in every control cycle.
    """

    time_step: timedelta
    """
    Duration of a control cycle.
    """

    @classmethod
    def from_joint_goal(
        cls,
        *,
        world: World,
        config: QPControllerConfig,
        start_position: float,
        start_velocity: float,
        goal: float,
        duration: timedelta,
    ) -> JointMotionRecording:
        """
        Runs a joint goal on the only joint of ``world`` for ``duration`` and records
        it.

        :param world: World with a single controlled joint, which starts without
            acceleration.
        :param config: Controller configuration to run the joint goal with.
        :param start_position: Position the joint starts at.
        :param start_velocity: Velocity the joint starts moving at.
        :param goal: Position the joint goal asks for.
        :param duration: Time to run the joint goal for.
        :return: Recording of the joint during the joint goal.
        """
        connection = _connection(world)
        connection.position = start_position
        world.state[connection.dof.id].velocity = start_velocity
        executor = StatechartExecutor(
            context=StatechartContext(world=world),
            extensions=[MotionControl(qp_controller_config=config)],
        )
        statechart = Statechart(context=executor.context)
        statechart.add_node(
            JointPositionList(goal_state=JointState.from_mapping({connection: goal}))
        )
        executor.compile(statechart=statechart)
        positions, velocities = [start_position], [start_velocity]
        for _ in range(round(duration / config.control_time_step)):
            executor.tick()
            positions.append(connection.position)
            velocities.append(world.state[connection.dof.id].velocity)
        return cls(
            positions=np.array(positions),
            velocities=np.array(velocities),
            time_step=config.control_time_step,
        )

    @property
    def peak_jerk(self) -> float:
        """
        Largest jerk magnitude of the commanded velocities, starting from zero
        acceleration.
        """
        time_step = self.time_step.total_seconds()
        accelerations = np.diff(self.velocities, prepend=self.velocities[0]) / time_step
        return float(np.max(np.abs(np.diff(accelerations) / time_step)))


def _config_covering_braking(
    world: World, *, target_frequency: float, braking_time: timedelta
) -> QPControllerConfig:
    """
    :return: Configuration whose prediction horizon also covers a braking of the only
        joint of ``world`` with a jerk or acceleration limit of its own.
    """
    config = QPControllerConfig(
        target_frequency=target_frequency, braking_time=braking_time
    )
    velocity_limit = _connection(world).dof.limits.upper.velocity
    braking = JerkLimitedBraking(
        velocity_limit=velocity_limit,
        jerk_limit=_jerk_limit(world, config=config),
        time_step=config.control_time_step,
    )
    return QPControllerConfig(
        target_frequency=target_frequency,
        braking_time=braking_time,
        prediction_horizon=max(
            config.prediction_horizon,
            braking.number_of_steps + config.number_of_resting_steps,
        ),
    )


def _jerk_limit(world: World, *, config: QPControllerConfig) -> float:
    """
    :return: Jerk limit the controller keeps for the only joint of ``world``.
    """
    return (
        DegreeOfFreedomLimitProfiler(config)
        .resolve_limits(_connection(world).dof)
        .upper.jerk
    )


@pytest.mark.parametrize(
    "start_position, start_velocity",
    [
        pytest.param(-0.5, 0.0, id="from_rest"),
        pytest.param(0.85, VELOCITY_LIMIT, id="at_full_speed_near_the_limit"),
        pytest.param(0.99, 0.3, id="too_close_to_brake_in_time"),
    ],
)
@pytest.mark.parametrize(
    "target_frequency, braking_time", CONTROL_FREQUENCIES_AND_BRAKING_TIMES
)
def test_joint_never_passes_its_upper_limit(
    prismatic_bot, target_frequency, braking_time, start_position, start_velocity
):
    """
    A joint heading for a goal beyond its upper limit stays within the limit in every
    control cycle, even when it starts too close to brake within its jerk limit.
    """
    upper = _connection(prismatic_bot).dof.limits.upper.position

    recording = JointMotionRecording.from_joint_goal(
        world=prismatic_bot,
        config=QPControllerConfig(
            target_frequency=target_frequency, braking_time=braking_time
        ),
        start_position=start_position,
        start_velocity=start_velocity,
        goal=upper + 0.5,
        duration=timedelta(seconds=3) + 2 * braking_time,
    )

    assert np.max(recording.positions) <= upper + POSITION_LIMIT_TOLERANCE


@pytest.mark.parametrize(
    "target_frequency, braking_time", CONTROL_FREQUENCIES_AND_BRAKING_TIMES
)
def test_joint_approaching_its_limit_from_rest_keeps_its_jerk_limit(
    prismatic_bot, target_frequency, braking_time
):
    """
    A joint that approaches its upper limit from rest brakes onto it within its jerk
    limit, so the jerk limit is never relaxed.
    """
    config = QPControllerConfig(
        target_frequency=target_frequency, braking_time=braking_time
    )

    recording = JointMotionRecording.from_joint_goal(
        world=prismatic_bot,
        config=config,
        start_position=-0.5,
        start_velocity=0.0,
        goal=_connection(prismatic_bot).dof.limits.upper.position,
        duration=timedelta(seconds=3) + 2 * braking_time,
    )

    jerk_limit = _jerk_limit(prismatic_bot, config=config)
    assert recording.peak_jerk <= jerk_limit * (1 + JERK_LIMIT_TOLERANCE)


def test_joint_that_can_brake_before_its_limit_keeps_its_jerk_limit(
    prismatic_bot_with_acceleration_limit,
):
    """
    A joint heading for its upper limit from a distance it can still brake within keeps
    its jerk limit while it brakes, rather than stopping at once.

    Its acceleration limit lowers the jerk limit so far that braking from the start
    velocity takes most of the distance to the limit.
    """
    world = prismatic_bot_with_acceleration_limit
    config = _config_covering_braking(
        world, target_frequency=100, braking_time=timedelta(seconds=0.3)
    )

    recording = JointMotionRecording.from_joint_goal(
        world=world,
        config=config,
        start_position=0.85,
        start_velocity=0.3,
        goal=0.0,
        duration=timedelta(seconds=1),
    )

    jerk_limit = _jerk_limit(world, config=config)
    assert recording.peak_jerk <= jerk_limit * (1 + JERK_LIMIT_TOLERANCE)
