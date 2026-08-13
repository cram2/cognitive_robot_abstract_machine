import numpy as np
import pytest

from giskardpy.executor import Executor
from giskardpy.motion_statechart.context import MotionStatechartContext
from giskardpy.motion_statechart.graph_node import EndMotion
from giskardpy.motion_statechart.motion_statechart import MotionStatechart
from giskardpy.motion_statechart.tasks.joint_tasks import JointPositionList, JointState
from giskardpy.qp.dof_limits import QuadraticProgramDegreeOfFreedomLimits
from giskardpy.qp.qp_controller_config import QPControllerConfig
from semantic_digital_twin.world import World
from semantic_digital_twin.world_description.connections import ActiveConnection

TARGET_FREQUENCY = 20
PREDICTION_HORIZON = 10
DT = 1 / TARGET_FREQUENCY
NUMBER_OF_VELOCITY_STEPS = PREDICTION_HORIZON - 2
EXPECTED_JERK_LIMIT = DT
POSITION_UPPER_LIMIT = 1.0
VELOCITY_LIMIT = 1.0


def test_joint_goal_inside_limits_reached(pr2_world_state_reset):
    connection = pr2_world_state_reset.get_connection_by_name("head_pan_joint")
    dof = connection.dof
    lower = dof.limits.lower.position
    upper = dof.limits.upper.position
    goal = 1.0

    msc = MotionStatechart()
    joint_goal = JointPositionList(
        goal_state=JointState.from_mapping({connection: goal})
    )
    msc.add_node(joint_goal)
    end = EndMotion()
    msc.add_node(end)
    end.start_condition = joint_goal.observation_variable

    kin_sim = Executor(MotionStatechartContext(world=pr2_world_state_reset))
    kin_sim.compile(motion_statechart=msc)
    kin_sim.tick_until_end()

    assert np.isclose(connection.position, goal, atol=0.01)
    assert lower <= connection.position <= upper


def test_joint_goal_clamped_to_upper_limit(pr2_world_state_reset):
    connection = pr2_world_state_reset.get_connection_by_name("head_pan_joint")
    dof = connection.dof
    upper = dof.limits.upper.position
    goal_beyond_limit = upper + 2.0

    msc = MotionStatechart()
    joint_goal = JointPositionList(
        goal_state=JointState.from_mapping({connection: goal_beyond_limit})
    )
    msc.add_node(joint_goal)
    end = EndMotion()
    msc.add_node(end)
    end.start_condition = joint_goal.observation_variable

    kin_sim = Executor(MotionStatechartContext(world=pr2_world_state_reset))
    kin_sim.compile(motion_statechart=msc)
    kin_sim.tick_until_end()

    assert np.isclose(connection.position, upper, atol=0.01)


def test_joint_goal_clamped_to_lower_limit(pr2_world_state_reset):
    connection = pr2_world_state_reset.get_connection_by_name("head_pan_joint")
    dof = connection.dof
    lower = dof.limits.lower.position
    goal_beyond_limit = lower - 2.0

    msc = MotionStatechart()
    joint_goal = JointPositionList(
        goal_state=JointState.from_mapping({connection: goal_beyond_limit})
    )
    msc.add_node(joint_goal)
    end = EndMotion()
    msc.add_node(end)
    end.start_condition = joint_goal.observation_variable

    kin_sim = Executor(MotionStatechartContext(world=pr2_world_state_reset))
    kin_sim.compile(motion_statechart=msc)
    kin_sim.tick_until_end()

    assert np.isclose(connection.position, lower, atol=0.01)


def test_joint_above_upper_limit_recovers(pr2_world_state_reset):
    connection = pr2_world_state_reset.get_connection_by_name("head_pan_joint")
    dof = connection.dof
    lower = dof.limits.lower.position
    upper = dof.limits.upper.position

    connection.position = upper + 0.5
    goal = 1.0

    msc = MotionStatechart()
    joint_goal = JointPositionList(
        goal_state=JointState.from_mapping({connection: goal})
    )
    msc.add_node(joint_goal)
    end = EndMotion()
    msc.add_node(end)
    end.start_condition = joint_goal.observation_variable

    kin_sim = Executor(MotionStatechartContext(world=pr2_world_state_reset))
    kin_sim.compile(motion_statechart=msc)
    kin_sim.tick_until_end()

    assert np.isclose(connection.position, goal, atol=0.01)
    assert lower <= connection.position <= upper + 0.01


def test_joint_below_lower_limit_recovers(pr2_world_state_reset):
    connection = pr2_world_state_reset.get_connection_by_name("head_pan_joint")
    dof = connection.dof
    lower = dof.limits.lower.position
    upper = dof.limits.upper.position

    connection.position = lower - 0.5
    goal = -1.0

    msc = MotionStatechart()
    joint_goal = JointPositionList(
        goal_state=JointState.from_mapping({connection: goal})
    )
    msc.add_node(joint_goal)
    end = EndMotion()
    msc.add_node(end)
    end.start_condition = joint_goal.observation_variable

    kin_sim = Executor(MotionStatechartContext(world=pr2_world_state_reset))
    kin_sim.compile(motion_statechart=msc)
    kin_sim.tick_until_end()

    assert np.isclose(connection.position, goal, atol=0.01)
    assert lower - 0.01 <= connection.position <= upper


def test_multiple_joints_outside_limits_recover(pr2_world_state_reset):
    head_pan = pr2_world_state_reset.get_connection_by_name("head_pan_joint")
    head_tilt = pr2_world_state_reset.get_connection_by_name("head_tilt_joint")
    upper_arm_roll = pr2_world_state_reset.get_connection_by_name(
        "r_upper_arm_roll_joint"
    )

    head_pan.position = head_pan.dof.limits.upper.position + 0.5
    head_tilt.position = head_tilt.dof.limits.upper.position + 0.3
    upper_arm_roll.position = upper_arm_roll.dof.limits.lower.position - 0.5

    goals = {
        head_pan: 0.0,
        head_tilt: 0.5,
        upper_arm_roll: 0.0,
    }

    msc = MotionStatechart()
    joint_goal = JointPositionList(goal_state=JointState.from_mapping(goals))
    msc.add_node(joint_goal)
    end = EndMotion()
    msc.add_node(end)
    end.start_condition = joint_goal.observation_variable

    kin_sim = Executor(MotionStatechartContext(world=pr2_world_state_reset))
    kin_sim.compile(motion_statechart=msc)
    kin_sim.tick_until_end()

    for conn, goal in goals.items():
        lower = conn.dof.limits.lower.position
        upper = conn.dof.limits.upper.position
        assert np.isclose(conn.position, goal, atol=0.01), f"{conn.name} not at goal"
        assert (
            lower - 0.01 <= conn.position <= upper + 0.01
        ), f"{conn.name} outside limits"


def _default_config() -> QPControllerConfig:
    return QPControllerConfig(
        target_frequency=TARGET_FREQUENCY, prediction_horizon=PREDICTION_HORIZON
    )


def _compute_limits(world: World) -> QuadraticProgramDegreeOfFreedomLimits:
    return QuadraticProgramDegreeOfFreedomLimits.create(
        world.active_degrees_of_freedom, qp_controller_config=_default_config()
    )


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

    assert (
        upper_vel[0] < 0.9
    ), "Upper velocity at t=0 must be reduced near the upper position limit"
    assert np.isclose(
        lower_vel[0], -VELOCITY_LIMIT, atol=1e-3
    ), "Lower velocity is unconstrained going away from the limit"


def test_dof_limits_joint_near_lower_limit(prismatic_bot):
    _connection(prismatic_bot).position = -0.9

    limits = _compute_limits(prismatic_bot)
    upper_vel = limits.upper_bounds.evaluate()[:NUMBER_OF_VELOCITY_STEPS]
    lower_vel = limits.lower_bounds.evaluate()[:NUMBER_OF_VELOCITY_STEPS]

    assert (
        lower_vel[0] > -0.9
    ), "Lower velocity at t=0 must be reduced (less negative) near the lower position limit"
    assert np.isclose(
        upper_vel[0], VELOCITY_LIMIT, atol=1e-3
    ), "Upper velocity is unconstrained going away from the limit"


def test_dof_limits_joint_above_upper_limit(prismatic_bot):
    _connection(prismatic_bot).position = 1.2

    limits = _compute_limits(prismatic_bot)
    upper_vel = limits.upper_bounds.evaluate()[:NUMBER_OF_VELOCITY_STEPS]

    assert (
        upper_vel[0] <= 0.0
    ), "Upper velocity must be non-positive when joint is above the upper limit"


def test_dof_limits_joint_below_lower_limit(prismatic_bot):
    _connection(prismatic_bot).position = -1.2

    limits = _compute_limits(prismatic_bot)
    lower_vel = limits.lower_bounds.evaluate()[:NUMBER_OF_VELOCITY_STEPS]

    assert (
        lower_vel[0] >= 0.0
    ), "Lower velocity must be non-negative when joint is below the lower limit"


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
    assert (
        final_position <= POSITION_UPPER_LIMIT + 1e-3
    ), "Integrating the upper velocity bounds must not overshoot the upper position limit"


def test_dof_limits_jerk_is_relaxed_when_braking_is_insufficient(prismatic_bot):
    connection = _connection(prismatic_bot)
    connection.position = 0.9
    connection.velocity = 0.9

    limits = _compute_limits(prismatic_bot)
    jerk_upper_at_t0 = limits.upper_bounds.evaluate()[NUMBER_OF_VELOCITY_STEPS]

    assert (
        jerk_upper_at_t0 > EXPECTED_JERK_LIMIT
    ), "Jerk limit at t=0 must be relaxed when normal braking cannot prevent a position limit violation"


def test_dof_limits_no_position_limits(prismatic_world_no_position_limits):
    limits = _compute_limits(prismatic_world_no_position_limits)
    upper_vel = limits.upper_bounds.evaluate()[:NUMBER_OF_VELOCITY_STEPS]
    lower_vel = limits.lower_bounds.evaluate()[:NUMBER_OF_VELOCITY_STEPS]

    assert np.allclose(
        upper_vel, VELOCITY_LIMIT, atol=1e-3
    ), "Without position limits, upper velocity bounds must be flat"
    assert np.allclose(
        lower_vel, -VELOCITY_LIMIT, atol=1e-3
    ), "Without position limits, lower velocity bounds must be flat"
