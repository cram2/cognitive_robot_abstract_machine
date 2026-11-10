"""
Tests for taking a snapshot of a stepped MuJoCo simulation and continuing from it again.
"""

from __future__ import annotations

from datetime import timedelta

import pytest

from semantic_digital_twin.adapters.multi_sim import MujocoSim
from semantic_digital_twin.exceptions import SnapshotDoesNotFitSimulationError

from ...pytest_environment import runs_in_continuous_integration
from .test_mujoco_servos import _pendulum_world, _servoed_pendulum_world

pytestmark = pytest.mark.skipif(
    not runs_in_continuous_integration(), reason="MuJoCo tests only run in CI"
)

# %% continuing from a snapshot


def test_a_restored_snapshot_puts_the_joint_back_where_it_was():
    pendulum = _servoed_pendulum_world()
    world = pendulum.world
    degree_of_freedom = pendulum.hinge.raw_dof.id
    simulation = MujocoSim(world=world, headless=True)
    simulation.start_stepped_simulation()
    try:
        world.state[degree_of_freedom].position = 0.5
        world.notify_state_change()
        simulation.step_simulation(timedelta(seconds=0.2))
        snapshot = simulation.take_snapshot()
        position_at_snapshot = simulation.simulator.get_joint_value("hinge").result
        world.state[degree_of_freedom].position = -0.5
        world.notify_state_change()
        simulation.step_simulation(timedelta(seconds=1))

        simulation.restore_snapshot(snapshot)

        restored_position = simulation.simulator.get_joint_value("hinge").result
    finally:
        simulation.stop_simulation()

    assert restored_position == position_at_snapshot
    assert world.state[degree_of_freedom].position == position_at_snapshot


def test_continuing_from_a_restored_snapshot_repeats_what_followed_it():
    """
    The joint's velocity and its servo's set point come back with the snapshot, so the
    physics after a restore runs exactly as it ran after the snapshot was taken.
    """
    pendulum = _servoed_pendulum_world()
    world = pendulum.world
    degree_of_freedom = pendulum.hinge.raw_dof.id
    simulation = MujocoSim(world=world, headless=True)
    simulation.start_stepped_simulation()
    try:
        world.state[degree_of_freedom].position = 0.5
        world.notify_state_change()
        simulation.step_simulation(timedelta(seconds=0.1))
        snapshot = simulation.take_snapshot()
        simulation.step_simulation(timedelta(seconds=0.3))
        first_continuation = simulation.simulator.get_joint_value("hinge").result
        world.state[degree_of_freedom].position = -0.5
        world.notify_state_change()
        simulation.step_simulation(timedelta(seconds=1))

        simulation.restore_snapshot(snapshot)
        simulation.step_simulation(timedelta(seconds=0.3))

        second_continuation = simulation.simulator.get_joint_value("hinge").result
    finally:
        simulation.stop_simulation()

    assert second_continuation == pytest.approx(first_continuation, abs=1e-9)


# %% snapshots of another simulation


def test_a_snapshot_of_a_differently_built_simulation_is_refused():
    servoed = MujocoSim(world=_servoed_pendulum_world().world, headless=True)
    unservoed = MujocoSim(world=_pendulum_world().world, headless=True)
    servoed.start_stepped_simulation()
    unservoed.start_stepped_simulation()
    try:
        snapshot = servoed.take_snapshot()
        with pytest.raises(SnapshotDoesNotFitSimulationError):
            unservoed.restore_snapshot(snapshot)
    finally:
        servoed.stop_simulation()
        unservoed.stop_simulation()
