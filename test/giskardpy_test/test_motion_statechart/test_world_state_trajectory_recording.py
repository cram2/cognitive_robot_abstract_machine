"""
Tests of switching the world state trajectory recording on and off.
"""

from __future__ import annotations

import pytest

from cramph.context import StatechartContext
from cramph.executor import StatechartExecutor
from cramph.nodes_for_testing import ConstTrueNode
from cramph.statechart import Statechart
from giskardpy.motion_control import MotionControl, WorldStateTrajectoryRecording
from giskardpy.qp.qp_controller_config import QPControllerConfig
from semantic_digital_twin.world import World

# %% helpers


def compile_and_tick(
    *, recording: WorldStateTrajectoryRecording, ticks: int
) -> StatechartExecutor:
    """
    Compile a statechart in an empty world with `recording` after motion control and
    tick it `ticks` times.

    :return: The executor that ticked.
    """
    executor = StatechartExecutor(
        context=StatechartContext(world=World()),
        extensions=[
            MotionControl(
                qp_controller_config=QPControllerConfig.create_with_simulation_defaults()
            ),
            recording,
        ],
    )
    statechart = Statechart(context=executor.context)
    statechart.add_node(node=ConstTrueNode())
    executor.compile(statechart=statechart)
    for _ in range(ticks):
        executor.tick()
    return executor


# %% switching the recording


def test_a_recording_that_is_off_keeps_only_the_state_of_the_compile():
    """
    While switched off, ticking appends nothing to the trajectory the compile started.
    """
    recording = WorldStateTrajectoryRecording(records_trajectory=False)

    compile_and_tick(recording=recording, ticks=3)

    assert len(recording.plotter.world_state_trajectory.times) == 1


def test_a_recording_that_is_on_records_every_tick():
    """
    While switched on, every tick appends its world state.
    """
    recording = WorldStateTrajectoryRecording()
    ticks = 3

    compile_and_tick(recording=recording, ticks=ticks)

    assert len(recording.plotter.world_state_trajectory.times) == 1 + ticks


@pytest.mark.parametrize("records_trajectory", [True, False])
def test_compiling_drops_the_trajectory_of_the_previous_statechart(
    *, records_trajectory: bool
):
    """
    A compile starts a new trajectory whether or not the recording is switched on.
    """
    recording = WorldStateTrajectoryRecording(records_trajectory=True)
    executor = compile_and_tick(recording=recording, ticks=3)
    recording.records_trajectory = records_trajectory
    statechart = Statechart(context=executor.context)
    statechart.add_node(node=ConstTrueNode())

    executor.compile(statechart=statechart)

    assert len(recording.plotter.world_state_trajectory.times) == 1
