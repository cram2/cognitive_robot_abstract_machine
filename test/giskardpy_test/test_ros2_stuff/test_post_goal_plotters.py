import pytest

from cramph.exceptions import MissingExecutorExtensionError
from giskardpy.motion_control import WorldStateTrajectoryRecording
from giskardpy.middleware.ros2.post_goal_plotters import (
    GoalGanttChartPlotter,
    GoalTrajectoryPlotter,
)

from .test_motion_server import create_executor

# %% recording is opt in


def test_creating_a_trajectory_plotter_does_not_record_yet():
    executor = create_executor()

    GoalTrajectoryPlotter(executor=executor)

    with pytest.raises(MissingExecutorExtensionError):
        executor.require_extension(WorldStateTrajectoryRecording)


def test_start_recording_hands_the_trajectory_plotter_to_the_executor():
    executor = create_executor()
    plotter = GoalTrajectoryPlotter(executor=executor)

    plotter.start_recording()

    assert (
        executor.require_extension(WorldStateTrajectoryRecording).plotter
        is plotter.trajectory_plotter
    )


def test_a_plotter_without_own_recording_leaves_the_executor_alone():
    executor = create_executor()

    GoalGanttChartPlotter(executor=executor).start_recording()

    with pytest.raises(MissingExecutorExtensionError):
        executor.require_extension(WorldStateTrajectoryRecording)
