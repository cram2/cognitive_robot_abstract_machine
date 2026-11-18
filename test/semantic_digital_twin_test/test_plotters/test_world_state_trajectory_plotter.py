"""
Tests of the plot of a world state trajectory.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pytest

from krrood_spatial_math.spatial_types import Vector3
from semantic_digital_twin import logger
from semantic_digital_twin.datastructures.prefixed_name import PrefixedName
from semantic_digital_twin.world import World
from semantic_digital_twin.world_description.connection_properties import (
    NormalizedTwist,
)
from semantic_digital_twin.world_description.connections import ScrewAxisConnection
from semantic_digital_twin.world_description.degree_of_freedom import (
    DegreeOfFreedom,
    DegreeOfFreedomLimits,
)
from semantic_digital_twin.world_description.derivatives import Derivatives
from semantic_digital_twin.testing import world_setup
from semantic_digital_twin.world_description.world_entity import Body
from semantic_digital_twin.world_description.world_state import WorldStateTrajectory
from semantic_digital_twin.world_description.world_state_trajectory_plotter import (
    TimeAxisTickSpacing,
    WorldStateTrajectoryPlotter,
)

# %% helpers


@dataclass(kw_only=True)
class RenderedPlot:
    """
    The figure a plotter drew, kept open for inspection.
    """

    figure: plt.Figure
    """
    The whole figure.
    """

    axes: list[plt.Axes]
    """
    The subplots of :attr:`figure`, top to bottom.
    """


def _axes_width_in_inches(axis: plt.Axes) -> float:
    """
    Return the drawable width of an axes in inches based on its position box.
    """
    figure = axis.figure
    figure.canvas.draw()
    position = axis.get_position()  # fraction of figure
    return position.width * figure.get_figwidth()


def _record_trajectory(
    world_setup, duration: float = 3.0, time_step: float = 0.1
) -> WorldStateTrajectory:
    """
    Record `duration` seconds of the test world, in which one degree of freedom moves
    and all others stay where they are; the recording starts at a time other than zero.
    """
    world, _, _, _, right_parent, right_child = world_setup

    # Start at a non-zero time to verify normalization in the plotter
    start_time = 10.0

    moving_degree_of_freedom_id = world.get_connection(right_parent, right_child).dof.id

    # Offset initial position of the moving DOF to exercise center_positions
    world.state[moving_degree_of_freedom_id].position = 5.0
    world.notify_state_change()

    trajectory = WorldStateTrajectory.from_world_state(world.state, start_time)

    # Apply a jerk command to the moving DOF only; keep the others zero
    command = np.zeros(len(world.state))
    command[trajectory._index[moving_degree_of_freedom_id]] = 10.0

    step_count = int(np.ceil(duration / time_step))
    time = start_time
    for _ in range(step_count):
        time += time_step
        world.apply_control_commands(command, time_step, Derivatives.jerk)
        trajectory.append(world.state, time)

    return trajectory


def _record_a_few_of_many_joints_moving(
    *, joints: int, moving: int
) -> WorldStateTrajectory:
    """
    Record a second of a world of `joints` prismatic joints named ``joint_00`` and up,
    of which only the last `moving` in name order move.
    """
    world = World()
    root = Body(name=PrefixedName("root"))
    degrees_of_freedom = []
    with world.modify_world():
        world.add_kinematic_structure_entity(root)
        for index in range(joints):
            degree_of_freedom = DegreeOfFreedom(
                name=PrefixedName(f"joint_{index:02d}"),
                limits=DegreeOfFreedomLimits.from_position_range_and_speed(
                    maximum_speed=1
                ),
            )
            world.add_degree_of_freedom(degree_of_freedom)
            world.add_connection(
                ScrewAxisConnection(
                    normalized_twist=NormalizedTwist.create_prismatic(),
                    parent=root,
                    child=Body(name=PrefixedName(f"link_{index:02d}")),
                    raw_dof=degree_of_freedom,
                    axis=Vector3.X(reference_frame=root),
                )
            )
            degrees_of_freedom.append(degree_of_freedom)
    trajectory = WorldStateTrajectory.from_world_state(world.state, 0.0)
    command = np.zeros(len(world.state))
    for degree_of_freedom in degrees_of_freedom[-moving:]:
        command[trajectory._index[degree_of_freedom.id]] = 1.0
    time_step = 0.1
    for step in range(1, 11):
        world.apply_control_commands(command, time_step, Derivatives.jerk)
        trajectory.append(world.state, step * time_step)
    return trajectory


def _render_plot(
    plotter: WorldStateTrajectoryPlotter,
    trajectory: WorldStateTrajectory,
    monkeypatch: pytest.MonkeyPatch,
) -> RenderedPlot:
    """
    Plot `trajectory` with `plotter` without writing a file.
    """
    # Avoid file output and figure closing during tests
    monkeypatch.setattr(plt, "savefig", lambda *args, **kwargs: None)
    monkeypatch.setattr(plt, "close", lambda *args, **kwargs: None)
    plotter.world_state_trajectory = trajectory
    plotter.plot_trajectory(file_name=Path("/dev/null"))

    figure = plt.gcf()
    return RenderedPlot(figure=figure, axes=figure.get_axes())


# %% layout


def test_physical_width_and_time_normalization(world_setup, monkeypatch):
    """
    The time axis starts at zero whenever the recording started, and one second is as
    wide as configured.
    """
    trajectory = _record_trajectory(world_setup, duration=5.0, time_step=0.2)

    second_width_in_cm = 2.0
    plotter = WorldStateTrajectoryPlotter(
        derivatives_to_plot=[Derivatives.position, Derivatives.velocity],
        legend=False,
        second_width_in_cm=second_width_in_cm,
        subplot_height_in_cm=4.0,
    )

    axes = _render_plot(plotter, trajectory, monkeypatch).axes
    assert len(axes) == 2  # two stacked subplots

    # All subplots share the same x-limits; they should start at 0 (normalized time)
    x_start, x_end = axes[-1].get_xlim()
    assert x_start == pytest.approx(0.0, abs=1e-9)
    expected_duration = trajectory.times[-1] - trajectory.times[0]
    assert x_end == pytest.approx(expected_duration, rel=1e-6, abs=1e-6)

    # Check the physical width in inches matches seconds * cm-per-second / 2.54
    actual_width = _axes_width_in_inches(axes[0])
    expected_width = expected_duration * (second_width_in_cm / 2.54)
    assert actual_width == pytest.approx(expected_width, rel=0.1, abs=0.1)


def test_subplot_titles_labels_and_shared_xlabel(world_setup, monkeypatch):
    """
    Each subplot is titled and labelled after its derivative, and only the bottom one
    carries the time label.
    """
    trajectory = _record_trajectory(world_setup, duration=2.0, time_step=0.1)

    plotter = WorldStateTrajectoryPlotter(
        derivatives_to_plot=[
            Derivatives.position,
            Derivatives.velocity,
            Derivatives.acceleration,
        ],
        legend=False,
    )

    axes = _render_plot(plotter, trajectory, monkeypatch).axes
    assert len(axes) == 3

    # Titles should be derivative names; y-labels are derivative-specific
    titles = [axis.get_title() for axis in axes]
    assert titles[0].lower().find("position") >= 0
    assert titles[1].lower().find("velocity") >= 0
    assert titles[2].lower().find("acceleration") >= 0

    y_labels = [axis.get_ylabel() for axis in axes]
    # Expect non-empty labels and they may differ per derivative
    assert all(isinstance(label, str) and len(label) > 0 for label in y_labels)

    # Only the bottom subplot should carry the shared x label
    x_labels = [axis.get_xlabel() for axis in axes]
    assert x_labels[0] == ""
    assert x_labels[1] == ""
    assert x_labels[2] == "Time [s]"

    # Grid should be visible on both axes directions
    for axis in axes:
        assert any(line.get_visible() for line in axis.get_xgridlines())
        assert any(line.get_visible() for line in axis.get_ygridlines())


def test_a_short_trajectory_is_widened_to_the_minimum_plot_width(
    world_setup, monkeypatch
):
    """
    A trajectory too short to fill :attr:`minimum_plot_width_in_cm` at
    :attr:`second_width_in_cm` gets wider seconds instead.
    """
    trajectory = _record_trajectory(world_setup, duration=3.1, time_step=0.1)
    plotter = WorldStateTrajectoryPlotter(legend=False)

    axes = _render_plot(plotter, trajectory, monkeypatch).axes

    assert _axes_width_in_inches(axes[0]) == pytest.approx(
        plotter.minimum_plot_width_in_cm / 2.54, rel=0.01
    )


@pytest.mark.parametrize(
    "duration, time_step, expected_spacing",
    [
        (3.1, 0.1, TimeAxisTickSpacing.ONE_SECOND),
        (6.1, 0.1, TimeAxisTickSpacing.TWO_SECONDS),
        (30.2, 0.2, TimeAxisTickSpacing.FIVE_SECONDS),
    ],
)
def test_the_time_axis_numbers_ticks_at_least_the_minimum_distance_apart(
    world_setup,
    monkeypatch,
    duration: float,
    time_step: float,
    expected_spacing: TimeAxisTickSpacing,
):
    """
    The time axis takes the finest spacing whose numbered ticks are at least
    :attr:`minimum_distance_between_numbered_ticks_in_cm` apart; every subplot draws
    the grid lines of that spacing, and those between the numbered ticks carry no number.
    """
    trajectory = _record_trajectory(world_setup, duration=duration, time_step=time_step)
    plotter = WorldStateTrajectoryPlotter(legend=False)
    recorded_duration = trajectory.times[-1] - trajectory.times[0]

    axes = _render_plot(plotter, trajectory, monkeypatch).axes

    np.testing.assert_allclose(
        axes[-1].get_xticks(),
        np.arange(
            0.0,
            recorded_duration,
            expected_spacing.seconds_between_numbered_ticks,
        ),
    )
    np.testing.assert_allclose(
        np.union1d(axes[-1].get_xticks(), axes[-1].get_xticks(minor=True)),
        np.arange(0.0, recorded_duration, expected_spacing.seconds_between_grid_lines),
    )
    for axis in axes:
        minor_ticks = axis.xaxis.get_minor_ticks()
        assert all(tick.gridline.get_visible() for tick in minor_ticks)
        assert all(tick.label1.get_text() == "" for tick in minor_ticks)


# %% which degrees of freedom are drawn


def test_center_positions_and_suppression_rules(world_setup, monkeypatch):
    """
    Centered positions start at zero, and degrees of freedom that never move are left
    out.
    """
    trajectory = _record_trajectory(world_setup, duration=1.0, time_step=0.1)

    plotter = WorldStateTrajectoryPlotter(
        derivatives_to_plot=[Derivatives.position, Derivatives.velocity],
        legend=False,
        center_positions=True,
        sort_degrees_of_freedom=False,
        plot_constant_lines=False,
    )

    position_axis, velocity_axis = _render_plot(plotter, trajectory, monkeypatch).axes

    # For center_positions=True, the position series should be shifted by initial value
    # Extract first plotted line data and verify it has zero at time 0 approximately
    position_lines = position_axis.get_lines()
    assert len(position_lines) >= 1
    times = position_lines[0].get_xdata()
    positions = position_lines[0].get_ydata()
    # The first y should be approx 0 after centering (time normalized to 0)
    # Note: Due to potential downsampling, check the minimum x index
    first_index = int(np.argmin(times))
    assert positions[first_index] == pytest.approx(0.0, abs=1e-6)

    # With plot_constant_lines=False, DOFs that are always equal to their initial
    # position are suppressed in the position subplot, and strictly zero series in
    # velocity; the moving DOF appears once, the constant DOF not at all
    assert len(position_lines) == 1
    assert len(velocity_axis.get_lines()) == 1


# %% the saved plot


def test_the_plot_logs_the_file_it_was_saved_to(world_setup, tmp_path, caplog):
    """
    Saving the plot tells where to find it.
    """
    plotter = WorldStateTrajectoryPlotter()
    plotter.world_state_trajectory = _record_trajectory(world_setup)
    file_name = tmp_path / "trajectory.pdf"

    with caplog.at_level(logging.INFO, logger=logger.name):
        plotter.plot_trajectory(file_name=file_name)

    assert file_name.exists()
    assert any(str(file_name) in record.getMessage() for record in caplog.records)


# %% telling the lines apart


def test_every_drawn_line_has_a_style_of_its_own(monkeypatch):
    """
    Degrees of freedom that never move take no style from those that are drawn, so in a
    world of many joints every drawn line still differs in color or line style.
    """
    moving = 15
    trajectory = _record_a_few_of_many_joints_moving(joints=60, moving=moving)
    plotter = WorldStateTrajectoryPlotter(legend=False)

    position_axis = _render_plot(plotter, trajectory, monkeypatch).axes[0]

    styles = {
        (line.get_color(), line.get_linestyle()) for line in position_axis.get_lines()
    }
    assert len(position_axis.get_lines()) == moving
    assert len(styles) == moving


# %% trajectories too short to plot


def test_a_trajectory_of_one_state_is_not_plotted(world_setup, tmp_path):
    """
    A single state spans no time, so there is no plot to draw.
    """
    world, *_ = world_setup
    plotter = WorldStateTrajectoryPlotter()
    plotter.reset(world.state)
    file_name = tmp_path / "trajectory.pdf"

    plotter.plot_trajectory(file_name=file_name)

    assert not plotter.has_trajectory_to_plot
    assert not file_name.exists()
