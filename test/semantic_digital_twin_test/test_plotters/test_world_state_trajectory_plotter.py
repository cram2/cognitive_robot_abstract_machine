import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pytest

from semantic_digital_twin.spatial_types.derivatives import Derivatives
from semantic_digital_twin.testing import world_setup
from semantic_digital_twin.world_description.world_state import WorldStateTrajectory
from semantic_digital_twin.world_description.world_state_trajectory_plotter import (
    WorldStateTrajectoryPlotter,
)


def _axes_width_in(ax: plt.Axes) -> float:
    """
    Return the drawable width of an axes in inches based on its position box.
    """
    fig = ax.figure
    fig.canvas.draw()
    bbox = ax.get_position()  # fraction of figure
    return bbox.width * fig.get_figwidth()


def _build_simple_trajectory(world_setup, duration_s: float = 3.0, dt: float = 0.1):
    world, l1, l2, bf, r1, r2 = world_setup

    # Start at a non-zero time to verify normalization in the plotter
    t0 = 10.0

    # Use the prismatic connection r1->r2 for a changing DOF
    prismatic = world.get_connection(r1, r2)
    dof_change = prismatic.dof_id

    # Create a second DOF that remains constant to test suppression behavior
    second = world.get_connection(l1, l2)
    dof_constant = second.dof_id

    # Offset initial position of the changing DOF to exercise center_positions
    world.state[dof_change].position = 5.0
    world.notify_state_change()

    traj = WorldStateTrajectory.from_world_state(world.state, t0)

    # Apply jerk command to first DOF only; keep others zero
    cmd = np.zeros(len(world.state))
    # First index in state is r1->r2 dof (by construction of test world), but be robust by using index map
    first_idx = traj._index[dof_change]
    cmd[first_idx] = 10.0

    steps = int(np.ceil(duration_s / dt))
    t = t0
    for _ in range(steps):
        t += dt
        world.apply_control_commands(cmd, dt, Derivatives.jerk)
        traj.append(world.state, t)

    return world, traj, dof_change, dof_constant, dt


def _render_plot(
    plotter: WorldStateTrajectoryPlotter, traj: WorldStateTrajectory, monkeypatch
):
    captured = {}

    # Avoid file output and figure closing during tests
    monkeypatch.setattr(plt, "savefig", lambda *args, **kwargs: None)
    monkeypatch.setattr(plt, "close", lambda *args, **kwargs: None)

    plotter.plot_trajectory(traj, "/dev/null")

    # Get the current figure created by the plotter
    fig = plt.gcf()
    captured["fig"] = fig
    captured["axes"] = fig.get_axes()
    return captured


def test_physical_width_and_time_normalization(world_setup, monkeypatch):
    world, traj, dof_change, dof_constant, dt = _build_simple_trajectory(
        world_setup, duration_s=5.0, dt=0.2
    )

    second_width_in_cm = 2.0
    plotter = WorldStateTrajectoryPlotter(
        derivatives_to_plot=[Derivatives.position, Derivatives.velocity],
        legend=False,
        second_width_in_cm=second_width_in_cm,
        subplot_height_in_cm=4.0,
    )

    rendered = _render_plot(plotter, traj, monkeypatch)
    axes = rendered["axes"]
    assert len(axes) == 2  # two stacked subplots

    # All subplots share the same x-limits; they should start at 0 (normalized time)
    x0, x1 = axes[-1].get_xlim()
    assert x0 == pytest.approx(0.0, abs=1e-9)
    expected_duration = traj.times[-1] - traj.times[0]
    assert x1 == pytest.approx(expected_duration, rel=1e-6, abs=1e-6)

    # Check the physical width in inches matches seconds * cm-per-second / 2.54
    fig = rendered["fig"]
    main_ax = axes[0]
    actual_w_in = _axes_width_in(main_ax)
    expected_w_in = expected_duration * (second_width_in_cm / 2.54)
    assert actual_w_in == pytest.approx(expected_w_in, rel=0.1, abs=0.1)


def test_subplot_titles_labels_and_shared_xlabel(world_setup, monkeypatch):
    _, traj, *_ = _build_simple_trajectory(world_setup, duration_s=2.0, dt=0.1)

    plotter = WorldStateTrajectoryPlotter(
        derivatives_to_plot=[
            Derivatives.position,
            Derivatives.velocity,
            Derivatives.acceleration,
        ],
        legend=False,
    )

    rendered = _render_plot(plotter, traj, monkeypatch)
    axes = rendered["axes"]
    assert len(axes) == 3

    # Titles should be derivative names; y-labels are derivative-specific
    titles = [ax.get_title() for ax in axes]
    assert titles[0].lower().find("position") >= 0
    assert titles[1].lower().find("velocity") >= 0
    assert titles[2].lower().find("acceleration") >= 0

    ylabels = [ax.get_ylabel() for ax in axes]
    # Expect non-empty labels and they may differ per derivative
    assert all(isinstance(lbl, str) and len(lbl) > 0 for lbl in ylabels)

    # Only the bottom subplot should carry the shared x label
    xlabels = [ax.get_xlabel() for ax in axes]
    assert xlabels[0] == ""
    assert xlabels[1] == ""
    assert xlabels[2] == "Time [s]"

    # Grid should be visible on both axes directions
    for ax in axes:
        assert any(gl.get_visible() for gl in ax.get_xgridlines())
        assert any(gl.get_visible() for gl in ax.get_ygridlines())


def test_center_positions_and_suppression_rules(world_setup, monkeypatch):
    world, traj, dof_change, dof_constant, dt = _build_simple_trajectory(
        world_setup, duration_s=1.0, dt=0.1
    )

    plotter = WorldStateTrajectoryPlotter(
        derivatives_to_plot=[Derivatives.position, Derivatives.velocity],
        legend=False,
        center_positions=True,
        sort_degrees_of_freedom=False,
        plot_constant_lines=False,
    )

    rendered = _render_plot(plotter, traj, monkeypatch)
    pos_ax, vel_ax = rendered["axes"]

    # For center_positions=True, the position series should be shifted by initial value
    # Extract first plotted line data and verify it has zero at time 0 approximately
    pos_lines = pos_ax.get_lines()
    assert len(pos_lines) >= 1
    t_data, y_data = pos_lines[0].get_xdata(), pos_lines[0].get_ydata()
    # The first y should be approx 0 after centering (time normalized to 0)
    # Note: Due to potential downsampling, check the minimum x index
    idx0 = int(np.argmin(t_data))
    assert y_data[idx0] == pytest.approx(0.0, abs=1e-6)

    # With plot_0_lines=False, DOFs that are always equal to their initial position
    # should be suppressed in the position subplot, and strictly zero series suppressed in velocity
    # Changing DOF should appear once; constant DOF should not
    assert len(pos_lines) == 1
    assert len(vel_ax.get_lines()) == 1
