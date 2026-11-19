"""
Tests of where Giskard keeps the debug plots of its goals.
"""

import os
import time
from datetime import timedelta
from pathlib import Path

from giskardpy.middleware.ros2.plot_archive import (
    GoalPlotKind,
    PlotArchive,
    RunPlotDirectory,
)

# %% helpers


def write_plot(*, run: RunPlotDirectory, age: timedelta) -> Path:
    """
    :return: A plot of the first goal of `run`, last modified `age` ago.
    """
    plot = run.goal_plot_path(plot_kind=GoalPlotKind.TRAJECTORY, goal_id=0)
    plot.write_bytes(b"")
    modification_time = time.time() - age.total_seconds()
    os.utime(plot, (modification_time, modification_time))
    return plot


def age_directory(*, directory: Path, age: timedelta) -> None:
    """
    Set the modification time of `directory` to `age` ago.
    """
    modification_time = time.time() - age.total_seconds()
    os.utime(directory, (modification_time, modification_time))


# %% one directory per run


def test_every_run_gets_its_own_directory(tmp_path: Path):
    """
    Two servers started at once write their plots apart, so neither overwrites the
    other's.
    """
    archive = PlotArchive(path=tmp_path)

    first_run = archive.start_run()
    second_run = archive.start_run()

    assert first_run.goal_plot_path(
        plot_kind=GoalPlotKind.TRAJECTORY, goal_id=0
    ) != second_run.goal_plot_path(plot_kind=GoalPlotKind.TRAJECTORY, goal_id=0)


def test_each_plot_kind_gets_its_own_file(tmp_path: Path):
    """
    The trajectory and the Gantt chart of one goal do not overwrite each other.
    """
    run = PlotArchive(path=tmp_path).start_run()

    assert run.goal_plot_path(
        plot_kind=GoalPlotKind.TRAJECTORY, goal_id=0
    ) != run.goal_plot_path(plot_kind=GoalPlotKind.GANTT_CHART, goal_id=0)


# %% deleting old runs


def test_starting_a_run_deletes_runs_older_than_the_retention(tmp_path: Path):
    """
    A run whose plots and folders were all last changed before the retention is gone.
    """
    archive = PlotArchive(path=tmp_path)
    old_run = archive.start_run()
    expired_age = archive.retention + timedelta(days=1)
    plot = write_plot(run=old_run, age=expired_age)
    age_directory(directory=plot.parent, age=expired_age)
    age_directory(directory=old_run.path, age=expired_age)

    archive.start_run()

    assert not old_run.path.exists()


def test_a_run_with_a_recent_plot_is_kept(tmp_path: Path):
    """
    A run is judged by its newest plot, so a server that runs for longer than the
    retention keeps its recent plots.
    """
    archive = PlotArchive(path=tmp_path)
    long_run = archive.start_run()
    recent_plot = write_plot(run=long_run, age=timedelta(hours=1))
    age_directory(directory=long_run.path, age=archive.retention + timedelta(days=1))

    archive.start_run()

    assert recent_plot.exists()


def test_starting_a_run_in_a_missing_archive_creates_nothing(tmp_path: Path):
    """
    A run that never plots leaves nothing behind.
    """
    archive_path = tmp_path / "archive"

    PlotArchive(path=archive_path).start_run()

    assert not archive_path.exists()
