"""
Where Giskard keeps the debug plots of its goals.
"""

from __future__ import annotations

import getpass
import os
import shutil
import tempfile
import time
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import StrEnum
from pathlib import Path


class GoalPlotKind(StrEnum):
    """
    The kinds of debug plot Giskard writes per goal, each named after the folder that
    holds them.
    """

    TRAJECTORY = "trajectories"
    """
    How the degrees of freedom moved.
    """

    GANTT_CHART = "gantt_charts"
    """
    When each motion statechart node was active.
    """

    MOTION_STATECHART = "motion_statecharts"
    """
    The structure of the motion statechart.
    """


@dataclass(kw_only=True)
class RunPlotDirectory:
    """
    The directory that holds the debug plots of one Giskard run, one folder per
    :class:`GoalPlotKind` and one pdf per goal.
    """

    path: Path
    """
    The directory; it is created with the first plot.
    """

    def goal_plot_path(self, *, plot_kind: GoalPlotKind, goal_id: int) -> Path:
        """
        :return: The pdf of the `plot_kind` of the goal with id `goal_id`; its folder
            is created if missing.
        """
        directory = self.path / plot_kind
        directory.mkdir(parents=True, exist_ok=True)
        return directory / f"goal_{goal_id}.pdf"


@dataclass(kw_only=True)
class PlotArchive:
    """
    The directory of one user that holds a :class:`RunPlotDirectory` per Giskard run,
    and deletes the runs whose plots are all older than :attr:`retention`.
    """

    path: Path = field(
        default_factory=lambda: (
            Path(tempfile.gettempdir()) / f"giskard_{getpass.getuser()}"
        )
    )
    """
    The directory; it is created with the first plot.
    """

    retention: timedelta = timedelta(weeks=1)
    """
    How long the plots of a run are kept after its last plot was written.
    """

    def start_run(self) -> RunPlotDirectory:
        """
        Delete the expired runs and name the directory of a new one, distinct from that
        of every other run, also of another process.
        """
        self.delete_expired_runs()
        started = datetime.now().strftime("%Y-%m-%d_%H-%M-%S-%f")
        return RunPlotDirectory(path=self.path / f"{started}_{os.getpid()}")

    def delete_expired_runs(self) -> None:
        """
        Delete every run whose newest file or folder is older than :attr:`retention`.
        """
        if not self.path.is_dir():
            return
        expiry = time.time() - self.retention.total_seconds()
        for run in self.path.iterdir():
            if not run.is_dir():
                continue
            entries = [run, *run.rglob("*")]
            if max(entry.stat().st_mtime for entry in entries) < expiry:
                shutil.rmtree(run, ignore_errors=True)
