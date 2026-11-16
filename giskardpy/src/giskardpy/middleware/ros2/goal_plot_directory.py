"""
Where Giskard writes the debug plots of its goals.
"""

from __future__ import annotations

import tempfile
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path
from typing_extensions import Self


class GoalPlotKind(StrEnum):
    """
    The kinds of debug plot Giskard writes per goal, each named after the folder in the
    temporary directory that holds them.
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
class GoalPlotDirectory:
    """
    A directory that holds one plot per goal, named after the id of the goal.
    """

    path: Path
    """
    The directory.
    """

    @classmethod
    def from_plot_kind(cls, *, plot_kind: GoalPlotKind) -> Self:
        """
        :return: The folder in the temporary directory that holds plots of `plot_kind`.
        """
        return cls(path=Path(tempfile.gettempdir()) / plot_kind)

    def goal_plot_path(self, *, goal_id: int) -> Path:
        """
        :return: The pdf of the goal with id `goal_id`; its directory is created if
            missing.
        """
        self.path.mkdir(parents=True, exist_ok=True)
        return self.path / f"goal_{goal_id}.pdf"
