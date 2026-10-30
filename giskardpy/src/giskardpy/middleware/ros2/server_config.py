from __future__ import annotations

from dataclasses import dataclass

from giskardpy.middleware.ros2.utils.utils import is_in_github_workflow


@dataclass
class GiskardServerConfig:
    """
    Configures the goal lifecycle of the Giskard motion server.
    """

    debug_mode: bool = False
    """
    Master switch for all debug output; disabled in github workflows.
    """

    plot_trajectory: bool = False
    """
    Plot the trajectory of every goal, requires ``debug_mode``.
    """

    plot_gantt_chart: bool = False
    """
    Plot when each motion statechart node was active, requires ``debug_mode``.
    """

    plot_motion_statechart: bool = False
    """
    Draw the structure of every executed motion statechart, requires ``debug_mode``.
    """

    idle_frequency: float = 20.0
    """
    Frequency in hertz at which Giskard waits for goals.
    """

    def __post_init__(self):
        if is_in_github_workflow():
            self.debug_mode = False
