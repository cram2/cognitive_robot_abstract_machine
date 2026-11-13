"""
Configuration of the goal lifecycle of the Giskard motion server.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class GiskardServerConfig:
    """
    Configures the goal lifecycle of the Giskard motion server.
    """

    debug_mode: bool = False
    """
    Master switch for all debug output.
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
