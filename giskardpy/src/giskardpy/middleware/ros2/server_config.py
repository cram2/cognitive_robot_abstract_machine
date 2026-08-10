from __future__ import annotations

from dataclasses import dataclass
from enum import Enum, auto

from giskardpy.executor import NoPacing, Pacer, RealTimePacer
from giskardpy.middleware.ros2.utils.utils import is_in_github_workflow


class ExecutionMode(Enum):
    """
    Decides whether Giskard simulates the motion or commands the real robot.
    """

    STANDALONE = auto()
    """
    The motion is simulated in place, as fast as the hardware allows.
    """

    CLOSED_LOOP = auto()
    """
    The commands of the controller are sent to the robot in real time.
    """


@dataclass
class GiskardServerConfig:
    """
    Configures the goal lifecycle of the Giskard motion server.
    """

    execution_mode: ExecutionMode = ExecutionMode.STANDALONE
    """
    Whether the motion is simulated or executed on the robot.
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

    @property
    def is_standalone(self) -> bool:
        """
        Whether the motion is simulated in place.
        """
        return self.execution_mode == ExecutionMode.STANDALONE

    @property
    def is_closed_loop(self) -> bool:
        """
        Whether the commands of the controller are sent to the robot.
        """
        return self.execution_mode == ExecutionMode.CLOSED_LOOP

    def create_pacer(self) -> Pacer:
        """
        Build the pacer of the control loop.

        A simulated motion may run as fast as the hardware allows, while a real robot
        has to be commanded in real time, so a closed loop setup can never be sped up or
        slowed down.
        """
        if self.is_standalone:
            return NoPacing()
        return RealTimePacer()
