"""
When a control cycle started and how long cycles take to publish their commands.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from rclpy.clock import Clock

from semantic_digital_twin.adapters.ros.background_node import ros_time_seconds

# %% cycle timing


@dataclass(kw_only=True)
class CycleTiming:
    """
    The start of the running control cycle and the computation time of the previous one,
    on the ROS clock that header stamps are comparable with.

    The only clock the control loop reads, so the times it stamps commands with and
    compares measurements against cannot drift apart.
    """

    clock: Clock
    """
    The ROS clock the cycle is timed with.
    """

    cycle_start_time: float = field(init=False, default=0.0)
    """
    Seconds of :attr:`clock` at which the running cycle started.
    """

    computation_time: float = field(init=False, default=0.0)
    """
    Seconds of :attr:`clock` the previous cycle took from its start to publishing its
    commands.
    """

    @property
    def expected_publish_time(self) -> float:
        """
        Seconds of :attr:`clock` at which the running cycle is expected to publish its
        commands, assuming it takes as long as the previous one.
        """
        return self.cycle_start_time + self.computation_time

    def now(self) -> float:
        """
        :return: The current seconds of :attr:`clock`.
        """
        return ros_time_seconds(self.clock.now())

    def start_cycle(self) -> None:
        """
        Mark the start of a cycle.
        """
        self.cycle_start_time = self.now()

    def finish_cycle(self) -> float:
        """
        Mark that the running cycle publishes its commands now.

        A clock that jumped back during the cycle, as a restarted simulation's does,
        counts as no computation time.

        :return: Seconds of :attr:`clock` at which the commands are published.
        """
        publish_time = self.now()
        self.computation_time = max(publish_time - self.cycle_start_time, 0.0)
        return publish_time
