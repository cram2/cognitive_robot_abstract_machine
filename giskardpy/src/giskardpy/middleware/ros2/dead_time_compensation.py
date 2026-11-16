"""
Compensating the dead time between measuring a robot and the robot executing the command
computed from that measurement.
"""

from __future__ import annotations

from dataclasses import dataclass

from giskardpy.middleware.ros2.command_publishing import CommandPublisher
from giskardpy.middleware.ros2.cycle_timing import CycleTiming
from semantic_digital_twin.adapters.ros.input_synchronization import (
    JointStateInputSynchronizer,
)
from semantic_digital_twin.input_synchronization import InputSynchronizer

# %% dead time compensation


@dataclass(kw_only=True)
class DeadTimeCompensation(InputSynchronizer):
    """
    Moves measured joint positions forward to where the robot will be once the command
    of the running cycle takes effect.

    Applied after the joint states it moves forward, so that it overwrites what they
    wrote; each publisher decides how far its joints move.
    """

    joint_states: list[JointStateInputSynchronizer]
    """
    The inputs whose latest measurements are moved forward.
    """

    command_publishers: list[CommandPublisher]
    """
    The publishers whose commands move the robot.
    """

    cycle_timing: CycleTiming
    """
    When the command of the running cycle will be published.
    """

    def apply(self) -> bool:
        wrote_any = False
        for joint_state in self.joint_states:
            if joint_state.latest_measurement is None:
                continue
            for publisher in self.command_publishers:
                wrote_any |= publisher.write_predicted_positions(
                    measurement=joint_state.latest_measurement,
                    cycle_timing=self.cycle_timing,
                )
        return wrote_any
