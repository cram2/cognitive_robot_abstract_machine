"""
Compensating the dead time between measuring a robot and the robot executing the command
computed from that measurement.
"""

from __future__ import annotations

from dataclasses import dataclass
from uuid import UUID

from giskardpy.middleware.ros2.command_publishing import CommandPublisher
from giskardpy.middleware.ros2.cycle_timing import CycleTiming
from semantic_digital_twin.adapters.ros.input_synchronization import (
    JointPositionMeasurement,
    JointStateInputSynchronizer,
)
from semantic_digital_twin.input_synchronization import WorldStateInputs

# %% dead time compensation


@dataclass(kw_only=True)
class DeadTimeCompensation:
    """
    Moves measured joint positions forward to where the robot will be once the command
    of the running cycle takes effect.

    Applied after all :attr:`inputs`, so that it overwrites what they wrote; each
    publisher decides how far its joints move.
    """

    inputs: WorldStateInputs
    """
    The inputs of the control loop; the latest measurements of its joint state inputs
    are moved forward, including inputs added after this compensation was created.
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
        """
        Overwrite the latest measured joint positions with their predictions.

        :return: Whether any position was overwritten.
        """
        wrote_any = False
        for measurement in self.latest_measurements():
            for publisher in self.command_publishers:
                wrote_any |= publisher.write_predicted_positions(
                    measurement=measurement, cycle_timing=self.cycle_timing
                )
        return wrote_any

    def latest_measurements(self) -> list[JointPositionMeasurement]:
        """
        The latest measurement of every joint state input, with the measurements sampled
        at the same time merged into one, as the inputs of the parts of a robot that
        share a topic report them.

        :return: One measurement per distinct measurement time.
        """
        positions_by_time: dict[float, dict[UUID, float]] = {}
        for synchronizer in self.inputs.synchronizers:
            if not isinstance(synchronizer, JointStateInputSynchronizer):
                continue
            measurement = synchronizer.latest_measurement
            if measurement is None:
                continue
            positions_by_time.setdefault(measurement.measurement_time, {}).update(
                measurement.positions
            )
        return [
            JointPositionMeasurement(
                measurement_time=measurement_time, positions=positions
            )
            for measurement_time, positions in positions_by_time.items()
        ]
