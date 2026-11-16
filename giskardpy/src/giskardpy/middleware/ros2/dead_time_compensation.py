"""
Compensating the dead time between measuring a robot and the robot executing the command
computed from that measurement.
"""

from __future__ import annotations

from dataclasses import dataclass

from giskardpy.middleware.ros2.command_publishing import (
    CommandPublisher,
    JointGroupVelocityCommandPublisher,
)
from semantic_digital_twin.adapters.ros.input_synchronization import (
    JointPositionMeasurement,
    JointStateInputSynchronizer,
)
from semantic_digital_twin.input_synchronization import WorldStateInputs
from semantic_digital_twin.world import World

# %% dead time compensation


@dataclass(kw_only=True)
class DeadTimeCompensation:
    """
    Moves measured joint positions forward to where the robot will be once the next
    command takes effect, by adding how far the commands published until then move it.

    Only the joints of :class:`JointGroupVelocityCommandPublisher` instances with an
    :attr:`~giskardpy.middleware.ros2.command_publishing.JointGroupVelocityCommandPublisher.actuation_delay`
    are moved.
    """

    world: World
    """
    The world the measured positions are written into.
    """

    inputs: WorldStateInputs
    """
    The inputs whose joint state measurements are moved forward.
    """

    command_publishers: list[CommandPublisher]
    """
    The publishers whose commands move the robot.
    """

    def predict_positions(self, *, command_publish_time: float) -> bool:
        """
        Overwrite the measured position of every compensated joint with its position at
        the time the next command takes effect.

        :param command_publish_time: Seconds of the ROS clock at which the next command
            will be published.
        :return: Whether any position was overwritten.
        """
        wrote_any = False
        for measurement in self._joint_position_measurements():
            for publisher in self.command_publishers:
                wrote_any |= self._predict_positions_commanded_by(
                    publisher=publisher,
                    measurement=measurement,
                    command_publish_time=command_publish_time,
                )
        return wrote_any

    def _joint_position_measurements(self) -> list[JointPositionMeasurement]:
        """
        :return: The latest measurement of every joint state input that received one.
        """
        return [
            synchronizer.latest_measurement
            for synchronizer in self.inputs.synchronizers
            if isinstance(synchronizer, JointStateInputSynchronizer)
            and synchronizer.latest_measurement is not None
        ]

    def _predict_positions_commanded_by(
        self,
        *,
        publisher: CommandPublisher,
        measurement: JointPositionMeasurement,
        command_publish_time: float,
    ) -> bool:
        """
        Overwrite the measured positions of the joints one publisher commands.

        The robot executes every command one actuation delay after it was published, so
        from the measurement until the next command takes effect it executes the
        commands published from one actuation delay before the measurement on.

        :param publisher: The publisher whose commands move the joints.
        :param measurement: The measured positions to move forward.
        :param command_publish_time: Seconds of the ROS clock at which the next command
            will be published.
        :return: Whether any position was overwritten.
        """
        if not isinstance(publisher, JointGroupVelocityCommandPublisher):
            return False
        if publisher.actuation_delay is None:
            return False
        if not publisher.command_history.held_commands:
            return False
        displacement = publisher.command_history.displacement(
            start_time=measurement.measurement_time
            - publisher.actuation_delay.total_seconds(),
            end_time=command_publish_time,
        )
        wrote_any = False
        for connection, distance in zip(publisher.connections, displacement):
            measured_position = measurement.positions.get(connection.raw_dof.id)
            if measured_position is None:
                continue
            self.world.state[connection.raw_dof.id].position = (
                measured_position + distance
            )
            wrote_any = True
        return wrote_any
