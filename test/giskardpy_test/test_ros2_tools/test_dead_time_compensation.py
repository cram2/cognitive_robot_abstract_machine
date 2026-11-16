"""
Tests for moving the measured joint positions forward to where the robot will be once
the next command takes effect.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import timedelta

import pytest
from sensor_msgs.msg import JointState as JointStateMessage

from giskardpy.middleware.ros2.command_publishing import (
    JointGroupVelocityCommandPublisher,
)
from giskardpy.middleware.ros2.dead_time_compensation import DeadTimeCompensation
from semantic_digital_twin.adapters.ros import background_node
from semantic_digital_twin.adapters.ros.input_synchronization import (
    PendingJointStateSynchronizer,
)
from semantic_digital_twin.input_synchronization import WorldStateInputs
from semantic_digital_twin.world import World
from semantic_digital_twin.world_description.connections import ActiveConnection1DOF

from .test_command_publishing import ConnectionSpec, build_world_with_joints

# %% a measured and commanded joint


@dataclass(kw_only=True)
class CompensatedJoint:
    """
    A joint whose position is measured, whose velocity is commanded, and whose
    measurement is compensated for the dead time of the loop.
    """

    world: World
    """
    The world holding the joint.
    """

    connection: ActiveConnection1DOF
    """
    The measured and commanded joint.
    """

    synchronizer: PendingJointStateSynchronizer
    """
    Writes the measured position of the joint.
    """

    publisher: JointGroupVelocityCommandPublisher
    """
    Commands the velocity of the joint.
    """

    compensation: DeadTimeCompensation
    """
    The compensation under test.
    """

    @property
    def position(self) -> float:
        """
        The position of the joint in the world state.
        """
        return self.world.state[self.connection.raw_dof.id].position

    def measure(self, *, position: float, measurement_time: float) -> None:
        """
        Write a measured position, sampled at ``measurement_time``, into the world.
        """
        message = JointStateMessage()
        message.header.stamp.sec = int(measurement_time)
        message.header.stamp.nanosec = round(
            (measurement_time - int(measurement_time)) * 1e9
        )
        message.name = [self.connection.name.name]
        message.position = [position]
        self.synchronizer.latest_message = message
        self.synchronizer.apply()


def build_compensated_joint(*, actuation_delay: timedelta | None) -> CompensatedJoint:
    """
    Build a single joint that is measured and commanded with the given actuation delay.
    """
    spec = ConnectionSpec(name="joint_a", velocity=0.0)
    world = build_world_with_joints([spec])
    connection = world.get_connection_by_name(spec.name)
    synchronizer = PendingJointStateSynchronizer(
        world=world, node=background_node.get_node(), topic_name="joint_states"
    )
    publisher = JointGroupVelocityCommandPublisher(
        world=world,
        command_topic="test_cmd",
        connections=[connection],
        actuation_delay=actuation_delay,
    )
    return CompensatedJoint(
        world=world,
        connection=connection,
        synchronizer=synchronizer,
        publisher=publisher,
        compensation=DeadTimeCompensation(
            world=world,
            inputs=WorldStateInputs(world=world, synchronizers=[synchronizer]),
            command_publishers=[publisher],
        ),
    )


# %% predicting positions


def test_the_position_moves_by_the_commands_executed_until_the_next_one_takes_effect(
    init_background_node,
):
    """
    The robot executes the commands published from one actuation delay before the
    measurement until the next command is published.
    """
    actuation_delay = timedelta(seconds=0.3)
    joint = build_compensated_joint(actuation_delay=actuation_delay)
    joint.publisher.command_history.record(publish_time=9.0, velocities=[0.5])
    joint.measure(position=1.0, measurement_time=10.0)

    joint.compensation.predict_positions(command_publish_time=10.1)

    executed_since_measurement = 10.1 - (10.0 - actuation_delay.total_seconds())
    assert joint.position == pytest.approx(1.0 + 0.5 * executed_since_measurement)


def test_predicting_twice_starts_both_times_from_the_measurement(
    init_background_node,
):
    """
    A prediction never builds on an earlier prediction.
    """
    joint = build_compensated_joint(actuation_delay=timedelta(seconds=0.3))
    joint.publisher.command_history.record(publish_time=9.0, velocities=[0.5])
    joint.measure(position=1.0, measurement_time=10.0)

    joint.compensation.predict_positions(command_publish_time=10.1)
    predicted_once = joint.position
    joint.compensation.predict_positions(command_publish_time=10.1)

    assert joint.position == pytest.approx(predicted_once)


def test_a_publisher_without_actuation_delay_leaves_the_measurement_alone(
    init_background_node,
):
    """
    Compensation is opt-in per publisher.
    """
    joint = build_compensated_joint(actuation_delay=None)
    joint.publisher.command_history.record(publish_time=9.0, velocities=[0.5])
    joint.measure(position=1.0, measurement_time=10.0)

    assert joint.compensation.predict_positions(command_publish_time=10.1) is False
    assert joint.position == 1.0


def test_nothing_is_predicted_before_the_first_measurement(init_background_node):
    """
    Without a measurement there is nothing to move forward.
    """
    joint = build_compensated_joint(actuation_delay=timedelta(seconds=0.3))
    joint.publisher.command_history.record(publish_time=9.0, velocities=[0.5])
    position_before = joint.position

    assert joint.compensation.predict_positions(command_publish_time=10.1) is False
    assert joint.position == position_before


def test_a_joint_without_published_commands_keeps_its_measurement(
    init_background_node,
):
    """
    A joint that was never commanded is resting where it was measured.
    """
    joint = build_compensated_joint(actuation_delay=timedelta(seconds=0.3))
    joint.measure(position=1.0, measurement_time=10.0)

    assert joint.compensation.predict_positions(command_publish_time=10.1) is False
    assert joint.position == 1.0
