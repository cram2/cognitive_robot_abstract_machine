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
from giskardpy.middleware.ros2.cycle_timing import CycleTiming
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

    inputs: WorldStateInputs
    """
    The inputs of the loop, :attr:`synchronizer` among them.
    """

    other_connection: ActiveConnection1DOF
    """
    A measured joint that :attr:`publisher` does not command.
    """

    publisher: JointGroupVelocityCommandPublisher
    """
    Commands the velocity of the joint.
    """

    cycle_timing: CycleTiming
    """
    When the running cycle started and how long the previous one took.
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
        self.synchronizer.latest_message = stamped_joint_state(
            positions={self.connection.name.name: position},
            measurement_time=measurement_time,
        )
        self.synchronizer.apply()

    def add_joint_state_input(
        self, *, positions: dict[str, float], measurement_time: float
    ) -> PendingJointStateSynchronizer:
        """
        Add another joint state input to the loop and let it write a measurement.

        :param positions: The measured position of every joint, by joint name.
        :param measurement_time: The time the positions were sampled at.
        :return: The added input.
        """
        synchronizer = PendingJointStateSynchronizer(
            world=self.world, node=background_node.get_node(), topic_name="joint_states"
        )
        self.inputs.add_input(synchronizer)
        synchronizer.latest_message = stamped_joint_state(
            positions=positions, measurement_time=measurement_time
        )
        synchronizer.apply()
        return synchronizer

    def predict(self, *, cycle_start_time: float, computation_time: float) -> bool:
        """
        Apply the compensation in a cycle that started at ``cycle_start_time``, after a
        cycle that took ``computation_time``.

        :return: Whether the compensation wrote a position.
        """
        self.cycle_timing.cycle_start_time = cycle_start_time
        self.cycle_timing.computation_time = computation_time
        return self.compensation.apply()


def build_compensated_joint(*, actuation_delay: timedelta | None) -> CompensatedJoint:
    """
    Build a single joint that is measured and commanded with the given actuation delay.
    """
    world = build_world_with_joints(
        [
            ConnectionSpec(name="joint_a", velocity=0.0),
            ConnectionSpec(name="joint_b", velocity=0.0),
        ]
    )
    connection = world.get_connection_by_name("joint_a")
    synchronizer = PendingJointStateSynchronizer(
        world=world, node=background_node.get_node(), topic_name="joint_states"
    )
    inputs = WorldStateInputs(world=world)
    inputs.add_input(synchronizer)
    publisher = JointGroupVelocityCommandPublisher(
        world=world,
        command_topic="test_cmd",
        connections=[connection],
        actuation_delay=actuation_delay,
    )
    cycle_timing = CycleTiming(clock=background_node.get_node().get_clock())
    return CompensatedJoint(
        world=world,
        connection=connection,
        synchronizer=synchronizer,
        inputs=inputs,
        other_connection=world.get_connection_by_name("joint_b"),
        publisher=publisher,
        cycle_timing=cycle_timing,
        compensation=DeadTimeCompensation(
            inputs=inputs, command_publishers=[publisher], cycle_timing=cycle_timing
        ),
    )


def stamped_joint_state(
    *, positions: dict[str, float], measurement_time: float
) -> JointStateMessage:
    """
    A joint state message reporting ``positions``, sampled at ``measurement_time``.
    """
    message = JointStateMessage()
    message.header.stamp.sec = int(measurement_time)
    message.header.stamp.nanosec = round(
        (measurement_time - int(measurement_time)) * 1e9
    )
    message.name = list(positions)
    message.position = list(positions.values())
    return message


# %% predicting positions


def test_the_position_moves_by_the_commands_executed_until_the_next_one_takes_effect(
    init_background_node,
):
    """
    The robot executes the commands published from one actuation delay before the
    measurement until the command of the running cycle is published.
    """
    actuation_delay = timedelta(seconds=0.3)
    joint = build_compensated_joint(actuation_delay=actuation_delay)
    joint.publisher.command_history.record(publish_time=9.0, velocities=[0.5])
    joint.measure(position=1.0, measurement_time=10.0)

    joint.predict(cycle_start_time=10.05, computation_time=0.05)

    executed_since_measurement = joint.cycle_timing.expected_publish_time - (
        10.0 - actuation_delay.total_seconds()
    )
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

    joint.predict(cycle_start_time=10.05, computation_time=0.05)
    predicted_once = joint.position
    joint.predict(cycle_start_time=10.05, computation_time=0.05)

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

    assert joint.predict(cycle_start_time=10.05, computation_time=0.05) is False
    assert joint.position == 1.0


def test_nothing_is_predicted_before_the_first_measurement(init_background_node):
    """
    Without a measurement there is nothing to move forward.
    """
    joint = build_compensated_joint(actuation_delay=timedelta(seconds=0.3))
    joint.publisher.command_history.record(publish_time=9.0, velocities=[0.5])
    position_before = joint.position

    assert joint.predict(cycle_start_time=10.05, computation_time=0.05) is False
    assert joint.position == position_before


def test_a_joint_without_published_commands_keeps_its_measurement(
    init_background_node,
):
    """
    A joint that was never commanded is resting where it was measured.
    """
    joint = build_compensated_joint(actuation_delay=timedelta(seconds=0.3))
    joint.measure(position=1.0, measurement_time=10.0)

    assert joint.predict(cycle_start_time=10.05, computation_time=0.05) is False
    assert joint.position == 1.0


# %% measurements that cannot be moved forward


def test_a_measurement_without_a_stamp_is_not_moved_forward(init_background_node):
    """
    A driver that leaves the header stamp at zero gives no time to compensate from.
    """
    joint = build_compensated_joint(actuation_delay=timedelta(seconds=0.3))
    joint.publisher.command_history.record(publish_time=9.0, velocities=[0.5])
    joint.measure(position=1.0, measurement_time=0.0)

    assert joint.predict(cycle_start_time=10.05, computation_time=0.05) is False
    assert joint.position == 1.0


def test_a_measurement_stamped_after_the_expected_publish_time_is_not_moved_forward(
    init_background_node,
):
    """
    A robot whose clock runs ahead reports measurements from the future.
    """
    joint = build_compensated_joint(actuation_delay=timedelta(seconds=0.3))
    joint.publisher.command_history.record(publish_time=9.0, velocities=[0.5])
    joint.measure(position=1.0, measurement_time=10.2)

    assert joint.predict(cycle_start_time=10.05, computation_time=0.05) is False
    assert joint.position == 1.0


def test_a_measurement_received_after_the_cycle_start_is_moved_forward(
    init_background_node,
):
    """
    A joint state may arrive while the cycle is already running.
    """
    actuation_delay = timedelta(seconds=0.3)
    joint = build_compensated_joint(actuation_delay=actuation_delay)
    joint.publisher.command_history.record(publish_time=9.0, velocities=[0.5])
    joint.measure(position=1.0, measurement_time=10.06)

    joint.predict(cycle_start_time=10.05, computation_time=0.05)

    executed_since_measurement = joint.cycle_timing.expected_publish_time - (
        10.06 - actuation_delay.total_seconds()
    )
    assert joint.position == pytest.approx(1.0 + 0.5 * executed_since_measurement)


def test_a_measurement_older_than_the_remembered_commands_is_not_moved_forward(
    init_background_node,
):
    """
    The commands executed since such a measurement are partly forgotten.
    """
    joint = build_compensated_joint(actuation_delay=timedelta(seconds=0.3))
    joint.publisher.command_history.record(publish_time=9.0, velocities=[0.5])
    cycle_start_time = 10.05
    joint.measure(
        position=1.0,
        measurement_time=cycle_start_time
        - joint.publisher.command_history_margin.total_seconds()
        - 0.01,
    )

    assert (
        joint.predict(cycle_start_time=cycle_start_time, computation_time=0.05) is False
    )
    assert joint.position == 1.0


def test_a_stale_measurement_of_a_commanded_joint_is_warned_about(
    init_background_node, monkeypatch
):
    """
    The warning names the publisher whose joints stay uncompensated.
    """
    joint = build_compensated_joint(actuation_delay=timedelta(seconds=0.3))
    joint.publisher.command_history.record(publish_time=9.0, velocities=[0.5])
    joint.measure(position=1.0, measurement_time=3.0)
    warnings = record_warnings(monkeypatch)

    joint.predict(cycle_start_time=10.05, computation_time=0.05)

    assert len(warnings) == 1
    assert joint.publisher.command_topic in warnings[0]


def test_a_stale_input_holding_none_of_the_commanded_joints_is_not_warned_about(
    init_background_node, monkeypatch
):
    """
    A gripper that publishes rarely does not report the compensation of the arm as
    failing.
    """
    joint = build_compensated_joint(actuation_delay=timedelta(seconds=0.3))
    joint.publisher.command_history.record(publish_time=9.0, velocities=[0.5])
    joint.measure(position=1.0, measurement_time=10.0)
    joint.add_joint_state_input(
        positions={joint.other_connection.name.name: 0.2}, measurement_time=3.0
    )
    warnings = record_warnings(monkeypatch)

    assert joint.predict(cycle_start_time=10.05, computation_time=0.05) is True
    assert warnings == []


def record_warnings(monkeypatch) -> list[str]:
    """
    Record the warnings of the background node instead of logging them.

    :return: The recorded warnings, filled as they are logged.
    """
    warnings = []
    monkeypatch.setattr(
        background_node.get_node().get_logger(),
        "warning",
        lambda message, **_: warnings.append(message),
    )
    return warnings


# %% the inputs that are moved forward


def test_a_joint_state_input_added_after_the_compensation_is_moved_forward(
    init_background_node,
):
    """
    The inputs are read every cycle, so a source added later is compensated as well.
    """
    actuation_delay = timedelta(seconds=0.3)
    joint = build_compensated_joint(actuation_delay=actuation_delay)
    joint.publisher.command_history.record(publish_time=9.0, velocities=[0.5])
    joint.add_joint_state_input(
        positions={joint.connection.name.name: 1.0}, measurement_time=10.0
    )

    joint.predict(cycle_start_time=10.05, computation_time=0.05)

    executed_since_measurement = joint.cycle_timing.expected_publish_time - (
        10.0 - actuation_delay.total_seconds()
    )
    assert joint.position == pytest.approx(1.0 + 0.5 * executed_since_measurement)


def test_inputs_sampled_at_the_same_time_are_moved_forward_by_one_displacement(
    init_background_node, monkeypatch
):
    """
    The parts of a robot that share a topic report the same stamp, so their joints move
    by the same commands.
    """
    joint = build_compensated_joint(actuation_delay=timedelta(seconds=0.3))
    publisher = JointGroupVelocityCommandPublisher(
        world=joint.world,
        command_topic="test_cmd_of_both_joints",
        connections=[joint.connection, joint.other_connection],
        actuation_delay=timedelta(seconds=0.3),
    )
    joint.compensation.command_publishers = [publisher]
    publisher.command_history.record(publish_time=9.0, velocities=[0.5, -0.5])
    joint.measure(position=1.0, measurement_time=10.0)
    joint.add_joint_state_input(
        positions={joint.other_connection.name.name: 2.0}, measurement_time=10.0
    )
    displacements = []
    displacement = publisher.command_history.displacement
    monkeypatch.setattr(
        publisher.command_history,
        "displacement",
        lambda **kwargs: displacements.append(kwargs) or displacement(**kwargs),
    )

    joint.predict(cycle_start_time=10.05, computation_time=0.05)

    assert len(displacements) == 1
    assert joint.position > 1.0
    assert joint.world.state[joint.other_connection.raw_dof.id].position < 2.0
