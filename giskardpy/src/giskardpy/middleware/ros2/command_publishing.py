"""
Publishes commanded velocities from the world state to the robot's hardware interfaces.

Throughout this module, *joint* refers to the real robot's hardware actuators and the
ROS topics/parameters used to command them, while *connection* refers to our internal
kinematic representation (:class:`~semantic_digital_twin.world_description.connections.ActiveConnection1DOF`
and related types). A connection is read from the world state and translated into a
velocity published to a joint.
"""

from __future__ import annotations

import math
from abc import ABC, abstractmethod
from collections import deque
from dataclasses import dataclass, field
from datetime import timedelta

import numpy as np
from typing_extensions import TYPE_CHECKING, Self, TypeVar, Generic, Union

from control_msgs.msg import MultiDOFCommand
from geometry_msgs.msg import Twist, TwistStamped
from rclpy.publisher import Publisher
from std_msgs.msg import Float64MultiArray

from krrood.patterns.subclass_safe_generic import SubClassSafeGeneric
from semantic_digital_twin.adapters.ros import background_node
from giskardpy.middleware.ros2.exceptions import UnknownMinimumVelocityJointError
from semantic_digital_twin.world import World
from semantic_digital_twin.world_description.connections import (
    ActiveConnection1DOF,
    DifferentialDrive,
    OmniDrive,
)
from semantic_digital_twin.world_description.degree_of_freedom import DegreeOfFreedom

if TYPE_CHECKING:
    from giskardpy.middleware.ros2.cycle_timing import CycleTiming
    from semantic_digital_twin.adapters.ros.input_synchronization import (
        JointPositionMeasurement,
    )

# %% minimum velocities


@dataclass(kw_only=True)
class MinimumVelocity:
    """
    Smallest velocity magnitude a hardware interface still reacts to.
    """

    magnitude: float = 0.0
    """
    Velocities below this magnitude are increased to it. ``0.0`` disables the minimum.
    """

    def enforce_on_vector(self, *, velocities: list[float]) -> list[float]:
        """
        Scale a velocity vector up to :attr:`magnitude`, keeping its direction.

        :param velocities: The velocity vector to scale.
        :return: The scaled velocity vector.
        """
        norm = math.hypot(*velocities)
        if not 0.0 < norm < self.magnitude:
            return velocities
        scale = self.magnitude / norm
        return [velocity * scale for velocity in velocities]

    def enforce_on_scalar(self, *, velocity: float) -> float:
        """
        Increase a single velocity to :attr:`magnitude`, keeping its sign.

        :param velocity: The velocity to increase.
        :return: The velocity, raised to the minimum magnitude if necessary.
        """
        if 0.0 < velocity < self.magnitude:
            return self.magnitude
        if -self.magnitude < velocity < 0.0:
            return -self.magnitude
        return velocity


@dataclass(kw_only=True)
class JointMinimumVelocity(MinimumVelocity):
    """
    Minimum velocity of a single joint, overriding the default of its publisher.
    """

    joint_name: str = field(kw_only=True)
    """
    Name of the joint this minimum applies to.
    """


@dataclass(kw_only=True)
class JointMinimumVelocities:
    """
    Minimum velocities of the joints a publisher commands.
    """

    default: MinimumVelocity = field(default_factory=MinimumVelocity)
    """
    Minimum applied to every joint without an override.
    """

    overrides: list[JointMinimumVelocity] = field(default_factory=list)
    """
    Minimums of individual joints, e.g. to exempt hardware without a velocity deadband.
    """

    @classmethod
    def from_magnitudes(
        cls, *, default: float = 0.0, overrides: dict[str, float] | None = None
    ) -> Self:
        """
        Build minimum velocities from a default magnitude and per-joint magnitudes.

        :param default: Minimum magnitude applied to every joint without an override.
        :param overrides: Minimum magnitudes of individual joints, keyed by joint name.
        :return: The built minimum velocities.
        """
        return cls(
            default=MinimumVelocity(magnitude=default),
            overrides=[
                JointMinimumVelocity(magnitude=magnitude, joint_name=joint_name)
                for joint_name, magnitude in (overrides or {}).items()
            ],
        )

    def of(self, *, connection: ActiveConnection1DOF) -> MinimumVelocity:
        """
        Minimum velocity that applies to the given connection.

        :param connection: The connection to look up the minimum velocity for.
        :return: The minimum velocity that applies to the connection.
        """
        for override in self.overrides:
            if override.joint_name == connection.name.name:
                return override
        return self.default

    def validate_overrides_apply_to(
        self, *, connections: list[ActiveConnection1DOF]
    ) -> None:
        """
        Make sure every override is used by one of the given connections.

        An override for a joint that is not commanded reads like that joint is exempt,
        while the hardware keeps receiving the default minimum.

        :param connections: The connections that are actually commanded.
        :raises UnknownMinimumVelocityJointError: If an override applies to no
            connection.
        :return: None
        """
        commanded_joint_names = [connection.name.name for connection in connections]
        for override in self.overrides:
            if override.joint_name in commanded_joint_names:
                continue
            raise UnknownMinimumVelocityJointError(
                joint_name=override.joint_name,
                commanded_joint_names=commanded_joint_names,
            )


# %% reading the commanded velocities


@dataclass(kw_only=True)
class StateVelocityReader:
    """
    Reads the velocities of a fixed set of degrees of freedom.

    Where the degrees of freedom live in the state data is resolved once and refreshed
    whenever the world model changes, so that a control cycle needs a single read of the
    state instead of one lookup per degree of freedom.
    """

    world: World
    """
    The world holding the commanded velocities.
    """

    degrees_of_freedom: list[DegreeOfFreedom]
    """
    The degrees of freedom to read, in the order the velocities are returned in.
    """

    columns: list[int] = field(init=False, default_factory=list)
    """
    Column of every degree of freedom in the state data.
    """

    model_version: int = field(init=False, default=-1)
    """
    Model version the columns were resolved for.
    """

    def velocities(self) -> list[float]:
        """
        The current velocity of every degree of freedom.

        :return: The velocities, in the order :attr:`degrees_of_freedom` lists them.
        """
        if self.model_version != self.world.get_world_model_manager().version:
            self.bind_to_state()
        state_velocities = self.world.state.velocities.tolist()
        return [state_velocities[column] for column in self.columns]

    def bind_to_state(self) -> None:
        """
        Resolve where the degrees of freedom live in the state data.
        """
        self.columns = self.world.state.column_indices(self.degrees_of_freedom)
        self.model_version = self.world.get_world_model_manager().version


@dataclass(kw_only=True)
class JointVelocityCommand:
    """
    The velocity to send for a single joint.
    """

    connection: ActiveConnection1DOF
    """
    The commanded connection.
    """

    minimum_velocity: MinimumVelocity
    """
    Minimum velocity of this joint.
    """

    def from_state_velocity(self, *, state_velocity: float) -> float:
        """
        Scale a raw state velocity to the joint and raise it to the minimum.

        :param state_velocity: The raw velocity read from the world state.
        :return: The velocity to send to the joint's hardware interface.
        """
        return self.minimum_velocity.enforce_on_scalar(
            velocity=state_velocity * self.connection.multiplier
        )

    def executed_state_velocity(self, *, sent_velocity: float) -> float:
        """
        The raw state velocity the joint moves at while it executes a sent velocity.

        :param sent_velocity: The velocity sent to the joint's hardware interface.
        :return: The velocity of the raw degree of freedom.
        """
        return sent_velocity / self.connection.multiplier

    @classmethod
    def for_connections(
        cls,
        *,
        connections: list[ActiveConnection1DOF],
        minimum_velocities: JointMinimumVelocities,
    ) -> list[Self]:
        """
        Resolve the minimum velocity of every connection once.

        :param connections: The connections to build commands for.
        :param minimum_velocities: The minimum velocities to resolve per connection.
        :raises UnknownMinimumVelocityJointError: If an override applies to no
            connection.
        :return: One command per connection, in the same order.
        """
        minimum_velocities.validate_overrides_apply_to(connections=connections)
        return [
            cls(
                connection=connection,
                minimum_velocity=minimum_velocities.of(connection=connection),
            )
            for connection in connections
        ]


# %% remembering the published commands


@dataclass(kw_only=True)
class HeldCommand:
    """
    Velocities a publisher sent, which the robot holds until the next command.
    """

    publish_time: float
    """
    Seconds of the ROS clock at which the command was published.
    """

    velocities: np.ndarray
    """
    The velocity of every commanded degree of freedom.
    """


@dataclass(kw_only=True)
class CommandHistory:
    """
    The commands a publisher sent recently, oldest first.

    Before the oldest remembered command, the commanded degrees of freedom count as
    resting.
    """

    kept_duration: timedelta
    """
    How long a command is remembered after the next one replaced it.
    """

    held_commands: deque[HeldCommand] = field(init=False, default_factory=deque)
    """
    The remembered commands, oldest first.
    """

    def record(self, *, publish_time: float, velocities: list[float]) -> None:
        """
        Remember a command and forget the ones replaced longer than
        :attr:`kept_duration` ago.

        :param publish_time: Seconds of the ROS clock at which the command was
            published.
        :param velocities: The velocity of every commanded degree of freedom.
        """
        self.held_commands.append(
            HeldCommand(publish_time=publish_time, velocities=np.array(velocities))
        )
        oldest_needed_time = publish_time - self.kept_duration.total_seconds()
        while (
            len(self.held_commands) > 1
            and self.held_commands[1].publish_time < oldest_needed_time
        ):
            self.held_commands.popleft()

    def displacement(self, *, start_time: float, end_time: float) -> np.ndarray:
        """
        How far the commands move every degree of freedom between two times, each
        command being held until the next one and the newest until ``end_time``.

        :param start_time: Seconds of the ROS clock the displacement starts at.
        :param end_time: Seconds of the ROS clock the displacement ends at.
        :return: The displacement of every commanded degree of freedom.
        """
        displacement = np.zeros_like(self.held_commands[-1].velocities)
        held_until = end_time
        for command in reversed(self.held_commands):
            held_since = max(command.publish_time, start_time)
            if held_since < held_until:
                displacement += command.velocities * (held_until - held_since)
            if command.publish_time <= start_time:
                break
            held_until = min(held_until, command.publish_time)
        return displacement


# %% message formats

CommandMessage = TypeVar("CommandMessage")
"""
The ROS message type a controller consumes.
"""


@dataclass(kw_only=True)
class CommandFormat(Generic[CommandMessage], SubClassSafeGeneric, ABC):
    """
    The message format a controller consumes.

    Each subclass binds the message type it builds, so the type a format publishes and
    the type it packs are the same declaration.
    """

    @classmethod
    def message_type(cls) -> type[CommandMessage]:
        """
        The ROS message type published on the command topic.
        """
        return cls.get_generic_type_parameters()[0]


# %% message formats of group controllers


@dataclass(kw_only=True)
class GroupCommandFormat(CommandFormat[CommandMessage], ABC):
    """
    Builds the message a joint group velocity controller consumes from the velocities of
    the joints it commands.
    """

    @abstractmethod
    def create_message(self, *, velocities: list[float]) -> CommandMessage:
        """
        Pack the velocities into a fresh message of :meth:`message_type`.

        :param velocities: The velocity of every commanded joint, in the order the
            controller expects them.
        :return: The message to publish.
        """


@dataclass(kw_only=True)
class Float64MultiArrayFormat(GroupCommandFormat[Float64MultiArray]):
    """
    Commands a controller that consumes :class:`std_msgs.msg.Float64MultiArray`.
    """

    def create_message(self, *, velocities: list[float]) -> Float64MultiArray:
        message = Float64MultiArray()
        message.data = velocities
        return message


@dataclass(kw_only=True)
class MultiDOFCommandFormat(GroupCommandFormat[MultiDOFCommand]):
    """
    Commands a controller that consumes :class:`control_msgs.msg.MultiDOFCommand`.

    ..note:: ``dof_names`` is left empty, so the controller applies the values in its own
        joint order.
    """

    def create_message(self, *, velocities: list[float]) -> MultiDOFCommand:
        message = MultiDOFCommand()
        message.values = velocities
        return message


# %% message formats of drives


@dataclass(kw_only=True)
class DriveCommandFormat(CommandFormat[CommandMessage], ABC):
    """
    Builds the message a drive controller consumes from the twist the drive is
    commanded.
    """

    @abstractmethod
    def create_message(self, *, twist: Twist, frame_id: str) -> CommandMessage:
        """
        :param twist: The commanded twist.
        :param frame_id: The frame the twist is expressed in.
        :return: The message to publish.
        """


@dataclass(kw_only=True)
class TwistFormat(DriveCommandFormat[Twist]):
    """
    Commands a drive controller that consumes :class:`geometry_msgs.msg.Twist`.
    """

    def create_message(self, *, twist: Twist, frame_id: str) -> Twist:
        return twist


@dataclass(kw_only=True)
class TwistStampedFormat(DriveCommandFormat[TwistStamped]):
    """
    Commands a drive controller that consumes :class:`geometry_msgs.msg.TwistStamped`,
    stamped with the current time.
    """

    def create_message(self, *, twist: Twist, frame_id: str) -> TwistStamped:
        message = TwistStamped()
        message.header.stamp = background_node.get_node().get_clock().now().to_msg()
        message.header.frame_id = frame_id
        message.twist = twist
        return message


# %% publishers


@dataclass(kw_only=True)
class CommandPublisher(ABC):
    """
    Sends the velocities computed by the controller to the robot.
    """

    @abstractmethod
    def publish(self) -> None:
        """
        Publish the velocities currently stored in the world state.
        """

    @abstractmethod
    def stop(self) -> None:
        """
        Publish zero velocities so the robot comes to a halt.
        """

    def write_predicted_positions(
        self, *, measurement: JointPositionMeasurement, cycle_timing: CycleTiming
    ) -> bool:
        """
        Overwrite the measured positions of the joints this publisher commands with
        where they will be once the command of the running cycle takes effect.

        Does nothing by default, as a publisher that knows no actuation delay predicts
        nothing.

        :param measurement: The measured positions to move forward.
        :param cycle_timing: When the command of the running cycle will be published.
        :return: Whether any position was overwritten.
        """
        return False


@dataclass(kw_only=True)
class JointGroupVelocityCommandPublisher(CommandPublisher):
    """
    Publishes the velocities of a group of joints as a single message.
    """

    world: World
    """
    The world holding the commanded velocities.
    """

    command_topic: str
    """
    Topic the velocity array is published on.
    """

    connections: list[ActiveConnection1DOF]
    """
    The controlled connections, in the order expected by the controller.
    """

    minimum_velocities: JointMinimumVelocities = field(
        default_factory=JointMinimumVelocities
    )
    """
    Minimum velocities of the commanded joints.
    """

    command_format: GroupCommandFormat = field(default_factory=Float64MultiArrayFormat)
    """
    Message format the controller on ``command_topic`` consumes.
    """

    actuation_delay: timedelta | None = None
    """
    How long the robot takes to execute a published command, compensated by
    :meth:`write_predicted_positions`; ``None`` compensates nothing.
    """

    command_history_margin: timedelta = timedelta(seconds=1)
    """
    How much longer than :attr:`actuation_delay` the published commands are remembered,
    and so the age a measurement may have to still be moved forward.
    """

    unusable_measurement_warning_period: timedelta = timedelta(seconds=5)
    """
    How often at most a measurement that cannot be moved forward is warned about.
    """

    command_history: CommandHistory = field(init=False)
    """
    The commands published recently, as velocities of the raw degrees of freedom; kept
    only with an :attr:`actuation_delay`.
    """

    commands: list[JointVelocityCommand] = field(init=False, default_factory=list)
    """
    What to send for every connection.
    """

    velocity_reader: StateVelocityReader = field(init=False)
    """
    Reads the commanded velocities from the world state.
    """

    command_publisher: Publisher = field(init=False)
    """
    The publisher for ``cmd_topic``.
    """

    def __post_init__(self) -> None:
        """
        Create the ROS publisher, resolve what to send for every connection and size
        the command history to the actuation delay.

        :raises UnknownMinimumVelocityJointError: If a minimum velocity override applies
            to no connection.
        """
        self.command_history = CommandHistory(
            kept_duration=(self.actuation_delay or timedelta(0))
            + self.command_history_margin
        )
        self.command_publisher = background_node.get_node().create_publisher(
            self.command_format.message_type(), self.command_topic, 10
        )
        with self.world.modify_world():
            self.world.set_dofs_has_hardware_interface(
                [
                    dof
                    for connection in self.connections
                    for dof in connection.active_dofs
                ],
                True,
            )
        self.commands = JointVelocityCommand.for_connections(
            connections=self.connections, minimum_velocities=self.minimum_velocities
        )
        self.velocity_reader = StateVelocityReader(
            world=self.world,
            degrees_of_freedom=[connection.raw_dof for connection in self.connections],
        )
        background_node.get_node().get_logger().info(
            f"Created publisher for {self.command_topic} for "
            f"{[connection.name.name for connection in self.connections]}"
        )

    def publish(self) -> None:
        state_velocities = self.velocity_reader.velocities()
        velocities = [
            command.from_state_velocity(state_velocity=state_velocity)
            for command, state_velocity in zip(self.commands, state_velocities)
        ]
        self.command_publisher.publish(
            self.command_format.create_message(velocities=velocities)
        )
        self._record_command(sent_velocities=velocities)

    def stop(self) -> None:
        velocities = [0.0] * len(self.commands)
        self.command_publisher.publish(
            self.command_format.create_message(velocities=velocities)
        )
        self._record_command(sent_velocities=velocities)

    def _record_command(self, *, sent_velocities: list[float]) -> None:
        """
        Remember the command just published, stamped with the current ROS time, as the
        raw state velocities the joints execute; nothing without an actuation delay.

        :param sent_velocities: The velocity sent to every joint.
        """
        if self.actuation_delay is None:
            return
        self.command_history.record(
            publish_time=background_node.ros_clock_seconds(),
            velocities=[
                command.executed_state_velocity(sent_velocity=sent_velocity)
                for command, sent_velocity in zip(self.commands, sent_velocities)
            ],
        )

    def write_predicted_positions(
        self, *, measurement: JointPositionMeasurement, cycle_timing: CycleTiming
    ) -> bool:
        """
        Unlike :meth:`CommandPublisher.write_predicted_positions`, moves the positions
        forward by the commands executed from one :attr:`actuation_delay` before the
        measurement on. A measurement stamped after the expected publish time, or older
        than :attr:`command_history_margin`, is left as it is, with a warning.
        """
        if self.actuation_delay is None or not self.command_history.held_commands:
            return False
        earliest_usable_time = (
            cycle_timing.cycle_start_time - self.command_history_margin.total_seconds()
        )
        latest_usable_time = cycle_timing.expected_publish_time
        if not (
            earliest_usable_time <= measurement.measurement_time <= latest_usable_time
        ):
            background_node.get_node().get_logger().warning(
                f"Not compensating the actuation delay of {self.command_topic}: the "
                f"joint state stamped {measurement.measurement_time:.3f} s lies outside "
                f"[{earliest_usable_time:.3f} s, {latest_usable_time:.3f} s]. "
                f"Check the header stamps of the joint states and the clocks.",
                throttle_duration_sec=self.unusable_measurement_warning_period.total_seconds(),
            )
            return False
        displacement = self.command_history.displacement(
            start_time=measurement.measurement_time
            - self.actuation_delay.total_seconds(),
            end_time=latest_usable_time,
        )
        wrote_any = False
        for connection, distance in zip(self.connections, displacement):
            measured_position = measurement.positions.get(connection.raw_dof.id)
            if measured_position is None:
                continue
            self.world.state[connection.raw_dof.id].position = (
                measured_position + distance
            )
            wrote_any = True
        return wrote_any


@dataclass(kw_only=True)
class DriveVelocityCommandPublisher(CommandPublisher):
    """
    Publishes the velocity of a drive connection as a twist, expressed in the frame of
    the body the drive moves.
    """

    world: World
    """
    The world holding the commanded velocities.
    """

    command_topic: str
    """
    Topic the twist is published on.
    """

    connection: Union[OmniDrive, DifferentialDrive]
    """
    The drive connection that is commanded.
    """

    minimum_linear_velocity: MinimumVelocity = field(default_factory=MinimumVelocity)
    """
    Minimum magnitude of the commanded linear velocity.
    """

    minimum_angular_velocity: MinimumVelocity = field(default_factory=MinimumVelocity)
    """
    Minimum magnitude of the commanded rotational velocity.
    """

    command_format: DriveCommandFormat = field(default_factory=TwistFormat)
    """
    Message format the controller on ``command_topic`` consumes.
    """

    velocity_reader: StateVelocityReader = field(init=False)
    """
    Reads the commanded velocities of the drivable axes, rotation last.
    """

    twist: Twist = field(init=False, default_factory=Twist)
    """
    The twist reused for every publication, which :attr:`command_format` turns into the
    message it publishes.
    """

    velocity_publisher: Publisher = field(init=False)
    """
    The publisher for ``cmd_topic``.
    """

    def __post_init__(self):
        self.velocity_publisher = background_node.get_node().create_publisher(
            self.command_format.message_type(), self.command_topic, 10
        )
        with self.world.modify_world():
            self.world.set_dofs_has_hardware_interface(
                self.connection.active_dofs, True
            )
        self.velocity_reader = StateVelocityReader(
            world=self.world,
            degrees_of_freedom=self.translation_degrees_of_freedom()
            + [self.connection.yaw],
        )
        background_node.get_node().get_logger().info(
            f"Created publisher for {self.command_topic}."
        )

    def translation_degrees_of_freedom(self) -> list[DegreeOfFreedom]:
        """
        The degrees of freedom the drive can translate along.

        :return: The translation degrees of freedom, in publish order.
        """
        if isinstance(self.connection, OmniDrive):
            return [self.connection.x_velocity, self.connection.y_velocity]
        return [self.connection.x_velocity]

    def publish(self) -> None:
        velocities = self.velocity_reader.velocities()
        linear_velocities = self.minimum_linear_velocity.enforce_on_vector(
            velocities=velocities[:-1]
        )
        self.twist.linear.x = linear_velocities[0]
        if len(linear_velocities) > 1:
            self.twist.linear.y = linear_velocities[1]
        self.twist.angular.z = self.minimum_angular_velocity.enforce_on_scalar(
            velocity=velocities[-1]
        )
        self._publish_twist()

    def stop(self) -> None:
        self.twist.linear.x = 0.0
        self.twist.linear.y = 0.0
        self.twist.angular.z = 0.0
        self._publish_twist()

    def _publish_twist(self) -> None:
        """
        Publish :attr:`twist` in the format the controller consumes.
        """
        self.velocity_publisher.publish(
            self.command_format.create_message(
                twist=self.twist, frame_id=self.connection.child.name.name
            )
        )
