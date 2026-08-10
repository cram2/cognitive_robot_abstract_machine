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
from dataclasses import dataclass, field
from typing import Dict, List, Union

from geometry_msgs.msg import Twist
from rclpy.publisher import Publisher
from std_msgs.msg import Float64, Float64MultiArray
from typing_extensions import Self

from giskardpy.middleware.ros2 import rospy
from giskardpy.middleware.ros2.exceptions import UnknownMinimumVelocityJointError
from giskardpy.middleware.ros2.ros2_interface import get_parameters
from semantic_digital_twin.world import World
from semantic_digital_twin.world_description.connections import (
    ActiveConnection1DOF,
    DifferentialDrive,
    OmniDrive,
)
from semantic_digital_twin.world_description.degree_of_freedom import DegreeOfFreedom

# %% minimum velocities


@dataclass
class MinimumVelocity:
    """
    Smallest velocity magnitude a hardware interface still reacts to.
    """

    magnitude: float = 0.0
    """
    Velocities below this magnitude are increased to it. ``0.0`` disables the minimum.
    """

    def enforce_on_vector(self, velocities: List[float]) -> List[float]:
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

    def enforce_on_scalar(self, velocity: float) -> float:
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


@dataclass
class JointMinimumVelocity(MinimumVelocity):
    """
    Minimum velocity of a single joint, overriding the default of its publisher.
    """

    joint_name: str = field(kw_only=True)
    """
    Name of the joint this minimum applies to.
    """


@dataclass
class JointMinimumVelocities:
    """
    Minimum velocities of the joints a publisher commands.
    """

    default: MinimumVelocity = field(default_factory=MinimumVelocity)
    """
    Minimum applied to every joint without an override.
    """

    overrides: List[JointMinimumVelocity] = field(default_factory=list)
    """
    Minimums of individual joints, e.g. to exempt hardware without a velocity deadband.
    """

    @classmethod
    def from_magnitudes(
        cls, default: float = 0.0, overrides: Dict[str, float] | None = None
    ) -> Self:
        """
        Build minimum velocities from a default magnitude and per-joint magnitudes.

        :param default: Minimum magnitude applied to every joint without an override.
        :param overrides: Minimum magnitudes of individual joints, keyed by joint name.
        :return: The built minimum velocities.
        """
        return cls(
            default=MinimumVelocity(default),
            overrides=[
                JointMinimumVelocity(magnitude, joint_name=joint_name)
                for joint_name, magnitude in (overrides or {}).items()
            ],
        )

    def of(self, connection: ActiveConnection1DOF) -> MinimumVelocity:
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
        self, connections: List[ActiveConnection1DOF]
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


@dataclass
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

    degrees_of_freedom: List[DegreeOfFreedom]
    """
    The degrees of freedom to read, in the order the velocities are returned in.
    """

    columns: List[int] = field(init=False, default_factory=list)
    """
    Column of every degree of freedom in the state data.
    """

    model_version: int = field(init=False, default=-1)
    """
    Model version the columns were resolved for.
    """

    def velocities(self) -> List[float]:
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


@dataclass
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

    def from_state_velocity(self, state_velocity: float) -> float:
        """
        Scale a raw state velocity to the joint and raise it to the minimum.

        :param state_velocity: The raw velocity read from the world state.
        :return: The velocity to send to the joint's hardware interface.
        """
        return self.minimum_velocity.enforce_on_scalar(
            state_velocity * self.connection.multiplier
        )

    @classmethod
    def for_connections(
        cls,
        connections: List[ActiveConnection1DOF],
        minimum_velocities: JointMinimumVelocities,
    ) -> List[Self]:
        """
        Resolve the minimum velocity of every connection once.

        :param connections: The connections to build commands for.
        :param minimum_velocities: The minimum velocities to resolve per connection.
        :raises UnknownMinimumVelocityJointError: If an override applies to no
            connection.
        :return: One command per connection, in the same order.
        """
        minimum_velocities.validate_overrides_apply_to(connections)
        return [
            cls(connection, minimum_velocities.of(connection))
            for connection in connections
        ]


# %% publishers


@dataclass
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


@dataclass
class JointVelocityCommandPublisher(CommandPublisher):
    """
    Publishes one velocity per joint, each on its own topic.
    """

    world: World
    """
    The world holding the commanded velocities.
    """

    namespaces: List[str]
    """
    Namespaces of the velocity controllers; each offers a ``command`` topic and a
    ``joint`` parameter.
    """

    minimum_velocities: JointMinimumVelocities = field(
        default_factory=JointMinimumVelocities
    )
    """
    Minimum velocities of the commanded joints.
    """

    connections: List[ActiveConnection1DOF] = field(init=False, default_factory=list)
    """
    The controlled connections, in the same order as ``namespaces``.
    """

    publishers: List[Publisher] = field(init=False, default_factory=list)
    """
    The command publishers, in the same order as ``namespaces``.
    """

    commands: List[JointVelocityCommand] = field(init=False, default_factory=list)
    """
    What to send for every connection.
    """

    velocity_reader: StateVelocityReader = field(init=False)
    """
    Reads the commanded velocities from the world state.
    """

    message: Float64 = field(init=False, default_factory=Float64)
    """
    The message reused for every publication.
    """

    def __post_init__(self):
        for namespace in self.namespaces:
            self.publishers.append(
                rospy.node.create_publisher(Float64, f"/{namespace}/command", 10)
            )
            joint_name = (
                get_parameters(parameters=["joint"], node_name=namespace)
                .values[0]
                .string_value
            )
            connection: ActiveConnection1DOF = self.world.get_connection_by_name(
                joint_name
            )
            connection.has_hardware_interface = True
            self.connections.append(connection)
        self.commands = JointVelocityCommand.for_connections(
            self.connections, self.minimum_velocities
        )
        self.velocity_reader = StateVelocityReader(
            world=self.world,
            degrees_of_freedom=[connection.raw_dof for connection in self.connections],
        )

    def publish(self) -> None:
        state_velocities = self.velocity_reader.velocities()
        for publisher, command, state_velocity in zip(
            self.publishers, self.commands, state_velocities
        ):
            self.message.data = command.from_state_velocity(state_velocity)
            publisher.publish(self.message)

    def stop(self) -> None:
        self.message.data = 0.0
        for publisher in self.publishers:
            publisher.publish(self.message)


@dataclass
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

    connections: List[ActiveConnection1DOF]
    """
    The controlled connections, in the order expected by the controller.
    """

    minimum_velocities: JointMinimumVelocities = field(
        default_factory=JointMinimumVelocities
    )
    """
    Minimum velocities of the commanded joints.
    """

    commands: List[JointVelocityCommand] = field(init=False, default_factory=list)
    """
    What to send for every connection.
    """

    velocity_reader: StateVelocityReader = field(init=False)
    """
    Reads the commanded velocities from the world state.
    """

    message: Float64MultiArray = field(init=False, default_factory=Float64MultiArray)
    """
    The message reused for every publication.
    """

    command_publisher: Publisher = field(init=False)
    """
    The publisher for ``cmd_topic``.
    """

    def __post_init__(self):
        self.command_publisher = rospy.node.create_publisher(
            Float64MultiArray, self.command_topic, 10
        )
        for connection in self.connections:
            connection.has_hardware_interface = True
        self.commands = JointVelocityCommand.for_connections(
            self.connections, self.minimum_velocities
        )
        self.velocity_reader = StateVelocityReader(
            world=self.world,
            degrees_of_freedom=[connection.raw_dof for connection in self.connections],
        )
        rospy.node.get_logger().info(
            f"Created publisher for {self.command_topic} for "
            f"{[connection.name.name for connection in self.connections]}"
        )

    def publish(self) -> None:
        self.message.data = [
            command.from_state_velocity(state_velocity)
            for command, state_velocity in zip(
                self.commands, self.velocity_reader.velocities()
            )
        ]
        self.command_publisher.publish(self.message)

    def stop(self) -> None:
        self.message.data = [0.0] * len(self.commands)
        self.command_publisher.publish(self.message)


@dataclass
class DriveVelocityCommandPublisher(CommandPublisher):
    """
    Publishes the velocity of a drive connection as a twist.
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

    velocity_reader: StateVelocityReader = field(init=False)
    """
    Reads the commanded velocities of the drivable axes, rotation last.
    """

    message: Twist = field(init=False, default_factory=Twist)
    """
    The message reused for every publication.
    """

    velocity_publisher: Publisher = field(init=False)
    """
    The publisher for ``cmd_topic``.
    """

    def __post_init__(self):
        self.velocity_publisher = rospy.node.create_publisher(
            Twist, self.command_topic, 10
        )
        self.connection.has_hardware_interface = True
        self.velocity_reader = StateVelocityReader(
            world=self.world,
            degrees_of_freedom=self.translation_degrees_of_freedom()
            + [self.connection.yaw],
        )
        rospy.node.get_logger().info(f"Created publisher for {self.command_topic}.")

    def translation_degrees_of_freedom(self) -> List[DegreeOfFreedom]:
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
            velocities[:-1]
        )
        self.message.linear.x = linear_velocities[0]
        if len(linear_velocities) > 1:
            self.message.linear.y = linear_velocities[1]
        self.message.angular.z = self.minimum_angular_velocity.enforce_on_scalar(
            velocities[-1]
        )
        self.velocity_publisher.publish(self.message)

    def stop(self) -> None:
        self.message.linear.x = 0.0
        self.message.linear.y = 0.0
        self.message.angular.z = 0.0
        self.velocity_publisher.publish(self.message)
