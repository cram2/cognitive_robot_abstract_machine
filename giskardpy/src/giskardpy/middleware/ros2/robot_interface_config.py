from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing_extensions import TYPE_CHECKING

from nav_msgs.msg import Odometry

from cramph.executor import NoPacing, Pacer, RealTimePacer
from semantic_digital_twin.adapters.ros import rospy
from giskardpy.middleware.ros2.command_publishing import (
    DriveVelocityCommandPublisher,
    Float64MultiArrayFormat,
    DriveCommandFormat,
    GroupCommandFormat,
    TwistFormat,
    TwistStampedFormat,
    JointGroupVelocityCommandPublisher,
    JointMinimumVelocities,
    JointVelocityCommandPublisher,
    MinimumVelocity,
)
from giskardpy.middleware.ros2.control_loop import ControlLoop
from giskardpy.middleware.ros2.controller_manager import (
    ControllerManager,
    ControllerType,
)
from giskardpy.middleware.ros2.exceptions import NoUniqueDriveError
from semantic_digital_twin.adapters.ros.input_synchronization import (
    PendingJointStateSynchronizer,
    OdometrySynchronizer,
    TfFrameSynchronizer,
)
from giskardpy.middleware.ros2.motion_server import MotionServer
from giskardpy.middleware.ros2.ros2_interface import (
    search_for_unique_publisher_of_type,
    search_for_unique_subscriber_of_type,
)
from giskardpy.middleware.ros2.server_config import GiskardServerConfig
from semantic_digital_twin.robots.robot_part_mixins import HasInputSource
from semantic_digital_twin.robots.robot_parts import AbstractRobot
from semantic_digital_twin.world import World
from semantic_digital_twin.world_description.connections import (
    ActiveConnection1DOF,
    Connection6DoF,
    DifferentialDrive,
    OmniDrive,
    WheeledDrive,
)

if TYPE_CHECKING:
    from giskardpy.middleware.ros2.giskard import Giskard


@dataclass
class RobotInterfaceConfig(ABC):
    """
    Describes how Giskard reads the state of a robot and how it commands it.

    The kind of interface decides whether the motion is simulated or executed on the
    robot.
    """

    giskard: Giskard = field(init=False, repr=False, compare=False)
    """
    The Giskard instance this interface belongs to, set by :meth:`attach`.

    ..note:: Kept out of ``__repr__`` and ``__eq__`` because Giskard refers back to this
        interface.
    """

    def attach(self, giskard: Giskard) -> None:
        """
        Bind this interface to the Giskard instance it configures.
        """
        self.giskard = giskard

    @abstractmethod
    def create_pacer(self) -> Pacer:
        """
        Build the pacer of the control loop.
        """

    @abstractmethod
    def connect(self) -> None:
        """
        Decide which joints Giskard controls and how it reaches them.
        """

    @property
    def world(self) -> World:
        return self.giskard.executor.context.world

    @property
    def robot(self) -> AbstractRobot:
        return self.giskard.robot

    @property
    def server_config(self) -> GiskardServerConfig:
        return self.giskard.server_config

    @property
    def motion_server(self) -> MotionServer:
        return self.giskard.motion_server

    @property
    def control_loop(self) -> ControlLoop:
        return self.giskard.motion_server.control_loop


@dataclass
class StandAloneRobotInterfaceConfig(RobotInterfaceConfig):
    """
    Simulates the motion in place, without talking to any hardware.

    Giskard controls the joints the robot class declares and the base drive of the world
    config.
    """

    def create_pacer(self) -> Pacer:
        """
        A simulated motion runs as fast as the computer allows.
        """
        return NoPacing()

    def connect(self) -> None:
        pass


@dataclass
class ClosedLoopRobotInterfaceConfig(RobotInterfaceConfig, ABC):
    """
    Reads the state of a real robot and sends it the commands of the controller.

    Giskard controls exactly the joints that :meth:`setup` adds a command publisher for.
    """

    tf_frame_synchronizer: TfFrameSynchronizer | None = field(
        init=False, default=None, repr=False, compare=False
    )
    """
    Created on demand, tracks all 6 degree of freedom connections that follow tf.
    """

    def create_pacer(self) -> Pacer:
        """
        A real robot has to be commanded in real time.
        """
        return RealTimePacer()

    def connect(self) -> None:
        """
        Release every joint the robot class declared as controlled, then let
        :meth:`setup` declare the ones that are commanded.
        """
        with self.world.modify_world():
            self.world.set_dofs_has_hardware_interface(
                self.world.degrees_of_freedom, False
            )
        self.setup()

    @abstractmethod
    def setup(self) -> None:
        """
        Implement this method to configure how Giskard reads the state of the robot and
        how it sends commands, using the ``sync_*`` and ``add_*`` methods.
        """

    # %% reading the state of the robot

    def sync_odometry_topic(
        self,
        odometry_topic: str | None = None,
        joint: OmniDrive | DifferentialDrive = None,
        sync_in_control_loop: bool = True,
    ):
        """
        Tell Giskard to sync an odometry joint added during by the world config.
        """
        if odometry_topic is None:
            odometry_topic = search_for_unique_publisher_of_type(Odometry)
        assert isinstance(joint, (OmniDrive, DifferentialDrive))
        synchronizer = OdometrySynchronizer(
            world=self.world,
            node=rospy.get_node(),
            topic_name=odometry_topic,
            connection=joint,
        )
        self.motion_server.inputs.synchronizers.append(synchronizer)
        if sync_in_control_loop:
            self.control_loop.inputs.synchronizers.append(synchronizer)

    def sync_6dof_joint_with_tf_frame(
        self, joint: Connection6DoF, tf_parent_frame: str, tf_child_frame: str
    ):
        """
        Tell Giskard to sync a 6dof joint with a tf frame.
        """
        if self.tf_frame_synchronizer is None:
            self.tf_frame_synchronizer = TfFrameSynchronizer(
                world=self.world, node=rospy.get_node()
            )
            self.motion_server.inputs.synchronizers.insert(
                0, self.tf_frame_synchronizer
            )
            self.control_loop.inputs.synchronizers.insert(0, self.tf_frame_synchronizer)
        self.tf_frame_synchronizer.track(joint, tf_parent_frame, tf_child_frame)

    def sync_robot_parts(self):
        """
        Tell Giskard to read every part of the robot that declares a topic from the
        robot itself, instead of from the world it stands in.

        The topics come from the parts, so a part mounted on a different robot is read
        wherever that robot publishes it.
        """
        self.robot.use_real_sources(rospy.get_node())
        self.motion_server.inputs.add_robot_inputs(self.robot)
        self.control_loop.inputs.add_robot_inputs(self.robot)

    def sync_robot_part(self, robot_part: HasInputSource):
        """
        Tell Giskard to read one part of the robot from the robot itself, on the topic
        that part declares.
        """
        robot_part.use_real_source(rospy.get_node())
        self.motion_server.inputs.add_robot_part_input(robot_part)
        self.control_loop.inputs.add_robot_part_input(robot_part)

    def sync_joint_state_topic(self, topic_name: str, group_name: str | None = None):
        """
        Tell Giskard to sync the world state with a joint state topic.
        """
        if group_name is None:
            group_name = self.robot.name
        self.motion_server.inputs.synchronizers.append(
            PendingJointStateSynchronizer(
                world=self.world, node=rospy.get_node(), topic_name=topic_name
            )
        )
        if group_name != self.robot.name:
            return
        self.control_loop.inputs.synchronizers.append(
            PendingJointStateSynchronizer(
                world=self.world, node=rospy.get_node(), topic_name=topic_name
            )
        )

    # %% commanding the robot

    def add_drive_velocity_command_topic(
        self,
        command_topic: str | None = None,
        drive: WheeledDrive | None = None,
        minimum_linear_velocity: float = 0.0,
        minimum_angular_velocity: float = 0.0,
        command_format: DriveCommandFormat | None = None,
    ) -> None:
        """
        Tell Giskard to send the velocity of a drive to a topic.

        :param command_topic: The topic the velocity is sent to, the only topic of the
            message type of ``command_format`` if ``None``.
        :param drive: The drive to command, the only drive of the world if ``None``.
        :param minimum_linear_velocity: magnitude that smaller linear velocities are
            raised to so the hardware moves, without changing the driving direction.
            ``0.0`` disables raising.
        :param minimum_angular_velocity: magnitude that smaller rotational velocities
            are raised to. ``0.0`` disables raising.
        :param command_format: message format the drive controller consumes; defaults to
            :class:`~giskardpy.middleware.ros2.command_publishing.TwistFormat`.
        :raises NoUniqueDriveError: If ``drive`` is ``None`` and the world does not hold
            exactly one drive.
        """
        if drive is None:
            drives = self.world.get_connections_by_type(WheeledDrive)
            if len(drives) != 1:
                raise NoUniqueDriveError(drives=drives)
            [drive] = drives
        command_format = command_format or TwistFormat()
        if command_topic is None:
            command_topic = search_for_unique_subscriber_of_type(
                command_format.message_type()
            )
        self.control_loop.command_publishers.append(
            DriveVelocityCommandPublisher(
                world=self.world,
                command_topic=command_topic,
                connection=drive,
                minimum_linear_velocity=MinimumVelocity(minimum_linear_velocity),
                minimum_angular_velocity=MinimumVelocity(minimum_angular_velocity),
                command_format=command_format,
            )
        )

    def add_joint_velocity_controller(
        self,
        namespaces: list[str],
        minimum_valid_velocity: float = 0.0,
        minimum_velocity_overrides: dict[str, float] | None = None,
    ):
        """
        Tell Giskard how it can send velocities to joints.

        :param namespaces: A list of namespaces where Giskard can find the topics and
            parameters.
        :param minimum_valid_velocity: magnitude that smaller joint velocities are
            raised to so the hardware moves. ``0.0`` disables raising.
        :param minimum_velocity_overrides: minimum magnitude per joint name, overriding
            ``minimum_valid_velocity``; ``0.0`` exempts a joint.
        """
        self.control_loop.command_publishers.append(
            JointVelocityCommandPublisher(
                world=self.world,
                namespaces=namespaces,
                minimum_velocities=JointMinimumVelocities.from_magnitudes(
                    minimum_valid_velocity, minimum_velocity_overrides
                ),
            )
        )

    def add_joint_group_velocity_command_topic(
        self,
        command_topic: str,
        connections: list[str],
        minimum_valid_velocity: float = 0.0,
        minimum_velocity_overrides: dict[str, float] | None = None,
        command_format: GroupCommandFormat | None = None,
    ) -> None:
        """
        Tell Giskard to send the velocities of a group of joints, all in one message, to
        a topic.

        :param command_topic: The topic the velocities are sent to.
        :param connections: The names of the joints, in the order the message lists their
            velocities.
        :param minimum_valid_velocity: magnitude that smaller joint velocities are
            raised to so the hardware moves. ``0.0`` disables raising.
        :param minimum_velocity_overrides: minimum magnitude per joint name, overriding
            ``minimum_valid_velocity``; ``0.0`` exempts a joint.
        :param command_format: message format the controller consumes; defaults to
            :class:`~giskardpy.middleware.ros2.command_publishing.Float64MultiArrayFormat`.
        """
        controlled_connections: list[ActiveConnection1DOF] = [
            self.world.get_connection_by_name(connection_name)
            for connection_name in connections
        ]
        self.control_loop.command_publishers.append(
            JointGroupVelocityCommandPublisher(
                world=self.world,
                command_topic=command_topic,
                connections=controlled_connections,
                minimum_velocities=JointMinimumVelocities.from_magnitudes(
                    minimum_valid_velocity, minimum_velocity_overrides
                ),
                command_format=command_format or Float64MultiArrayFormat(),
            )
        )

    # %% discovery

    def discover_interfaces_from_controller_manager(
        self,
        controller_manager_name: str = "controller_manager",
        whitelist: list[str] | None = None,
    ) -> None:
        """
        Tell Giskard to read and command the robot through the active controllers of a
        controller manager.

        :param controller_manager_name: The controller manager node, including the
            namespace it runs in.
        :param whitelist: The names of the controllers to consider, all if ``None``.
        """
        controller_manager = ControllerManager(controller_manager_name)
        for controller in controller_manager.active_controllers(whitelist):
            match controller.type:
                case ControllerType.JOINT_STATE_BROADCASTER:
                    self.sync_joint_state_topic(
                        controller_manager.joint_state_topic_of(controller.name)
                    )
                case ControllerType.JOINT_GROUP_VELOCITY:
                    command_format = Float64MultiArrayFormat()
                    self.add_joint_group_velocity_command_topic(
                        command_topic=controller_manager.command_topic_of(
                            controller.name, command_format.message_type()
                        ),
                        connections=controller_manager.joints_of(controller.name),
                        command_format=command_format,
                    )
                case ControllerType.DIFF_DRIVE:
                    command_format = TwistStampedFormat()
                    self.add_drive_velocity_command_topic(
                        command_topic=controller_manager.command_topic_of(
                            controller.name, command_format.message_type()
                        ),
                        command_format=command_format,
                    )
