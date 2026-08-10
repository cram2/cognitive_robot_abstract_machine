from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Dict, List, Union

from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry
from sensor_msgs.msg import JointState
from std_msgs.msg import Float64MultiArray

from giskardpy.data_types.exceptions import (
    JointRegistrationRequiresStandaloneModeError,
)
from giskardpy.middleware.ros2 import rospy
from giskardpy.middleware.ros2.command_publishing import (
    DriveVelocityCommandPublisher,
    JointGroupVelocityCommandPublisher,
    JointMinimumVelocities,
    JointVelocityCommandPublisher,
    MinimumVelocity,
)
from giskardpy.middleware.ros2.control_loop import ControlLoop
from giskardpy.middleware.ros2.input_synchronization import (
    LatestJointStateSynchronizer,
    PendingJointStateSynchronizer,
    OdometrySynchronizer,
    TfFrameSynchronizer,
)
from giskardpy.middleware.ros2.motion_server import MotionServer
from giskardpy.middleware.ros2.ros2_interface import (
    search_for_subscriber_of_node_with_type,
    get_parameters,
    search_for_publisher_of_node_with_type,
    search_for_unique_publisher_of_type,
    search_for_unique_subscriber_of_type,
)
from giskardpy.middleware.ros2.server_config import GiskardServerConfig
from semantic_digital_twin.datastructures.prefixed_name import PrefixedName
from semantic_digital_twin.robots.robot_parts import AbstractRobot
from semantic_digital_twin.world import World
from semantic_digital_twin.world_description.connections import (
    ActiveConnection,
    ActiveConnection1DOF,
    Connection6DoF,
    DifferentialDrive,
    OmniDrive,
)

if TYPE_CHECKING:
    from giskardpy.middleware.ros2.giskard import Giskard


@dataclass
class RobotInterfaceConfig(ABC):
    """
    Describes how Giskard reads the state of a robot and how it commands it.
    """

    giskard: Giskard = field(init=False, repr=False, compare=False)
    """
    The Giskard instance this interface belongs to, set by :meth:`attach`.

    ..note:: Kept out of ``__repr__`` and ``__eq__`` because Giskard refers back to this
        interface.
    """

    tf_frame_synchronizer: TfFrameSynchronizer | None = field(
        init=False, default=None, repr=False, compare=False
    )
    """
    Created on demand, tracks all 6 degree of freedom connections that follow tf.
    """

    def attach(self, giskard: Giskard) -> None:
        """
        Bind this interface to the Giskard instance it configures.
        """
        self.giskard = giskard

    @abstractmethod
    def setup(self):
        """
        Implement this method to configure how Giskard can talk to the robot using it's
        self.

        methods.
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

    # %% reading the state of the robot

    def sync_odometry_topic(
        self,
        odometry_topic: str | None = None,
        joint: Union[OmniDrive, DifferentialDrive] = None,
        sync_in_control_loop: bool = True,
    ):
        """
        Tell Giskard to sync an odometry joint added during by the world config.
        """
        if odometry_topic is None:
            odometry_topic = search_for_unique_publisher_of_type(Odometry)
        assert isinstance(joint, (OmniDrive, DifferentialDrive))
        synchronizer = OdometrySynchronizer(
            world=self.world, topic_name=odometry_topic, connection=joint
        )
        self.motion_server.inputs.synchronizers.append(synchronizer)
        if sync_in_control_loop and self.server_config.is_closed_loop:
            self.control_loop.inputs.synchronizers.append(synchronizer)

    def sync_6dof_joint_with_tf_frame(
        self, joint: Connection6DoF, tf_parent_frame: str, tf_child_frame: str
    ):
        """
        Tell Giskard to sync a 6dof joint with a tf frame.
        """
        if self.tf_frame_synchronizer is None:
            self.tf_frame_synchronizer = TfFrameSynchronizer(world=self.world)
            self.motion_server.inputs.synchronizers.insert(
                0, self.tf_frame_synchronizer
            )
            if self.server_config.is_closed_loop:
                self.control_loop.inputs.synchronizers.insert(
                    0, self.tf_frame_synchronizer
                )
        self.tf_frame_synchronizer.track(joint, tf_parent_frame, tf_child_frame)

    def sync_joint_state_topic(self, topic_name: str, group_name: str | None = None):
        """
        Tell Giskard to sync the world state with a joint state topic.
        """
        if group_name is None:
            group_name = self.robot.name
        self.motion_server.inputs.synchronizers.append(
            PendingJointStateSynchronizer(world=self.world, topic_name=topic_name)
        )
        if not self.server_config.is_closed_loop or group_name != self.robot.name:
            return
        self.control_loop.inputs.synchronizers.append(
            LatestJointStateSynchronizer(world=self.world, topic_name=topic_name)
        )

    # %% commanding the robot

    def add_base_cmd_velocity(
        self,
        cmd_vel_topic: str | None = None,
        joint: Union[OmniDrive, DifferentialDrive] = None,
        minimum_linear_velocity: float = 0.0,
        minimum_angular_velocity: float = 0.0,
    ):
        """
        Tell Giskard how it can control an odom joint of the robot.

        :param cmd_vel_topic: a Twist topic
        :param joint: omni or diff drive joint. Doesn't need to be specified if there is
            only one.
        :param minimum_linear_velocity: magnitude that smaller linear velocities are
            raised to so the hardware moves, without changing the driving direction.
            ``0.0`` disables raising.
        :param minimum_angular_velocity: magnitude that smaller rotational velocities
            are raised to. ``0.0`` disables raising.
        """
        if cmd_vel_topic is None:
            cmd_vel_topic = search_for_unique_subscriber_of_type(Twist)
        if not self.server_config.is_closed_loop:
            return
        self.control_loop.command_publishers.append(
            DriveVelocityCommandPublisher(
                world=self.world,
                command_topic=cmd_vel_topic,
                connection=joint,
                minimum_linear_velocity=MinimumVelocity(minimum_linear_velocity),
                minimum_angular_velocity=MinimumVelocity(minimum_angular_velocity),
            )
        )

    def add_joint_velocity_controller(
        self,
        namespaces: List[str],
        minimum_valid_velocity: float = 0.0,
        minimum_velocity_overrides: Dict[str, float] | None = None,
    ):
        """
        For closed loop mode.

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

    def add_joint_velocity_group_controller(
        self,
        cmd_topic: str,
        connections: List[str],
        minimum_valid_velocity: float = 0.0,
        minimum_velocity_overrides: Dict[str, float] | None = None,
    ):
        """
        For closed loop mode.

        Tell Giskard how it can send velocities for a group of connections.
        :param minimum_valid_velocity: magnitude that smaller joint velocities are
            raised to so the hardware moves. ``0.0`` disables raising.
        :param minimum_velocity_overrides: minimum magnitude per joint name, overriding
            ``minimum_valid_velocity``; ``0.0`` exempts a joint.
        """
        controlled_connections: List[ActiveConnection1DOF] = [
            self.world.get_connection_by_name(connection_name)
            for connection_name in connections
        ]
        self.control_loop.command_publishers.append(
            JointGroupVelocityCommandPublisher(
                world=self.world,
                command_topic=cmd_topic,
                connections=controlled_connections,
                minimum_velocities=JointMinimumVelocities.from_magnitudes(
                    minimum_valid_velocity, minimum_velocity_overrides
                ),
            )
        )

    def register_controlled_joints(
        self, joint_names: List[Union[str, PrefixedName]]
    ) -> None:
        """
        Flag the given joints as controlled by Giskard itself.

        :raises JointRegistrationRequiresStandaloneModeError: If Giskard is not in
            standalone mode.
        """
        if not self.server_config.is_standalone:
            raise JointRegistrationRequiresStandaloneModeError()
        for joint_name in joint_names:
            connection: ActiveConnection = self.world.get_connection_by_name(joint_name)
            if not isinstance(connection, ActiveConnection):
                raise Exception(
                    f"{joint_name} is not an active connection and cannot be controlled."
                )
            connection.has_hardware_interface = True

    # %% discovery

    def discover_interfaces_from_controller_manager(
        self,
        controller_manager_name: str = "controller_manager",
        whitelist: List[str] | None = None,
    ) -> None:
        """
        :param whitelist: list all controllers that should get added, if None, giskard will search automatically
        """
        import controller_manager as cm
        from controller_manager_msgs.srv._list_controllers import (
            ListControllers_Response,
        )

        controllers: ListControllers_Response = cm.list_controllers(
            node=rospy.node, controller_manager_name=controller_manager_name
        )

        controllers_to_add = self.__filter_controllers_with_whitelist(
            controllers.controller, whitelist
        )

        for controller in controllers_to_add:
            if controller.state == "active":
                if controller.type == "joint_state_broadcaster/JointStateBroadcaster":
                    topic_name = search_for_publisher_of_node_with_type(
                        topic_type=JointState, node_name=controller.name
                    )
                    self.sync_joint_state_topic(topic_name)
                elif (
                    controller.type
                    == "velocity_controllers/JointGroupVelocityController"
                ):
                    cmt_topic = search_for_subscriber_of_node_with_type(
                        topic_type=Float64MultiArray, node_name=controller.name
                    )
                    joints = (
                        get_parameters(parameters=["joints"], node_name=controller.name)
                        .values[0]
                        .string_array_value
                    )
                    self.add_joint_velocity_group_controller(
                        cmd_topic=cmt_topic, connections=joints
                    )
                elif controller.type == "diff_drive_controller/DiffDriveController":
                    self.add_base_cmd_velocity(controller.name)

    def __filter_controllers_with_whitelist(
        self, controllers: list, whitelist: List[str] | None
    ) -> list:
        from controller_manager_msgs.msg import ControllerState

        controllers_to_add: List[ControllerState]
        if whitelist is None:
            return controllers
        else:
            available_controllers = {controller.name for controller in controllers}
            missing_controllers = [
                controller
                for controller in whitelist
                if controller not in available_controllers
            ]
            if missing_controllers:
                raise ValueError(
                    f"The following controllers from the whitelist are not available: {missing_controllers}"
                )
            return [
                controller for controller in controllers if controller.name in whitelist
            ]


@dataclass
class StandAloneRobotInterfaceConfig(RobotInterfaceConfig):
    """
    Controls the given joints without talking to any hardware.
    """

    joint_names: List[str]
    """
    The joints Giskard is allowed to control.
    """

    def setup(self):
        self.register_controlled_joints(self.joint_names)
