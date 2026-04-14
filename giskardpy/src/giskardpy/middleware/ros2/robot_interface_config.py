from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Optional, List, Dict, Union

from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry
from sensor_msgs.msg import JointState
from std_msgs.msg import Float64MultiArray

from giskardpy.data_types.exceptions import SetupException
from giskardpy.middleware.ros2 import rospy
from giskardpy.middleware.ros2.ros2_interface import (
    search_for_subscriber_of_node_with_type,
    get_parameters,
    search_for_publisher_of_node_with_type,
    search_for_unique_publisher_of_type,
    search_for_unique_subscriber_of_type,
)
from giskardpy.tree.blackboard_utils import GiskardBlackboard
from giskardpy.tree.branches.giskard_bt import GiskardBT
from semantic_digital_twin.datastructures.prefixed_name import PrefixedName
from semantic_digital_twin.robots.abstract_robot import AbstractRobot
from semantic_digital_twin.spatial_types.derivatives import Derivatives
from semantic_digital_twin.world import World
from semantic_digital_twin.world_description.connections import (
    OmniDrive,
    ActiveConnection,
    Connection6DoF,
    DifferentialDrive,
)
from semantic_digital_twin.world_description.world_entity import Connection


class RobotInterfaceConfig(ABC):
    @abstractmethod
    def setup(self):
        """
        Implement this method to configure how Giskard can talk to the robot using it's self. methods.
        """

    @property
    def world(self) -> World:
        return GiskardBlackboard().executor.context.world

    @property
    def robot(self) -> AbstractRobot:
        return GiskardBlackboard().giskard.robot

    @property
    def tree(self) -> GiskardBT:
        return GiskardBlackboard().tree

    def sync_odometry_topic(
        self,
        odometry_topic: Optional[str] = None,
        joint: Union[OmniDrive, DifferentialDrive] = None,
        sync_in_control_loop: bool = True,
    ):
        """
        Tell Giskard to sync an odometry joint added during by the world config.
        """
        if odometry_topic is None:
            odometry_topic = search_for_unique_publisher_of_type(Odometry)
        assert isinstance(joint, (OmniDrive, DifferentialDrive))
        self.tree.wait_for_goal.synchronization.sync_odometry_topic(
            odometry_topic, joint
        )
        if sync_in_control_loop and GiskardBlackboard().tree_config.is_closed_loop():
            self.tree.control_loop_branch.closed_loop_synchronization.sync_odometry_topic(
                odometry_topic, joint
            )

    def sync_6dof_joint_with_tf_frame(
        self, joint: Connection6DoF, tf_parent_frame: str, tf_child_frame: str
    ):
        """
        Tell Giskard to sync a 6dof joint with a tf frame.
        """
        self.tree.wait_for_goal.synchronization.sync_6dof_joint_with_tf_frame(
            joint, tf_parent_frame, tf_child_frame
        )
        if GiskardBlackboard().tree_config.is_closed_loop():
            self.tree.control_loop_branch.closed_loop_synchronization.sync_6dof_joint_with_tf_frame(
                joint, tf_parent_frame, tf_child_frame
            )

    def sync_joint_state_topic(self, topic_name: str, group_name: Optional[str] = None):
        """
        Tell Giskard to sync the world state with a joint state topic
        """
        if group_name is None:
            group_name = self.robot.name
        self.tree.wait_for_goal.synchronization.sync_joint_state_topic(
            group_name=group_name, topic_name=topic_name
        )
        if (
            GiskardBlackboard().tree_config.is_closed_loop()
            and group_name == self.robot.name
        ):
            self.tree.control_loop_branch.closed_loop_synchronization.sync_joint_state2_topic(
                group_name=group_name, topic_name=topic_name
            )

    def add_base_cmd_velocity(
        self,
        cmd_vel_topic: Optional[str] = None,
        joint: OmniDrive = None,
        track_only_velocity: bool = False,
    ):
        """
        Tell Giskard how it can control an odom joint of the robot.
        :param cmd_vel_topic: a Twist topic
        :param track_only_velocity: The tracking mode. If true, any position error is not considered which makes
                                    the tracking smoother but less accurate.
        :param joint: omni or diff drive joint. Doesn't need to be specified if there is only one.
        """
        if cmd_vel_topic is None:
            cmd_vel_topic = search_for_unique_subscriber_of_type(Twist)
        if GiskardBlackboard().tree_config.is_closed_loop():
            self.tree.control_loop_branch.send_controls.add_send_cmd_velocity(
                topic_name=cmd_vel_topic, joint=joint
            )
        elif GiskardBlackboard().tree_config.is_open_loop():
            self.tree.execute_traj.add_base_traj_action_server(
                cmd_vel_topic=cmd_vel_topic, track_only_velocity=track_only_velocity
            )

    def register_controlled_joints(
        self, joint_names: List[Union[str, PrefixedName]]
    ) -> None:
        if not GiskardBlackboard().tree_config.is_standalone():
            raise SetupException(
                f"Joints only need to be registered in StandAlone mode."
            )
        for joint_name in joint_names:
            connection: ActiveConnection = self.world.get_connection_by_name(joint_name)
            if not isinstance(connection, ActiveConnection):
                raise Exception(
                    f"{joint_name} is not an active connection and cannot be controlled."
                )
            connection.has_hardware_interface = True

    def add_follow_joint_trajectory_server(
        self,
        namespace: str,
        group_name: Optional[str] = None,
        fill_velocity_values: bool = False,
        path_tolerance: Dict[Derivatives, float] = None,
    ):
        """
        Connect Giskard to a follow joint trajectory server. It will automatically figure out which joints are offered
        and can be controlled.
        :param namespace: namespace of the action server
        :param group_name: set if there are multiple robots
        :param fill_velocity_values: whether to fill the velocity entries in the message send to the robot
        """
        if group_name is None:
            group_name = self.world.robot_name
        if not GiskardBlackboard().tree_config.is_open_loop():
            raise SetupException(
                "add_follow_joint_trajectory_server only works in planning mode"
            )
        self.tree.execute_traj.add_follow_joint_traj_action_server(
            namespace=namespace,
            group_name=group_name,
            fill_velocity_values=fill_velocity_values,
            path_tolerance=path_tolerance,
        )

    def discover_interfaces_from_controller_manager(
        self,
        controller_manager_name: str = "controller_manager",
        whitelist: Optional[List[str]] = None,
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
                        cmd_topic=cmt_topic, joints=joints
                    )
                elif controller.type == "diff_drive_controller/DiffDriveController":
                    self.add_base_cmd_velocity(controller.name)

    def __filter_controllers_with_whitelist(
        self, controllers: list, whitelist: Optional[List[str]]
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

    def add_joint_velocity_controller(self, namespaces: List[str]):
        """
        For closed loop mode. Tell Giskard how it can send velocities to joints.
        :param namespaces: A list of namespaces where Giskard can find the topics and rosparams.
        """
        self.tree.control_loop_branch.send_controls.add_joint_velocity_controllers(
            namespaces
        )

    def add_joint_velocity_group_controller(
        self, cmd_topic: str, connections: List[str]
    ):
        """
        For closed loop mode. Tell Giskard how it can send velocities for a group of connections.
        """
        controlled_connections: List[Connection] = []
        for i in range(len(connections)):
            controlled_connections.append(
                GiskardBlackboard().executor.context.world.get_connection_by_name(
                    connections[i]
                )
            )
        self.tree.control_loop_branch.send_controls.add_joint_velocity_group_controllers(
            cmd_topic=cmd_topic, connections=controlled_connections
        )


class StandAloneRobotInterfaceConfig(RobotInterfaceConfig):
    joint_names: List[str]

    def __init__(self, joint_names: List[str]):
        self.joint_names = joint_names

    def setup(self):
        self.register_controlled_joints(self.joint_names)
