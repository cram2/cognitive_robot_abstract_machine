"""
Reading what a ros2_control controller manager knows about the robot it drives.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum

from controller_manager_msgs.srv import ListControllers
from sensor_msgs.msg import JointState
from std_msgs.msg import Float64MultiArray
from typing_extensions import TYPE_CHECKING

from giskardpy.middleware.ros2.exceptions import (
    ControllersNotActiveError,
    ControllersNotLoadedError,
)
from giskardpy.middleware.ros2.ros2_interface import (
    call_service,
    get_parameters,
    search_for_publisher_of_node_with_type,
    search_for_subscriber_of_node_with_type,
)

if TYPE_CHECKING:
    from controller_manager_msgs.msg import ControllerState as ControllerStateMessage


class ControllerType(StrEnum):
    """
    The kinds of ros2_control controllers Giskard can talk to, as the controller manager
    names them.
    """

    JOINT_STATE_BROADCASTER = "joint_state_broadcaster/JointStateBroadcaster"
    JOINT_GROUP_VELOCITY = "velocity_controllers/JointGroupVelocityController"
    DIFF_DRIVE = "diff_drive_controller/DiffDriveController"


class ControllerState(StrEnum):
    """
    The lifecycle states of a controller, as the controller manager names them.
    """

    ACTIVE = "active"
    INACTIVE = "inactive"


class ControllerParameter(StrEnum):
    """
    Parameters Giskard reads from a controller.
    """

    JOINTS = "joints"
    """
    The joints a controller commands, in the order of its command layout.
    """


@dataclass
class ControllerManager:
    """
    A ros2_control controller manager and the controllers it loaded.
    """

    name: str = "controller_manager"
    """
    The name of the controller manager node, including the namespace it runs in.
    """

    @property
    def node_name(self) -> str:
        """
        The name of the controller manager node without its namespace.
        """
        return self.name.strip("/").rpartition("/")[2]

    @property
    def namespace(self) -> str:
        """
        The namespace the controller manager and its controllers run in.
        """
        return "/" + self.name.strip("/").rpartition("/")[0]

    def active_controllers(
        self, whitelist: list[str] | None = None
    ) -> list[ControllerStateMessage]:
        """
        :param whitelist: The names of the controllers to consider, all if ``None``.
        :return: The controllers that currently run.
        :raises ControllersNotLoadedError: If the whitelist names a controller the
            manager did not load.
        :raises ControllersNotActiveError: If the whitelist names a controller that is
            loaded but does not run.
        """
        controllers = call_service(
            ListControllers, f"{self.name}/list_controllers", ListControllers.Request()
        ).controller
        if whitelist is not None:
            loaded_controllers = {controller.name for controller in controllers}
            missing_controllers = [
                name for name in whitelist if name not in loaded_controllers
            ]
            if missing_controllers:
                raise ControllersNotLoadedError(
                    controller_manager_name=self.name,
                    missing_controllers=missing_controllers,
                )
            controllers = [
                controller for controller in controllers if controller.name in whitelist
            ]
            inactive_controllers = [
                controller.name
                for controller in controllers
                if controller.state != ControllerState.ACTIVE
            ]
            if inactive_controllers:
                raise ControllersNotActiveError(
                    controller_manager_name=self.name,
                    inactive_controllers=inactive_controllers,
                )
        return [
            controller
            for controller in controllers
            if controller.state == ControllerState.ACTIVE
        ]

    def joints_of(self, controller_name: str) -> list[str]:
        """
        :return: The joints the controller commands, in the order of its command layout.
        """
        return (
            get_parameters(
                parameters=[ControllerParameter.JOINTS],
                node_name=f"{self.namespace.rstrip('/')}/{controller_name}",
            )
            .values[0]
            .string_array_value
        )

    def command_topic_of(self, controller_name: str) -> str:
        """
        :return: The topic a joint group controller receives its commands on.
        """
        return search_for_subscriber_of_node_with_type(
            node_name=controller_name,
            topic_type=Float64MultiArray,
            namespace=self.namespace,
        )

    def joint_state_topic_of(self, controller_name: str) -> str:
        """
        :return: The topic a joint state broadcaster publishes on.
        """
        return search_for_publisher_of_node_with_type(
            node_name=controller_name, topic_type=JointState, namespace=self.namespace
        )
