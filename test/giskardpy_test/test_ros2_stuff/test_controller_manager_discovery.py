"""
Coverage for reading the interfaces of a robot from its ros2_control controller manager.

The controller manager and its controllers are mimicked by nodes inside the test
process, so no driver has to run.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import timedelta
from threading import Thread
from time import sleep, time
from typing import Iterator

import pytest
from rclpy.executors import SingleThreadedExecutor
from rclpy.node import Node
from sensor_msgs.msg import JointState
from std_msgs.msg import Float64MultiArray

from giskardpy.middleware.ros2 import rospy
from giskardpy.middleware.ros2.controller_manager import (
    ControllerManager,
    ControllerParameter,
    ControllerState,
    ControllerType,
)
from giskardpy.middleware.ros2.exceptions import (
    ControllersNotActiveError,
    ControllersNotLoadedError,
)
from giskardpy.middleware.ros2.giskard import Giskard
from giskardpy.middleware.ros2.robot_interface_config import (
    ClosedLoopRobotInterfaceConfig,
)
from giskardpy.middleware.ros2.scripts.iai_robots.daisy.configs import (
    DAiSyVelocityInterface,
    WorldWithDaisyConfig,
)
from giskardpy.middleware.ros2.scripts.iai_robots.tracy.configs import (
    TracyVelocityInterface,
    WorldWithTracyConfig,
)
from giskardpy.middleware.ros2.server_config import GiskardServerConfig
from giskardpy.middleware.ros2.utils.utils import load_xacro
from giskardpy.qp.qp_controller_config import QPControllerConfig
from semantic_digital_twin.adapters.ros.input_synchronization import (
    PendingJointStateSynchronizer,
)
from semantic_digital_twin.robots.daisy import DAiSy, DAiSyController, DAiSyJoint
from semantic_digital_twin.robots.pr2 import PR2Joint
from semantic_digital_twin.robots.tracy import (
    Tracy,
    TracyController,
    TracyControllerManager,
    TracyJoint,
)

from .test_robot_interface_config import closed_loop_pr2, controlled_joint_names

controller_manager_messages = pytest.importorskip("controller_manager_msgs.msg")
controller_manager_services = pytest.importorskip("controller_manager_msgs.srv")

# %% the mimicked controller manager


@dataclass
class ControllerMimic:
    """
    A controller as a controller manager reports it.
    """

    name: str
    """
    The name the controller was loaded under.
    """

    controller_type: ControllerType
    """
    The kind of controller.
    """

    state: ControllerState = ControllerState.ACTIVE
    """
    Whether the controller currently runs.
    """

    joints: list[str] = field(default_factory=list)
    """
    The joints the controller commands, in the order of its command layout.
    """


@dataclass
class ControllerManagerMimic:
    """
    Nodes that answer like a controller manager and the controllers it loaded.
    """

    name: str
    """
    The name of the controller manager, including the namespace it runs in.
    """

    controllers: list[ControllerMimic]
    """
    The controllers the manager reports.
    """

    nodes: list[Node] = field(init=False, default_factory=list)
    """
    The node of the manager followed by one node per controller.
    """

    def __post_init__(self):
        manager = ControllerManager(self.name)
        manager_node = Node(manager.node_name, namespace=manager.namespace)
        manager_node.create_service(
            controller_manager_services.ListControllers,
            "~/list_controllers",
            self.list_controllers,
        )
        self.nodes.append(manager_node)
        for controller in self.controllers:
            self.nodes.append(self.create_controller_node(controller))

    def create_controller_node(self, controller: ControllerMimic) -> Node:
        """
        :return: A node offering the parameters and topics of ``controller``.
        """
        node = Node(controller.name, namespace=ControllerManager(self.name).namespace)
        match controller.controller_type:
            case ControllerType.JOINT_GROUP_VELOCITY:
                node.declare_parameter(ControllerParameter.JOINTS, controller.joints)
                node.create_subscription(
                    Float64MultiArray, "~/commands", self.ignore_command, 10
                )
            case ControllerType.JOINT_STATE_BROADCASTER:
                node.create_publisher(JointState, "joint_states", 10)
        return node

    def ignore_command(self, command: Float64MultiArray) -> None:
        """
        Accept a command without acting on it.
        """

    def list_controllers(self, request, response):
        """
        Report the controllers of this manager.
        """
        response.controller = [
            controller_manager_messages.ControllerState(
                name=controller.name,
                type=str(controller.controller_type),
                state=str(controller.state),
            )
            for controller in self.controllers
        ]
        return response

    def controller_node(self, controller: ControllerMimic) -> Node:
        """
        :return: The node standing in for ``controller``.
        """
        return self.nodes[1 + self.controllers.index(controller)]

    def command_topic_of(self, controller: ControllerMimic) -> str:
        """
        :return: The topic ``controller`` receives its commands on.
        """
        [subscription] = [
            subscription
            for subscription in self.controller_node(controller).subscriptions
            if subscription.msg_type is Float64MultiArray
        ]
        return subscription.topic_name

    def joint_state_topic_of(self, controller: ControllerMimic) -> str:
        """
        :return: The topic ``controller`` publishes the joint states on.
        """
        [publisher] = [
            publisher
            for publisher in self.controller_node(controller).publishers
            if publisher.msg_type is JointState
        ]
        return publisher.topic_name

    def is_visible_to(self, node: Node) -> bool:
        """
        Whether ``node`` already knows every node of this mimic.
        """
        known_nodes = node.get_node_names_and_namespaces()
        return all(
            (mimic_node.get_name(), mimic_node.get_namespace()) in known_nodes
            for mimic_node in self.nodes
        )


@dataclass
class ControllerManagerMimics:
    """
    Serves several mimicked controller managers while a test runs.
    """

    managers: list[ControllerManagerMimic]
    """
    The controller managers that are served.
    """

    graph_timeout: timedelta = timedelta(seconds=10)
    """
    How long the mimicked nodes may take to become visible to Giskard's node.
    """

    executor: SingleThreadedExecutor = field(
        init=False, default_factory=SingleThreadedExecutor
    )
    """
    Spins the nodes of all managers.
    """

    serving: bool = field(init=False, default=True)
    """
    Whether the nodes are still spun.
    """

    thread: Thread = field(init=False)
    """
    The thread the executor spins on.
    """

    def __post_init__(self):
        for manager in self.managers:
            for node in manager.nodes:
                self.executor.add_node(node)
        self.thread = Thread(target=self.serve, daemon=True)
        self.thread.start()
        deadline = time() + self.graph_timeout.total_seconds()
        while not all(
            manager.is_visible_to(rospy.get_node()) for manager in self.managers
        ):
            assert time() < deadline, "The mimicked controllers did not appear."
            sleep(0.05)

    def serve(self) -> None:
        """
        Spin the nodes until :meth:`stop` is called.
        """
        while self.serving:
            self.executor.spin_once(timeout_sec=0.05)

    def stop(self) -> None:
        """
        Stop serving and destroy the nodes.
        """
        self.serving = False
        self.thread.join()
        for manager in self.managers:
            for node in manager.nodes:
                self.executor.remove_node(node)
                node.destroy_node()


@pytest.fixture()
def serve(init_rospy) -> Iterator[list[ControllerManagerMimics]]:
    """
    Collects the mimics a test starts and stops them before the ros context goes down.
    """
    served: list[ControllerManagerMimics] = []
    yield served
    for mimics in served:
        mimics.stop()


def start(
    serve: list[ControllerManagerMimics], *managers: ControllerManagerMimic
) -> None:
    """
    Serve ``managers`` until the test that requested ``serve`` ends.
    """
    serve.append(ControllerManagerMimics(managers=list(managers)))


# %% interfaces that only discover


@dataclass
class CommandDiscoveringInterface(ClosedLoopRobotInterfaceConfig):
    """
    An interface that takes the joints it commands from a controller manager.
    """

    controller_manager_name: str
    """
    The controller manager that is asked for its controllers.
    """

    whitelist: list[str] | None = None
    """
    The controllers to consider, all if ``None``.
    """

    def setup(self):
        self.add_velocity_controllers_from_controller_manager(
            self.controller_manager_name, whitelist=self.whitelist
        )


@dataclass
class FullyDiscoveringInterface(CommandDiscoveringInterface):
    """
    An interface that takes both its state and its commands from a controller manager.
    """

    def setup(self):
        self.discover_interfaces_from_controller_manager(
            self.controller_manager_name, whitelist=self.whitelist
        )


def joint_state_topics(giskard: Giskard) -> list[str]:
    """
    The joint state topics that are read while ``giskard`` executes no goal.
    """
    return [
        synchronizer.topic_name
        for synchronizer in giskard.motion_server.inputs.synchronizers
        if isinstance(synchronizer, PendingJointStateSynchronizer)
    ]


def commanded_joint_names_by_topic(giskard: Giskard) -> dict[str, list[str]]:
    """
    The joints ``giskard`` commands, in command order, keyed by their command topic.
    """
    return {
        publisher.command_topic: [
            connection.name.name for connection in publisher.connections
        ]
        for publisher in giskard.motion_server.control_loop.command_publishers
    }


def joints_by_command_topic(
    manager: ControllerManagerMimic,
) -> dict[str, list[str]]:
    """
    The joints the active velocity controllers of ``manager`` command, keyed by their
    command topic.
    """
    return {
        manager.command_topic_of(controller): controller.joints
        for controller in manager.controllers
        if controller.controller_type is ControllerType.JOINT_GROUP_VELOCITY
        and controller.state is ControllerState.ACTIVE
    }


# %% discovering the commanded joints


def test_a_controller_manager_in_a_namespace_yields_its_velocity_controller(serve):
    manager = ControllerManagerMimic(
        name="head/controller_manager",
        controllers=[
            ControllerMimic(
                name="forward_velocity_controller",
                controller_type=ControllerType.JOINT_GROUP_VELOCITY,
                joints=[PR2Joint.HEAD_TILT, PR2Joint.HEAD_PAN],
            )
        ],
    )
    start(serve, manager)
    giskard = closed_loop_pr2(CommandDiscoveringInterface(manager.name))

    giskard.setup()

    assert commanded_joint_names_by_topic(giskard) == joints_by_command_topic(manager)


def test_an_inactive_velocity_controller_is_not_commanded(serve):
    active = ControllerMimic(
        name="head_velocity_controller",
        controller_type=ControllerType.JOINT_GROUP_VELOCITY,
        joints=[PR2Joint.HEAD_PAN],
    )
    inactive = ControllerMimic(
        name="torso_velocity_controller",
        controller_type=ControllerType.JOINT_GROUP_VELOCITY,
        state=ControllerState.INACTIVE,
        joints=[PR2Joint.TORSO_LIFT],
    )
    manager = ControllerManagerMimic(
        name="controller_manager", controllers=[active, inactive]
    )
    start(serve, manager)
    giskard = closed_loop_pr2(CommandDiscoveringInterface(manager.name))

    giskard.setup()

    assert controlled_joint_names(giskard.world_config.world) == set(active.joints)


def test_a_whitelist_limits_the_commanded_controllers(serve):
    listed = ControllerMimic(
        name="head_velocity_controller",
        controller_type=ControllerType.JOINT_GROUP_VELOCITY,
        joints=[PR2Joint.HEAD_PAN],
    )
    unlisted = ControllerMimic(
        name="torso_velocity_controller",
        controller_type=ControllerType.JOINT_GROUP_VELOCITY,
        joints=[PR2Joint.TORSO_LIFT],
    )
    manager = ControllerManagerMimic(
        name="controller_manager", controllers=[listed, unlisted]
    )
    start(serve, manager)
    giskard = closed_loop_pr2(
        CommandDiscoveringInterface(manager.name, whitelist=[listed.name])
    )

    giskard.setup()

    assert controlled_joint_names(giskard.world_config.world) == set(listed.joints)


def test_a_whitelisted_controller_that_is_not_loaded_raises(serve):
    manager = ControllerManagerMimic(
        name="controller_manager",
        controllers=[
            ControllerMimic(
                name="head_velocity_controller",
                controller_type=ControllerType.JOINT_GROUP_VELOCITY,
                joints=[PR2Joint.HEAD_PAN],
            )
        ],
    )
    start(serve, manager)
    giskard = closed_loop_pr2(
        CommandDiscoveringInterface(
            manager.name, whitelist=["controller_that_is_not_loaded"]
        )
    )

    with pytest.raises(ControllersNotLoadedError):
        giskard.setup()


def test_a_whitelisted_controller_that_is_not_active_raises(serve):
    inactive = ControllerMimic(
        name="head_velocity_controller",
        controller_type=ControllerType.JOINT_GROUP_VELOCITY,
        state=ControllerState.INACTIVE,
        joints=[PR2Joint.HEAD_PAN],
    )
    manager = ControllerManagerMimic(name="controller_manager", controllers=[inactive])
    start(serve, manager)
    giskard = closed_loop_pr2(
        CommandDiscoveringInterface(manager.name, whitelist=[inactive.name])
    )

    with pytest.raises(ControllersNotActiveError):
        giskard.setup()


# %% discovering the joint states


def manager_with_broadcaster_and_velocity_controller() -> ControllerManagerMimic:
    """
    A controller manager in a namespace that publishes joint states and commands the head.
    """
    return ControllerManagerMimic(
        name="head/controller_manager",
        controllers=[
            ControllerMimic(
                name="joint_state_broadcaster",
                controller_type=ControllerType.JOINT_STATE_BROADCASTER,
            ),
            ControllerMimic(
                name="forward_velocity_controller",
                controller_type=ControllerType.JOINT_GROUP_VELOCITY,
                joints=[PR2Joint.HEAD_PAN],
            ),
        ],
    )


def test_discovering_only_commands_reads_no_joint_state_topic(serve):
    manager = manager_with_broadcaster_and_velocity_controller()
    start(serve, manager)
    giskard = closed_loop_pr2(CommandDiscoveringInterface(manager.name))

    giskard.setup()

    assert joint_state_topics(giskard) == []


def test_discovering_everything_reads_the_topic_of_the_joint_state_broadcaster(serve):
    manager = manager_with_broadcaster_and_velocity_controller()
    start(serve, manager)
    [broadcaster, _] = manager.controllers
    giskard = closed_loop_pr2(FullyDiscoveringInterface(manager.name))

    giskard.setup()

    assert joint_state_topics(giskard) == [manager.joint_state_topic_of(broadcaster)]


# %% robots that are commanded through their controller managers


def arm_velocity_controller(name: str, joints: list[str]) -> ControllerMimic:
    """
    An active velocity controller of one arm.
    """
    return ControllerMimic(
        name=name, controller_type=ControllerType.JOINT_GROUP_VELOCITY, joints=joints
    )


def tracy_left_arm_manager(
    *other_controllers: ControllerMimic,
) -> ControllerManagerMimic:
    """
    The controller manager of Tracy's left arm with its velocity controller, followed by
    ``other_controllers``.
    """
    return ControllerManagerMimic(
        name=TracyControllerManager.LEFT_ARM,
        controllers=[
            arm_velocity_controller(
                TracyController.FORWARD_VELOCITY,
                [
                    TracyJoint.LEFT_SHOULDER_PAN,
                    TracyJoint.LEFT_SHOULDER_LIFT,
                    TracyJoint.LEFT_ELBOW,
                    TracyJoint.LEFT_WRIST_1,
                    TracyJoint.LEFT_WRIST_2,
                    TracyJoint.LEFT_WRIST_3,
                ],
            ),
            *other_controllers,
        ],
    )


def tracy_right_arm_manager() -> ControllerManagerMimic:
    """
    The controller manager of Tracy's right arm with its velocity controller.
    """
    return ControllerManagerMimic(
        name=TracyControllerManager.RIGHT_ARM,
        controllers=[
            arm_velocity_controller(
                TracyController.FORWARD_VELOCITY,
                [
                    TracyJoint.RIGHT_SHOULDER_PAN,
                    TracyJoint.RIGHT_SHOULDER_LIFT,
                    TracyJoint.RIGHT_ELBOW,
                    TracyJoint.RIGHT_WRIST_1,
                    TracyJoint.RIGHT_WRIST_2,
                    TracyJoint.RIGHT_WRIST_3,
                ],
            )
        ],
    )


def closed_loop_tracy() -> Giskard:
    """
    A Giskard that commands Tracy through its velocity interface.
    """
    return Giskard(
        world_config=WorldWithTracyConfig(urdf=load_xacro(Tracy.get_ros_file_path())),
        robot_interface_config=TracyVelocityInterface(),
        server_config=GiskardServerConfig(),
        qp_controller_config=QPControllerConfig(target_frequency=25),
    )


def test_tracy_commands_the_joints_its_two_controller_managers_report(serve):
    left = tracy_left_arm_manager()
    right = tracy_right_arm_manager()
    start(serve, left, right)
    giskard = closed_loop_tracy()

    giskard.setup()

    assert commanded_joint_names_by_topic(giskard) == (
        joints_by_command_topic(left) | joints_by_command_topic(right)
    )


def test_tracy_does_not_command_a_velocity_controller_of_a_gripper(serve):
    gripper = arm_velocity_controller(
        "gripper_velocity_controller", [TracyJoint.LEFT_GRIPPER_LEFT_KNUCKLE]
    )
    left = tracy_left_arm_manager(gripper)
    start(serve, left, tracy_right_arm_manager())
    giskard = closed_loop_tracy()

    giskard.setup()

    assert left.command_topic_of(gripper) not in commanded_joint_names_by_topic(giskard)


def test_tracy_fails_when_an_arm_has_not_loaded_its_velocity_controller(serve):
    left_without_controllers = ControllerManagerMimic(
        name=TracyControllerManager.LEFT_ARM, controllers=[]
    )
    start(serve, left_without_controllers, tracy_right_arm_manager())
    giskard = closed_loop_tracy()

    with pytest.raises(ControllersNotLoadedError):
        giskard.setup()


def daisy_left_arm_controller() -> ControllerMimic:
    """
    The velocity controller of DAiSy's left arm.
    """
    return arm_velocity_controller(
        DAiSyController.LEFT_FORWARD_VELOCITY,
        [
            DAiSyJoint.LEFT_SHOULDER_PAN,
            DAiSyJoint.LEFT_SHOULDER_LIFT,
            DAiSyJoint.LEFT_ELBOW,
            DAiSyJoint.LEFT_WRIST_1,
            DAiSyJoint.LEFT_WRIST_2,
            DAiSyJoint.LEFT_WRIST_3,
        ],
    )


def daisy_right_arm_controller() -> ControllerMimic:
    """
    The velocity controller of DAiSy's right arm.
    """
    return arm_velocity_controller(
        DAiSyController.RIGHT_FORWARD_VELOCITY,
        [
            DAiSyJoint.RIGHT_SHOULDER_PAN,
            DAiSyJoint.RIGHT_SHOULDER_LIFT,
            DAiSyJoint.RIGHT_ELBOW,
            DAiSyJoint.RIGHT_WRIST_1,
            DAiSyJoint.RIGHT_WRIST_2,
            DAiSyJoint.RIGHT_WRIST_3,
        ],
    )


def closed_loop_daisy() -> Giskard:
    """
    A Giskard that commands DAiSy through its velocity interface.
    """
    return Giskard(
        world_config=WorldWithDaisyConfig(urdf=load_xacro(DAiSy.get_ros_file_path())),
        robot_interface_config=DAiSyVelocityInterface(),
        server_config=GiskardServerConfig(),
        qp_controller_config=QPControllerConfig(target_frequency=25),
    )


def test_daisy_commands_the_joints_its_controller_manager_reports(serve):
    manager = ControllerManagerMimic(
        name="controller_manager",
        controllers=[daisy_left_arm_controller(), daisy_right_arm_controller()],
    )
    start(serve, manager)
    giskard = closed_loop_daisy()

    giskard.setup()

    assert commanded_joint_names_by_topic(giskard) == joints_by_command_topic(manager)


def test_daisy_does_not_command_a_velocity_controller_of_a_gripper(serve):
    gripper = arm_velocity_controller(
        "gripper_velocity_controller", [DAiSyJoint.LEFT_GRIPPER_FINGER]
    )
    manager = ControllerManagerMimic(
        name="controller_manager",
        controllers=[
            daisy_left_arm_controller(),
            daisy_right_arm_controller(),
            gripper,
        ],
    )
    start(serve, manager)
    giskard = closed_loop_daisy()

    giskard.setup()

    assert manager.command_topic_of(gripper) not in commanded_joint_names_by_topic(
        giskard
    )


def test_daisy_fails_when_an_arm_has_not_loaded_its_velocity_controller(serve):
    manager = ControllerManagerMimic(
        name="controller_manager", controllers=[daisy_left_arm_controller()]
    )
    start(serve, manager)
    giskard = closed_loop_daisy()

    with pytest.raises(ControllersNotLoadedError):
        giskard.setup()
