"""
Closed-loop coverage for a robot that is commanded through ros2_control.

The Universal Robots driver is started with mock hardware, which integrates the
commanded velocities, so Giskard talks to the same controller manager and controllers it
finds on a real arm.
"""

import signal
import subprocess
from collections import deque
from dataclasses import dataclass, field
from datetime import timedelta
from threading import Event, Thread
from time import monotonic, sleep, time
from typing_extensions import Iterator

import numpy as np
import pytest

from rclpy.executors import SingleThreadedExecutor
from rclpy.node import Node
from rclpy.publisher import Publisher
from sensor_msgs.msg import JointState as JointStateMessage
from std_msgs.msg import Float64MultiArray

from cramph.monitors import CountSeconds
from krrood.patterns.called_positionally import called_positionally
from cramph.statechart import Statechart
from semantic_digital_twin.adapters.ros import background_node
from semantic_digital_twin.adapters.ros.ros_graph import wait_for_future_to_complete
from giskardpy.middleware.ros2.command_publishing import (
    JointGroupVelocityCommandPublisher,
)
from giskardpy.middleware.ros2.controller_manager import ControllerState
from giskardpy.middleware.ros2.giskard import Giskard
from giskardpy.middleware.ros2.robot_interface_config import (
    ClosedLoopRobotInterfaceConfig,
)
from giskardpy.middleware.ros2.scripts.tutorial.joint_goal_client import (
    UniversalRobotJoint,
)
from giskardpy.middleware.ros2.scripts.tutorial.universal_robot_velocity import (
    create_giskard,
)
from giskardpy.middleware.ros2.giskard_tester import GiskardTester
from giskardpy.motion_statechart.graph_node import EndMotion
from giskardpy.motion_statechart.tasks.joint_tasks import JointPositionList, JointState

ament_index_packages = pytest.importorskip("ament_index_python.packages")
controller_manager_services = pytest.importorskip("controller_manager_msgs.srv")

# %% the mock robot


@dataclass(kw_only=True)
class MockRobotDriver:
    """
    The Universal Robots driver running a UR5e on mock hardware.
    """

    package: str = "ur_robot_driver"
    """
    The package whose launch file starts the controller manager of the arm.
    """

    velocity_controller: str = "forward_velocity_controller"
    """
    The driver's controller that forwards joint velocities to the hardware.
    """

    startup_timeout: timedelta = timedelta(seconds=60)
    """
    How long the driver may take until its velocity controller is active.
    """

    @property
    def launch_command(self) -> list[str]:
        """
        Starts the driver without a robot, with the velocity controller active.
        """
        return [
            "ros2",
            "launch",
            self.package,
            "ur_control.launch.py",
            "ur_type:=ur5e",
            "robot_ip:=0.0.0.0",
            "use_mock_hardware:=true",
            f"initial_joint_controller:={self.velocity_controller}",
            "launch_rviz:=false",
        ]

    def velocity_controller_is_active(self) -> bool:
        """
        Whether the controller manager is up and reports the velocity controller as
        active.
        """
        node = background_node.get_node()
        client = node.create_client(
            controller_manager_services.ListControllers,
            "controller_manager/list_controllers",
        )
        if not client.service_is_ready():
            node.destroy_client(client)
            return False
        future = client.call_async(
            controller_manager_services.ListControllers.Request()
        )
        wait_for_future_to_complete(future)
        node.destroy_client(client)
        return any(
            controller.name == self.velocity_controller
            and controller.state == ControllerState.ACTIVE
            for controller in future.result().controller
        )


@pytest.fixture(scope="module")
def mock_robot_driver() -> Iterator[MockRobotDriver]:
    """
    The driver of a UR5e running on mock hardware for all tests of this module.
    """
    driver = MockRobotDriver()
    if driver.package not in ament_index_packages.get_packages_with_prefixes():
        pytest.skip(f"{driver.package} is not installed")
    process = subprocess.Popen(
        driver.launch_command,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        start_new_session=True,
    )
    yield driver
    process.send_signal(signal.SIGINT)
    process.wait()


@pytest.fixture()
def mock_robot(
    mock_robot_driver: MockRobotDriver, init_background_node
) -> MockRobotDriver:
    """
    The mock robot, once it can be commanded.
    """
    deadline = time() + mock_robot_driver.startup_timeout.total_seconds()
    while not mock_robot_driver.velocity_controller_is_active():
        assert time() < deadline, "The mock robot did not start in time."
        sleep(0.1)
    return mock_robot_driver


# %% discovering the controllers


@pytest.fixture()
def connected_giskard(mock_robot) -> Iterator[Giskard]:
    """
    A Giskard that discovered the controllers of the mock robot.
    """
    giskard = create_giskard(background_node.get_node())
    giskard.setup()
    yield giskard
    giskard.close_world_model_ros_interface()


def test_discovery_leaves_the_node_on_the_executor_of_giskard(
    connected_giskard,
) -> None:
    """
    Giskard only hears the robot and its clients while its own executor spins the node.
    """
    assert background_node.get_node().executor is background_node.executor


def test_discovery_commands_the_joints_of_the_active_velocity_controller(
    connected_giskard, mock_robot: MockRobotDriver
) -> None:
    [publisher] = connected_giskard.motion_server.control_loop.command_publishers

    assert isinstance(publisher, JointGroupVelocityCommandPublisher)
    assert publisher.command_topic == f"/{mock_robot.velocity_controller}/commands"
    assert [connection.name.name for connection in publisher.connections] == list(
        UniversalRobotJoint
    )


# %% goal convergence


@dataclass(kw_only=True)
class ControllerManagerRobotTester(GiskardTester):
    """
    A closed-loop Giskard commanding a robot through its controller manager.
    """

    def setup_giskard(self) -> Giskard:
        return create_giskard(background_node.get_node())


@pytest.fixture()
def robot(mock_robot) -> Iterator[ControllerManagerRobotTester]:
    tester = ControllerManagerRobotTester()
    yield tester
    tester.close()


def test_a_joint_goal_moves_the_mock_robot(robot: ControllerManagerRobotTester) -> None:
    """
    In a closed loop the world is overwritten with what the robot reports, so the joint
    only arrives if the commands reached the hardware and its state came back.
    """
    shoulder_pan_goal = 0.5
    shoulder_pan = robot.world.get_connection_by_name(UniversalRobotJoint.SHOULDER_PAN)

    motion_statechart = Statechart(context=robot.api.context)
    motion_statechart.add_node(
        node=(
            joint_goal := JointPositionList(
                goal_state=JointState.from_str_dict(
                    {UniversalRobotJoint.SHOULDER_PAN: shoulder_pan_goal},
                    robot.api.world,
                )
            )
        )
    )
    motion_statechart.add_node(node=EndMotion.when_true(node=joint_goal))
    robot.api.compile(statechart=motion_statechart)
    robot.api.execute()

    np.testing.assert_allclose(shoulder_pan.position, shoulder_pan_goal, atol=1e-2)


# %% a robot that reacts late to its commands


@dataclass(kw_only=True)
class RelayedCommand:
    """
    A command waiting in the relay.
    """

    arrival_time: float
    """
    Monotonic time the command reached the relay.
    """

    message: Float64MultiArray
    """
    The command as Giskard sent it.
    """


@dataclass(kw_only=True)
class CommandDelayRelay:
    """
    Forwards velocity commands to the mock robot only after an actuation delay, like
    hardware that reacts late to what it is sent.

    Runs on a node of its own, so that its timing does not depend on the load of the
    node Giskard runs on.
    """

    node: Node
    """
    The node the relay subscribes and publishes on.
    """

    input_topic: str
    """
    The topic Giskard sends its commands to.
    """

    output_topic: str
    """
    The topic of the mock robot's velocity controller.
    """

    actuation_delay: timedelta
    """
    How long every command is held back.
    """

    polling_period: timedelta = timedelta(milliseconds=0.5)
    """
    How long the forwarding thread waits before it looks for a due command again.
    """

    held_commands: deque[RelayedCommand] = field(init=False, default_factory=deque)
    """
    The commands not forwarded yet, oldest first.
    """

    stopped: Event = field(init=False, default_factory=Event)
    """
    Set to end the forwarding thread.
    """

    publisher: Publisher = field(init=False)
    """
    Publishes the due commands on :attr:`output_topic`.
    """

    forwarding_thread: Thread = field(init=False)
    """
    Forwards the commands once they are due.
    """

    def __post_init__(self):
        """
        Start relaying from :attr:`input_topic` to :attr:`output_topic`.
        """
        self.publisher = self.node.create_publisher(
            Float64MultiArray, self.output_topic, 10
        )
        self.node.create_subscription(
            Float64MultiArray, self.input_topic, self.hold_command, 10
        )
        self.forwarding_thread = Thread(target=self.forward_due_commands, daemon=True)
        self.forwarding_thread.start()

    @called_positionally
    def hold_command(self, message: Float64MultiArray) -> None:
        """
        Keep a command until it is due.
        """
        self.held_commands.append(
            RelayedCommand(arrival_time=monotonic(), message=message)
        )

    def forward_due_commands(self) -> None:
        """
        Forward every command once its actuation delay has passed.
        """
        delay = self.actuation_delay.total_seconds()
        while not self.stopped.is_set():
            if (
                self.held_commands
                and self.held_commands[0].arrival_time + delay <= monotonic()
            ):
                self.publisher.publish(self.held_commands.popleft().message)
                continue
            sleep(self.polling_period.total_seconds())

    def stop(self) -> None:
        """
        End the forwarding thread.
        """
        self.stopped.set()
        self.forwarding_thread.join()


@dataclass(kw_only=True)
class JointPositionRecorder:
    """
    Records the positions the mock robot reports for one joint.
    """

    node: Node
    """
    The node the recorder subscribes on.
    """

    joint_name: str
    """
    The joint whose positions are recorded.
    """

    positions: list[float] = field(init=False, default_factory=list)
    """
    The reported positions, oldest first.
    """

    def __post_init__(self):
        """
        Start recording the joint states of the mock robot.
        """
        self.node.create_subscription(
            JointStateMessage, "/joint_states", self.record, 100
        )

    @called_positionally
    def record(self, message: JointStateMessage) -> None:
        """
        Keep the position of :attr:`joint_name`.
        """
        self.positions.append(
            message.position[list(message.name).index(self.joint_name)]
        )


@dataclass(kw_only=True)
class RelayedCommandInterface(ClosedLoopRobotInterfaceConfig):
    """
    Reads the joint states of the mock robot and commands its arm through a topic other
    than the one of its velocity controller.
    """

    command_topic: str
    """
    The topic the velocities of the arm are sent to.
    """

    actuation_delay: timedelta | None
    """
    The actuation delay Giskard compensates, ``None`` for none.
    """

    def setup(self) -> None:
        self.sync_joint_state_topic(topic_name="/joint_states")
        self.add_joint_group_velocity_command_topic(
            command_topic=self.command_topic,
            connections=list(UniversalRobotJoint),
            actuation_delay=self.actuation_delay,
        )


@dataclass(kw_only=True)
class RelayedCommandRobotTester(GiskardTester):
    """
    A closed-loop Giskard commanding the mock robot through a relay.
    """

    robot_interface_config: RelayedCommandInterface
    """
    How Giskard reads and commands the mock robot.
    """

    def setup_giskard(self) -> Giskard:
        return create_giskard(
            background_node.get_node(),
            robot_interface_config=self.robot_interface_config,
        )


@dataclass(kw_only=True)
class LateReactingRobot:
    """
    The mock robot behind a relay that delays its commands, observed by a recorder.
    """

    tester: RelayedCommandRobotTester
    """
    The Giskard commanding the robot.
    """

    recorder: JointPositionRecorder
    """
    Records what the robot reports for the moved joint.
    """


@pytest.fixture()
def relay_node(init_background_node) -> Iterator[Node]:
    """
    A node with an executor of its own, for the relay and the recorder.
    """
    node = Node("late_reacting_robot")
    executor = SingleThreadedExecutor()
    executor.add_node(node)
    spinning_thread = Thread(target=executor.spin, daemon=True)
    spinning_thread.start()
    yield node
    executor.shutdown()
    node.destroy_node()


@pytest.fixture()
def late_reacting_robot(
    mock_robot: MockRobotDriver, relay_node: Node
) -> Iterator[LateReactingRobot]:
    """
    The mock robot reacting 0.3 s late, as a Stretch was measured to, with Giskard
    compensating that delay.
    """
    actuation_delay = timedelta(seconds=0.3)
    relay = CommandDelayRelay(
        node=relay_node,
        input_topic="/relayed_commands",
        output_topic=f"/{mock_robot.velocity_controller}/commands",
        actuation_delay=actuation_delay,
    )
    tester = RelayedCommandRobotTester(
        robot_interface_config=RelayedCommandInterface(
            command_topic=relay.input_topic, actuation_delay=actuation_delay
        )
    )
    yield LateReactingRobot(
        tester=tester,
        recorder=JointPositionRecorder(
            node=relay_node, joint_name=UniversalRobotJoint.SHOULDER_PAN
        ),
    )
    tester.close()
    relay.stop()


def test_dead_time_compensation_keeps_a_late_reacting_robot_from_overshooting(
    late_reacting_robot: LateReactingRobot,
) -> None:
    """
    The joint is held at its goal for a while after reaching it, so that an overshoot is
    not cut short by the end of the motion.
    """
    tester = late_reacting_robot.tester
    shoulder_pan = tester.world.get_connection_by_name(UniversalRobotJoint.SHOULDER_PAN)
    distance = 0.5
    hold_seconds = 3.0
    shoulder_pan_goal = shoulder_pan.position + distance

    motion_statechart = Statechart(context=tester.api.context)
    motion_statechart.add_node(
        node=(
            joint_goal := JointPositionList(
                goal_state=JointState.from_str_dict(
                    {UniversalRobotJoint.SHOULDER_PAN: shoulder_pan_goal},
                    tester.api.world,
                )
            )
        )
    )
    motion_statechart.add_node(node=(hold_time := CountSeconds(seconds=hold_seconds)))
    motion_statechart.add_node(node=EndMotion.when_true(node=hold_time))
    tester.api.compile(statechart=motion_statechart)
    tester.api.execute()

    overshoot = max(late_reacting_robot.recorder.positions) - shoulder_pan_goal
    assert overshoot < joint_goal.threshold
    assert late_reacting_robot.recorder.positions[-1] == pytest.approx(
        shoulder_pan_goal, abs=joint_goal.threshold
    )
