"""
Closed-loop coverage for a robot that is commanded through ros2_control.

The Universal Robots driver is started with mock hardware, which integrates the
commanded velocities, so Giskard talks to the same controller manager and controllers it
finds on a real arm.
"""

import signal
import subprocess
from dataclasses import dataclass
from datetime import timedelta
from enum import StrEnum
from time import sleep, time
from typing import Iterator

import numpy as np
import pytest

from giskardpy.middleware.ros2 import rospy
from giskardpy.middleware.ros2.command_publishing import (
    JointGroupVelocityCommandPublisher,
)
from giskardpy.middleware.ros2.controller_manager import ControllerState
from giskardpy.middleware.ros2.giskard import Giskard
from giskardpy.middleware.ros2.scripts.tutorial.universal_robot_velocity import (
    create_giskard,
)
from giskardpy.middleware.ros2.utils.utils_for_tests import GiskardTester
from giskardpy.motion_statechart.graph_node import EndMotion
from giskardpy.motion_statechart.motion_statechart import MotionStatechart
from giskardpy.motion_statechart.tasks.joint_tasks import JointPositionList, JointState

ament_index_packages = pytest.importorskip("ament_index_python.packages")
controller_manager_services = pytest.importorskip("controller_manager_msgs.srv")

# %% the mock robot


class UniversalRobotJoint(StrEnum):
    """
    The joints of the arm, in the order the velocity controller expects its commands.
    """

    SHOULDER_PAN = "shoulder_pan_joint"
    SHOULDER_LIFT = "shoulder_lift_joint"
    ELBOW = "elbow_joint"
    WRIST_1 = "wrist_1_joint"
    WRIST_2 = "wrist_2_joint"
    WRIST_3 = "wrist_3_joint"


@dataclass
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
        node = rospy.get_node()
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
        rospy.wait_for_future_to_complete(future)
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
def mock_robot(mock_robot_driver: MockRobotDriver, init_rospy) -> MockRobotDriver:
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
    giskard = create_giskard()
    giskard.setup()
    yield giskard
    giskard.close_world_model_ros_interface()


def test_discovery_leaves_the_node_on_the_executor_of_giskard(connected_giskard):
    """
    Giskard only hears the robot and its clients while its own executor spins the node.
    """
    assert rospy.get_node().executor is rospy.executor


def test_discovery_commands_the_joints_of_the_active_velocity_controller(
    connected_giskard, mock_robot: MockRobotDriver
):
    [publisher] = connected_giskard.motion_server.control_loop.command_publishers

    assert isinstance(publisher, JointGroupVelocityCommandPublisher)
    assert publisher.command_topic == f"/{mock_robot.velocity_controller}/commands"
    assert [connection.name.name for connection in publisher.connections] == list(
        UniversalRobotJoint
    )


# %% goal convergence


@dataclass
class ControllerManagerRobotTester(GiskardTester):
    """
    A closed-loop Giskard commanding a robot through its controller manager.
    """

    def setup_giskard(self) -> Giskard:
        return create_giskard()


@pytest.fixture()
def robot(mock_robot) -> Iterator[ControllerManagerRobotTester]:
    tester = ControllerManagerRobotTester()
    yield tester
    tester.close()


def test_a_joint_goal_moves_the_mock_robot(robot: ControllerManagerRobotTester):
    """
    In a closed loop the world is overwritten with what the robot reports, so the joint
    only arrives if the commands reached the hardware and its state came back.
    """
    shoulder_pan_goal = 0.5
    shoulder_pan = robot.world.get_connection_by_name(UniversalRobotJoint.SHOULDER_PAN)

    motion_statechart = MotionStatechart()
    motion_statechart.add_node(
        joint_goal := JointPositionList(
            goal_state=JointState.from_str_dict(
                {UniversalRobotJoint.SHOULDER_PAN: shoulder_pan_goal}, robot.api.world
            )
        )
    )
    motion_statechart.add_node(EndMotion.when_true(joint_goal))
    robot.api.execute(motion_statechart)

    np.testing.assert_allclose(shoulder_pan.position, shoulder_pan_goal, atol=1e-2)
