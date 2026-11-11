from dataclasses import dataclass

import numpy as np
import pytest
from giskardpy.middleware.ros2.robot_interface_config import (
    StandAloneRobotInterfaceConfig,
)
from giskardpy.middleware.ros2.server_config import GiskardServerConfig
from giskardpy.middleware.ros2.giskard import Giskard
from giskardpy.data_types.exceptions import MaxTrajectoryLengthException
from giskardpy.middleware.ros2.giskard_tester import GiskardTester
from giskardpy.motion_statechart.graph_node import EndMotion
from cramph.monitors import CountSimulationTimeSeconds
from cramph.node import CancelStatechart
from cramph.statechart import (
    Statechart,
)
from giskardpy.motion_statechart.tasks.joint_tasks import JointPositionList, JointState
from giskardpy.qp.qp_controller_config import QPControllerConfig
from krrood_spatial_math.spatial_types import HomogeneousTransformationMatrix
from semantic_digital_twin.robots.daisy import DAiSy
from semantic_digital_twin.specifications.robots import RobotSpecification
from semantic_digital_twin.specifications.worlds import WorldSpecification


# %% trajectory length limit

MAX_TRAJECTORY_LENGTH_SECONDS = 60.0
"""
Simulation time after which a motion is cancelled.

Twice the longest motion planned here, so that a motion which fails to terminate aborts
with a :class:`MaxTrajectoryLengthException` instead of running forever.
"""


def add_trajectory_length_limit(motion_statechart: Statechart) -> None:
    """
    Cancels the motion once its trajectory grows past
    :data:`MAX_TRAJECTORY_LENGTH_SECONDS`.
    """
    motion_statechart.add_node(
        max_length := CountSimulationTimeSeconds(seconds=MAX_TRAJECTORY_LENGTH_SECONDS)
    )
    motion_statechart.add_node(
        CancelStatechart.when_true(max_length, MaxTrajectoryLengthException())
    )


# %% fixtures


@pytest.fixture()
def default_joint_state(default_daisy_pose: dict[str, float]) -> dict[str, float]:
    return default_daisy_pose


@dataclass
class DAiSyTester(GiskardTester):
    def setup_giskard(self) -> Giskard:
        return Giskard(
            world_source=WorldSpecification(
                robots=[RobotSpecification.specify_real_robot(DAiSy)]
            ),
            robot_interface_config=StandAloneRobotInterfaceConfig(),
            server_config=GiskardServerConfig(),
            qp_controller_config=QPControllerConfig.create_with_simulation_defaults(),
        )


@pytest.fixture()
def robot():
    c = DAiSyTester()
    try:
        yield c
    finally:
        print("tear down")
        c.close()


# %% motions through the action server


def test_joints1(giskard: DAiSyTester) -> None:
    statechart = Statechart(context=giskard.api.context)
    statechart.add_node(
        joint_goal := JointPositionList(
            goal_state=JointState.from_str_dict(
                {
                    "left_wrist_1_joint": 1.23,
                    "left_wrist_2_joint": 1.23,
                    "left_wrist_3_joint": 1.23,
                },
                giskard.api.world,
            )
        ),
    )
    statechart.add_node(EndMotion.when_true(joint_goal))
    add_trajectory_length_limit(statechart)
    giskard.api.compile(statechart)
    giskard.api.execute()

    base = giskard.world.get_kinematic_structure_entity_by_name("left_base_link")
    finger_tip = giskard.world.get_kinematic_structure_entity_by_name(
        "left_gripper_left_finger_tip_link"
    )
    base_T_finger_current = giskard.world.compute_forward_kinematics(
        root=base, tip=finger_tip
    )
    base_T_finger_expected = HomogeneousTransformationMatrix.from_xyz_quaternion(
        pos_x=0.3807,
        pos_y=0.1862,
        pos_z=0.5557,
        quat_x=0.8587,
        quat_y=0.2248,
        quat_z=0.4539,
        quat_w=-0.0767,
        reference_frame=base,
    )
    assert np.allclose(base_T_finger_current, base_T_finger_expected, atol=1e-2)
