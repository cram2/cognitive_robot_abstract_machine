"""
Standalone-controller coverage for Stretch.

This runs the real QP controller, motion server and robot description in a closed loop
with no hardware, which is the closest offline proxy for whether a goal the demo sends
will actually be reached on the robot.
"""

from dataclasses import dataclass, field

import numpy as np
import pytest

from giskardpy.middleware.ros2.giskard import Giskard
from giskardpy.middleware.ros2.giskard_tester import GiskardTester
from giskardpy.middleware.ros2.robot_interface_config import (
    StandAloneRobotInterfaceConfig,
)
from giskardpy.middleware.ros2.server_config import GiskardServerConfig
from giskardpy.qp.qp_controller_config import QPControllerConfig
from giskardpy.motion_statechart.graph_node import EndMotion
from giskardpy.motion_statechart.monitors.monitors import LocalMinimumReached
from cramph.statechart import Statechart
from giskardpy.motion_statechart.tasks.joint_tasks import JointPositionList, JointState
from semantic_digital_twin.robots.robot_parts import AbstractRobot
from semantic_digital_twin.robots.stretch import Stretch
from semantic_digital_twin.specifications.robots import RobotSpecification
from semantic_digital_twin.specifications.worlds import WorldSpecification
from semantic_digital_twin.world_description.connections import DifferentialDrive
from semantic_digital_twin.world_description.world_entity import (
    KinematicStructureEntity,
)

# %% standalone stretch


@dataclass
class StretchTester(GiskardTester):
    """
    A standalone Giskard driving Stretch through the same configuration the robot's own
    ``stretch_standalone`` launcher uses.
    """

    tool_frame: KinematicStructureEntity = field(init=False)
    """
    The frame the demo's pick and place actions drive.
    """

    def __post_init__(self):
        super().__post_init__()
        self.tool_frame = self.api.world.get_kinematic_structure_entity_by_name(
            "link_grasp_center"
        )

    def setup_giskard(self) -> Giskard:
        return Giskard(
            world_source=WorldSpecification(
                robots=[RobotSpecification.specify_real_robot(Stretch)]
            ),
            robot_interface_config=StandAloneRobotInterfaceConfig(),
            server_config=GiskardServerConfig(),
            qp_controller_config=QPControllerConfig.create_with_simulation_defaults(),
        )

    @property
    def robot(self) -> AbstractRobot:
        return self.world.get_semantic_annotations_by_type(Stretch)[0]


# %% fixtures


@pytest.fixture()
def default_joint_state():
    """
    A neutral arm configuration to seed each motion from.
    """
    return {
        "joint_lift": 0.5,
        "joint_arm_l0": 0.0,
        "joint_arm_l1": 0.0,
        "joint_arm_l2": 0.0,
        "joint_arm_l3": 0.0,
        "joint_wrist_yaw": 1.0,
        "joint_head_pan": 0.0,
        "joint_head_tilt": 0.0,
    }


@pytest.fixture()
def robot():
    tester = StretchTester()
    try:
        yield tester
    finally:
        tester.close()


def tool_frame_height(giskard: StretchTester) -> float:
    """
    The height of the tool frame above the base, which the lift motions change.
    """
    base_link = giskard.world.get_kinematic_structure_entity_by_name("base_link")
    return giskard.world.compute_forward_kinematics(
        root=base_link, tip=giskard.tool_frame
    ).position.to_np()[2]


# %% controller setup


def test_standalone_interface_controls_the_base_drive(giskard: StretchTester):
    """
    The base drive is controlled in standalone, so the controller can actually move the
    base rather than silently holding it.
    """
    drive = giskard.api.world.get_connections_by_type(DifferentialDrive)[0]

    assert drive.has_hardware_interface


# %% goal convergence


LIFT_SEED_HEIGHT = 0.5
"""
Lift position the motions start from, matching ``default_joint_state``.
"""

LIFT_GOAL_HEIGHT = 0.9
"""
Lift position the motions drive to.
"""


def test_lift_goal_moves_the_tool_frame(giskard: StretchTester):
    """
    A joint goal reaches the controller and moves the arm: raising the lift raises the
    tool frame by the commanded amount.
    """
    height_before = tool_frame_height(giskard)

    motion_statechart = Statechart(context=giskard.api.context)
    motion_statechart.add_node(
        lift_goal := JointPositionList(
            goal_state=JointState.from_str_dict(
                {"joint_lift": LIFT_GOAL_HEIGHT}, giskard.api.world
            )
        )
    )
    motion_statechart.add_node(EndMotion.when_true(lift_goal))
    giskard.api.compile(motion_statechart)
    giskard.api.execute()

    height_after = tool_frame_height(giskard)
    np.testing.assert_allclose(
        height_after - height_before, LIFT_GOAL_HEIGHT - LIFT_SEED_HEIGHT, atol=1e-2
    )


def test_local_minimum_ends_a_motion(giskard: StretchTester):
    """
    The demo accepts a local minimum as success for its cartesian stage, so a motion
    ended only by :class:`LocalMinimumReached` has to terminate on its own and still
    arrive at the goal.
    """
    height_before = tool_frame_height(giskard)

    motion_statechart = Statechart(context=giskard.api.context)
    motion_statechart.add_node(
        JointPositionList(
            goal_state=JointState.from_str_dict(
                {"joint_lift": LIFT_GOAL_HEIGHT}, giskard.api.world
            )
        )
    )
    motion_statechart.add_node(
        local_minimum := LocalMinimumReached(joint_convergence_threshold=0.025)
    )
    motion_statechart.add_node(EndMotion.when_true(local_minimum))

    giskard.api.compile(motion_statechart)
    giskard.api.execute()

    height_after = tool_frame_height(giskard)
    np.testing.assert_allclose(
        height_after - height_before, LIFT_GOAL_HEIGHT - LIFT_SEED_HEIGHT, atol=5e-2
    )
