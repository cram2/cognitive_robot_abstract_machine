from dataclasses import dataclass, field

import pytest
from giskardpy.middleware.ros2.behavior_tree_config import StandAloneBTConfig
from giskardpy.middleware.ros2.giskard import Giskard
from giskardpy.middleware.ros2.scripts.other_robots.xarm5.configs import (
    XArm5StandAloneRobotInterfaceConfig,
    WorldWithXArm5Config,
)
from giskardpy.middleware.ros2.utils.utils import load_xacro
from giskardpy.middleware.ros2.utils.utils_for_tests import compare_poses, GiskardTester
from giskardpy.motion_statechart.graph_node import EndMotion
from giskardpy.motion_statechart.motion_statechart import MotionStatechart
from giskardpy.motion_statechart.tasks.joint_tasks import JointPositionList, JointState
from giskardpy.qp.qp_controller_config import QPControllerConfig
from giskardpy.tree.blackboard_utils import GiskardBlackboard
from semantic_digital_twin.robots.xarm5 import XArm5
from semantic_digital_twin.world_description.world_entity import (
    KinematicStructureEntity,
)


@pytest.fixture()
def default_joint_state():
    return {}


@pytest.fixture()
def better_pose(default_joint_state):
    return default_joint_state


@dataclass
class XArm5Tester(GiskardTester):
    tip: KinematicStructureEntity = field(init=False)
    base_footprint: KinematicStructureEntity = field(init=False)
    map: KinematicStructureEntity = field(init=False)

    def __post_init__(self):
        super().__post_init__()
        self.tip = self.api.world.get_kinematic_structure_entity_by_name("link_eef")
        self.base_footprint = self.api.world.get_kinematic_structure_entity_by_name(
            "link_base"
        )
        self.map = self.api.world.root

    def setup_giskard(self) -> Giskard:
        robot_desc = load_xacro(XArm5.get_ros_file_path())
        return Giskard(
            world_config=WorldWithXArm5Config(urdf=robot_desc),
            robot_interface_config=XArm5StandAloneRobotInterfaceConfig(),
            behavior_tree_config=StandAloneBTConfig(
                debug_mode=True,
                add_debug_marker_publisher=True,
                add_gantt_chart_plotter=True,
                add_trajectory_plotter=True,
            ),
            qp_controller_config=QPControllerConfig.create_with_simulation_defaults(),
        )

    @property
    def robot(self) -> XArm5:
        return (
            GiskardBlackboard().executor.context.world.get_semantic_annotations_by_type(
                XArm5
            )[0]
        )


@pytest.fixture()
def robot():
    c = XArm5Tester()
    try:
        yield c
    finally:
        print("tear down")
        c.print_stats()


class TestSetup:

    def test_small_msc(self, giskard: XArm5Tester):
        msc = MotionStatechart()
        msc.add_node(
            joint_goal := JointPositionList(
                goal_state=JointState.from_str_dict(
                    {
                        f"joint1": 1.23,
                        f"joint2": 1.23,
                        f"joint3": 1.23,
                    },
                    giskard.api.world,
                )
            ),
        )
        msc.add_node(EndMotion.when_true(joint_goal))
        giskard.api.execute(msc)
