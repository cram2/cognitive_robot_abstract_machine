from dataclasses import dataclass, field

import numpy as np
import pytest
from geometry_msgs.msg import PoseStamped
from giskardpy.middleware.ros2.behavior_tree_config import StandAloneBTConfig
from giskardpy.middleware.ros2.giskard import Giskard
from giskardpy.middleware.ros2.scripts.iai_robots.daisy.configs import (
    WorldWithDaisyConfig,
    DaisyStandAloneRobotInterfaceConfig,
)
from giskardpy.middleware.ros2.utils.utils import load_xacro
from giskardpy.middleware.ros2.utils.utils_for_tests import compare_poses, GiskardTester
from giskardpy.motion_statechart.data_types import ObservationStateValues
from giskardpy.motion_statechart.goals.collision_avoidance import SelfCollisionAvoidance
from giskardpy.motion_statechart.goals.templates import Sequence, Parallel
from giskardpy.motion_statechart.graph_node import EndMotion
from giskardpy.motion_statechart.monitors.monitors import LocalMinimumReached
from giskardpy.motion_statechart.motion_statechart import (
    MotionStatechart,
)
from giskardpy.motion_statechart.tasks.cartesian_tasks import CartesianPose
from giskardpy.motion_statechart.tasks.joint_tasks import JointPositionList, JointState
from giskardpy.qp.qp_controller_config import QPControllerConfig
from giskardpy.tree.blackboard_utils import GiskardBlackboard
from semantic_digital_twin.datastructures.definitions import StaticJointState
from semantic_digital_twin.robots.daisy import DAiSy
from semantic_digital_twin.spatial_types import (
    HomogeneousTransformationMatrix,
)
from semantic_digital_twin.spatial_types.spatial_types import Pose
from semantic_digital_twin.world_description.world_entity import (
    KinematicStructureEntity,
)


@pytest.fixture()
def default_joint_state():
    return {
        "left_shoulder_pan_joint": 0,
        "left_shoulder_lift_joint": -1.57,
        "left_elbow_joint": 1,
        "left_wrist_1_joint": 0,
        "left_wrist_2_joint": 0,
        "left_wrist_3_joint": 0,
        "right_shoulder_pan_joint": 0,
        "right_shoulder_lift_joint": -1.57,
        "right_elbow_joint": 1,
        "right_wrist_1_joint": 0,
        "right_wrist_2_joint": 0,
        "right_wrist_3_joint": 0,
    }


@pytest.fixture()
def better_pose(default_joint_state):
    return {
        "left_shoulder_pan_joint": 0,
        "left_shoulder_lift_joint": -1.57,
        "left_elbow_joint": 1,
        "left_wrist_1_joint": 0,
        "left_wrist_2_joint": 0,
        "left_wrist_3_joint": np.pi / 4,
        "right_shoulder_pan_joint": 3 / 4 * 3.14,
        "right_shoulder_lift_joint": -1.57,
        "right_elbow_joint": 1,
        "right_wrist_1_joint": 0,
        "right_wrist_2_joint": 0,
        "right_wrist_3_joint": np.pi / 4,
    }


@dataclass
class DAiSyTester(GiskardTester):
    left_base: KinematicStructureEntity = field(init=False)
    left_tip: KinematicStructureEntity = field(init=False)
    right_base: KinematicStructureEntity = field(init=False)
    right_tip: KinematicStructureEntity = field(init=False)
    map: KinematicStructureEntity = field(init=False)

    def __post_init__(self):
        super().__post_init__()
        self.left_base = self.api.world.get_kinematic_structure_entity_by_name(
            "left_base_link"
        )
        self.left_tip = self.api.world.get_kinematic_structure_entity_by_name(
            "left_gripper_tool_frame"
        )
        self.right_base = self.api.world.get_kinematic_structure_entity_by_name(
            "right_base_link"
        )
        self.right_tip = self.api.world.get_kinematic_structure_entity_by_name(
            "right_gripper_tool_frame"
        )
        self.map = self.api.world.root

    def setup_giskard(self) -> Giskard:
        robot_desc = load_xacro(
            "package://iai_daisy_description/robots/daisy.urdf.xacro"
        )
        return Giskard(
            world_config=WorldWithDaisyConfig(urdf=robot_desc),
            robot_interface_config=DaisyStandAloneRobotInterfaceConfig(),
            behavior_tree_config=StandAloneBTConfig(
                debug_mode=True,
                add_debug_marker_publisher=True,
                add_gantt_chart_plotter=True,
                add_trajectory_plotter=True,
            ),
            qp_controller_config=QPControllerConfig.create_with_simulation_defaults(),
        )

    @property
    def robot(self) -> DAiSy:
        return (
            GiskardBlackboard().executor.context.world.get_semantic_annotations_by_type(
                DAiSy
            )[0]
        )


@pytest.fixture()
def robot():
    c = DAiSyTester()
    try:
        yield c
    finally:
        print("tear down")
        c.print_stats()


@pytest.fixture()
def box_setup(giskard: DAiSyTester) -> DAiSyTester:
    giskard.add_box_to_world(
        name="box",
        size=(1.0, 1.0, 1.0),
        pose=HomogeneousTransformationMatrix.from_xyz_rpy(
            x=1.2, z=0.1, reference_frame=giskard.map
        ),
    )
    return giskard


class TestJointGoals:

    @pytest.mark.parametrize(
        "arm",
        ["left", "right"],
    )
    def test_mimic_joints(self, giskard: DAiSyTester, arm: str):
        msc = MotionStatechart()
        msc.add_node(
            joint_goal := JointPositionList(
                goal_state=JointState.from_str_dict(
                    {
                        f"{arm}_wrist_1_joint": 1.23,
                        f"{arm}_wrist_2_joint": 1.23,
                        f"{arm}_wrist_3_joint": 1.23,
                    },
                    giskard.api.world,
                )
            ),
        )
        msc.add_node(EndMotion.when_true(joint_goal))
        giskard.api.execute(msc)

        hand_T_finger_current = giskard.compute_fk_pose(
            f"{arm}_base_link", f"{arm}_gripper_left_finger_tip_link"
        )
        hand_T_finger_expected = PoseStamped()
        hand_T_finger_expected.header.frame_id = f"{arm}_base_link"
        hand_T_finger_expected.pose.position.x = 0.3802
        hand_T_finger_expected.pose.position.y = 0.1486
        hand_T_finger_expected.pose.position.z = 0.5331
        hand_T_finger_expected.pose.orientation.x = 0.4482
        hand_T_finger_expected.pose.orientation.y = 0.7662
        hand_T_finger_expected.pose.orientation.z = 0.3752
        hand_T_finger_expected.pose.orientation.w = 0.2666
        compare_poses(hand_T_finger_current.pose, hand_T_finger_expected.pose)

    @pytest.mark.parametrize(
        "arm",
        ["left", "right"],
    )
    def test_mimic_joints2(self, giskard: DAiSyTester, arm):
        base = giskard.api.world.get_kinematic_structure_entity_by_name(
            f"{arm}_base_link"
        )
        tip = giskard.api.world.get_kinematic_structure_entity_by_name(
            f"{arm}_gripper_left_finger_tip_link"
        )
        msc = MotionStatechart()
        msc.add_node(
            node := CartesianPose(
                root_link=base,
                tip_link=tip,
                goal_pose=Pose.from_xyz_axis_angle(
                    z=0.2,
                    reference_frame=tip,
                ),
            ),
        )
        msc.add_node(EndMotion.when_true(node))

        giskard.api.execute(msc)

        base_T_tip = PoseStamped()
        base_T_tip.header.frame_id = base
        base_T_tip.pose.position.x = 0.4027
        base_T_tip.pose.position.y = 0.4686
        base_T_tip.pose.position.z = 0.6614
        base_T_tip.pose.orientation.x = -0.6180
        base_T_tip.pose.orientation.y = 0.3386
        base_T_tip.pose.orientation.z = 0.3400
        base_T_tip.pose.orientation.w = 0.6226
        base_T_tip2 = giskard.compute_fk_pose(base.name.name, tip.name.name)
        compare_poses(base_T_tip2.pose, base_T_tip.pose)

    @pytest.mark.parametrize(
        "arm",
        [0, 1],  # 0: left, 1: right
    )
    def test_mimic_joint3(self, giskard: DAiSyTester, arm):
        for state in giskard.api.robot.arms[arm].joint_states:
            if state.state_type == StaticJointState.PARK:
                park_state = state
                break
        else:
            assert False

        msc = MotionStatechart()
        msc.add_node(node := JointPositionList(goal_state=park_state))
        msc.add_node(EndMotion.when_true(node))

        giskard.api.execute(msc)

        assert park_state.is_achieved()


class TestCollisionAvoidanceGoals:
    def test_self_collision_avoidance(self, giskard_better_pose: DAiSyTester):
        msc = MotionStatechart()

        offset_x = 0.1
        offset_y = 0.8
        offset_z = -0.1

        msc.add_nodes(
            [
                parallel := Parallel(
                    [
                        CartesianPose(
                            root_link=giskard_better_pose.map,
                            tip_link=giskard_better_pose.left_tip,
                            goal_pose=Pose.from_xyz_axis_angle(
                                x=offset_x,
                                y=offset_y,
                                z=offset_z,
                                reference_frame=giskard_better_pose.left_tip,
                            ),
                        ),
                        CartesianPose(
                            root_link=giskard_better_pose.map,
                            tip_link=giskard_better_pose.right_tip,
                            goal_pose=Pose.from_xyz_axis_angle(
                                x=offset_x,
                                y=offset_y,
                                z=offset_z,
                                reference_frame=giskard_better_pose.right_tip,
                            ),
                        ),
                    ],
                ),
                SelfCollisionAvoidance(),
                local_min := LocalMinimumReached(),
            ]
        )
        msc.add_node(EndMotion.when_true(local_min))
        giskard_better_pose.api.execute(msc)

        assert parallel.observation_state == ObservationStateValues.FALSE

    def test_self_collision_avoidance2(self, giskard_better_pose: DAiSyTester):
        msc = MotionStatechart()
        goal = Pose.from_xyz_axis_angle(
            x=-0.65, y=-0.3, z=1.0, reference_frame=giskard_better_pose.map
        )
        msc.add_nodes(
            [
                parallel := Parallel(
                    [
                        CartesianPose(
                            root_link=giskard_better_pose.map,
                            tip_link=giskard_better_pose.left_tip,
                            goal_pose=goal,
                        ),
                        CartesianPose(
                            root_link=giskard_better_pose.map,
                            tip_link=giskard_better_pose.right_tip,
                            goal_pose=goal,
                        ),
                    ],
                ),
                SelfCollisionAvoidance(),
                local_min := LocalMinimumReached(),
            ]
        )
        msc.add_node(EndMotion.when_true(local_min))
        giskard_better_pose.api.execute(msc)

        assert parallel.observation_state == ObservationStateValues.FALSE
