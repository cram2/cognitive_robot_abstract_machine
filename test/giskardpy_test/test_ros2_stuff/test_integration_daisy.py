from dataclasses import dataclass, field
from time import sleep

import numpy as np
import pytest
from geometry_msgs.msg import PoseStamped, PointStamped
from giskardpy.middleware.ros2.behavior_tree_config import StandAloneBTConfig
from giskardpy.middleware.ros2.giskard import Giskard
from giskardpy.middleware.ros2.scripts.iai_robots.daisy.configs import (
    WorldWithDaisyConfig,
    DaisyStandAloneRobotInterfaceConfig,
)
from giskardpy.middleware.ros2.scripts.iai_robots.hsr.configs import (
    WorldWithHSRConfig,
    HSRStandaloneInterface,
)
from giskardpy.middleware.ros2.utils.utils import load_xacro
from giskardpy.middleware.ros2.utils.utils_for_tests import compare_poses, GiskardTester
from giskardpy.motion_statechart.goals.collision_avoidance import SelfCollisionAvoidance
from giskardpy.motion_statechart.goals.open_close import Open, Close
from giskardpy.motion_statechart.goals.templates import Sequence, Parallel
from giskardpy.motion_statechart.goals.test import Cutting
from giskardpy.motion_statechart.graph_node import EndMotion
from giskardpy.motion_statechart.monitors.overwrite_state_monitors import (
    SetOdometry,
    SetSeedConfiguration,
)
from giskardpy.motion_statechart.monitors.payload_monitors import (
    Pulse,
    CheckControlCycleCount,
)
from giskardpy.motion_statechart.motion_statechart import MotionStatechart
from giskardpy.motion_statechart.tasks.cartesian_tasks import CartesianPose
from giskardpy.motion_statechart.tasks.joint_tasks import JointPositionList, JointState
from giskardpy.motion_statechart.tasks.pointing import Pointing
from giskardpy.qp.qp_controller_config import QPControllerConfig
from giskardpy.tree.blackboard_utils import GiskardBlackboard
from krrood.symbolic_math.symbolic_math import trinary_logic_not
from numpy import pi

from semantic_digital_twin.adapters.ros.visualization.viz_marker import (
    VizMarkerPublisher,
)
from semantic_digital_twin.robots.daisy import DAiSy
from semantic_digital_twin.robots.hsrb import HSRB
from semantic_digital_twin.spatial_types import (
    HomogeneousTransformationMatrix,
    Vector3,
    Point3,
    RotationMatrix,
)
from semantic_digital_twin.spatial_types.spatial_types import Pose
from semantic_digital_twin.world_description.connections import ActiveConnection1DOF
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
    return default_joint_state


@dataclass
class DAiSyTester(GiskardTester):
    left_tip: KinematicStructureEntity = field(init=False)
    right_tip: KinematicStructureEntity = field(init=False)
    map: KinematicStructureEntity = field(init=False)

    def __post_init__(self):
        super().__post_init__()
        self.left_tip = self.api.world.get_kinematic_structure_entity_by_name(
            "left_gripper_tool_frame"
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

    def test_mimic_joints(self, giskard: DAiSyTester):
        msc = MotionStatechart()
        msc.add_node(
            joint_goal := JointPositionList(
                goal_state=JointState.from_str_dict(
                    {"torso_lift_joint": 0.1, "hand_motor_joint": 1.23},
                    giskard.api.world,
                )
            ),
        )
        msc.add_node(EndMotion.when_true(joint_goal))
        giskard.api.execute(msc)

        arm_lift_joint: (
            ActiveConnection1DOF
        ) = GiskardBlackboard().giskard.world_config.world.get_connection_by_name(
            "arm_lift_joint"
        )
        hand_T_finger_current = giskard.compute_fk_pose(
            "hand_palm_link", "hand_l_distal_link"
        )
        hand_T_finger_expected = PoseStamped()
        hand_T_finger_expected.header.frame_id = "hand_palm_link"
        hand_T_finger_expected.pose.position.x = -0.01675
        hand_T_finger_expected.pose.position.y = -0.0907
        hand_T_finger_expected.pose.position.z = 0.0052
        hand_T_finger_expected.pose.orientation.x = -0.0434
        hand_T_finger_expected.pose.orientation.y = 0.0
        hand_T_finger_expected.pose.orientation.z = 0.0
        hand_T_finger_expected.pose.orientation.w = 0.999
        compare_poses(hand_T_finger_current.pose, hand_T_finger_expected.pose)

        np.testing.assert_almost_equal(
            arm_lift_joint.position,
            0.2,
            decimal=2,
        )
        base_T_torso = PoseStamped()
        base_T_torso.header.frame_id = "base_footprint"
        base_T_torso.pose.position.x = 0.0
        base_T_torso.pose.position.y = 0.0
        base_T_torso.pose.position.z = 0.8518
        base_T_torso.pose.orientation.x = 0.0
        base_T_torso.pose.orientation.y = 0.0
        base_T_torso.pose.orientation.z = 0.0
        base_T_torso.pose.orientation.w = 1.0
        base_T_torso2 = giskard.compute_fk_pose("base_footprint", "torso_lift_link")
        compare_poses(base_T_torso2.pose, base_T_torso.pose)

    def test_mimic_joints2(self, giskard: DAiSyTester):
        msc = MotionStatechart()
        msc.add_node(
            node := CartesianPose(
                root_link=giskard.base_footprint,
                tip_link=giskard.tip,
                goal_pose=Pose.from_xyz_axis_angle(
                    z=0.2,
                    reference_frame=giskard.tip,
                ),
            ),
        )
        msc.add_node(EndMotion.when_true(node))

        giskard.api.execute(msc)

        arm_lift_joint: (
            ActiveConnection1DOF
        ) = GiskardBlackboard().giskard.world_config.world.get_connection_by_name(
            "arm_lift_joint"
        )
        np.testing.assert_almost_equal(
            arm_lift_joint.position,
            0.2,
            decimal=2,
        )
        base_T_torso = PoseStamped()
        base_T_torso.header.frame_id = "base_footprint"
        base_T_torso.pose.position.x = 0.0
        base_T_torso.pose.position.y = 0.0
        base_T_torso.pose.position.z = 0.8518
        base_T_torso.pose.orientation.x = 0.0
        base_T_torso.pose.orientation.y = 0.0
        base_T_torso.pose.orientation.z = 0.0
        base_T_torso.pose.orientation.w = 1.0
        base_T_torso2 = giskard.compute_fk_pose("base_footprint", "torso_lift_link")
        compare_poses(base_T_torso2.pose, base_T_torso.pose)

    def test_mimic_joints3(self, giskard: DAiSyTester):
        head = giskard.api.world.get_body_by_name("head_pan_link")
        msc = MotionStatechart()
        msc.add_node(
            node := CartesianPose(
                root_link=giskard.base_footprint,
                tip_link=head,
                goal_pose=Pose.from_xyz_axis_angle(
                    z=0.15,
                    reference_frame=head,
                ),
            ),
        )
        msc.add_node(EndMotion.when_true(node))

        giskard.api.execute(msc)

        arm_lift_joint: (
            ActiveConnection1DOF
        ) = GiskardBlackboard().giskard.world_config.world.get_connection_by_name(
            "arm_lift_joint"
        )
        np.testing.assert_almost_equal(
            arm_lift_joint.position,
            0.3,
            decimal=2,
        )
        base_T_torso = PoseStamped()
        base_T_torso.header.frame_id = "base_footprint"
        base_T_torso.pose.position.x = 0.0
        base_T_torso.pose.position.y = 0.0
        base_T_torso.pose.position.z = 0.902
        base_T_torso.pose.orientation.x = 0.0
        base_T_torso.pose.orientation.y = 0.0
        base_T_torso.pose.orientation.z = 0.0
        base_T_torso.pose.orientation.w = 1.0
        base_T_torso2 = giskard.compute_fk_pose("base_footprint", "torso_lift_link")
        compare_poses(base_T_torso2.pose, base_T_torso.pose)

    def test_mimic_joints4(self, giskard: DAiSyTester):
        arm_lift_joints: ActiveConnection1DOF = (
            giskard.api.world.get_connection_by_name("arm_lift_joint")
        )
        assert arm_lift_joints.dof.limits.lower.velocity == -0.15
        assert arm_lift_joints.dof.limits.upper.velocity == 0.15
        torso_lift_joints: ActiveConnection1DOF = (
            giskard.api.world.get_connection_by_name("torso_lift_joint")
        )
        assert torso_lift_joints.dof.limits.lower.velocity == -0.075
        assert torso_lift_joints.dof.limits.upper.velocity == 0.075
        msc = MotionStatechart()
        msc.add_node(
            joint_goal := JointPositionList(
                goal_state=JointState.from_str_dict(
                    {"torso_lift_joint": 0.25},
                    giskard.api.world,
                )
            ),
        )
        msc.add_node(EndMotion.when_true(joint_goal))
        state_version = giskard.api.world.state.version
        giskard.api.execute(msc)
        for i in range(1000):
            if giskard.api.world.state.version != state_version:
                break
            sleep(0.01)
        np.testing.assert_almost_equal(
            giskard.api.world.state[arm_lift_joints.dof.id].position,
            0.5,
            decimal=2,
        )


class TestCartGoals:
    def test_rotate_gripper(self, giskard: DAiSyTester):
        # viz_marker = VizMarkerPublisher(_world=giskard.api.world, node=giskard.api.node)
        msc = MotionStatechart()
        msc.add_node(
            node := CartesianPose(
                root_link=giskard.default_root,
                tip_link=giskard.left_tip,
                goal_pose=Pose.from_xyz_axis_angle(
                    axis=Vector3.Z(),
                    angle=pi,
                    reference_frame=giskard.left_tip,
                ),
            ),
        )
        msc.add_node(EndMotion.when_true(node))
        giskard.api.execute(msc)


class TestConstraints:
    def test_pointing(self, giskard: DAiSyTester):
        kopf = giskard.api.world.get_body_by_name("head_rgbd_sensor_gazebo_frame")

        msc = MotionStatechart()
        msc.add_node(
            node := Pointing(
                tip_link=kopf,
                root_link=giskard.map,
                goal_point=Point3(1, -1, reference_frame=giskard.map),
                pointing_axis=Vector3.X(reference_frame=kopf),
            )
        )
        msc.add_node(EndMotion.when_true(node))
        giskard.api.execute(msc)


class TestCollisionAvoidanceGoals:
    def test_self_collision_avoidance(self, giskard: DAiSyTester):
        msc = MotionStatechart()

        offset_x = 0.1
        offset_y = -0.8

        msc.add_nodes(
            [
                parallel := Parallel(
                    [
                        CartesianPose(
                            root_link=giskard.map,
                            tip_link=giskard.left_tip,
                            goal_pose=Pose.from_xyz_axis_angle(
                                x=offset_x,
                                y=offset_y,
                                reference_frame=giskard.left_tip,
                            ),
                        ),
                        CartesianPose(
                            root_link=giskard.map,
                            tip_link=giskard.right_tip,
                            goal_pose=Pose.from_xyz_axis_angle(
                                x=offset_x,
                                y=offset_y,
                                reference_frame=giskard.right_tip,
                            ),
                        ),
                    ],
                ),
                SelfCollisionAvoidance(),
            ]
        )
        msc.add_node(EndMotion.when_true(parallel))
        giskard.api.execute(msc)

    def test_self_collision_avoidance2(self, giskard: DAiSyTester):
        msc = MotionStatechart()
        goal = Pose.from_xyz_axis_angle(
            x=-0.5, y=-0.2, z=1.0, reference_frame=giskard.map
        )
        msc.add_nodes(
            [
                parallel := Parallel(
                    [
                        CartesianPose(
                            root_link=giskard.map,
                            tip_link=giskard.left_tip,
                            goal_pose=goal,
                        ),
                        CartesianPose(
                            root_link=giskard.map,
                            tip_link=giskard.right_tip,
                            goal_pose=goal,
                        ),
                    ],
                ),
                SelfCollisionAvoidance(),
            ]
        )
        msc.add_node(EndMotion.when_true(parallel))
        giskard.api.execute(msc)


class TestAddObject:
    def test_add(self, giskard: DAiSyTester):
        box1_name = "box1"
        giskard.add_box_to_world(
            name=box1_name,
            size=(1, 1, 1),
            pose=HomogeneousTransformationMatrix.from_xyz_rpy(
                x=1, reference_frame=giskard.map
            ),
            parent_link=giskard.api.world.get_body_by_name("hand_palm_link"),
        )

        msc = MotionStatechart()
        msc.add_node(
            joint_goal := JointPositionList(
                goal_state=JointState.from_str_dict(
                    {"arm_flex_joint": -0.7},
                    giskard.api.world,
                )
            ),
        )
        msc.add_node(EndMotion.when_true(joint_goal))
        giskard.api.execute(msc)
