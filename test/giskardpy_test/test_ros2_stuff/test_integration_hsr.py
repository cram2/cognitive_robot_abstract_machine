import time
from dataclasses import dataclass, field
from time import sleep

import numpy as np
import pytest
from geometry_msgs.msg import PoseStamped, PointStamped
from giskardpy.middleware.ros2.behavior_tree_config import StandAloneBTConfig
from giskardpy.middleware.ros2.giskard import Giskard
from giskardpy.middleware.ros2.scripts.iai_robots.hsr.configs import (
    WorldWithHSRConfig,
    HSRStandaloneInterface,
)
from giskardpy.middleware.ros2.utils.utils import load_xacro
from giskardpy.middleware.ros2.utils.utils_for_tests import compare_poses, GiskardTester
from giskardpy.motion_statechart.goals.collision_avoidance import SelfCollisionAvoidance
from giskardpy.motion_statechart.goals.open_close import Open, Close
from giskardpy.motion_statechart.goals.templates import Sequence
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
        "arm_flex_joint": -0.03,
        "arm_lift_joint": 0.01,
        "arm_roll_joint": 0.0,
        "head_pan_joint": 0.0,
        "head_tilt_joint": 0.0,
        "wrist_flex_joint": 0.0,
        "wrist_roll_joint": 0.0,
    }


@pytest.fixture()
def better_pose(default_joint_state):
    return default_joint_state


@dataclass
class HSRTester(GiskardTester):
    tip: KinematicStructureEntity = field(init=False)
    base_footprint: KinematicStructureEntity = field(init=False)
    map: KinematicStructureEntity = field(init=False)

    def __post_init__(self):
        super().__post_init__()
        self.tip = self.api.world.get_kinematic_structure_entity_by_name(
            "hand_gripper_tool_frame"
        )
        self.base_footprint = self.api.world.get_kinematic_structure_entity_by_name(
            "base_footprint"
        )
        self.map = self.api.world.root

    def setup_giskard(self) -> Giskard:
        robot_desc = load_xacro("package://hsr_description/robots/hsrb4s.urdf.xacro")
        return Giskard(
            world_config=WorldWithHSRConfig(urdf=robot_desc),
            robot_interface_config=HSRStandaloneInterface(),
            behavior_tree_config=StandAloneBTConfig(
                debug_mode=True,
                add_debug_marker_publisher=True,
                add_gantt_chart_plotter=True,
                add_trajectory_plotter=True,
            ),
            qp_controller_config=QPControllerConfig.create_with_simulation_defaults(),
        )

    @property
    def robot(self) -> HSRB:
        return (
            GiskardBlackboard().executor.context.world.get_semantic_annotations_by_type(
                HSRB
            )[0]
        )


@pytest.fixture()
def robot():
    c = HSRTester()
    try:
        yield c
    finally:
        print("tear down")
        c.print_stats()


@pytest.fixture()
def box_setup(giskard: HSRTester) -> HSRTester:
    giskard.add_box_to_world(
        name="box",
        size=(1.0, 1.0, 1.0),
        pose=HomogeneousTransformationMatrix.from_xyz_rpy(
            x=1.2, z=0.1, reference_frame=giskard.map
        ),
    )
    return giskard


class TestJointGoals:

    def test_mimic_joints(self, giskard: HSRTester):
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

    def test_mimic_joints2(self, giskard: HSRTester):
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

    def test_mimic_joints3(self, giskard: HSRTester):
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

    def test_mimic_joints4(self, giskard: HSRTester):
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
        else:
            assert False
            #state version never changed
        np.testing.assert_almost_equal(
            giskard.api.world.state[arm_lift_joints.dof.id].position,
            0.5,
            decimal=2,
        )



class TestCartGoals:
    def test_move_base(self, giskard: HSRTester):
        msc = MotionStatechart()
        msc.add_node(
            node := Sequence(
                [
                    SetOdometry(
                        base_pose=HomogeneousTransformationMatrix.from_xyz_axis_angle(
                            x=1.0,
                            y=1.0,
                            axis=Vector3.Z(),
                            angle=pi / 3,
                            reference_frame=giskard.map,
                        ),
                    ),
                    CartesianPose(
                        root_link=giskard.default_root,
                        tip_link=giskard.base_footprint,
                        goal_pose=Pose.from_xyz_axis_angle(
                            x=1.0,
                            axis=Vector3.Z(),
                            angle=pi,
                            reference_frame=giskard.map,
                        ),
                    ),
                ]
            )
        )
        msc.add_node(EndMotion.when_true(node))
        giskard.api.execute(msc)

    def test_move_base_1m_forward(self, giskard: HSRTester):
        msc = MotionStatechart()
        msc.add_node(
            node := CartesianPose(
                root_link=giskard.default_root,
                tip_link=giskard.base_footprint,
                goal_pose=Pose.from_xyz_axis_angle(
                    x=1.0,
                    reference_frame=giskard.map,
                ),
            ),
        )
        msc.add_node(EndMotion.when_true(node))
        giskard.api.execute(msc)

    def test_move_base_1m_left(self, giskard: HSRTester):
        msc = MotionStatechart()
        msc.add_node(
            node := CartesianPose(
                root_link=giskard.default_root,
                tip_link=giskard.base_footprint,
                goal_pose=Pose.from_xyz_axis_angle(
                    y=1.0,
                    reference_frame=giskard.map,
                ),
            ),
        )
        msc.add_node(EndMotion.when_true(node))
        giskard.api.execute(msc)

    def test_move_base_1m_diagonal(self, giskard: HSRTester):
        msc = MotionStatechart()
        msc.add_node(
            node := CartesianPose(
                root_link=giskard.default_root,
                tip_link=giskard.base_footprint,
                goal_pose=Pose.from_xyz_axis_angle(
                    x=1.0,
                    y=1.0,
                    reference_frame=giskard.map,
                ),
            ),
        )
        msc.add_node(EndMotion.when_true(node))
        giskard.api.execute(msc)

    def test_move_base_rotate(self, giskard: HSRTester):
        msc = MotionStatechart()
        msc.add_node(
            node := CartesianPose(
                root_link=giskard.default_root,
                tip_link=giskard.base_footprint,
                goal_pose=Pose.from_xyz_axis_angle(
                    axis=Vector3.Z(),
                    angle=pi / 3,
                    reference_frame=giskard.map,
                ),
            ),
        )
        msc.add_node(EndMotion.when_true(node))
        giskard.api.execute(msc)

    def test_move_base_forward_rotate(self, giskard: HSRTester):
        msc = MotionStatechart()
        msc.add_node(
            node := CartesianPose(
                root_link=giskard.default_root,
                tip_link=giskard.base_footprint,
                goal_pose=Pose.from_xyz_axis_angle(
                    x=1.0,
                    axis=Vector3.Z(),
                    angle=pi / 3,
                    reference_frame=giskard.map,
                ),
            ),
        )
        msc.add_node(EndMotion.when_true(node))
        giskard.api.execute(msc)

    def test_rotate_gripper(self, giskard: HSRTester):
        msc = MotionStatechart()
        msc.add_node(
            node := CartesianPose(
                root_link=giskard.default_root,
                tip_link=giskard.tip,
                goal_pose=Pose.from_xyz_axis_angle(
                    y=1.0,
                    axis=Vector3.Z(),
                    angle=pi,
                    reference_frame=giskard.tip,
                ),
            ),
        )
        msc.add_node(EndMotion.when_true(node))
        giskard.api.execute(msc)

    @pytest.mark.skip(reason="not yet fixed")
    def test_wiggle_insert(self, default_pose_giskard: HSRTester):
        goal_state = {
            "arm_flex_joint": -1.5,
            "arm_lift_joint": 0.5,
            "arm_roll_joint": 0.0,
            "head_pan_joint": 0.0,
            "head_tilt_joint": 0.0,
            "wrist_flex_joint": -1.5,
            "wrist_roll_joint": 0.0,
        }

        default_pose_giskard.api.monitors.add_set_seed_configuration(
            seed_configuration=goal_state
        )
        default_pose_giskard.execute()

        hpl = (
            default_pose_giskard.apdefault_pose_giskard.api.world.search_for_link_name(
                link_name="hand_gripper_tool_frame", group_name="hsrb"
            )
        )
        root_link = default_pose_giskard.api.world.search_for_link_name(link_name="map")
        hole_point = PointStamped()
        hole_point.header.frame_id = "map"
        hole_point.point.x = 0.5
        hole_point.point.z = 0.3
        wiggle = "wiggle"
        default_pose_giskard.api.motion_goals.add_wiggle_insert(
            name=wiggle,
            root_link=root_link,
            tip_link=hpl,
            hole_point=hole_point,
            end_condition=wiggle,
        )
        resistence_point = PointStamped()
        resistence_point.header.frame_id = "map"
        resistence_point.point.x = 0.5
        resistence_point.point.z = 0.4
        timer = default_pose_giskard.api.monitors.add_sleep(5)
        default_pose_giskard.api.motion_goals.add_cartesian_position(
            root_link=root_link,
            tip_link=hpl,
            goal_point=resistence_point,
            end_condition=timer,
        )
        default_pose_giskard.api.monitors.add_end_motion(start_condition=wiggle)
        default_pose_giskard.execute(local_min_end=False)


class TestConstraints:

    @pytest.mark.skip(reason="suturo must fix")
    def test_schnibbeln_sequence(self, box_setup: HSRTester):
        box = box_setup.api.world.get_body_by_name("box")

        box_setup.add_box_to_world(
            name="Schnibbler",
            size=(0.05, 0.01, 0.15),
            pose=HomogeneousTransformationMatrix.from_xyz_rpy(
                z=0.06, reference_frame=box_setup.tip
            ),
            parent_link=box_setup.tip,
        )
        schnibbler = box_setup.api.world.get_body_by_name("Schnibbler")
        box_setup.add_box_to_world(
            name="Bernd",
            size=(0.1, 0.2, 0.06),
            pose=HomogeneousTransformationMatrix.from_xyz_rpy(
                x=0.91, y=0.25, z=0.62, reference_frame=box_setup.map
            ),
            parent_link=box,
        )

        pre_schnibble_pose = Pose.from_point_rotation_matrix(
            point=Point3(0.85, 0.2, 0.75, reference_frame=box_setup.map),
            rotation_matrix=RotationMatrix(
                [[0, 0, 1, 0], [0, -1, 0, 0], [1, 0, 0, 0], [0, 0, 0, 1]],
                reference_frame=box_setup.map,
            ),
            reference_frame=box_setup.map,
        )

        msc = MotionStatechart()
        msc.add_nodes(
            [
                pre_schnibble := CartesianPose(
                    name="Position Knife",
                    goal_pose=pre_schnibble_pose,
                    tip_link=box_setup.tip,
                    root_link=box_setup.map,
                ),
                human_close := Pulse(name="Human Close?", delay=15, length=15),
                cutting := Cutting(
                    name="Cut",
                    root_link=box_setup.map,
                    tip_link=schnibbler,
                    depth=0.1,
                    right_shift=-0.1,
                ),
                schnibbel_done := CheckControlCycleCount(name="Done?", threshold=120),
            ]
        )
        pre_schnibble.end_condition = pre_schnibble.observation_variable
        cutting.start_condition = pre_schnibble.observation_variable
        schnibbel_done.start_condition = cutting.observation_variable
        human_close.start_condition = pre_schnibble.observation_variable

        reset = trinary_logic_not(schnibbel_done.observation_variable)
        cutting.reset_condition = reset
        schnibbel_done.reset_condition = reset
        human_close.end_condition = schnibbel_done.observation_variable
        cutting.pause_condition = human_close.observation_variable

        msc.add_node(EndMotion.when_true(schnibbel_done))
        box_setup.api.execute(msc)

    def test_Pointing(self, giskard: HSRTester):
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

    def test_open_fridge(self, kitchen_setup: HSRTester, better_pose):
        handle_frame_id = kitchen_setup.api.world.get_body_by_name(
            "iai_fridge_door_handle"
        )
        handle_name = kitchen_setup.api.world.get_body_by_name("iai_fridge_door_handle")

        msc = MotionStatechart()
        msc.add_nodes(
            [
                sequence := Sequence(
                    [
                        CartesianPose(
                            root_link=kitchen_setup.map,
                            tip_link=kitchen_setup.base_footprint,
                            goal_pose=Pose.from_xyz_rpy(
                                x=0.3, y=-0.5, z=0.0, reference_frame=kitchen_setup.map
                            ),
                        ),
                        CartesianPose(
                            root_link=kitchen_setup.map,
                            tip_link=kitchen_setup.tip,
                            goal_pose=Pose.from_xyz_rpy(
                                x=0,
                                y=0,
                                z=0.0,
                                pitch=-np.pi / 2,
                                reference_frame=handle_frame_id,
                            ),
                        ),
                        Open(
                            tip_link=kitchen_setup.tip,
                            environment_link=handle_name,
                            goal_joint_state=1.5,
                        ),
                        Close(
                            tip_link=kitchen_setup.tip,
                            environment_link=handle_name,
                            goal_joint_state=0.1,
                        ),
                        JointPositionList(
                            goal_state=JointState.from_str_dict(
                                better_pose, world=kitchen_setup.api.world
                            )
                        ),
                    ]
                )
            ]
        )
        msc.add_node(EndMotion.when_true(sequence))
        kitchen_setup.api.execute(msc)


class TestCollisionAvoidanceGoals:

    def test_self_collision_avoidance(self, giskard: HSRTester):
        msc = MotionStatechart()
        msc.add_nodes(
            [
                cart_goal := CartesianPose(
                    root_link=giskard.map,
                    tip_link=giskard.tip,
                    goal_pose=Pose.from_xyz_axis_angle(
                        z=0.5,
                        reference_frame=giskard.tip,
                    ),
                ),
                SelfCollisionAvoidance(),
            ]
        )
        msc.add_node(EndMotion.when_true(cart_goal))
        giskard.api.execute(msc)

    def test_self_collision_avoidance2(self, giskard: HSRTester):
        hand_palm_link = giskard.api.world.get_body_by_name("hand_palm_link")

        msc = MotionStatechart()
        msc.add_nodes(
            [
                sequence := Sequence(
                    [
                        SetSeedConfiguration(
                            seed_configuration=JointState.from_str_dict(
                                {
                                    "arm_flex_joint": 0.0,
                                    "arm_lift_joint": 0.0,
                                    "arm_roll_joint": -1.52,
                                    "head_pan_joint": -0.09,
                                    "head_tilt_joint": -0.62,
                                    "wrist_flex_joint": -1.55,
                                    "wrist_roll_joint": 0.11,
                                },
                                giskard.api.world,
                            )
                        ),
                        CartesianPose(
                            root_link=giskard.map,
                            tip_link=giskard.tip,
                            goal_pose=Pose.from_xyz_axis_angle(
                                x=0.5,
                                reference_frame=hand_palm_link,
                            ),
                        ),
                    ]
                ),
                SelfCollisionAvoidance(),
            ]
        )
        msc.add_node(EndMotion.when_true(sequence))
        giskard.api.execute(msc)


class TestAddObject:
    def test_add(self, giskard: HSRTester):
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
