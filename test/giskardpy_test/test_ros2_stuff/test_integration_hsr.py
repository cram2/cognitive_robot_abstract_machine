from dataclasses import dataclass, field
from time import sleep

import numpy as np
import pytest
from giskardpy.middleware.ros2.server_config import ExecutionMode, GiskardServerConfig
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
from giskardpy.motion_statechart.graph_node import EndMotion
from giskardpy.motion_statechart.monitors.overwrite_state_monitors import (
    SetOdometry,
    SetSeedConfiguration,
)
from giskardpy.motion_statechart.motion_statechart import MotionStatechart
from giskardpy.motion_statechart.tasks.cartesian_tasks import CartesianPose
from giskardpy.motion_statechart.tasks.joint_tasks import JointPositionList, JointState
from giskardpy.motion_statechart.tasks.pointing import Pointing
from giskardpy.qp.qp_controller_config import QPControllerConfig
from numpy import pi
from semantic_digital_twin.robots.hsrb import HSRB, HSRBJoint
from semantic_digital_twin.spatial_types import (
    HomogeneousTransformationMatrix,
    Vector3,
    Point3,
)
from semantic_digital_twin.spatial_types.spatial_types import Pose
from semantic_digital_twin.world_description.connections import ActiveConnection1DOF
from semantic_digital_twin.world_description.world_entity import (
    KinematicStructureEntity,
)


@pytest.fixture()
def default_joint_state():
    return {
        HSRBJoint.ARM_FLEX: -0.03,
        HSRBJoint.ARM_LIFT: 0.01,
        HSRBJoint.ARM_ROLL: 0.0,
        HSRBJoint.HEAD_PAN: 0.0,
        HSRBJoint.HEAD_TILT: 0.0,
        HSRBJoint.WRIST_FLEX: 0.0,
        HSRBJoint.WRIST_ROLL: 0.0,
    }


@pytest.fixture()
def better_pose(default_joint_state):
    return default_joint_state


@dataclass
class HSRTester(GiskardTester):
    tip: KinematicStructureEntity = field(init=False)
    base_footprint: KinematicStructureEntity = field(init=False)
    torso_lift_link: KinematicStructureEntity = field(init=False)
    map: KinematicStructureEntity = field(init=False)

    def __post_init__(self):
        super().__post_init__()
        self.tip = self.api.world.get_kinematic_structure_entity_by_name(
            "hand_gripper_tool_frame"
        )
        self.base_footprint = self.api.world.get_kinematic_structure_entity_by_name(
            "base_footprint"
        )
        self.torso_lift_link = self.api.world.get_kinematic_structure_entity_by_name(
            "torso_lift_link"
        )
        self.map = self.api.world.root

    def setup_giskard(self) -> Giskard:
        robot_desc = load_xacro("package://hsr_description/robots/hsrb4s.urdf.xacro")
        return Giskard(
            world_config=WorldWithHSRConfig(urdf=robot_desc),
            robot_interface_config=HSRStandaloneInterface(),
            server_config=GiskardServerConfig(
                execution_mode=ExecutionMode.STANDALONE,
                debug_mode=True,
                plot_gantt_chart=True,
                plot_trajectory=True,
            ),
            qp_controller_config=QPControllerConfig.create_with_simulation_defaults(),
        )

    @property
    def robot(self) -> HSRB:
        return self.giskard.executor.context.world.get_semantic_annotations_by_type(
            HSRB
        )[0]


@pytest.fixture()
def robot():
    c = HSRTester()
    try:
        yield c
    finally:
        print("tear down")
        c.close()


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
                    {HSRBJoint.TORSO_LIFT: 0.1, HSRBJoint.HAND_MOTOR: 1.23},
                    giskard.api.world,
                )
            ),
        )
        msc.add_node(EndMotion.when_true(joint_goal))
        giskard.api.execute(msc)

        arm_lift_joint: ActiveConnection1DOF = giskard.world.get_connection_by_name(
            HSRBJoint.ARM_LIFT
        )
        hand_palm_link = giskard.world.get_kinematic_structure_entity_by_name(
            "hand_palm_link"
        )
        hand_T_finger_current = giskard.world.compute_forward_kinematics(
            root=hand_palm_link,
            tip=giskard.world.get_kinematic_structure_entity_by_name(
                "hand_l_distal_link"
            ),
        )
        hand_T_finger_expected = HomogeneousTransformationMatrix.from_xyz_quaternion(
            pos_x=-0.01675,
            pos_y=-0.0907,
            pos_z=0.0052,
            quat_x=-0.0434,
            quat_w=0.999,
            reference_frame=hand_palm_link,
        )
        compare_poses(hand_T_finger_current, hand_T_finger_expected)

        np.testing.assert_almost_equal(
            arm_lift_joint.position,
            0.2,
            decimal=2,
        )
        base_T_torso_expected = HomogeneousTransformationMatrix.from_xyz_rpy(
            z=0.8518, reference_frame=giskard.base_footprint
        )
        base_T_torso_current = giskard.world.compute_forward_kinematics(
            root=giskard.base_footprint, tip=giskard.torso_lift_link
        )
        compare_poses(base_T_torso_current, base_T_torso_expected)

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

        arm_lift_joint: ActiveConnection1DOF = giskard.world.get_connection_by_name(
            HSRBJoint.ARM_LIFT
        )
        np.testing.assert_almost_equal(
            arm_lift_joint.position,
            0.2,
            decimal=2,
        )
        base_T_torso_expected = HomogeneousTransformationMatrix.from_xyz_rpy(
            z=0.8518, reference_frame=giskard.base_footprint
        )
        base_T_torso_current = giskard.world.compute_forward_kinematics(
            root=giskard.base_footprint, tip=giskard.torso_lift_link
        )
        compare_poses(base_T_torso_current, base_T_torso_expected)

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

        arm_lift_joint: ActiveConnection1DOF = giskard.world.get_connection_by_name(
            HSRBJoint.ARM_LIFT
        )
        np.testing.assert_almost_equal(
            arm_lift_joint.position,
            0.3,
            decimal=2,
        )
        base_T_torso_expected = HomogeneousTransformationMatrix.from_xyz_rpy(
            z=0.902, reference_frame=giskard.base_footprint
        )
        base_T_torso_current = giskard.world.compute_forward_kinematics(
            root=giskard.base_footprint, tip=giskard.torso_lift_link
        )
        compare_poses(base_T_torso_current, base_T_torso_expected)

    def test_mimic_joints4(self, giskard: HSRTester):
        arm_lift_joints: ActiveConnection1DOF = (
            giskard.api.world.get_connection_by_name(HSRBJoint.ARM_LIFT)
        )
        assert arm_lift_joints.dof.limits.lower.velocity == -0.15
        assert arm_lift_joints.dof.limits.upper.velocity == 0.15
        torso_lift_joints: ActiveConnection1DOF = (
            giskard.api.world.get_connection_by_name(HSRBJoint.TORSO_LIFT)
        )
        assert torso_lift_joints.dof.limits.lower.velocity == -0.075
        assert torso_lift_joints.dof.limits.upper.velocity == 0.075
        msc = MotionStatechart()
        msc.add_node(
            joint_goal := JointPositionList(
                goal_state=JointState.from_str_dict(
                    {HSRBJoint.TORSO_LIFT: 0.25},
                    giskard.api.world,
                )
            ),
        )
        msc.add_node(EndMotion.when_true(joint_goal))
        state_version = giskard.api.world.state.version
        giskard.api.execute(msc)
        for i in range(1000):
            try:
                np.testing.assert_almost_equal(
                    giskard.api.world.state[arm_lift_joints.dof.id].position,
                    0.5,
                    decimal=2,
                )
                break
            except AssertionError as e:
                pass
            sleep(0.01)
        else:
            assert False


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
            HSRBJoint.ARM_FLEX: -1.5,
            HSRBJoint.ARM_LIFT: 0.5,
            HSRBJoint.ARM_ROLL: 0.0,
            HSRBJoint.HEAD_PAN: 0.0,
            HSRBJoint.HEAD_TILT: 0.0,
            HSRBJoint.WRIST_FLEX: -1.5,
            HSRBJoint.WRIST_ROLL: 0.0,
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
        hole_point = Point3(x=0.5, z=0.3, reference_frame=default_pose_giskard.map)
        wiggle = "wiggle"
        default_pose_giskard.api.motion_goals.add_wiggle_insert(
            name=wiggle,
            root_link=root_link,
            tip_link=hpl,
            hole_point=hole_point,
            end_condition=wiggle,
        )
        resistence_point = Point3(
            x=0.5, z=0.4, reference_frame=default_pose_giskard.map
        )
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
                                    HSRBJoint.ARM_FLEX: 0.0,
                                    HSRBJoint.ARM_LIFT: 0.0,
                                    HSRBJoint.ARM_ROLL: -1.52,
                                    HSRBJoint.HEAD_PAN: -0.09,
                                    HSRBJoint.HEAD_TILT: -0.62,
                                    HSRBJoint.WRIST_FLEX: -1.55,
                                    HSRBJoint.WRIST_ROLL: 0.11,
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
                    {HSRBJoint.ARM_FLEX: -0.7},
                    giskard.api.world,
                )
            ),
        )
        msc.add_node(EndMotion.when_true(joint_goal))
        giskard.api.execute(msc)
