from __future__ import division

import asyncio
from copy import deepcopy
from dataclasses import dataclass
from time import sleep
from typing import Set

import numpy as np
import pytest
from docutils.nodes import field
from geometry_msgs.msg import (
    PoseStamped,
    Point,
    Quaternion,
    Vector3Stamped,
    PointStamped,
)
from numpy import pi
from rclpy.duration import Duration

import semantic_digital_twin.spatial_types.spatial_types as cas
from giskardpy.data_types.exceptions import (
    MaxTrajectoryLengthException,
)
from giskardpy.middleware.ros2 import rospy
from giskardpy.middleware.ros2.behavior_tree_config import StandAloneBTConfig
from giskardpy.middleware.ros2.scripts.iai_robots.pr2.configs import (
    PR2StandaloneInterface,
    WorldWithPR2Config,
)
from giskardpy.middleware.ros2.giskard import Giskard
from giskardpy.middleware.ros2.rospy import wait_for_future_to_complete
from giskardpy.middleware.ros2.utils.utils import load_xacro
from giskardpy.middleware.ros2.utils.utils_for_tests import (
    GiskardTester,
)
from giskardpy.motion_statechart.data_types import (
    DefaultWeights,
    ObservationStateValues,
)
from giskardpy.motion_statechart.exceptions import EmptyMotionStatechartError
from giskardpy.motion_statechart.goals.collision_avoidance import (
    ExternalCollisionAvoidance,
    SelfCollisionAvoidance,
    UpdateTemporaryCollisionRules,
)
from giskardpy.motion_statechart.goals.templates import Parallel, Sequence
from giskardpy.motion_statechart.goals.tracebot import InsertCylinder
from giskardpy.motion_statechart.graph_node import EndMotion, CancelMotion
from giskardpy.motion_statechart.monitors.monitors import LocalMinimumReached
from giskardpy.motion_statechart.monitors.overwrite_state_monitors import (
    SetOdometry,
    SetSeedConfiguration,
)
from giskardpy.motion_statechart.motion_statechart import MotionStatechart
from giskardpy.motion_statechart.tasks.align_planes import AlignPlanes
from giskardpy.motion_statechart.tasks.cartesian_tasks import (
    CartesianPose,
)
from giskardpy.motion_statechart.tasks.joint_tasks import (
    JointPositionList,
    JointState,
)
from giskardpy.motion_statechart.tasks.pointing import Pointing
from giskardpy.qp.qp_controller_config import QPControllerConfig
from giskardpy.tree.blackboard_utils import GiskardBlackboard
from giskardpy.utils.math import (
    quaternion_from_axis_angle,
    quaternion_from_rotation_matrix,
)
from giskardpy.middleware.ros2.exceptions import (
    ExecutionCanceledException,
    ExecutionAbortedException,
)
from krrood.symbolic_math import symbolic_math as sm
from semantic_digital_twin.collision_checking.collision_rules import (
    AvoidExternalCollisions,
    AvoidCollisionBetweenGroups,
)
from semantic_digital_twin.exceptions import WorldEntityNotFoundError
from semantic_digital_twin.robots.abstract_robot import AbstractRobot
from semantic_digital_twin.robots.robot_parts import ParallelGripper
from semantic_digital_twin.spatial_types import (
    HomogeneousTransformationMatrix,
    Point3,
    RotationMatrix,
    Vector3,
)
from semantic_digital_twin.spatial_types.spatial_types import Pose
from semantic_digital_twin.world_description.connections import (
    RevoluteConnection,
    PrismaticConnection,
    ActiveConnection1DOF,
)
from semantic_digital_twin.world_description.world_entity import (
    Body,
    KinematicStructureEntity,
)


@dataclass
class PR2Tester(GiskardTester):

    better_pose_right = {
        "r_shoulder_pan_joint": -1.7125,
        "r_shoulder_lift_joint": -0.25672,
        "r_upper_arm_roll_joint": -1.46335,
        "r_elbow_flex_joint": -2.12,
        "r_forearm_roll_joint": 1.76632,
        "r_wrist_flex_joint": -0.10001,
        "r_wrist_roll_joint": 0.05106,
    }

    better_pose_left = {
        "l_shoulder_pan_joint": 1.9652,
        "l_shoulder_lift_joint": -0.26499,
        "l_upper_arm_roll_joint": 1.3837,
        "l_elbow_flex_joint": -2.12,
        "l_forearm_roll_joint": 16.99,
        "l_wrist_flex_joint": -0.10001,
        "l_wrist_roll_joint": 0,
    }

    r_tip: KinematicStructureEntity = field(init=False)
    l_tip: KinematicStructureEntity = field(init=False)
    odom_combined: KinematicStructureEntity = field(init=False)

    def __post_init__(self):
        super().__post_init__()
        self.l_tip = self.api.world.get_kinematic_structure_entity_by_name(
            "l_gripper_tool_frame"
        )
        self.r_tip = self.api.world.get_kinematic_structure_entity_by_name(
            "r_gripper_tool_frame"
        )
        self.camera = self.api.world.get_kinematic_structure_entity_by_name(
            "head_mount_kinect_rgb_link"
        )
        self.odom_combined = self.api.world.get_kinematic_structure_entity_by_name(
            "odom_combined"
        )
        self.base_footprint = self.api.world.get_kinematic_structure_entity_by_name(
            "base_footprint"
        )
        self.base_link = self.api.world.get_kinematic_structure_entity_by_name(
            "base_link"
        )
        self.torso_lift_link = self.api.world.get_kinematic_structure_entity_by_name(
            "torso_lift_link"
        )
        self.map = self.api.world.get_kinematic_structure_entity_by_name("map")

    def setup_giskard(self) -> Giskard:
        robot_desc = load_xacro(
            "package://iai_pr2_description/robots/pr2_with_ft2_cableguide.xacro"
        )
        return Giskard(
            world_config=WorldWithPR2Config(urdf=robot_desc),
            robot_interface_config=PR2StandaloneInterface(),
            behavior_tree_config=StandAloneBTConfig(
                debug_mode=True, add_debug_marker_publisher=True
            ),
            qp_controller_config=QPControllerConfig(
                target_frequency=20,
                retries_with_relaxed_constraints=15,
            ),
        )

    @property
    def robot(self) -> AbstractRobot:
        return (
            GiskardBlackboard().executor.context.world.get_semantic_annotation_by_name(
                self.api.robot_name
            )
        )

    @property
    def l_gripper_annotation(self) -> ParallelGripper:
        return next(sa for sa in self.robot.manipulators if "left" in str(sa.name))

    @property
    def r_gripper_annotation(self) -> ParallelGripper:
        return next(sa for sa in self.robot.manipulators if "right" in str(sa.name))

    def get_l_gripper_links(self) -> Set[Body]:
        return set(b for b in self.l_gripper_annotation.bodies if b.has_collision())

    def get_r_gripper_links(self) -> Set[Body]:
        return set(b for b in self.r_gripper_annotation.bodies if b.has_collision())

    def get_r_forearm_links(self):
        return [
            "r_wrist_flex_link",
            "r_wrist_roll_link",
            "r_forearm_roll_link",
            "r_forearm_link",
            "r_forearm_link",
        ]

    def open_r_gripper(self):
        return

    def close_r_gripper(self):
        return

    def open_l_gripper(self):
        return

    def close_l_gripper(self):
        return


@pytest.fixture()
def robot():
    c = PR2Tester()
    try:
        yield c
    finally:
        print("tear down")
        c.print_stats()


@pytest.fixture()
def default_joint_state():
    return {
        "r_elbow_flex_joint": -0.15,
        "r_forearm_roll_joint": 0,
        "r_shoulder_lift_joint": 0,
        "r_shoulder_pan_joint": 0,
        "r_upper_arm_roll_joint": 0,
        "r_wrist_flex_joint": -0.10001,
        "r_wrist_roll_joint": 0,
        "l_elbow_flex_joint": -0.15,
        "l_forearm_roll_joint": 0,
        "l_shoulder_lift_joint": 0,
        "l_shoulder_pan_joint": 0,
        "l_upper_arm_roll_joint": 0,
        "l_wrist_flex_joint": -0.10001,
        "l_wrist_roll_joint": 0,
        "torso_lift_joint": 0.2,
        "head_pan_joint": 0,
        "head_tilt_joint": 0,
        "l_gripper_l_finger_joint": 0.55,
        "r_gripper_l_finger_joint": 0.55,
    }


@pytest.fixture()
def pocky_pose_state():
    return {
        "r_elbow_flex_joint": -1.29610152504,
        "r_forearm_roll_joint": -0.0301682323805,
        "r_shoulder_lift_joint": 1.20324921318,
        "r_shoulder_pan_joint": -0.73456435706,
        "r_upper_arm_roll_joint": -0.70790051778,
        "r_wrist_flex_joint": -0.10001,
        "r_wrist_roll_joint": 0.258268529825,
        "l_elbow_flex_joint": -1.29610152504,
        "l_forearm_roll_joint": 0.0301682323805,
        "l_shoulder_lift_joint": 1.20324921318,
        "l_shoulder_pan_joint": 0.73456435706,
        "l_upper_arm_roll_joint": 0.70790051778,
        "l_wrist_flex_joint": -0.1001,
        "l_wrist_roll_joint": -0.258268529825,
        "torso_lift_joint": 0.2,
        "head_pan_joint": 0,
        "head_tilt_joint": 0,
        "l_gripper_l_finger_joint": 0.55,
        "r_gripper_l_finger_joint": 0.55,
    }


@pytest.fixture()
def better_pose():
    return {
        "r_shoulder_pan_joint": -1.7125,
        "r_shoulder_lift_joint": -0.25672,
        "r_upper_arm_roll_joint": -1.46335,
        "r_elbow_flex_joint": -2.12,
        "r_forearm_roll_joint": 1.76632,
        "r_wrist_flex_joint": -0.10001,
        "r_wrist_roll_joint": 0.05106,
        "l_shoulder_pan_joint": 1.9652,
        "l_shoulder_lift_joint": -0.26499,
        "l_upper_arm_roll_joint": 1.3837,
        "l_elbow_flex_joint": -2.12,
        "l_forearm_roll_joint": 16.99,
        "l_wrist_flex_joint": -0.10001,
        "l_wrist_roll_joint": 0,
        "torso_lift_joint": 0.2,
        "l_gripper_l_finger_joint": 0.55,
        "r_gripper_l_finger_joint": 0.55,
        "head_pan_joint": 0,
        "head_tilt_joint": 0,
    }


@pytest.fixture()
def pocky_pose_setup(giskard_factory, pocky_pose_state):
    return giskard_factory(pocky_pose_state)


@pytest.fixture()
def box_setup(pocky_pose_setup: PR2Tester) -> PR2Tester:
    pocky_pose_setup.add_box_to_world(
        name="box",
        size=(1.0, 1.0, 1.0),
        pose=HomogeneousTransformationMatrix.from_xyz_rpy(
            x=1.2, z=0.5, reference_frame=pocky_pose_setup.map
        ),
    )
    return pocky_pose_setup


@pytest.fixture()
def fake_table_setup(pocky_pose_setup: PR2Tester) -> PR2Tester:
    pocky_pose_setup.add_box_to_world(
        name="box",
        size=(1.0, 1.0, 1.0),
        pose=HomogeneousTransformationMatrix.from_xyz_rpy(
            x=1.2, z=0.3, reference_frame=pocky_pose_setup.map
        ),
    )
    return pocky_pose_setup


class TestJointGoals:
    def test_joint_goal(self, giskard: PR2Tester):
        js = {
            "torso_lift_joint": 0.2999225173357618,
            "head_pan_joint": 0.041880780651479044,
            "head_tilt_joint": -0.37,
            "r_upper_arm_roll_joint": -0.9487714747527726,
            "r_shoulder_pan_joint": -1.0047307505973626,
            "r_shoulder_lift_joint": 0.48736790658811985,
            "r_forearm_roll_joint": -14.895833882874182,
            "r_elbow_flex_joint": -1.392377908925028,
            "r_wrist_flex_joint": -0.4548695149411013,
            "r_wrist_roll_joint": 0.11426798984097819,
            "l_upper_arm_roll_joint": 1.7383062350263658,
            "l_shoulder_pan_joint": 1.8799810286792007,
            "l_shoulder_lift_joint": 0.011627231224188975,
            "l_forearm_roll_joint": 312.67276414458695,
            "l_elbow_flex_joint": -2.0300928925694675,
            "l_wrist_flex_joint": -0.1,
            "l_wrist_roll_joint": -6.062015047706399,
        }
        msc = MotionStatechart()
        msc.add_node(
            joint_goal := JointPositionList(
                goal_state=JointState.from_str_dict(js, giskard.api.world),
            )
        )
        msc.add_node(EndMotion.when_true(joint_goal))
        giskard.api.execute(msc)
        sleep(0.1)
        for joint, goal in js.items():
            connection = giskard.api.world.get_connection_by_name(joint)
            actual = giskard.api.world.state[connection.dof.id].position
            if (
                isinstance(connection, RevoluteConnection)
                and not connection.dof.has_position_limits()
            ):
                assert sm.shortest_angular_distance(actual, goal).to_np()[0] < 0.01
            else:
                try:
                    assert np.isclose(
                        actual,
                        goal,
                        atol=1e-2,
                    ), f"expected {actual} got {goal} for {connection.name}"
                except Exception as e:
                    pass

    def test_hard_joint_limits(self, giskard: PR2Tester):
        r_elbow_flex_joint: ActiveConnection1DOF = (
            giskard.api.world.get_connection_by_name("r_elbow_flex_joint")
        )
        torso_lift_joint: ActiveConnection1DOF = (
            giskard.api.world.get_connection_by_name("torso_lift_joint")
        )
        head_pan_joint: ActiveConnection1DOF = giskard.api.world.get_connection_by_name(
            "head_pan_joint"
        )
        msc = MotionStatechart()

        min_joint_goal = JointPositionList(
            goal_state=JointState.from_mapping(
                mapping={
                    r_elbow_flex_joint: r_elbow_flex_joint.dof.limits.lower.position
                    - 0.2,
                    torso_lift_joint: torso_lift_joint.dof.limits.lower.position - 0.2,
                    head_pan_joint: head_pan_joint.dof.limits.lower.position - 0.2,
                }
            )
        )
        msc.add_node(min_joint_goal)
        min_joint_goal.end_condition = min_joint_goal.observation_variable

        torso_joint_goal = JointPositionList(
            goal_state=JointState.from_mapping(mapping={torso_lift_joint: 3.2})
        )
        msc.add_node(torso_joint_goal)
        torso_joint_goal.start_condition = min_joint_goal.observation_variable
        torso_joint_goal.end_condition = torso_joint_goal.observation_variable

        max_joint_goal = JointPositionList(
            goal_state=JointState.from_mapping(
                mapping={
                    r_elbow_flex_joint: r_elbow_flex_joint.dof.limits.upper.position
                    + 0.2,
                    torso_lift_joint: torso_lift_joint.dof.limits.upper.position + 0.2,
                    head_pan_joint: head_pan_joint.dof.limits.upper.position + 0.2,
                }
            )
        )
        msc.add_node(max_joint_goal)
        max_joint_goal.start_condition = torso_joint_goal.observation_variable

        end = EndMotion()
        msc.add_node(end)
        end.start_condition = max_joint_goal.observation_variable
        giskard.api.execute(msc)


class TestConstraints:

    def test_drive_into_apartment(self, apartment_setup: PR2Tester):
        msc = MotionStatechart()
        msc.add_node(
            cart_goal := CartesianPose(
                root_link=apartment_setup.map,
                tip_link=apartment_setup.base_footprint,
                goal_pose=HomogeneousTransformationMatrix.from_xyz_rpy(
                    x=0.4, y=-2.0, reference_frame=apartment_setup.base_footprint
                ),
            )
        )
        msc.add_node(EndMotion.when_true(cart_goal))
        apartment_setup.api.execute(msc)

    @pytest.mark.skip(reason="me todo")
    def test_AvoidJointLimits(self, giskard: PR2Tester):
        percentage = 10.0
        giskard.api.motion_goals.allow_all_collisions()
        giskard.api.motion_goals.add_avoid_joint_limits(percentage=percentage)
        giskard.execute()

        joint_non_continuous = [
            j
            for j in giskard.robot.controlled_connections
            if isinstance(j, (PrismaticConnection, RevoluteConnection))
            and j.dof.has_position_limits()
        ]

        current_joint_state = giskard.api.world.state.to_position_dict()
        percentage *= (
            0.95  # it will not reach the exact percentage, because the weight is so low
        )
        for joint in joint_non_continuous:
            position = current_joint_state[joint.dof.name]
            lower_limit = joint.dof.limits.upper.position
            upper_limit = joint.dof.limits.upper.position
            joint_range = upper_limit - lower_limit
            center = (upper_limit + lower_limit) / 2.0
            upper_limit2 = center + joint_range / 2.0 * (1 - percentage / 100.0)
            lower_limit2 = center - joint_range / 2.0 * (1 - percentage / 100.0)
            assert upper_limit2 >= position >= lower_limit2

    @pytest.mark.skip(reason="fixme")
    def test_insert_cylinder1(self, giskard_better_pose: PR2Tester):
        cylinder_name = "C"
        cylinder_height = 0.121
        hole_point = PointStamped()
        hole_point.header.frame_id = "map"
        hole_point.point.x = 1.0
        hole_point.point.y = -1.0
        hole_point.point.z = 0.5
        pose = PoseStamped()
        pose.header.frame_id = "r_gripper_tool_frame"
        q = quaternion_from_rotation_matrix(
            np.array([[0, 0, 1, 0], [0, 1, 0, 0], [-1, 0, 0, 0], [0, 0, 0, 1]])
        )
        pose.pose.orientation = Quaternion(x=q[0], y=q[1], z=q[2], w=q[3])
        giskard_better_pose.add_cylinder_to_world(
            name=cylinder_name,
            height=cylinder_height,
            radius=0.0225,
            pose=pose,
            parent_link="r_gripper_tool_frame",
        )

        inserted = giskard_better_pose.api.motion_goals.add_motion_goal(
            class_name=InsertCylinder.__name__,
            name="Insert Cyclinder",
            cylinder_name=cylinder_name,
            cylinder_height=0.121,
            hole_point=hole_point,
        )
        giskard_better_pose.api.motion_goals.allow_all_collisions()
        giskard_better_pose.api.monitors.add_end_motion(start_condition=inserted)
        giskard_better_pose.execute(local_min_end=False)

    def test_pointing_kitchen(self, kitchen_setup: PR2Tester, better_pose):
        pointing_axis = Vector3.X(reference_frame=kitchen_setup.camera)
        gaya_pose2 = deepcopy(better_pose)
        del gaya_pose2["head_pan_joint"]
        del gaya_pose2["head_tilt_joint"]

        handle_point = kitchen_setup.api.world.compute_forward_kinematics(
            root=kitchen_setup.map,
            tip=kitchen_setup.api.world.get_kinematic_structure_entity_by_name(
                "iai_fridge_door_handle"
            ),
        ).to_position()

        msc = MotionStatechart()
        msc.add_node(
            sequence := Sequence(
                [
                    Pointing(
                        root_link=kitchen_setup.map,
                        tip_link=kitchen_setup.camera,
                        goal_point=handle_point,
                        pointing_axis=pointing_axis,
                    ),
                    Parallel(
                        [
                            Pointing(
                                root_link=kitchen_setup.map,
                                tip_link=kitchen_setup.camera,
                                goal_point=handle_point,
                                pointing_axis=pointing_axis,
                            ),
                            CartesianPose(
                                root_link=kitchen_setup.map,
                                tip_link=kitchen_setup.base_footprint,
                                goal_pose=HomogeneousTransformationMatrix.from_xyz_axis_angle(
                                    y=2.0,
                                    axis=Vector3.Z(),
                                    angle=1,
                                    reference_frame=kitchen_setup.base_footprint,
                                ),
                            ),
                            JointPositionList(
                                goal_state=JointState.from_str_dict(
                                    gaya_pose2, world=kitchen_setup.api.world
                                )
                            ),
                        ]
                    ),
                ]
            )
        )
        msc.add_node(EndMotion.when_true(sequence))
        kitchen_setup.api.execute(msc)

        msc = MotionStatechart()
        msc.add_node(
            parallel := Parallel(
                [
                    Pointing(
                        root_link=kitchen_setup.r_tip,
                        tip_link=kitchen_setup.camera,
                        goal_point=Point3(reference_frame=kitchen_setup.r_tip),
                        pointing_axis=pointing_axis,
                    ),
                    CartesianPose(
                        root_link=kitchen_setup.base_footprint,
                        tip_link=kitchen_setup.r_tip,
                        goal_pose=HomogeneousTransformationMatrix.from_point_rotation_matrix(
                            point=Point3(x=-0.3, z=0.6),
                            rotation_matrix=RotationMatrix(
                                [
                                    [0, 0, -1, 0],
                                    [0, 1, 0, 0],
                                    [1, 0, 0, 0],
                                    [0, 0, 0, 1],
                                ]
                            ),
                            reference_frame=kitchen_setup.r_tip,
                        ),
                    ),
                ]
            ),
        )
        msc.add_node(EndMotion.when_true(parallel))
        kitchen_setup.api.execute(msc)

    @pytest.mark.skip(reason="suturo")
    def test_open_close_dishwasher(self, kitchen_setup: PR2Tester):
        # TODO continue here
        p = PoseStamped()
        p.header.frame_id = "map"
        p.pose.orientation.w = 1.0
        p.pose.position.x = 0.5
        p.pose.position.y = 0.2
        kitchen_setup.teleport_base(p)

        hand = kitchen_setup.r_tip

        goal_angle = np.pi / 4
        handle_frame_id = "sink_area_dish_washer_door_handle"
        handle_name = "sink_area_dish_washer_door_handle"
        kitchen_setup.register_group(
            new_group_name="dishwasher",
            root_link_name="sink_area_dish_washer_main",
        )
        kitchen_setup.register_group(
            new_group_name="handle",
            root_link_name=handle_name,
        )
        bar_axis = Vector3Stamped()
        bar_axis.header.frame_id = handle_frame_id
        bar_axis.vector.y = 1.0

        bar_center = PointStamped()
        bar_center.header.frame_id = handle_frame_id

        tip_grasp_axis = Vector3Stamped()
        tip_grasp_axis.header.frame_id = hand
        tip_grasp_axis.vector.z = 1.0

        kitchen_setup.api.motion_goals.add_grasp_bar(
            root_link=kitchen_setup.default_root,
            tip_link=hand,
            tip_grasp_axis=tip_grasp_axis,
            bar_center=bar_center,
            bar_axis=bar_axis,
            bar_length=0.3,
        )
        # kitchen_setup.api.motion_goals.allow_collision([], 'kitchen', [handle_name])
        # kitchen_setup.api.motion_goals.allow_all_collisions()

        x_gripper = Vector3Stamped()
        x_gripper.header.frame_id = hand
        x_gripper.vector.x = 1.0

        x_goal = Vector3Stamped()
        x_goal.header.frame_id = handle_frame_id
        x_goal.vector.x = -1.0
        kitchen_setup.api.motion_goals.add_align_planes(
            tip_link=hand, root_link="map", tip_normal=x_gripper, goal_normal=x_goal
        )
        # kitchen_setup.api.motion_goals.allow_all_collisions()

        kitchen_setup.execute()

        kitchen_setup.api.motion_goals.add_open_container(
            tip_link=hand, environment_link=handle_name, goal_joint_state=goal_angle
        )
        # kitchen_setup.api.motion_goals.allow_all_collisions()
        kitchen_setup.api.motion_goals.allow_collision(
            group1=kitchen_setup.default_env_name, group2=kitchen_setup.r_gripper_group
        )
        kitchen_setup.execute()
        kitchen_setup.set_env_state({"sink_area_dish_washer_door_joint": goal_angle})

        kitchen_setup.api.motion_goals.add_open_container(
            tip_link=hand, environment_link=handle_name, goal_joint_state=0
        )
        kitchen_setup.api.motion_goals.allow_all_collisions()
        kitchen_setup.execute()
        kitchen_setup.set_env_state({"sink_area_dish_washer_door_joint": 0})

    def test_align_planes1(self, giskard: PR2Tester):
        msc = MotionStatechart()
        msc.add_node(
            parallel := Parallel(
                [
                    AlignPlanes(
                        root_link=giskard.map,
                        tip_link=giskard.r_tip,
                        tip_normal=Vector3.X(reference_frame=giskard.r_tip),
                        goal_normal=Vector3.X(reference_frame=giskard.map),
                    ),
                    AlignPlanes(
                        root_link=giskard.map,
                        tip_link=giskard.r_tip,
                        tip_normal=Vector3.Y(reference_frame=giskard.r_tip),
                        goal_normal=Vector3.Z(reference_frame=giskard.map),
                    ),
                ]
            )
        )
        msc.add_node(EndMotion.when_true(parallel))

        giskard.api.execute(msc)


class TestCartGoals:

    def test_cart_goal_1eef(self, giskard: PR2Tester):
        tip = giskard.api.world.get_kinematic_structure_entity_by_name(
            "r_gripper_tool_frame"
        )
        root = giskard.api.world.get_kinematic_structure_entity_by_name(
            "base_footprint"
        )
        tip_goal = HomogeneousTransformationMatrix.from_xyz_quaternion(
            pos_x=-0.2, reference_frame=tip
        )

        msc = MotionStatechart()
        msc.add_node(
            cart_goal := CartesianPose(
                root_link=root,
                tip_link=tip,
                goal_pose=tip_goal,
            )
        )
        msc.add_node(EndMotion.when_true(cart_goal))
        giskard.api.execute(msc)

    def test_cart_goal_unreachable(self, giskard: PR2Tester):
        msc = MotionStatechart()
        msc.add_node(
            cart_goal := CartesianPose(
                root_link=giskard.map,
                tip_link=giskard.base_footprint,
                goal_pose=HomogeneousTransformationMatrix.from_xyz_rpy(
                    z=-1,
                    reference_frame=giskard.map,
                ),
            )
        )
        msc.add_node(local_min := LocalMinimumReached())
        msc.add_node(CancelMotion.when_true(cart_goal))
        msc.add_node(EndMotion.when_true(local_min))
        giskard.api.execute(msc)

    def test_cart_goal_orientation_singularity(self, giskard: PR2Tester):
        msc = MotionStatechart()
        msc.add_node(
            parallel := Parallel(
                [
                    CartesianPose(
                        root_link=giskard.base_link,
                        tip_link=giskard.r_tip,
                        goal_pose=HomogeneousTransformationMatrix.from_xyz_quaternion(
                            pos_x=-0.1,
                            reference_frame=giskard.r_tip,
                        ),
                    ),
                    CartesianPose(
                        root_link=giskard.base_link,
                        tip_link=giskard.l_tip,
                        goal_pose=HomogeneousTransformationMatrix.from_xyz_quaternion(
                            pos_x=-0.05,
                            reference_frame=giskard.l_tip,
                        ),
                    ),
                ]
            )
        )
        msc.add_node(EndMotion.when_true(parallel))
        giskard.api.execute(msc)

    def test_cart_goal_left_right_chain(self, giskard: PR2Tester):
        msc = MotionStatechart()
        msc.add_node(
            cart_goal := CartesianPose(
                root_link=giskard.l_tip,
                tip_link=giskard.r_tip,
                goal_pose=HomogeneousTransformationMatrix.from_point_rotation_matrix(
                    point=Point3(
                        0.2,
                        0.0,
                        reference_frame=giskard.l_tip,
                    ),
                    rotation_matrix=RotationMatrix(
                        [[-1, 0, 0, 0], [0, -1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1.0]],
                        reference_frame=giskard.l_tip,
                    ),
                    reference_frame=giskard.l_tip,
                ),
            )
        )
        msc.add_node(EndMotion.when_true(cart_goal))
        giskard.api.execute(msc)

    def test_root_link_not_equal_chain_root(self, giskard: PR2Tester):
        msc = MotionStatechart()
        msc.add_node(
            cart_goal := CartesianPose(
                root_link=giskard.torso_lift_link,
                tip_link=giskard.r_tip,
                goal_pose=HomogeneousTransformationMatrix.from_xyz_rpy(
                    x=0.8,
                    y=-0.5,
                    z=1.0,
                    reference_frame=giskard.base_footprint,
                ),
            )
        )
        msc.add_node(EndMotion.when_true(cart_goal))
        giskard.api.execute(msc)


class TestSelfCollisionAvoidance:

    def test_cable_guide_collision(self, giskard: PR2Tester):
        msc = MotionStatechart()
        msc.add_nodes(
            [
                joint_goal := JointPositionList(
                    goal_state=JointState.from_str_dict(
                        {"head_pan_joint": 2.84, "head_tilt_joint": 1.0},
                        world=giskard.api.world,
                    )
                ),
                SelfCollisionAvoidance(),
                local_min := LocalMinimumReached(),
            ]
        )
        msc.add_node(EndMotion.when_true(local_min))
        giskard.api.execute(msc)
        assert joint_goal.observation_state == ObservationStateValues.FALSE

    def test_attached_self_collision_avoid_stick(self, giskard: PR2Tester):
        collision_pose = {
            "l_elbow_flex_joint": -1.1343683863086362,
            "l_forearm_roll_joint": 7.517553513504836,
            "l_shoulder_lift_joint": 0.5726770101613905,
            "l_shoulder_pan_joint": 0.1592669164939349,
            "l_upper_arm_roll_joint": 0.5532568387077381,
            "l_wrist_flex_joint": -1.215660155912625,
            "l_wrist_roll_joint": 4.249300323527076,
            "torso_lift_joint": 0.2,
        }

        msc = MotionStatechart()
        msc.add_nodes(
            [
                joint_goal := JointPositionList(
                    goal_state=JointState.from_str_dict(
                        collision_pose,
                        world=giskard.api.world,
                    )
                ),
            ]
        )
        msc.add_node(EndMotion.when_true(joint_goal))
        giskard.api.execute(msc)

        attached_link_name = "pocky"
        giskard.add_box_to_world(
            name=attached_link_name,
            size=(0.16, 0.04, 0.04),
            parent_link=giskard.l_tip,
            pose=HomogeneousTransformationMatrix.from_xyz_rpy(
                x=0.04, reference_frame=giskard.l_tip
            ),
        )

        msc = MotionStatechart()
        msc.add_nodes(
            [
                JointPositionList(
                    goal_state=JointState.from_str_dict(
                        {
                            "r_forearm_roll_joint": 0.0,
                            "r_shoulder_lift_joint": 0.0,
                            "r_shoulder_pan_joint": 0.0,
                            "r_upper_arm_roll_joint": 0.0,
                            "r_wrist_flex_joint": -0.10001,
                            "r_wrist_roll_joint": 0.0,
                            "r_elbow_flex_joint": -0.15,
                            "torso_lift_joint": 0.2,
                        },
                        world=giskard.api.world,
                    )
                ),
                cart_goal := CartesianPose(
                    root_link=giskard.map,
                    tip_link=giskard.l_tip,
                    goal_pose=HomogeneousTransformationMatrix.from_xyz_rpy(
                        z=0.2, reference_frame=giskard.l_tip
                    ),
                ),
                SelfCollisionAvoidance(),
            ]
        )
        msc.add_node(EndMotion.when_true(cart_goal))
        giskard.api.execute(msc)

        giskard.check_cpi_geq(giskard.get_l_gripper_links(), 0.048)
        giskard.check_cpi_geq(
            [
                giskard.api.world.get_kinematic_structure_entity_by_name(
                    attached_link_name
                )
            ],
            0.048,
        )

    def test_box_overlapping_with_gripper(self, giskard_better_pose: PR2Tester):
        box_name = "muh"
        giskard_better_pose.add_box_to_world(
            name=box_name,
            size=(0.2, 0.1, 0.1),
            pose=HomogeneousTransformationMatrix(
                reference_frame=giskard_better_pose.r_tip
            ),
            parent_link=giskard_better_pose.r_tip,
        )
        box = giskard_better_pose.api.world.get_kinematic_structure_entity_by_name(
            box_name
        )

        msc = MotionStatechart()
        msc.add_nodes(
            [
                cart_goal := CartesianPose(
                    root_link=giskard_better_pose.map,
                    tip_link=box,
                    goal_pose=HomogeneousTransformationMatrix.from_xyz_rpy(
                        x=-0.5, reference_frame=box
                    ),
                ),
                SelfCollisionAvoidance(robot=giskard_better_pose.api.robot),
            ]
        )
        msc.add_node(EndMotion.when_true(cart_goal))
        giskard_better_pose.api.execute(msc)

    def test_avoid_self_collision_with_l_arm(self, giskard: PR2Tester):
        msc = MotionStatechart()
        msc.add_nodes(
            [
                sequence := Sequence(
                    [
                        SetSeedConfiguration(
                            seed_configuration=JointState.from_str_dict(
                                {
                                    "r_elbow_flex_joint": -1.43286344265,
                                    "r_forearm_roll_joint": -1.26465060073,
                                    "r_shoulder_lift_joint": 0.47990329056,
                                    "r_shoulder_pan_joint": -0.281272240139,
                                    "r_upper_arm_roll_joint": -0.528415402668,
                                    "r_wrist_flex_joint": -1.18811419869,
                                    "r_wrist_roll_joint": 2.26884630124,
                                },
                                world=giskard.api.world,
                            )
                        ),
                        CartesianPose(
                            root_link=giskard.base_footprint,
                            tip_link=giskard.r_tip,
                            goal_pose=Pose.from_xyz_rpy(
                                0.2, reference_frame=giskard.r_tip
                            ),
                        ),
                    ]
                ),
                SelfCollisionAvoidance(robot=giskard.api.robot),
            ]
        )

        msc.add_node(EndMotion.when_true(sequence))

        giskard.api.execute(msc)
        giskard.check_cpi_geq(giskard.get_r_gripper_links(), 0.048)

    def test_avoid_self_collision_specific_link(self, giskard: PR2Tester):
        msc = MotionStatechart()
        msc.add_nodes(
            [
                sequence := Sequence(
                    [
                        SetSeedConfiguration(
                            seed_configuration=JointState.from_str_dict(
                                {
                                    "r_shoulder_pan_joint": -0.0672581793019,
                                    "r_shoulder_lift_joint": 0.429650469244,
                                    "r_upper_arm_roll_joint": -0.580889703636,
                                    "r_forearm_roll_joint": -101.948215412,
                                    "r_elbow_flex_joint": -1.35221928696,
                                    "r_wrist_flex_joint": -0.986144640142,
                                    "r_wrist_roll_joint": 0.7397216172451033,
                                },
                                world=giskard.api.world,
                            )
                        ),
                        CartesianPose(
                            root_link=giskard.base_footprint,
                            tip_link=giskard.r_tip,
                            goal_pose=Pose.from_xyz_rpy(
                                0.2, reference_frame=giskard.r_tip
                            ),
                        ),
                    ]
                ),
                UpdateTemporaryCollisionRules(
                    temporary_rules=[
                        AvoidCollisionBetweenGroups(
                            buffer_zone_distance=0.05,
                            body_group_a=list(giskard.get_r_gripper_links()),
                            body_group_b=[
                                giskard.api.world.get_kinematic_structure_entity_by_name(
                                    "l_forearm_link"
                                )
                            ],
                        )
                    ]
                ),
                SelfCollisionAvoidance(robot=giskard.api.robot),
            ]
        )

        msc.add_node(EndMotion.when_true(sequence))
        giskard.api.execute(msc)

        giskard.check_cpi_geq(giskard.get_r_gripper_links(), 0.048)

    def test_get_out_of_self_collision(self, giskard: PR2Tester):
        msc = MotionStatechart()
        msc.add_node(
            sequence := Sequence(
                [
                    JointPositionList(
                        goal_state=JointState.from_str_dict(
                            {
                                "l_elbow_flex_joint": -1.43286344265,
                                "l_forearm_roll_joint": 1.26465060073,
                                "l_shoulder_lift_joint": 0.47990329056,
                                "l_shoulder_pan_joint": 0.281272240139,
                                "l_upper_arm_roll_joint": 0.528415402668,
                                "l_wrist_flex_joint": -1.18811419869,
                                "l_wrist_roll_joint": 2.26884630124,
                            },
                            world=giskard.api.world,
                        )
                    ),
                    CartesianPose(
                        root_link=giskard.base_footprint,
                        tip_link=giskard.l_tip,
                        goal_pose=HomogeneousTransformationMatrix.from_xyz_rpy(
                            0.15, reference_frame=giskard.l_tip
                        ),
                    ),
                ]
            )
        )
        msc.add_node(EndMotion.when_true(sequence))
        giskard.api.execute(msc)

        msc = MotionStatechart()
        msc.add_nodes(
            [
                SelfCollisionAvoidance(robot=giskard.api.robot),
                local_min := LocalMinimumReached(),
            ]
        )
        msc.add_node(EndMotion.when_true(local_min))
        with pytest.raises(ExecutionAbortedException):
            giskard.api.execute(msc)


class TestCollisionAvoidanceGoals:

    def test_hard_constraints_violated(self, kitchen_setup: PR2Tester):
        msc = MotionStatechart()
        msc.add_node(
            Sequence(
                [
                    SetOdometry(
                        base_pose=HomogeneousTransformationMatrix.from_xyz_rpy(
                            x=2.0, reference_frame=kitchen_setup.map
                        )
                    ),
                    ExternalCollisionAvoidance(),
                ]
            )
        )
        with pytest.raises(ExecutionAbortedException):
            kitchen_setup.api.execute(msc)

    def test_avoid_collision_go_around_corner(self, fake_table_setup: PR2Tester):
        msc = MotionStatechart()

        cart_goal = CartesianPose(
            root_link=fake_table_setup.default_root,
            tip_link=fake_table_setup.r_tip,
            goal_pose=cas.Pose.from_xyz_axis_angle(
                x=0.8,
                y=-0.38,
                z=0.84,
                axis=cas.Vector3.Y(),
                angle=np.pi / 2.0,
                reference_frame=fake_table_setup.default_root,
            ),
        )
        msc.add_node(cart_goal)

        msc.add_node(
            UpdateTemporaryCollisionRules(
                temporary_rules=[
                    AvoidExternalCollisions(
                        buffer_zone_distance=0.1,
                        violated_distance=0.0,
                        robot=fake_table_setup.api.robot,
                    )
                ]
            )
        )
        msc.add_node(ExternalCollisionAvoidance(robot=fake_table_setup.api.robot))
        msc.add_node(local_min := LocalMinimumReached())
        msc.add_node(EndMotion.when_true(local_min))

        fake_table_setup.api.execute(msc)
        fake_table_setup.check_cpi_geq(fake_table_setup.get_l_gripper_links(), 0.05)
        fake_table_setup.check_cpi_geq(
            [
                GiskardBlackboard().executor.context.world.get_kinematic_structure_entity_by_name(
                    "r_gripper_l_finger_tip_link"
                )
            ],
            0.045,
        )
        fake_table_setup.check_cpi_geq(
            [
                GiskardBlackboard().executor.context.world.get_kinematic_structure_entity_by_name(
                    "r_gripper_r_finger_tip_link"
                )
            ],
            0.045,
        )

    def test_avoid_collision_box_between_3_boxes(self, pocky_pose_setup: PR2Tester):
        pocky_pose_setup.add_box_to_world(
            name="box",
            size=(0.2, 0.05, 0.05),
            parent_link=pocky_pose_setup.r_tip,
            pose=HomogeneousTransformationMatrix.from_xyz_rpy(
                x=0.08, reference_frame=pocky_pose_setup.r_tip
            ),
        )
        pocky_pose_setup.add_box_to_world(
            "b1",
            (0.01, 0.2, 0.2),
            pose=HomogeneousTransformationMatrix.from_xyz_rpy(
                x=0.2, reference_frame=pocky_pose_setup.r_tip
            ),
        )
        pocky_pose_setup.add_box_to_world(
            "bl",
            (0.1, 0.01, 0.2),
            pose=HomogeneousTransformationMatrix.from_xyz_rpy(
                x=0.15, y=0.04, reference_frame=pocky_pose_setup.r_tip
            ),
        )
        pocky_pose_setup.add_box_to_world(
            "br",
            (0.1, 0.01, 0.2),
            pose=HomogeneousTransformationMatrix.from_xyz_rpy(
                x=0.15, y=-0.04, reference_frame=pocky_pose_setup.r_tip
            ),
        )
        pocky_pose_setup.giskard.world_config.world.get_kinematic_structure_entity_by_name(
            "box"
        )
        pocky_pose_setup.giskard.world_config.world.get_kinematic_structure_entity_by_name(
            "b1"
        )
        pocky_pose_setup.giskard.world_config.world.get_kinematic_structure_entity_by_name(
            "bl"
        )
        pocky_pose_setup.giskard.world_config.world.get_kinematic_structure_entity_by_name(
            "br"
        )

        box = pocky_pose_setup.api.world.get_kinematic_structure_entity_by_name("box")

        msc = MotionStatechart()
        msc.add_nodes(
            [
                AlignPlanes(
                    root_link=pocky_pose_setup.map,
                    tip_link=box,
                    tip_normal=Vector3.X(reference_frame=box),
                    goal_normal=Vector3.X(reference_frame=pocky_pose_setup.map),
                ),
                AlignPlanes(
                    root_link=pocky_pose_setup.map,
                    tip_link=box,
                    tip_normal=Vector3.Y(reference_frame=box),
                    goal_normal=Vector3.Y(reference_frame=pocky_pose_setup.map),
                ),
                ExternalCollisionAvoidance(),
                local_min := LocalMinimumReached(),
            ]
        )
        msc.add_node(EndMotion.when_true(local_min))

        pocky_pose_setup.api.execute(msc)
        pocky_pose_setup.check_cpi_geq(pocky_pose_setup.get_r_gripper_links(), 0.04)

    def test_attached_two_items(self, giskard: PR2Tester):
        box1_name = "box1"
        box2_name = "box2"

        msc = MotionStatechart()
        msc.add_node(
            node := Sequence(
                [
                    JointPositionList(
                        goal_state=JointState.from_str_dict(
                            {
                                "r_elbow_flex_joint": -1.58118094489,
                                "r_forearm_roll_joint": -0.904933033043,
                                "r_shoulder_lift_joint": 0.822412440711,
                                "r_shoulder_pan_joint": -1.07866800992,
                                "r_upper_arm_roll_joint": -1.34905471854,
                                "r_wrist_flex_joint": -1.20182042644,
                                "r_wrist_roll_joint": 0.190433188769,
                            },
                            world=giskard.api.world,
                        )
                    ),
                    CartesianPose(
                        root_link=giskard.torso_lift_link,
                        tip_link=giskard.l_tip,
                        goal_pose=HomogeneousTransformationMatrix.from_point_rotation_matrix(
                            point=Point3(x=0.4),
                            rotation_matrix=RotationMatrix(
                                [
                                    [-1, 0, 0, 0],
                                    [0, 1, 0, 0],
                                    [0, 0, -1, 0],
                                    [0, 0, 0, 1.0],
                                ]
                            ),
                            reference_frame=giskard.r_tip,
                        ),
                    ),
                ]
            )
        )
        msc.add_node(EndMotion.when_true(node))
        giskard.api.execute(msc)

        giskard.add_box_to_world(
            box1_name,
            size=(0.2, 0.04, 0.04),
            parent_link=giskard.r_tip,
            pose=HomogeneousTransformationMatrix.from_xyz_rpy(
                x=0.1, reference_frame=giskard.r_tip
            ),
        )
        giskard.add_box_to_world(
            box2_name,
            size=(0.2, 0.04, 0.04),
            parent_link=giskard.l_tip,
            pose=HomogeneousTransformationMatrix.from_xyz_rpy(
                x=0.1, reference_frame=giskard.l_tip
            ),
        )
        msc = MotionStatechart()
        msc.add_nodes(
            [
                SelfCollisionAvoidance(robot=giskard.api.robot),
                local_min := LocalMinimumReached(),
            ]
        )
        msc.add_node(EndMotion.when_true(local_min))
        giskard.api.execute(msc)

        giskard.check_cpi_geq(
            [
                giskard.api.world.get_kinematic_structure_entity_by_name(box1_name),
                giskard.api.world.get_kinematic_structure_entity_by_name(box2_name),
            ],
            0.049,
        )

    @pytest.mark.skip("fix grasp bar")
    def test_get_milk_out_of_fridge(self, kitchen_setup: PR2Tester, better_pose):
        milk_name = "milk"
        iai_fridge_door_shelf1_bottom = (
            kitchen_setup.api.world.get_kinematic_structure_entity_by_name(
                "iai_fridge_door_shelf1_bottom"
            )
        )

        # take milk out of fridge
        hinge: ActiveConnection1DOF = kitchen_setup.api.world.get_connection_by_name(
            "iai_fridge_door_joint"
        )
        hinge.position = 1.56

        # spawn milk
        kitchen_setup.add_box_to_world(
            milk_name,
            (0.05, 0.05, 0.2),
            pose=HomogeneousTransformationMatrix.from_xyz_rpy(
                z=0.12, reference_frame=iai_fridge_door_shelf1_bottom
            ),
        )

        msc = MotionStatechart()
        msc.add_node(
            Sequence(
                [
                    SetOdometry(
                        base_pose=HomogeneousTransformationMatrix.from_xyz_quaternion(
                            pos_x=0.565,
                            pos_y=-0.5,
                            quat_z=-0.51152562713,
                            quat_w=0.85926802151,
                            reference_frame=kitchen_setup.map,
                        )
                    ),
                ]
            )
        )

        milk_pre_pose = PoseStamped()
        milk_pre_pose.header.frame_id = "iai_fridge_door_shelf1_bottom"
        milk_pre_pose.pose.position.z = 0.22
        milk_pre_pose.pose.orientation.w = 1.0

        # grasp milk
        kitchen_setup.open_l_gripper()

        bar_axis = Vector3Stamped()
        bar_axis.header.frame_id = "map"
        bar_axis.vector.z = 1.0

        bar_center = PointStamped()
        bar_center.header.frame_id = milk_pose.header.frame_id
        bar_center.point = deepcopy(milk_pose.pose.position)

        tip_grasp_axis = Vector3Stamped()
        tip_grasp_axis.header.frame_id = kitchen_setup.l_tip
        tip_grasp_axis.vector.z = 1.0
        kitchen_setup.api.motion_goals.add_grasp_bar(
            bar_center=bar_center,
            bar_axis=bar_axis,
            bar_length=0.12,
            tip_link=kitchen_setup.l_tip,
            tip_grasp_axis=tip_grasp_axis,
            root_link=kitchen_setup.default_root,
        )

        x = Vector3Stamped()
        x.header.frame_id = kitchen_setup.l_tip
        x.vector.x = 1.0
        x_map = Vector3Stamped()
        x_map.header.frame_id = "iai_fridge_door"
        x_map.vector.x = 1.0
        kitchen_setup.api.motion_goals.add_align_planes(
            tip_link=kitchen_setup.l_tip,
            tip_normal=x,
            goal_normal=x_map,
            root_link="map",
        )

        kitchen_setup.execute()

        kitchen_setup.update_parent_link_of_group(milk_name, kitchen_setup.l_tip)
        kitchen_setup.close_l_gripper()

        # Remove Milk
        kitchen_setup.api.motion_goals.add_cartesian_pose(
            milk_pre_pose, milk_name, kitchen_setup.default_root
        )
        kitchen_setup.execute()
        base_goal = PoseStamped()
        base_goal.header.frame_id = "base_footprint"
        base_goal.pose.orientation.w = 1.0
        kitchen_setup.api.motion_goals.add_joint_position(better_pose)
        kitchen_setup.move_base(base_goal)

        # place milk back
        kitchen_setup.api.motion_goals.add_cartesian_pose(
            milk_pre_pose, milk_name, kitchen_setup.default_root
        )
        kitchen_setup.execute()

        kitchen_setup.api.motion_goals.add_cartesian_pose(
            milk_pose, milk_name, kitchen_setup.default_root
        )
        kitchen_setup.execute()

        kitchen_setup.open_l_gripper()

        kitchen_setup.detach_group(milk_name)

        kitchen_setup.api.motion_goals.add_joint_position(better_pose)
        kitchen_setup.execute()

    @pytest.mark.skip("fix grasp bar")
    def test_bowl_and_cup(self, kitchen_setup: PR2Tester, better_pose):
        # kernprof -lv py.test -s test/test_integration_pr2.py::TestCollisionAvoidanceGoals::test_bowl_and_cup
        bowl_name = "bowl"
        cup_name = "cup"
        percentage = 50.0
        drawer_handle = "sink_area_left_middle_drawer_handle"
        drawer_joint = "sink_area_left_middle_drawer_main_joint"
        # spawn cup
        cup_pose = PoseStamped()
        cup_pose.header.frame_id = "sink_area_left_middle_drawer_main"
        cup_pose.header.stamp = (
            rospy.node.get_clock().now() + Duration(seconds=0.5)
        ).to_msg()
        cup_pose.pose.position = Point(x=0.1, y=0.2, z=-0.05)
        cup_pose.pose.orientation.w = 1.0

        kitchen_setup.add_cylinder_to_world(
            name=cup_name,
            height=0.07,
            radius=0.04,
            pose=cup_pose,
            parent_link="sink_area_left_middle_drawer_main",
        )

        # spawn bowl
        bowl_pose = PoseStamped()
        bowl_pose.header.frame_id = "sink_area_left_middle_drawer_main"
        bowl_pose.pose.position = Point(x=0.1, y=-0.2, z=-0.05)
        bowl_pose.pose.orientation.w = 1.0

        kitchen_setup.add_cylinder_to_world(
            name=bowl_name,
            height=0.05,
            radius=0.07,
            pose=bowl_pose,
            parent_link="sink_area_left_middle_drawer_main",
        )

        # grasp drawer handle
        bar_axis = Vector3Stamped()
        bar_axis.header.frame_id = drawer_handle
        bar_axis.vector.y = 1.0

        bar_center = PointStamped()
        bar_center.header.frame_id = drawer_handle

        tip_grasp_axis = Vector3Stamped()
        tip_grasp_axis.header.frame_id = kitchen_setup.l_tip
        tip_grasp_axis.vector.z = 1.0

        kitchen_setup.api.motion_goals.add_grasp_bar(
            bar_center=bar_center,
            bar_axis=bar_axis,
            bar_length=0.4,
            tip_link=kitchen_setup.l_tip,
            tip_grasp_axis=tip_grasp_axis,
            root_link=kitchen_setup.default_root,
        )
        x_gripper = Vector3Stamped()
        x_gripper.header.frame_id = kitchen_setup.l_tip
        x_gripper.vector.x = 1.0

        x_goal = Vector3Stamped()
        x_goal.header.frame_id = drawer_handle
        x_goal.vector.x = -1.0

        kitchen_setup.api.motion_goals.add_align_planes(
            tip_link=kitchen_setup.l_tip,
            tip_normal=x_gripper,
            root_link=kitchen_setup.default_root,
            goal_normal=x_goal,
        )
        # kitchen_setup.api.motion_goals.allow_all_collisions()
        kitchen_setup.execute()

        # open drawer
        kitchen_setup.api.motion_goals.add_open_container(
            tip_link=kitchen_setup.l_tip, environment_link=drawer_handle
        )
        kitchen_setup.execute()
        kitchen_setup.set_env_state({drawer_joint: 0.48})

        kitchen_setup.api.motion_goals.add_joint_position(better_pose)
        base_pose = PoseStamped()
        base_pose.header.frame_id = "map"
        base_pose.pose.position.y = 1.0
        base_pose.pose.position.x = 0.1
        base_pose.pose.orientation.w = 1.0
        kitchen_setup.move_base(base_pose)

        # grasp bowl
        l_goal = deepcopy(bowl_pose)
        l_goal.header.frame_id = "sink_area_left_middle_drawer_main"
        l_goal.pose.position.z += 0.2
        q = quaternion_from_rotation_matrix(
            [[0, 1, 0, 0], [0, 0, -1, 0], [-1, 0, 0, 0], [0, 0, 0, 1.0]]
        )
        l_goal.pose.orientation = Quaternion(x=q[0], y=q[1], z=q[2], w=q[3])
        kitchen_setup.api.motion_goals.add_cartesian_pose(
            goal_pose=l_goal,
            tip_link=kitchen_setup.l_tip,
            root_link=kitchen_setup.default_root,
        )
        kitchen_setup.api.motion_goals.allow_collision(
            kitchen_setup.l_gripper_group, bowl_name
        )

        # grasp cup
        r_goal = deepcopy(cup_pose)
        r_goal.header.frame_id = "sink_area_left_middle_drawer_main"
        r_goal.pose.position.z += 0.2
        q = quaternion_from_rotation_matrix(
            [[0, 1, 0, 0], [0, 0, -1, 0], [-1, 0, 0, 0], [0, 0, 0, 1.0]]
        )
        r_goal.pose.orientation = Quaternion(x=q[0], y=q[1], z=q[2], w=q[3])
        kitchen_setup.api.motion_goals.add_avoid_joint_limits(percentage=percentage)
        kitchen_setup.api.motion_goals.add_cartesian_pose(
            goal_pose=r_goal,
            tip_link=kitchen_setup.r_tip,
            root_link=kitchen_setup.default_root,
        )
        kitchen_setup.execute()

        l_goal.pose.position.z -= 0.2
        r_goal.pose.position.z -= 0.2
        kitchen_setup.api.motion_goals.add_cartesian_pose(
            goal_pose=l_goal,
            tip_link=kitchen_setup.l_tip,
            root_link=kitchen_setup.default_root,
        )
        kitchen_setup.api.motion_goals.add_cartesian_pose(
            goal_pose=r_goal,
            tip_link=kitchen_setup.r_tip,
            root_link=kitchen_setup.default_root,
        )
        kitchen_setup.api.motion_goals.add_avoid_joint_limits(percentage=percentage)
        kitchen_setup.api.motion_goals.avoid_all_collisions(0.05)
        kitchen_setup.api.motion_goals.allow_collision(
            group1=kitchen_setup.api.robot_name, group2=bowl_name
        )
        kitchen_setup.api.motion_goals.allow_collision(
            group1=kitchen_setup.api.robot_name, group2=cup_name
        )
        kitchen_setup.execute()

        kitchen_setup.update_parent_link_of_group(
            name=bowl_name, parent_link=kitchen_setup.l_tip
        )
        kitchen_setup.update_parent_link_of_group(
            name=cup_name, parent_link=kitchen_setup.r_tip
        )

        kitchen_setup.api.motion_goals.add_joint_position(better_pose)
        kitchen_setup.execute()
        base_goal = PoseStamped()
        base_goal.header.frame_id = "base_footprint"
        base_goal.pose.position.x = -0.1
        q = quaternion_from_axis_angle([0, 0, 1], pi)
        base_goal.pose.orientation = Quaternion(x=q[0], y=q[1], z=q[2], w=q[3])
        kitchen_setup.move_base(base_goal)

        # place bowl and cup
        bowl_goal = PoseStamped()
        bowl_goal.header.frame_id = "kitchen_island_surface"
        bowl_goal.pose.position = Point(x=0.2, y=0.0, z=0.05)
        bowl_goal.pose.orientation.w = 1.0

        cup_goal = PoseStamped()
        cup_goal.header.frame_id = "kitchen_island_surface"
        cup_goal.pose.position = Point(x=0.15, y=0.25, z=0.07)
        cup_goal.pose.orientation.w = 1.0

        kitchen_setup.api.motion_goals.add_cartesian_pose(
            goal_pose=bowl_goal,
            tip_link=bowl_name,
            root_link=kitchen_setup.default_root,
        )
        kitchen_setup.api.motion_goals.add_cartesian_pose(
            goal_pose=cup_goal, tip_link=cup_name, root_link=kitchen_setup.default_root
        )
        kitchen_setup.api.motion_goals.add_avoid_joint_limits(percentage=percentage)
        kitchen_setup.api.motion_goals.avoid_all_collisions(0.05)
        kitchen_setup.execute()

        kitchen_setup.detach_group(name=bowl_name)
        kitchen_setup.detach_group(name=cup_name)
        kitchen_setup.api.motion_goals.allow_collision(
            group1=kitchen_setup.api.robot_name, group2=cup_name
        )
        kitchen_setup.api.motion_goals.allow_collision(
            group1=kitchen_setup.api.robot_name, group2=bowl_name
        )
        kitchen_setup.api.motion_goals.add_joint_position(better_pose)
        kitchen_setup.execute()


class TestManipulability:

    @pytest.mark.skip(reason="future problem.")
    def test_manip1(self, giskard: PR2Tester):
        p = PoseStamped()
        p.header.stamp = rospy.node.get_clock().now().to_msg()
        p.header.frame_id = "map"
        p.pose.position = Point(x=0.8, y=-0.3, z=1.0)
        p.pose.orientation.w = 1.0
        giskard.api.motion_goals.allow_all_collisions()
        giskard.api.motion_goals.add_cartesian_pose(p, giskard.r_tip, "map")
        giskard.api.motion_goals.add_maximize_manipulability(
            root_link="torso_lift_link", tip_link="r_gripper_tool_frame"
        )
        giskard.execute()

    @pytest.mark.skip(reason="future problem.")
    def test_manip2(self, giskard: PR2Tester):
        p = PoseStamped()
        p.header.stamp = rospy.node.get_clock().now().to_msg()
        p.header.frame_id = giskard.r_tip
        p.pose.position = Point(x=1.0, y=-0.5, z=0.0)
        p.pose.orientation.w = 1.0
        giskard.api.motion_goals.allow_all_collisions()
        giskard.api.motion_goals.add_cartesian_pose(p, giskard.r_tip, "map")
        giskard.api.motion_goals.add_maximize_manipulability(
            root_link="torso_lift_link", tip_link="r_gripper_tool_frame"
        )
        p.pose.position = Point(x=1.0, y=0.1, z=0.0)
        giskard.api.motion_goals.add_cartesian_pose(p, giskard.l_tip, "map")
        giskard.api.motion_goals.add_maximize_manipulability(
            root_link="torso_lift_link", tip_link="l_gripper_tool_frame"
        )
        giskard.execute()


class TestWeightScaling:

    @pytest.mark.skip(reason="use debug expressions to check result.")
    def test_weight_scaling1(self, giskard):
        js = {
            # 'torso_lift_joint': 0.2999225173357618,
            "head_pan_joint": 0.041880780651479044,
            "head_tilt_joint": -0.37,
            "r_upper_arm_roll_joint": -0.9487714747527726,
            "r_shoulder_pan_joint": -1.0047307505973626,
            "r_shoulder_lift_joint": 0.48736790658811985,
            "r_forearm_roll_joint": -14.895833882874182,
            "r_elbow_flex_joint": -1.392377908925028,
            "r_wrist_flex_joint": -0.4548695149411013,
            "r_wrist_roll_joint": 0.11426798984097819,
            "l_upper_arm_roll_joint": 1.7383062350263658,
            "l_shoulder_pan_joint": 1.8799810286792007,
            "l_shoulder_lift_joint": 0.011627231224188975,
            "l_forearm_roll_joint": 312.67276414458695,
            "l_elbow_flex_joint": -2.0300928925694675,
            "l_wrist_flex_joint": -0.10014623223021513,
            "l_wrist_roll_joint": -6.062015047706399,
        }
        giskard.api.motion_goals.add_joint_position(js)
        giskard.api.motion_goals.allow_all_collisions()
        giskard.execute()

        goal_pose = PoseStamped()
        goal_pose.header.frame_id = "map"
        q = quaternion_from_rotation_matrix(
            [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1.0]]
        )
        goal_pose.pose.orientation = Quaternion(x=q[0], y=q[1], z=q[2], w=q[3])
        goal_pose.pose.position.x = 2.01
        goal_pose.pose.position.y = -0.2
        goal_pose.pose.position.z = 0.7

        goal_pose2 = deepcopy(goal_pose)
        goal_pose2.pose.position.y = -0.6
        goal_pose2.pose.position.z = 0.8
        q = quaternion_from_rotation_matrix(
            [[0, 0, 1, 0], [0, 1, 0, 0], [-1, 0, 0, 0], [0, 0, 0, 1.0]]
        )
        goal_pose2.pose.orientation = Quaternion(x=q[0], y=q[1], z=q[2], w=q[3])

        giskard.api.motion_goals.add_cartesian_pose(
            goal_pose, "l_gripper_tool_frame", "map"
        )
        giskard.api.motion_goals.add_cartesian_pose(
            goal_pose2, "r_gripper_tool_frame", "map"
        )

        goal_point = PointStamped()
        goal_point.header.frame_id = goal_pose.header.frame_id
        goal_point.point = goal_pose.pose.position
        pointing_axis = Vector3Stamped()
        pointing_axis.header.frame_id = "head_mount_kinect_rgb_optical_frame"
        pointing_axis.vector.z = 1.0
        giskard.api.motion_goals.add_pointing(
            goal_point, "head_mount_kinect_rgb_optical_frame", pointing_axis, "map"
        )

        x_base = Vector3Stamped()
        x_base.header.frame_id = "base_link"
        x_base.vector.x = 1.0
        x_goal = Vector3Stamped()
        x_goal.header.frame_id = "map"
        x_goal.vector.x = 1.0
        giskard.api.motion_goals.add_align_planes(
            tip_link="base_link", root_link="map", tip_normal=x_base, goal_normal=x_goal
        )

        tip_goal = PointStamped()
        tip_goal.header.frame_id = "map"
        tip_goal.point = goal_pose.pose.position
        giskard.api.motion_goals.add_base_arm_weight_scaling(
            root_link="map",
            tip_link="l_gripper_tool_frame",
            tip_goal=tip_goal,
            gain=100000,
            arm_joints=[
                "torso_lift_joint",
                # 'head_pan_joint',
                # 'head_tilt_joint',
                "r_upper_arm_roll_joint",
                "r_shoulder_pan_joint",
                "r_shoulder_lift_joint",
                "r_forearm_roll_joint",
                "r_elbow_flex_joint",
                "r_wrist_flex_joint",
                "r_wrist_roll_joint",
                "l_upper_arm_roll_joint",
                "l_shoulder_pan_joint",
                "l_shoulder_lift_joint",
                "l_forearm_roll_joint",
                "l_elbow_flex_joint",
                "l_wrist_flex_joint",
                "l_wrist_roll_joint",
            ],
            base_joints=["odom_combined_T_base_footprint"],
        )
        giskard.api.motion_goals.add_maximize_manipulability(
            root_link="torso_lift_link", tip_link="r_gripper_tool_frame"
        )
        giskard.api.motion_goals.add_maximize_manipulability(
            root_link="torso_lift_link", tip_link="l_gripper_tool_frame"
        )
        giskard.api.motion_goals.allow_all_collisions()
        giskard.execute()
        # assert (
        #     god_map.debug_expression_manager.evaluated_debug_expressions[
        #         PrefixedName(name="arm_scaling", prefix="")
        #     ][0]
        #     * 1000
        #     < god_map.debug_expression_manager.evaluated_debug_expressions[
        #         PrefixedName(name="base_scaling", prefix="")
        #     ][0]
        # )

    @pytest.mark.skip(reason="use debug expressions to check result.")
    def test_manip(self, giskard: PR2Tester):
        p = PoseStamped()
        p.header.stamp = rospy.node.get_clock().now().to_msg()
        p.header.frame_id = "map"
        p.pose.position = Point(x=0.8, y=-0.3, z=1.0)
        p.pose.orientation.w = 1.0
        giskard.api.motion_goals.allow_all_collisions()
        giskard.api.motion_goals.add_cartesian_pose(p, giskard.r_tip, "map")
        m_threshold = 0.16
        done = giskard.api.motion_goals.add_maximize_manipulability(
            root_link="torso_lift_link", tip_link=giskard.r_tip, m_threshold=m_threshold
        )
        giskard.api.monitors.add_end_motion(done)
        giskard.api.monitors.add_check_trajectory_length(20)
        giskard.execute(local_min_end=False)
        # assert (
        #     god_map.debug_expression_manager.evaluated_debug_expressions[
        #         PrefixedName(name=f"mIndex {giskard.r_tip}", prefix="")
        #     ][0]
        #     >= m_threshold - 0.01
        # )

    @pytest.mark.skip(reason="use debug expressions to check result.")
    def test_manip2(self, giskard: PR2Tester):
        m_threshold = 0.16
        p = PoseStamped()
        p.header.stamp = rospy.node.get_clock().now().to_msg()
        p.header.frame_id = giskard.r_tip
        p.pose.position = Point(x=1.0, y=-0.5, z=0.0)
        p.pose.orientation.w = 1.0
        giskard.api.motion_goals.allow_all_collisions()
        giskard.api.motion_goals.add_cartesian_pose(p, giskard.r_tip, "map")

        giskard.api.motion_goals.add_maximize_manipulability(
            root_link="torso_lift_link",
            tip_link=giskard.r_tip,
            m_threshold=m_threshold,
        )
        p.pose.position = Point(x=1.0, y=0.1, z=0.0)
        giskard.api.motion_goals.add_cartesian_pose(p, giskard.l_tip, "map")

        giskard.api.motion_goals.add_maximize_manipulability(
            root_link="torso_lift_link",
            tip_link=giskard.l_tip,
            m_threshold=m_threshold,
        )
        giskard.execute()
        # assert (
        #     god_map.debug_expression_manager.evaluated_debug_expressions[
        #         PrefixedName(name=f"mIndex {giskard.r_tip}", prefix="")
        #     ][0]
        #     >= m_threshold - 0.02
        # )
        # assert (
        #     god_map.debug_expression_manager.evaluated_debug_expressions[
        #         PrefixedName(name=f"mIndex {giskard.l_tip}", prefix="")
        #     ][0]
        #     >= m_threshold - 0.02
        # )


class TestActionServerEvents:
    def test_wrong_params1(self, giskard: PR2Tester):
        msc = MotionStatechart()
        msc.add_node(
            joint_goal := JointPositionList(
                goal_state=HomogeneousTransformationMatrix(),
            )
        )
        msc.add_node(EndMotion.when_true(joint_goal))
        with pytest.raises(ExecutionAbortedException):
            giskard.api.execute(msc)

        msc = MotionStatechart()
        msc.add_node(
            joint_goal := JointPositionList(
                goal_state=JointState.from_str_dict(
                    giskard.better_pose_left, world=giskard.api.world
                ),
            )
        )
        msc.add_node(EndMotion.when_true(joint_goal))
        giskard.api.execute(msc)

    @pytest.mark.asyncio
    async def test_cancel_with_new_goal(self, giskard: PR2Tester):
        msc = MotionStatechart()
        msc.add_node(
            CartesianPose(
                root_link=giskard.map,
                tip_link=giskard.base_footprint,
                goal_pose=HomogeneousTransformationMatrix.from_xyz_rpy(
                    x=100, reference_frame=giskard.base_footprint
                ),
            )
        )

        goal_accepted_future = giskard.api.execute_async(msc)
        wait_for_future_to_complete(goal_accepted_future)

        await asyncio.sleep(2)

        msc = MotionStatechart()
        msc.add_node(
            cart_goal := CartesianPose(
                root_link=giskard.map,
                tip_link=giskard.base_footprint,
                goal_pose=HomogeneousTransformationMatrix.from_xyz_rpy(
                    y=1, reference_frame=giskard.base_footprint
                ),
            )
        )
        msc.add_node(EndMotion.when_true(cart_goal))

        goal_accepted_future = giskard.api.execute_async(msc)
        wait_for_future_to_complete(goal_accepted_future)

        await giskard.api.get_result()

    @pytest.mark.asyncio
    async def test_interrupt(self, giskard: PR2Tester):
        msc = MotionStatechart()
        msc.add_node(
            CartesianPose(
                root_link=giskard.map,
                tip_link=giskard.base_footprint,
                goal_pose=HomogeneousTransformationMatrix.from_xyz_rpy(
                    x=0.5, reference_frame=giskard.base_footprint
                ),
            )
        )

        goal_accepted_future = giskard.api.execute_async(msc)
        wait_for_future_to_complete(goal_accepted_future)

        await asyncio.sleep(2)
        wait_for_future_to_complete(giskard.api.cancel_goal_async())

        with pytest.raises(ExecutionCanceledException):
            await giskard.api.get_result()

    def test_empty_goal(self, giskard: PR2Tester):
        with pytest.raises(EmptyMotionStatechartError):
            giskard.api.execute(MotionStatechart())

    @pytest.mark.asyncio
    async def test_world_updats_during_execution(self, giskard: PR2Tester):
        msc = MotionStatechart()
        msc.add_node(
            CartesianPose(
                root_link=giskard.map,
                tip_link=giskard.base_footprint,
                goal_pose=HomogeneousTransformationMatrix.from_xyz_rpy(
                    x=0.5, reference_frame=giskard.base_footprint
                ),
            )
        )

        goal_accepted_future = giskard.api.execute_async(msc)
        wait_for_future_to_complete(goal_accepted_future)

        await asyncio.sleep(1)

        giskard.add_box_to_world(
            name="box",
            size=(0.05, 0.01, 0.15),
            pose=HomogeneousTransformationMatrix.from_xyz_rpy(
                z=0.06, reference_frame=giskard.r_tip
            ),
            parent_link=giskard.r_tip,
        )
        # worlds should be out of sync until the motion is done
        assert giskard.api.world.get_kinematic_structure_entity_by_name("box")
        with pytest.raises(WorldEntityNotFoundError):
            GiskardBlackboard().executor.context.world.get_kinematic_structure_entity_by_name(
                "box"
            )
        wait_for_future_to_complete(giskard.api.cancel_goal_async())
        with pytest.raises(ExecutionCanceledException):
            await giskard.api.get_result()

        await asyncio.sleep(1)
        # they should be in sync after its over
        assert giskard.api.world.get_kinematic_structure_entity_by_name("box")
        assert GiskardBlackboard().executor.context.world.get_kinematic_structure_entity_by_name(
            "box"
        )


# kernprof -lv py.test -s test/test_integration_pr2.py


class TestFeatureFunctions:
    @pytest.mark.skip("suturo")
    def test_feature_perpendicular(self, giskard: PR2Tester):
        world_feature = Vector3Stamped()
        world_feature.header.frame_id = "map"
        world_feature.vector.x = 1.0

        robot_feature = Vector3Stamped()
        robot_feature.header.frame_id = giskard.r_tip
        robot_feature.vector.x = 1.0

        giskard.api.motion_goals.add_align_perpendicular(
            root_link="map",
            tip_link=giskard.r_tip,
            reference_normal=world_feature,
            tip_normal=robot_feature,
        )
        giskard.execute()

    @pytest.mark.skip("suturo")
    def test_feature_angle(self, giskard: PR2Tester):
        world_feature = Vector3Stamped()
        world_feature.header.frame_id = "map"
        world_feature.vector.z = 1.0

        robot_feature = Vector3Stamped()
        robot_feature.header.frame_id = giskard.r_tip
        robot_feature.vector.z = 1.0

        giskard.api.motion_goals.add_angle(
            root_link="map",
            tip_link=giskard.r_tip,
            reference_vector=world_feature,
            tip_vector=robot_feature,
            lower_angle=0.6,
            upper_angle=0.9,
        )
        giskard.execute()

    @pytest.mark.skip("suturo")
    def test_feature_height(self, giskard: PR2Tester):
        world_feature = PointStamped()
        world_feature.header.frame_id = "map"

        robot_feature = PointStamped()
        robot_feature.header.frame_id = giskard.r_tip

        giskard.api.motion_goals.add_height(
            root_link="map",
            tip_link=giskard.r_tip,
            reference_point=world_feature,
            tip_point=robot_feature,
            lower_limit=0.99,
            upper_limit=1,
        )
        giskard.execute()

    @pytest.mark.skip("suturo")
    def test_feature_distance(self, giskard: PR2Tester):
        world_feature = PointStamped()
        world_feature.header.frame_id = "map"

        robot_feature = PointStamped()
        robot_feature.header.frame_id = giskard.r_tip

        giskard.api.motion_goals.add_distance(
            root_link="map",
            tip_link=giskard.r_tip,
            reference_point=world_feature,
            tip_point=robot_feature,
            lower_limit=2,
            upper_limit=2,
        )
        mon = giskard.api.monitors.add_distance(
            root_link="map",
            tip_link=giskard.r_tip,
            reference_point=world_feature,
            tip_point=robot_feature,
            lower_limit=1.99,
            upper_limit=2.01,
        )

        giskard.api.monitors.add_end_motion(mon)
        giskard.execute()


class TestEndMotionReason:
    @pytest.mark.skip("malte")
    def test_get_end_motion_reason_simple(self, giskard: PR2Tester):
        goal_point = PointStamped()
        goal_point.header.frame_id = "map"
        goal_point.point = Point(x=2.0, y=2.0, z=2.0)
        controlled_point = PointStamped()
        controlled_point.header.frame_id = giskard.r_tip

        mon_distance = giskard.api.monitors.add_distance(
            root_link="map",
            tip_link=giskard.r_tip,
            reference_point=goal_point,
            name="distance",
            tip_point=controlled_point,
            lower_limit=0,
            upper_limit=0,
        )
        giskard.api.motion_goals.add_distance(
            root_link="base_link",
            tip_link=giskard.r_tip,
            reference_point=goal_point,
            tip_point=controlled_point,
            lower_limit=0,
            upper_limit=0,
            name="reach distance",
        )

        giskard.api.monitors.add_check_trajectory_length(1)
        giskard.api.monitors.add_end_motion(mon_distance)
        result = giskard.execute(
            expected_error_type=MaxTrajectoryLengthException, local_min_end=False
        )
        reason = giskard.api.get_end_motion_reason(move_result=result)
        assert len(reason) == 1.0 and list(reason.keys())[0] == mon_distance

    @pytest.mark.skip("malte")
    def test_get_end_motion_reason_convoluted(self, giskard: PR2Tester):
        goal_point = PointStamped()
        goal_point.header.frame_id = "map"
        goal_point.point = Point(x=2.0, y=2.0, z=2.0)
        controlled_point = PointStamped()
        controlled_point.header.frame_id = giskard.r_tip

        mon_sleep1 = giskard.api.monitors.add_sleep(seconds=10, name="sleep1")
        mon_sleep2 = giskard.api.monitors.add_sleep(
            seconds=10, start_condition=mon_sleep1, name="sleep2"
        )
        mon_distance = giskard.api.monitors.add_distance(
            root_link="map",
            tip_link=giskard.r_tip,
            reference_point=goal_point,
            name="mon_distance",
            tip_point=controlled_point,
            lower_limit=0,
            upper_limit=0,
            start_condition=mon_sleep2,
        )
        giskard.api.motion_goals.add_distance(
            root_link="base_link",
            tip_link=giskard.r_tip,
            reference_point=goal_point,
            name="distance",
            tip_point=controlled_point,
            lower_limit=0,
            upper_limit=0,
        )

        giskard.api.monitors.add_check_trajectory_length(1)
        giskard.api.monitors.add_end_motion(mon_distance)
        result = giskard.execute(
            expected_error_type=MaxTrajectoryLengthException, local_min_end=False
        )
        reason = giskard.api.get_end_motion_reason(move_result=result)
        print(reason)
        assert (
            len(reason) == 3
            and list(reason.keys())[0] == mon_distance
            and list(reason.keys())[2] == mon_sleep1
            and list(reason.keys())[1] == mon_sleep2
        )

    @pytest.mark.skip("malte")
    def test_multiple_end_motion_monitors(self, giskard: PR2Tester):
        goal_point = PointStamped()
        goal_point.header.frame_id = "map"
        goal_point.point = Point(x=2.0, y=2.0, z=2.0)
        controlled_point = PointStamped()
        controlled_point.header.frame_id = giskard.r_tip

        mon_sleep1 = giskard.api.monitors.add_sleep(seconds=10, name="sleep1")
        mon_sleep2 = giskard.api.monitors.add_sleep(
            seconds=10, start_condition=mon_sleep1, name="sleep2"
        )
        mon_distance = giskard.api.monitors.add_distance(
            root_link="map",
            tip_link=giskard.r_tip,
            reference_point=goal_point,
            name="g1",
            tip_point=controlled_point,
            lower_limit=0,
            upper_limit=0,
            start_condition=mon_sleep2,
        )
        giskard.api.motion_goals.add_distance(
            root_link="base_link",
            tip_link=giskard.r_tip,
            reference_point=goal_point,
            name="g2",
            tip_point=controlled_point,
            lower_limit=0,
            upper_limit=0,
        )

        giskard.api.monitors.add_check_trajectory_length(1)
        giskard.api.monitors.add_end_motion(mon_distance, name="endmotion 1")

        mon_sleep3 = giskard.api.monitors.add_sleep(seconds=20, name="sleep3")
        mon_sleep4 = giskard.api.monitors.add_sleep(
            seconds=20, start_condition=mon_sleep3, name="sleep4"
        )
        giskard.api.monitors.add_end_motion(mon_sleep4)

        result = giskard.execute(
            expected_error_type=MaxTrajectoryLengthException, local_min_end=False
        )
        reason = giskard.api.get_end_motion_reason(move_result=result)
        print(reason)
        assert (
            len(reason) == 5
            and list(reason.keys())[0] == mon_distance
            and list(reason.keys())[1] == mon_sleep2
            and list(reason.keys())[2] == mon_sleep1
            and list(reason.keys())[3] == mon_sleep4
            and list(reason.keys())[4] == mon_sleep3
        )


# kernprof -lv py.test -s test/test_integration_pr2.py
# time: [1-9][1-9]*.[1-9]* s
# import pytest
# pytest.main(['-s', __file__ + '::TestManipulability::test_manip1'])
# pytest.main(['-s', __file__ + '::TestJointGoals::test_joint_goal'])
# pytest.main(['-s', __file__ + '::TestConstraints::test_RelativePositionSequence'])
# pytest.main(['-s', __file__ + '::TestConstraints::test_open_dishwasher_apartment'])
# pytest.main(['-s', __file__ + '::TestCollisionAvoidanceGoals::test_bowl_and_cup'])
# pytest.main(['-s', __file__ + '::TestMonitors::test_joint_and_base_goal'])
# pytest.main(['-s', __file__ + '::TestCollisionAvoidanceGoals::test_avoid_collision_go_around_corner'])
# pytest.main(['-s', __file__ + '::TestCollisionAvoidanceGoals::test_avoid_collision_box_between_boxes'])
# pytest.main(['-s', __file__ + '::TestSelfCollisionAvoidance::test_avoid_self_collision_with_l_arm'])
# pytest.main(['-s', __file__ + '::TestCollisionAvoidanceGoals::test_avoid_collision_at_kitchen_corner'])
# pytest.main(['-s', __file__ + '::TestWayPoints::test_waypoints2'])
# pytest.main(['-s', __file__ + '::TestCartGoals::test_10_cart_goals'])
# pytest.main(['-s', __file__ + '::TestCartGoals::test_cart_goal_2eef2'])
# pytest.main(['-s', __file__ + '::TestWorld::test_compute_self_collision_matrix'])
