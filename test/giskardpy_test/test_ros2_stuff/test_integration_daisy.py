from dataclasses import dataclass, field
from math import ceil
from time import sleep

import numpy as np
import pytest
from giskardpy.middleware.ros2.server_config import ExecutionMode, GiskardServerConfig
from giskardpy.middleware.ros2.giskard import Giskard
from giskardpy.middleware.ros2.scripts.iai_robots.daisy.configs import (
    WorldWithDaisyConfig,
    DaisyStandAloneRobotInterfaceConfig,
)
from giskardpy.middleware.ros2.utils.utils import load_xacro
from giskardpy.middleware.ros2.utils.utils_for_tests import compare_poses, GiskardTester
from giskardpy.motion_statechart.data_types import ObservationStateValues
from giskardpy.motion_statechart.goals.collision_avoidance import SelfCollisionAvoidance
from giskardpy.motion_statechart.goals.templates import Parallel
from giskardpy.motion_statechart.graph_node import EndMotion
from giskardpy.motion_statechart.monitors.monitors import LocalMinimumReached
from giskardpy.motion_statechart.monitors.payload_monitors import (
    CountControlCycles,
)
from giskardpy.motion_statechart.motion_statechart import (
    MotionStatechart,
)
from giskardpy.motion_statechart.tasks.cartesian_tasks import CartesianPose
from giskardpy.motion_statechart.tasks.joint_tasks import JointPositionList, JointState
from giskardpy.qp.qp_controller_config import QPControllerConfig
from semantic_digital_twin.datastructures.definitions import StaticJointState
from semantic_digital_twin.robots.daisy import DAiSy, DAiSyJoint
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
        DAiSyJoint.LEFT_SHOULDER_PAN: 0,
        DAiSyJoint.LEFT_SHOULDER_LIFT: -1.57,
        DAiSyJoint.LEFT_ELBOW: 1,
        DAiSyJoint.LEFT_WRIST_1: 0,
        DAiSyJoint.LEFT_WRIST_2: 0,
        DAiSyJoint.LEFT_WRIST_3: 0,
        DAiSyJoint.RIGHT_SHOULDER_PAN: 0,
        DAiSyJoint.RIGHT_SHOULDER_LIFT: -1.57,
        DAiSyJoint.RIGHT_ELBOW: 1,
        DAiSyJoint.RIGHT_WRIST_1: 0,
        DAiSyJoint.RIGHT_WRIST_2: 0,
        DAiSyJoint.RIGHT_WRIST_3: 0,
    }


@pytest.fixture()
def better_pose(default_joint_state):
    return {
        DAiSyJoint.LEFT_SHOULDER_PAN: 0,
        DAiSyJoint.LEFT_SHOULDER_LIFT: -1.57,
        DAiSyJoint.LEFT_ELBOW: 1,
        DAiSyJoint.LEFT_WRIST_1: 0,
        DAiSyJoint.LEFT_WRIST_2: 0,
        DAiSyJoint.LEFT_WRIST_3: np.pi / 4,
        DAiSyJoint.RIGHT_SHOULDER_PAN: 3 / 4 * 3.14,
        DAiSyJoint.RIGHT_SHOULDER_LIFT: -1.57,
        DAiSyJoint.RIGHT_ELBOW: 1,
        DAiSyJoint.RIGHT_WRIST_1: 0,
        DAiSyJoint.RIGHT_WRIST_2: 0,
        DAiSyJoint.RIGHT_WRIST_3: np.pi / 4,
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
            server_config=GiskardServerConfig(
                execution_mode=ExecutionMode.STANDALONE,
                debug_mode=True,
                plot_gantt_chart=True,
                plot_trajectory=True,
            ),
            qp_controller_config=QPControllerConfig.create_with_simulation_defaults(),
        )

    @property
    def robot(self) -> DAiSy:
        return self.giskard.executor.context.world.get_semantic_annotations_by_type(
            DAiSy
        )[0]


@pytest.fixture()
def robot():
    c = DAiSyTester()
    try:
        yield c
    finally:
        print("tear down")
        c.close()


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
    def test_joints1(self, giskard: DAiSyTester, arm: str):
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

        base = giskard.world.get_kinematic_structure_entity_by_name(f"{arm}_base_link")
        finger_tip = giskard.world.get_kinematic_structure_entity_by_name(
            f"{arm}_gripper_left_finger_tip_link"
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
        compare_poses(base_T_finger_current, base_T_finger_expected)

    @pytest.mark.parametrize(
        "arm",
        ["left", "right"],
    )
    def test_joints2(self, giskard: DAiSyTester, arm):
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

        controlled_base = giskard.world.get_kinematic_structure_entity_by_name(
            f"{arm}_base_link"
        )
        controlled_tip = giskard.world.get_kinematic_structure_entity_by_name(
            f"{arm}_gripper_left_finger_tip_link"
        )
        base_T_tip_expected = HomogeneousTransformationMatrix.from_xyz_quaternion(
            pos_x=0.3945,
            pos_y=0.4690,
            pos_z=0.6194,
            quat_x=-0.1977,
            quat_y=0.6767,
            quat_z=0.6804,
            quat_w=0.1996,
            reference_frame=controlled_base,
        )
        base_T_tip_current = giskard.world.compute_forward_kinematics(
            root=controlled_base, tip=controlled_tip
        )
        compare_poses(base_T_tip_current, base_T_tip_expected)

    @pytest.mark.parametrize(
        "arm",
        [0, 1],  # 0: left, 1: right
    )
    def test_joint3(self, giskard: DAiSyTester, arm):
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

        for i in range(1000):
            try:
                assert park_state.is_achieved()
                break
            except AssertionError as e:
                pass
            sleep(0.01)
        else:
            assert False


class TestCollisionAvoidanceGoals:
    def test_self_collision_avoidance(self, giskard_better_pose: DAiSyTester):
        msc = MotionStatechart()

        offset_x = 0.8
        offset_y = -0.1
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
            x=0.3, y=0.6, z=1.0, reference_frame=giskard_better_pose.map
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
                cycles := CountControlCycles(
                    control_cycles=ceil(
                        30
                        * giskard_better_pose.giskard.qp_controller_config.target_frequency
                    )
                ),
            ]
        )
        msc.add_node(EndMotion.when_true(cycles))
        giskard_better_pose.api.execute(msc)

        assert parallel.observation_state == ObservationStateValues.FALSE
