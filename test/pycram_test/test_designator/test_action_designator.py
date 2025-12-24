import unittest

import pytest
import rclpy
import rustworkx

from giskardpy.utils.utils_for_tests import compare_axis_angle
from pycram.datastructures.dataclasses import Context
from pycram.motion_executor import MotionExecutor
from pycram.process_module import simulated_robot
from pycram.robot_plans.actions import *
from pycram.robot_plans.motions import MoveTCPWaypointsMotion
from pycram.testing import ApartmentWorldTestCase
from semantic_digital_twin.adapters.viz_marker import VizMarkerPublisher
from semantic_digital_twin.semantic_annotations.semantic_annotations import (
    Milk,
)


# class TestActionDesignatorGrounding(ApartmentWorldTestCase):
#    """Testcase for the grounding methods of action designators."""


def test_move_torso(immutable_model_world):
    world, pr2, context = immutable_model_world
    description = MoveTorsoActionDescription([TorsoState.HIGH])
    plan = SequentialPlan(context, description)
    assert description.resolve().torso_state == TorsoState.HIGH
    # self.assertEqual(description.resolve().torso_state, TorsoState.HIGH)
    with simulated_robot:
        plan.perform()
    dof = world.get_degree_of_freedom_by_name("torso_lift_joint")
    assert world.state[dof.id].position == pytest.approx(0.29, abs=0.01)
    # self.assertAlmostEqual(self.world.state[dof.id].position, 0.29, places=2)


def test_set_gripper(immutable_model_world):
    world, pr2, context = immutable_model_world
    description = SetGripperActionDescription(
        [Arms.LEFT], [GripperStateEnum.OPEN, GripperStateEnum.CLOSE]
    )
    plan = SequentialPlan(context, description)
    assert description.resolve().gripper == Arms.LEFT
    assert description.resolve().motion == GripperStateEnum.OPEN
    with simulated_robot:
        plan.perform()
    joint_state = JointStateManager().get_gripper_state(
        Arms.LEFT, GripperStateEnum.OPEN, pr2
    )
    for joint, state in zip(joint_state.joint_names, joint_state.joint_positions):
        dof = world.get_degree_of_freedom_by_name(joint)
        assert world.state[dof.id].position == pytest.approx(state, abs=0.01)
        # self.assertAlmostEqual(self.world.state[dof.id].position, state, places=2)


def test_park_arms(immutable_model_world):
    world, robot_view, context = immutable_model_world
    description = ParkArmsActionDescription([Arms.BOTH])
    plan = SequentialPlan(context, description)
    assert description.resolve().arm == Arms.BOTH
    with simulated_robot:
        plan.perform()
    joint_states_right = JointStateManager().get_arm_state(
        Arms.RIGHT, StaticJointState.Park, robot_view
    )
    joint_states_left = JointStateManager().get_arm_state(
        Arms.LEFT, StaticJointState.Park, robot_view
    )
    for joint_name, joint_state in zip(
        joint_states_right.joint_names, joint_states_right.joint_positions
    ):
        dof = world.get_degree_of_freedom_by_name(joint_name)
        compare_axis_angle(
            world.state[dof.id].position,
            np.array([1, 0, 0]),
            joint_state,
            np.array([1, 0, 0]),
            decimal=1,
        )
        # self.assertAlmostEqual(self.world.state[dof.id].position, joint_state % (2 * np.pi), places=1)
    for joint_name, joint_state in zip(
        joint_states_left.joint_names, joint_states_left.joint_positions
    ):
        dof = world.get_degree_of_freedom_by_name(joint_name)
        compare_axis_angle(
            world.state[dof.id].position,
            [1, 0, 0],
            joint_state,
            [1, 0, 0],
            decimal=1,
        )
        # self.assertAlmostEqual(self.world.state[dof.id].position, joint_state % (2 * np.pi), places=1)


def test_navigate(immutable_model_world):
    world, robot_view, context = immutable_model_world
    description = NavigateActionDescription(
        [PoseStamped.from_list([0.3, 0, 0], [0, 0, 0, 1], world.root)]
    )
    plan = SequentialPlan(context, description)
    with simulated_robot:
        plan.perform()
    assert description.resolve().target_location == PoseStamped.from_list(
        [0.3, 0, 0], [0, 0, 0, 1], world.root
    )
    expected_pose = np.eye(4)
    expected_pose[:3, 3] = [0.3, 0, 0]
    np.testing.assert_almost_equal(
        world.compute_forward_kinematics_np(
            world.root, world.get_body_by_name("base_footprint")
        ),
        expected_pose,
        decimal=2,
    )


def test_reach_to_pick_up(immutable_model_world):
    world, robot_view, context = immutable_model_world
    grasp_description = GraspDescription(
        ApproachDirection.FRONT, VerticalAlignment.NoAlignment, False
    )
    performable = ReachActionDescription(
        target_pose=PoseStamped.from_spatial_type(
            world.get_body_by_name("milk.stl").global_pose
        ),
        object_designator=world.get_body_by_name("milk.stl"),
        arm=Arms.LEFT,
        grasp_description=grasp_description,
    )
    plan = SequentialPlan(
        context,
        NavigateActionDescription(
            PoseStamped.from_list([1.7, 1.5, 0], [0, 0, 0, 1], world.root),
            True,
        ),
        ParkArmsActionDescription(Arms.BOTH),
        MoveTorsoActionDescription([TorsoState.HIGH]),
        SetGripperActionDescription(Arms.LEFT, GripperStateEnum.OPEN),
        performable,
    )
    with simulated_robot:
        plan.perform()
    gripper_pose = world.get_body_by_name("l_gripper_tool_frame").global_pose.to_np()[
        :3, 3
    ]
    np.testing.assert_almost_equal(gripper_pose, np.array([2.37, 2, 1.05]), decimal=2)


def test_pick_up(mutable_model_world):
    world, robot_view, context = mutable_model_world

    grasp_description = GraspDescription(
        ApproachDirection.FRONT, VerticalAlignment.NoAlignment, False
    )
    description = PickUpActionDescription(
        world.get_body_by_name("milk.stl"), [Arms.LEFT], [grasp_description]
    )

    plan = SequentialPlan(
        context,
        NavigateActionDescription(
            PoseStamped.from_list([1.7, 1.5, 0], [0, 0, 0, 1], world.root),
            True,
        ),
        MoveTorsoActionDescription([TorsoState.HIGH]),
        description,
    )
    with simulated_robot:
        plan.perform()
    assert (
        world.get_connection(
            world.get_body_by_name("l_gripper_tool_frame"),
            world.get_body_by_name("milk.stl"),
        )
        is not None
    )


def test_place(mutable_model_world):
    world, robot_view, context = mutable_model_world

    object_description = world.get_body_by_name("milk.stl")
    description = PlaceActionDescription(
        object_description,
        PoseStamped.from_list([2.2, 2, 1], [0, 0, 0, 1], world.root),
        [Arms.LEFT],
    )
    plan = SequentialPlan(
        Context.from_world(world),
        NavigateActionDescription(
            PoseStamped.from_list([1.7, 1.5, 0], [0, 0, 0, 1], world.root),
            True,
        ),
        MoveTorsoActionDescription([TorsoState.HIGH]),
        PickUpActionDescription(
            object_description,
            Arms.LEFT,
            GraspDescription(
                ApproachDirection.FRONT, VerticalAlignment.NoAlignment, False
            ),
        ),
        description,
    )
    with simulated_robot:
        plan.perform()
    with pytest.raises(rustworkx.NoEdgeBetweenNodes):
        assert (
            world.get_connection(
                world.get_body_by_name("l_gripper_tool_frame"),
                world.get_body_by_name("milk.stl"),
            )
            is None
        )


def test_look_at(immutable_model_world):
    world, robot_view, context = immutable_model_world

    description = LookAtAction.description(
        [PoseStamped.from_list([5, 0, 1], frame=world.root)]
    )
    assert description.resolve().target == PoseStamped.from_list(
        [5, 0, 1], frame=world.root
    )

    plan = SequentialPlan(context, description)
    with simulated_robot:
        # self._test_validate_action_pre_perform(description, LookAtGoalNotReached)
        plan.perform()


def test_detect(immutable_model_world):
    world, robot_view, context = immutable_model_world
    milk_body = world.get_body_by_name("milk.stl")
    with world.modify_world():
        world.add_semantic_annotation(Milk(body=milk_body))
    milk_body.parent_connection.origin = HomogeneousTransformationMatrix.from_xyz_rpy(
        2.5, 2, 1.2, reference_frame=world.root
    )

    description = DetectActionDescription(
        technique=DetectionTechnique.TYPES,
        object_sem_annotation=Milk,
    )
    plan = SequentialPlan(context, description)
    with simulated_robot:
        detected_object = plan.perform()

    assert detected_object[0].name.name == "milk.stl"
    assert detected_object[0] is milk_body


def test_open(immutable_model_world):
    world, robot_view, context = immutable_model_world

    plan = SequentialPlan(
        context,
        MoveTorsoActionDescription([TorsoState.HIGH]),
        ParkArmsActionDescription(Arms.BOTH),
        NavigateActionDescription(
            PoseStamped.from_list([1.75, 1.75, 0], [0, 0, 0.5, 1], world.root)
        ),
        OpenActionDescription(world.get_body_by_name("handle_cab10_t"), [Arms.LEFT]),
    )
    with simulated_robot:
        plan.perform()
    assert world.state[
        world.get_degree_of_freedom_by_name("cabinet10_drawer_top_joint").id
    ].position == pytest.approx(0.45, abs=0.1)


def test_close(immutable_model_world):
    world, robot_view, context = immutable_model_world

    world.state[
        world.get_degree_of_freedom_by_name("cabinet10_drawer_top_joint").id
    ].position = 0.45
    world.notify_state_change()
    plan = SequentialPlan(
        context,
        MoveTorsoActionDescription([TorsoState.HIGH]),
        ParkArmsActionDescription(Arms.BOTH),
        NavigateActionDescription(
            PoseStamped.from_list([1.75, 1.75, 0], [0, 0, 0.5, 1], world.root)
        ),
        CloseActionDescription(world.get_body_by_name("handle_cab10_t"), [Arms.LEFT]),
    )
    with simulated_robot:
        plan.perform()
    assert world.state[
        world.get_degree_of_freedom_by_name("cabinet10_drawer_top_joint").id
    ].position == pytest.approx(0, abs=0.1)


def test_transport(mutable_model_world):
    world, robot_view, context = mutable_model_world
    description = TransportActionDescription(
        world.get_body_by_name("milk.stl"),
        [PoseStamped.from_list([2.9, 2.2, 0.95], [0.0, 0.0, 1.0, 0.0], world.root)],
        [Arms.LEFT],
    )
    plan = SequentialPlan(
        context, MoveTorsoActionDescription([TorsoState.HIGH]), description
    )
    with simulated_robot:
        plan.perform()
    milk_position = world.get_body_by_name("milk.stl").global_pose.to_np()[:3, 3]
    dist = np.linalg.norm(milk_position - np.array([2.9, 2.2, 0.95]))
    assert dist <= 0.01


def test_grasping(immutable_model_world):
    world, robot_view, context = immutable_model_world
    description = GraspingActionDescription(
        world.get_body_by_name("milk.stl"), [Arms.RIGHT]
    )
    plan = SequentialPlan(
        context,
        NavigateActionDescription(
            PoseStamped.from_list([1.8, 1.8, 0], frame=world.root), True
        ),
        description,
    )
    with simulated_robot:
        plan.perform()
    dist = np.linalg.norm(world.get_body_by_name("milk.stl").global_pose.to_np()[3, :3])
    assert dist < 0.01


def test_facing(immutable_model_world):
    world, robot_view, context = immutable_model_world
    with simulated_robot:
        milk_pose = PoseStamped.from_spatial_type(
            world.get_body_by_name("milk.stl").global_pose
        )
        plan = SequentialPlan(context, FaceAtActionDescription(milk_pose, True))
        plan.perform()
        milk_in_robot_frame = world.transform(
            world.get_body_by_name("milk.stl").global_pose,
            robot_view.root,
        )
        milk_in_robot_frame = PoseStamped.from_spatial_type(milk_in_robot_frame)
        assert milk_in_robot_frame.position.y == pytest.approx(0.0, abs=0.01)


def test_move_tcp_waypoints(immutable_model_world):
    world, robot_view, context = immutable_model_world
    with world.modify_world():
        world.state[
            world.get_degree_of_freedom_by_name("torso_lift_joint").id
        ].position = 0.1
    world.notify_state_change()

    gripper_pose = PoseStamped.from_spatial_type(
        world.get_body_by_name("r_gripper_tool_frame").global_pose
    )
    path = []
    for i in range(1, 5):
        new_pose = deepcopy(gripper_pose)
        new_pose.position.z -= 0.05 * i
        path.append(new_pose)
    description = MoveTCPWaypointsMotion(path, Arms.RIGHT)
    plan = SequentialPlan(context, description)

    me = MotionExecutor([description._motion_chart], world)
    me.construct_msc()
    with simulated_robot:
        me.execute()
    gripper_position = PoseStamped.from_spatial_type(
        world.get_body_by_name("r_gripper_tool_frame").global_pose
    )
    assert path[-1].position.x == pytest.approx(gripper_position.position.x, abs=0.1)
    assert path[-1].position.y == pytest.approx(gripper_position.position.y, abs=0.1)
    assert path[-1].position.z == pytest.approx(gripper_position.position.z, abs=0.1)

    assert path[-1].orientation.x == pytest.approx(
        gripper_position.orientation.x, abs=0.1
    )
    assert path[-1].orientation.y == pytest.approx(
        gripper_position.orientation.y, abs=0.1
    )
    assert path[-1].orientation.z == pytest.approx(
        gripper_position.orientation.z, abs=0.1
    )
    assert path[-1].orientation.w == pytest.approx(
        gripper_position.orientation.w, abs=0.1
    )


@pytest.mark.skip
def test_search_action(self):
    plan = SequentialPlan(
        self.context,
        MoveTorsoActionDescription([TorsoState.HIGH]),
        SearchActionDescription(
            PoseStamped.from_list([2, 2, 1], self.world.root), Milk
        ),
    )
    with simulated_robot:
        milk = plan.perform()
    self.assertTrue(milk)
    self.assertEqual(milk.obj_type, Milk)
    self.assertEqual(self.milk.pose, milk.pose)
