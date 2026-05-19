from copy import deepcopy

import numpy as np
import pytest
import rclpy
from rustworkx import NoEdgeBetweenNodes

from giskardpy.utils.utils_for_tests import compare_axis_angle, compare_orientations
from pycram.datastructures.dataclasses import Context
from pycram.datastructures.enums import Arms, ApproachDirection, VerticalAlignment
from pycram.datastructures.grasp import GraspDescription
from pycram.datastructures.trajectory import PoseTrajectory

from pycram.motion_executor import simulated_robot
from pycram.plans.factories import execute_single, sequential
from pycram.robot_plans.actions.core.pick_up import (
    ReachAction,
    GraspingAction,
    PickUpAction,
)
from pycram.robot_plans.actions.core.placing import PlaceAction
from pycram.robot_plans.actions.core.robot_body import (
    ParkArmsAction,
    SetGripperAction,
    FollowToolCenterPointPathAction,
)
from pycram.testing import _make_sine_scan_poses
from pycram.view_manager import ViewManager

from semantic_digital_twin.adapters.ros.visualization.viz_marker import (
    VizMarkerPublisher,
)
from semantic_digital_twin.datastructures.definitions import (
    JointStateType,
    GripperState,
    StaticJointState,
)
from semantic_digital_twin.datastructures.prefixed_name import PrefixedName
from semantic_digital_twin.robots.tracy import Tracy
from semantic_digital_twin.spatial_types import HomogeneousTransformationMatrix, Point3
from semantic_digital_twin.spatial_types.spatial_types import Pose
from semantic_digital_twin.world_description.connections import Connection6DoF
from semantic_digital_twin.world_description.geometry import Box, Scale
from semantic_digital_twin.world_description.shape_collection import ShapeCollection
from semantic_digital_twin.world_description.world_entity import Body


@pytest.fixture(scope="session")
def tracy_block_world(tracy_world):
    box1 = Body(
        name=PrefixedName("box1"),
        collision=ShapeCollection([Box(scale=Scale(0.1, 0.1, 0.1))]),
        visual=ShapeCollection([Box(scale=Scale(0.1, 0.1, 0.1))]),
    )

    box2 = Body(
        name=PrefixedName("box2"),
        collision=ShapeCollection([Box(scale=Scale(0.1, 0.1, 0.1))]),
        visual=ShapeCollection([Box(scale=Scale(0.1, 0.1, 0.1))]),
    )

    with tracy_world.modify_world():
        box1_connection = Connection6DoF.create_with_dofs(
            tracy_world,
            tracy_world.root,
            box1,
            PrefixedName("box1_connection"),
            HomogeneousTransformationMatrix.from_xyz_rpy(0.8, 0.5, 0.93),
        )

        box2_connection = Connection6DoF.create_with_dofs(
            tracy_world,
            tracy_world.root,
            box2,
            PrefixedName("box2_connection"),
            HomogeneousTransformationMatrix.from_xyz_rpy(0.8, -0.5, 0.93),
        )
        tracy_world.add_connection(box1_connection)
        tracy_world.add_connection(box2_connection)
    return tracy_world


@pytest.fixture
def immutable_tracy_block_world(tracy_block_world):
    state = deepcopy(tracy_block_world.state._data)
    view = tracy_block_world.get_semantic_annotations_by_type(Tracy)[0]
    yield tracy_block_world, view, Context(tracy_block_world, view)
    tracy_block_world.state._data[:] = state
    tracy_block_world.notify_state_change()


@pytest.fixture
def mutable_tracy_block_world(tracy_block_world):
    copy_world = deepcopy(tracy_block_world)
    copy_view = copy_world.get_semantic_annotations_by_type(Tracy)[0]
    return copy_world, copy_view, Context(copy_world, copy_view)


def test_park_arms_tracy(immutable_tracy_block_world):
    world, view, context = immutable_tracy_block_world

    description = ParkArmsAction(Arms.BOTH)
    plan = execute_single(description, context=context).plan
    with simulated_robot:
        plan.perform()

    joints = []
    states = []
    for arm in view.arms:
        joint_state = arm.get_joint_state_by_type(StaticJointState.PARK)
        joints.extend(joint_state.connections)
        states.extend(joint_state.target_values)
    for connection, value in zip(joints, states):
        compare_axis_angle(
            connection.position,
            np.array([1, 0, 0]),
            value,
            np.array([1, 0, 0]),
            decimal=1,
        )


def test_reach_action_multi(immutable_tracy_block_world):
    world, view, context = immutable_tracy_block_world
    left_arm = ViewManager.get_arm_view(Arms.LEFT, view)

    grasp_description = GraspDescription(
        ApproachDirection.FRONT,
        VerticalAlignment.TOP,
        left_arm.manipulator,
    )
    box_body = world.get_body_by_name("box1")

    plan = sequential(
        [
            ParkArmsAction(Arms.BOTH),
            ReachAction(
                target_pose=Pose(
                    Point3.from_iterable([0.8, 0.5, 0.93]), reference_frame=world.root
                ),
                object_designator=box_body,
                arm=Arms.LEFT,
                grasp_description=grasp_description,
            ),
        ],
        context=context,
    ).plan

    with simulated_robot:
        plan.perform()

    manipulator_pose = left_arm.manipulator.tool_frame.global_transform
    manipulator_position = manipulator_pose.to_position().to_np()
    manipulator_orientation = manipulator_pose.to_quaternion().to_np()

    target_orientation = grasp_description.grasp_orientation()

    assert manipulator_position[:3] == pytest.approx([0.8, 0.5, 0.93], abs=0.01)
    compare_orientations(manipulator_orientation, target_orientation, decimal=2)


def test_move_gripper_multi(immutable_tracy_block_world):
    world, view, context = immutable_tracy_block_world

    plan = execute_single(
        SetGripperAction(Arms.LEFT, GripperState.OPEN), context=context
    ).plan

    with simulated_robot:
        plan.perform()

    arm = view.arms[0]
    open_state = arm.manipulator.get_joint_state_by_type(GripperState.OPEN)
    close_state = arm.manipulator.get_joint_state_by_type(GripperState.CLOSE)

    for connection, target in open_state.items():
        assert connection.position == pytest.approx(target, abs=0.01)

    plan = execute_single(
        SetGripperAction(Arms.LEFT, GripperState.CLOSE), context=context
    ).plan

    with simulated_robot:
        plan.perform()

    for connection, target in close_state.items():
        assert connection.position == pytest.approx(target, abs=0.01)


def test_grasping(immutable_tracy_block_world):
    world, robot_view, context = immutable_tracy_block_world
    left_arm = ViewManager.get_arm_view(Arms.LEFT, robot_view)

    grasp_description = GraspDescription(
        ApproachDirection.FRONT,
        VerticalAlignment.TOP,
        left_arm.manipulator,
    )
    description = GraspingAction(
        world.get_body_by_name("box1"), Arms.LEFT, grasp_description
    )
    plan = sequential(
        [ParkArmsAction(Arms.BOTH), description],
        context=context,
    ).plan
    with simulated_robot:
        plan.perform()
    dist = np.linalg.norm(
        world.get_body_by_name("box1").global_transform.to_np()[3, :3]
    )
    assert dist < 0.01


def test_pick_up_tracy(mutable_tracy_block_world):
    world, view, context = mutable_tracy_block_world

    left_arm = ViewManager.get_arm_view(Arms.LEFT, view)
    grasp_description = GraspDescription(
        ApproachDirection.FRONT,
        VerticalAlignment.TOP,
        left_arm.manipulator,
    )
    plan = sequential(
        [
            ParkArmsAction(Arms.BOTH),
            PickUpAction(world.get_body_by_name("box1"), Arms.LEFT, grasp_description),
        ],
        context=context,
    ).plan

    with simulated_robot:
        plan.perform()

    assert (
        world.get_connection(
            left_arm.manipulator.tool_frame,
            world.get_body_by_name("box1"),
        )
        is not None
    )

    plan.validate()


def test_place_tracy(mutable_tracy_block_world):
    world, view, context = mutable_tracy_block_world

    left_arm = ViewManager.get_arm_view(Arms.LEFT, view)
    grasp_description = GraspDescription(
        ApproachDirection.FRONT,
        VerticalAlignment.TOP,
        left_arm.manipulator,
    )

    plan = sequential(
        [
            ParkArmsAction(Arms.BOTH),
            PickUpAction(world.get_body_by_name("box1"), Arms.LEFT, grasp_description),
            PlaceAction(
                world.get_body_by_name("box1"),
                Pose(Point3.from_iterable([0.9, 0, 0.93]), reference_frame=world.root),
                Arms.LEFT,
            ),
        ],
        context=context,
    ).plan

    with simulated_robot:
        plan.perform()

    with pytest.raises(NoEdgeBetweenNodes):
        world.get_connection(
            left_arm.manipulator.tool_frame,
            world.get_body_by_name("box1"),
        )
    box_body = world.get_body_by_name("box1")
    milk_position = box_body.global_transform.to_position().to_np()

    assert milk_position[:3] == pytest.approx([0.9, 0, 0.93], abs=0.01)
    plan.validate()


def test_move_tcp_follows_sine_waypoints(immutable_tracy_block_world):
    world, view, context = immutable_tracy_block_world
    right_arm = ViewManager.get_arm_view(Arms.RIGHT, view)
    anchor = Pose(Point3.from_iterable([0.85, -0.25, 0.95]), reference_frame=world.root)
    anchor_T = anchor.to_homogeneous_matrix()
    offset_T = HomogeneousTransformationMatrix.from_xyz_axis_angle(
        z=-0.03,
        axis=(0, 1, 0),
        angle=np.pi / 2,
        reference_frame=world.root,
    )
    target_pose = (anchor_T @ offset_T).to_pose()
    waypoints = PoseTrajectory(_make_sine_scan_poses(target_pose, lane_axis="z"))

    plan = execute_single(
        FollowToolCenterPointPathAction(target_locations=waypoints, arm=Arms.RIGHT),
        context=context,
    )
    with simulated_robot:
        plan.perform()

    tip_pose = right_arm.manipulator.tool_frame.global_transform
    expected = waypoints.poses[-1]

    assert np.allclose(tip_pose.to_position(), expected.to_position(), atol=0.01)
    assert np.allclose(tip_pose.to_quaternion(), expected.to_quaternion(), atol=0.01)
