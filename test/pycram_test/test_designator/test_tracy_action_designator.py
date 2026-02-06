from copy import deepcopy

import numpy as np
import pytest
import rclpy
from rustworkx import NoEdgeBetweenNodes

from giskardpy.utils.utils_for_tests import compare_axis_angle, compare_orientations
from pycram.datastructures.dataclasses import Context
from pycram.datastructures.enums import Arms, ApproachDirection, VerticalAlignment
from pycram.datastructures.grasp import GraspDescription
from pycram.datastructures.pose import PoseStamped
from pycram.language import SequentialPlan
from pycram.process_module import simulated_robot
from pycram.robot_description import ViewManager
from pycram.robot_plans import (
    ParkArmsActionDescription,
    ReachActionDescription,
    GraspingActionDescription,
    PickUpActionDescription,
    PlaceActionDescription,
    SetGripperActionDescription,
)
from semantic_digital_twin.adapters.ros.visualization.viz_marker import (
    VizMarkerPublisher,
)
from semantic_digital_twin.datastructures.definitions import (
    JointStateType,
    GripperState,
)
from semantic_digital_twin.datastructures.prefixed_name import PrefixedName
from semantic_digital_twin.robots.tracy import Tracy
from semantic_digital_twin.spatial_types import HomogeneousTransformationMatrix
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
    state = deepcopy(tracy_block_world.state.data)
    view = tracy_block_world.get_semantic_annotations_by_type(Tracy)[0]
    yield tracy_block_world, view, Context(tracy_block_world, view)
    tracy_block_world.state.data = state


@pytest.fixture
def mutable_tracy_block_world(tracy_block_world):
    copy_world = deepcopy(tracy_block_world)
    copy_view = Tracy.from_world(copy_world)
    return copy_world, copy_view, Context(copy_world, copy_view)


def test_park_arms_tracy(immutable_tracy_block_world):
    world, view, context = immutable_tracy_block_world

    description = ParkArmsActionDescription([Arms.BOTH])
    plan = SequentialPlan(context, description)
    assert description.resolve().arm == Arms.BOTH
    with simulated_robot:
        plan.perform()

    joints = []
    states = []
    for arm in view.arms:
        joint_state = arm.get_joint_state_by_type(JointStateType.PARK)
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

    plan = SequentialPlan(
        context,
        ParkArmsActionDescription(Arms.BOTH),
        ReachActionDescription(
            target_pose=PoseStamped.from_list([0.8, 0.5, 0.93], frame=world.root),
            object_designator=box_body,
            arm=Arms.LEFT,
            grasp_description=grasp_description,
        ),
    )

    with simulated_robot:
        plan.perform()

    manipulator_pose = left_arm.manipulator.tool_frame.global_pose
    manipulator_position = manipulator_pose.to_position().to_np()
    manipulator_orientation = manipulator_pose.to_quaternion().to_np()

    target_orientation = grasp_description.grasp_orientation()

    assert manipulator_position[:3] == pytest.approx([0.8, 0.5, 0.93], abs=0.01)
    compare_orientations(manipulator_orientation, target_orientation, decimal=2)


def test_move_gripper_multi(immutable_tracy_block_world):
    world, view, context = immutable_tracy_block_world

    plan = SequentialPlan(
        context, SetGripperActionDescription(Arms.LEFT, GripperState.OPEN)
    )

    with simulated_robot:
        plan.perform()

    arm = view.arms[0]
    open_state = arm.manipulator.get_joint_state_by_type(GripperState.OPEN)
    close_state = arm.manipulator.get_joint_state_by_type(GripperState.CLOSE)

    for connection, target in open_state.items():
        assert connection.position == pytest.approx(target, abs=0.01)

    plan = SequentialPlan(
        context, SetGripperActionDescription(Arms.LEFT, GripperState.CLOSE)
    )

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
    description = GraspingActionDescription(
        world.get_body_by_name("box1"), [Arms.LEFT], grasp_description
    )
    plan = SequentialPlan(
        context,
        ParkArmsActionDescription(Arms.BOTH),
        description,
    )
    with simulated_robot:
        plan.perform()
    dist = np.linalg.norm(world.get_body_by_name("box1").global_pose.to_np()[3, :3])
    assert dist < 0.01


def test_pick_up_multi(mutable_tracy_block_world):
    world, view, context = mutable_tracy_block_world

    left_arm = ViewManager.get_arm_view(Arms.LEFT, view)
    grasp_description = GraspDescription(
        ApproachDirection.FRONT,
        VerticalAlignment.TOP,
        left_arm.manipulator,
    )
    plan = SequentialPlan(
        context,
        ParkArmsActionDescription(Arms.BOTH),
        PickUpActionDescription(
            world.get_body_by_name("box1"), Arms.LEFT, grasp_description
        ),
    )

    with simulated_robot:
        plan.perform()

    assert (
        world.get_connection(
            left_arm.manipulator.tool_frame,
            world.get_body_by_name("box1"),
        )
        is not None
    )

    assert len(plan.nodes) == len(plan.all_nodes)
    assert len(plan.edges) == len(plan.all_nodes) - 1


def test_place_multi(mutable_tracy_block_world):
    world, view, context = mutable_tracy_block_world

    left_arm = ViewManager.get_arm_view(Arms.LEFT, view)
    grasp_description = GraspDescription(
        ApproachDirection.FRONT,
        VerticalAlignment.NoAlignment,
        left_arm.manipulator,
    )

    plan = SequentialPlan(
        context,
        ParkArmsActionDescription(Arms.BOTH),
        PickUpActionDescription(
            world.get_body_by_name("box1"), Arms.LEFT, grasp_description
        ),
        PlaceActionDescription(
            world.get_body_by_name("box1"),
            PoseStamped.from_list([0.9, 0, 0.93], frame=world.root),
            Arms.LEFT,
        ),
    )

    with simulated_robot:
        plan.perform()

    with pytest.raises(NoEdgeBetweenNodes):
        world.get_connection(
            left_arm.manipulator.tool_frame,
            world.get_body_by_name("box1"),
        )
    box_body = world.get_body_by_name("box1")
    milk_position = box_body.global_pose.to_position().to_np()

    assert milk_position[:3] == pytest.approx([0.9, 0, 0.93], abs=0.01)

    assert len(plan.nodes) == len(plan.all_nodes)
    assert len(plan.edges) == len(plan.all_nodes) - 1
