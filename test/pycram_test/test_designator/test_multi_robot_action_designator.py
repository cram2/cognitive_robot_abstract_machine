from copy import deepcopy

import numpy as np
import pytest
from typing_extensions import Tuple, Generator

from giskardpy.utils.utils_for_tests import compare_axis_angle
from pycram.datastructures.dataclasses import Context
from pycram.datastructures.enums import (
    Arms,
    ApproachDirection,
    VerticalAlignment,
    DetectionTechnique,
)
from pycram.datastructures.grasp import GraspDescription
from pycram.datastructures.pose import PoseStamped
from pycram.language import SequentialPlan
from pycram.process_module import simulated_robot
from pycram.robot_description import ViewManager
from pycram.robot_plans import (
    MoveTorsoAction,
    MoveTorsoActionDescription,
    NavigateActionDescription,
    SetGripperActionDescription,
    PickUpActionDescription,
    ParkArmsActionDescription,
    ReachActionDescription,
    PlaceActionDescription,
    LookAtActionDescription,
    DetectActionDescription,
    OpenActionDescription,
    CloseActionDescription,
    FaceAtActionDescription,
    GraspingActionDescription,
)
from semantic_digital_twin.datastructures.definitions import (
    TorsoState,
    GripperState,
    JointStateType,
)
from semantic_digital_twin.robots.abstract_robot import AbstractRobot
from semantic_digital_twin.robots.hsrb import HSRB
from semantic_digital_twin.robots.stretch import Stretch
from semantic_digital_twin.robots.tiago import Tiago
from semantic_digital_twin.semantic_annotations.semantic_annotations import Milk
from semantic_digital_twin.spatial_types import HomogeneousTransformationMatrix
from semantic_digital_twin.world import World


@pytest.fixture(scope="session", params=["hsrb", "stretch", "tiago"])
def setup_multi_robot_apartment(
    request, hsr_world_setup, stretch_world, tiago_world, apartment_world_setup
):
    apartment_copy = deepcopy(apartment_world_setup)

    if request.param == "hsrb":
        hsr_copy = deepcopy(hsr_world_setup)
        apartment_copy.merge_world_at_pose(
            hsr_copy, HomogeneousTransformationMatrix.from_xyz_rpy(1.5, 2, 0)
        )
        view = HSRB.from_world(apartment_copy)
        return apartment_copy, view
    elif request.param == "stretch":
        stretch_copy = deepcopy(stretch_world)
        apartment_copy.merge_world_at_pose(
            stretch_copy, HomogeneousTransformationMatrix.from_xyz_rpy(1.5, 2, 0)
        )
        view = Stretch.from_world(apartment_copy)
        return apartment_copy, view

    elif request.param == "tiago":
        tiago_copy = deepcopy(tiago_world)
        apartment_copy.merge_world_at_pose(
            tiago_copy, HomogeneousTransformationMatrix.from_xyz_rpy(1.5, 2, 0)
        )
        view = Tiago.from_world(apartment_copy)
        return apartment_copy, view


@pytest.fixture
def immutable_multiple_robot_apartment(
    setup_multi_robot_apartment,
) -> Generator[Tuple[World, AbstractRobot, Context]]:
    world, view = setup_multi_robot_apartment
    state = deepcopy(world.state.data)
    yield world, view, Context(world, view)
    world.state.data = state


@pytest.fixture
def mutable_multiple_robot_apartment(setup_multi_robot_apartment):
    world, view = setup_multi_robot_apartment
    copy_world = deepcopy(world)
    copy_view = view.from_world(copy_world)
    return copy_world, copy_view, Context(world, view)


def test_move_torso_multi(immutable_multiple_robot_apartment):
    world, view, context = immutable_multiple_robot_apartment
    plan = SequentialPlan(context, MoveTorsoActionDescription(TorsoState.HIGH))

    with simulated_robot:
        plan.perform()

    joint_state = view.torso.get_joint_state_by_type(TorsoState.HIGH)

    for connection, target in joint_state.items():
        assert connection.position == pytest.approx(target, abs=0.01)


def test_navigate_multi(immutable_multiple_robot_apartment):
    world, view, context = immutable_multiple_robot_apartment
    plan = SequentialPlan(
        context,
        NavigateActionDescription(PoseStamped.from_list([1, 2, 0], frame=world.root)),
    )

    with simulated_robot:
        plan.perform()

    robot_base_pose = view.root.global_pose
    robot_base_position = robot_base_pose.to_position().to_np()
    robot_base_orientation = robot_base_pose.to_quaternion().to_np()

    assert robot_base_position[:3] == pytest.approx([1, 2, 0], abs=0.01)
    assert robot_base_orientation == pytest.approx([0, 0, 0, 1], abs=0.01)


def test_move_gripper_multi(immutable_multiple_robot_apartment):
    world, view, context = immutable_multiple_robot_apartment

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


def test_park_arms_multi(immutable_multiple_robot_apartment):
    world, robot_view, context = immutable_multiple_robot_apartment
    description = ParkArmsActionDescription([Arms.BOTH])
    plan = SequentialPlan(context, description)
    assert description.resolve().arm == Arms.BOTH
    with simulated_robot:
        plan.perform()
    joint_state_left = robot_view.left_arm.get_joint_state_by_type(JointStateType.PARK)
    joint_state_right = robot_view.right_arm.get_joint_state_by_type(
        JointStateType.PARK
    )
    for connection, value in joint_state_left.items():
        compare_axis_angle(
            connection.position,
            np.array([1, 0, 0]),
            value,
            np.array([1, 0, 0]),
            decimal=1,
        )
    for connection, value in joint_state_right.items():
        compare_axis_angle(
            connection.position,
            np.array([1, 0, 0]),
            value,
            np.array([1, 0, 0]),
            decimal=1,
        )


def test_reach_action_multi(immutable_multiple_robot_apartment):
    world, view, context = immutable_multiple_robot_apartment
    left_arm = ViewManager.get_arm_view(Arms.LEFT, view)
    grasp_description = GraspDescription(
        ApproachDirection.FRONT,
        VerticalAlignment.NoAlignment,
        left_arm.manipulator,
    )

    milk_body = world.get_body_by_name("milk.stl")
    milk_body.parent_connection.position = HomogeneousTransformationMatrix.from_xyz_rpy(
        4, 3, 1
    )
    view.root.parent_connection.position = HomogeneousTransformationMatrix.from_xyz_rpy(
        3.5, 2.7, 0
    )

    plan = SequentialPlan(
        context,
        ReachActionDescription(
            target_pose=PoseStamped.from_list([4, 3, 1], frame=world.root),
            object_designator=milk_body,
            arm=Arms.LEFT,
            grasp_description=grasp_description,
        ),
    )

    with simulated_robot:
        plan.perform()

    manipulator_pose = left_arm.manipulator.tool_frame.global_pose
    manipulator_position = manipulator_pose.to_position().to_np()
    manipulator_orientation = manipulator_pose.to_quaternion().to_np()

    assert manipulator_position[:3] == pytest.approx([4, 3, 1], abs=0.01)
    assert manipulator_orientation == pytest.approx([0, 0, 0, 1], abs=0.01)


def test_grasping(immutable_multiple_robot_apartment):
    world, robot_view, context = immutable_multiple_robot_apartment
    description = GraspingActionDescription(
        world.get_body_by_name("milk.stl"), [Arms.RIGHT]
    )

    milk_body = world.get_body_by_name("milk.stl")
    milk_body.parent_connection.position = HomogeneousTransformationMatrix.from_xyz_rpy(
        4, 3, 1
    )
    robot_view.root.parent_connection.position = (
        HomogeneousTransformationMatrix.from_xyz_rpy(3.5, 2.7, 0)
    )

    plan = SequentialPlan(
        context,
        description,
    )
    with simulated_robot:
        plan.perform()
    dist = np.linalg.norm(world.get_body_by_name("milk.stl").global_pose.to_np()[3, :3])
    assert dist < 0.01


def test_pick_up_multi(mutable_multiple_robot_apartment):
    world, view, context = mutable_multiple_robot_apartment

    left_arm = ViewManager.get_arm_view(Arms.LEFT, view)
    grasp_description = GraspDescription(
        ApproachDirection.FRONT,
        VerticalAlignment.NoAlignment,
        left_arm.manipulator,
    )

    milk_body = world.get_body_by_name("milk.stl")
    milk_body.parent_connection.position = HomogeneousTransformationMatrix.from_xyz_rpy(
        4, 3, 1
    )
    view.root.parent_connection.position = HomogeneousTransformationMatrix.from_xyz_rpy(
        3.5, 2.7, 0
    )

    plan = SequentialPlan(
        context,
        PickUpActionDescription(
            world.get_body_by_name("milk.stl"), Arms.LEFT, grasp_description
        ),
    )

    with simulated_robot:
        plan.perform()

    assert (
        world.get_connection(
            left_arm.manipulator.tool_frame,
            world.get_body_by_name("milk.stl"),
        )
        is not None
    )


def test_place_multi(mutable_multiple_robot_apartment):
    world, view, context = mutable_multiple_robot_apartment

    left_arm = ViewManager.get_arm_view(Arms.LEFT, view)
    grasp_description = GraspDescription(
        ApproachDirection.FRONT,
        VerticalAlignment.NoAlignment,
        left_arm.manipulator,
    )

    milk_body = world.get_body_by_name("milk.stl")
    milk_body.parent_connection.position = HomogeneousTransformationMatrix.from_xyz_rpy(
        4, 3, 1
    )
    view.root.parent_connection.position = HomogeneousTransformationMatrix.from_xyz_rpy(
        3.5, 2.7, 0
    )

    plan = SequentialPlan(
        context,
        PickUpActionDescription(
            world.get_body_by_name("milk.stl"), Arms.LEFT, grasp_description
        ),
        PlaceActionDescription(
            world.get_body_by_name("milk.stl"),
            PoseStamped.from_list([4, 3.1, 1], frame=world.root),
            Arms.LEFT,
        ),
    )

    assert (
        world.get_connection(
            left_arm.manipulator.tool_frame,
            world.get_body_by_name("milk.stl"),
        )
        is None
    )

    milk_position = milk_body.global_pose.to_position().to_np()

    assert milk_position[:3] == pytest.approx([4, 3.1, 1], abs=0.01)


def test_look_at(immutable_multiple_robot_apartment):
    world, robot_view, context = immutable_multiple_robot_apartment

    description = LookAtActionDescription(
        [PoseStamped.from_list([5, 0, 1], frame=world.root)]
    )
    assert description.resolve().target == PoseStamped.from_list(
        [5, 0, 1], frame=world.root
    )

    plan = SequentialPlan(context, description)
    with simulated_robot:
        plan.perform()


def test_detect(immutable_multiple_robot_apartment):
    world, robot_view, context = immutable_multiple_robot_apartment
    milk_body = world.get_body_by_name("milk.stl")
    with world.modify_world():
        world.add_semantic_annotation(Milk(root=milk_body))
    robot_view.root.parent_connection.position = (
        HomogeneousTransformationMatrix.from_xyz_rpy(1.5, -2, 0)
    )
    milk_body.parent_connection.origin = HomogeneousTransformationMatrix.from_xyz_rpy(
        2.5, -2, 1.2, reference_frame=world.root
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


def test_open(immutable_multiple_robot_apartment):
    world, robot_view, context = immutable_multiple_robot_apartment

    plan = SequentialPlan(
        context,
        MoveTorsoActionDescription([TorsoState.HIGH]),
        ParkArmsActionDescription(Arms.BOTH),
        NavigateActionDescription(
            PoseStamped.from_list([1.75, 1.75, 0], [0, 0, 0.5, 1], world.root)
        ),
        OpenActionDescription(world.get_body_by_name("handle_cab10_m"), [Arms.LEFT]),
    )
    with simulated_robot:
        plan.perform()
    assert world.get_connection_by_name(
        "handle_cab10_mid_joint"
    ).position == pytest.approx(0.45, abs=0.1)


def test_close(immutable_multiple_robot_apartment):
    world, robot_view, context = immutable_multiple_robot_apartment

    world.state[
        world.get_degree_of_freedom_by_name("handle_cab10_mid_joint").id
    ].position = 0.45
    world.notify_state_change()
    plan = SequentialPlan(
        context,
        MoveTorsoActionDescription([TorsoState.HIGH]),
        ParkArmsActionDescription(Arms.BOTH),
        NavigateActionDescription(
            PoseStamped.from_list([1.75, 1.75, 0], [0, 0, 0.5, 1], world.root)
        ),
        CloseActionDescription(world.get_body_by_name("handle_cab10_m"), [Arms.LEFT]),
    )
    with simulated_robot:
        plan.perform()
    assert world.state[
        world.get_degree_of_freedom_by_name("handle_cab10_mid_joint").id
    ].position == pytest.approx(0, abs=0.1)


def test_facing(immutable_multiple_robot_apartment):
    world, robot_view, context = immutable_multiple_robot_apartment
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
