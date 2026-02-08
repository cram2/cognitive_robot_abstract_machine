import time
from copy import deepcopy

import numpy as np
import pytest
import rclpy
from rustworkx.rustworkx import NoEdgeBetweenNodes
from typing_extensions import Tuple, Generator

from giskardpy.utils.utils_for_tests import compare_axis_angle, compare_orientations
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
    TransportActionDescription,
)
from semantic_digital_twin.adapters.ros.pose_publisher import PosePublisher
from semantic_digital_twin.adapters.ros.tf_publisher import TFPublisher
from semantic_digital_twin.adapters.ros.visualization.viz_marker import (
    VizMarkerPublisher,
)
from semantic_digital_twin.datastructures.definitions import (
    TorsoState,
    GripperState,
    JointStateType,
)
from semantic_digital_twin.robots.abstract_robot import AbstractRobot
from semantic_digital_twin.robots.hsrb import HSRB
from semantic_digital_twin.robots.pr2 import PR2
from semantic_digital_twin.robots.stretch import Stretch
from semantic_digital_twin.robots.tiago import Tiago
from semantic_digital_twin.semantic_annotations.semantic_annotations import Milk
from semantic_digital_twin.spatial_types import HomogeneousTransformationMatrix
from semantic_digital_twin.world import World


@pytest.fixture(scope="session", params=["hsrb", "stretch", "tiago", "pr2"])
def setup_multi_robot_apartment(
    request,
    hsr_world_setup,
    stretch_world,
    tiago_world,
    pr2_world_setup,
    apartment_world_setup,
):
    apartment_copy = deepcopy(apartment_world_setup)

    if request.param == "hsrb":
        hsr_copy = deepcopy(hsr_world_setup)
        apartment_copy.merge_world(hsr_copy)
        view = HSRB.from_world(apartment_copy)
        view.root.parent_connection.origin = (
            HomogeneousTransformationMatrix.from_xyz_rpy(1.5, 2, 0)
        )
        return apartment_copy, view
    elif request.param == "stretch":
        stretch_copy = deepcopy(stretch_world)
        apartment_copy.merge_world(
            stretch_copy,
        )
        view = Stretch.from_world(apartment_copy)
        view.root.parent_connection.origin = (
            HomogeneousTransformationMatrix.from_xyz_rpy(1.5, 2, 0)
        )
        return apartment_copy, view

    elif request.param == "tiago":
        tiago_copy = deepcopy(tiago_world)
        apartment_copy.merge_world(
            tiago_copy,
        )
        view = Tiago.from_world(apartment_copy)
        view.root.parent_connection.origin = (
            HomogeneousTransformationMatrix.from_xyz_rpy(1.5, 2, 0)
        )
        return apartment_copy, view

    elif request.param == "pr2":
        pr2_copy = deepcopy(pr2_world_setup)
        apartment_copy.merge_world(
            pr2_copy,
        )
        view = PR2.from_world(apartment_copy)
        view.root.parent_connection.origin = (
            HomogeneousTransformationMatrix.from_xyz_rpy(1.5, 2, 0)
        )
        return apartment_copy, view


@pytest.fixture
def immutable_multiple_robot_apartment(
    setup_multi_robot_apartment,
) -> Generator[Tuple[World, AbstractRobot, Context]]:
    world, view = setup_multi_robot_apartment
    state = deepcopy(world.state.data)
    yield world, view, Context(world, view)
    world.state.data = state
    world.notify_state_change()


@pytest.fixture
def mutable_multiple_robot_apartment(setup_multi_robot_apartment):
    world, view = setup_multi_robot_apartment
    copy_world = deepcopy(world)
    copy_view = view.from_world(copy_world)
    return copy_world, copy_view, Context(copy_world, copy_view)


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

    joints = []
    states = []
    for arm in robot_view.arms:
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


def test_reach_action_multi(immutable_multiple_robot_apartment):
    world, view, context = immutable_multiple_robot_apartment
    left_arm = ViewManager.get_arm_view(Arms.LEFT, view)

    grasp_description = GraspDescription(
        ApproachDirection.FRONT,
        VerticalAlignment.NoAlignment,
        left_arm.manipulator,
    )
    milk_body = world.get_body_by_name("milk.stl")
    milk_body.parent_connection.origin = HomogeneousTransformationMatrix.from_xyz_rpy(
        1, -2, 0.8, reference_frame=world.root
    )
    view.root.parent_connection.origin = HomogeneousTransformationMatrix.from_xyz_rpy(
        0.3, -2.5, 0, reference_frame=world.root
    )
    world.notify_state_change()

    plan = SequentialPlan(
        context,
        ParkArmsActionDescription(Arms.BOTH),
        ReachActionDescription(
            target_pose=PoseStamped.from_list([1, -2, 0.8], frame=world.root),
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

    target_orientation = grasp_description.grasp_orientation()

    assert manipulator_position[:3] == pytest.approx([1, -2, 0.8], abs=0.01)
    compare_orientations(manipulator_orientation, target_orientation, decimal=2)


def test_grasping(immutable_multiple_robot_apartment):
    world, robot_view, context = immutable_multiple_robot_apartment
    left_arm = ViewManager.get_arm_view(Arms.LEFT, robot_view)

    grasp_description = GraspDescription(
        ApproachDirection.FRONT,
        VerticalAlignment.NoAlignment,
        left_arm.manipulator,
    )
    description = GraspingActionDescription(
        world.get_body_by_name("milk.stl"), [Arms.LEFT], grasp_description
    )

    milk_body = world.get_body_by_name("milk.stl")
    milk_body.parent_connection.origin = HomogeneousTransformationMatrix.from_xyz_rpy(
        1, -2, 0.8, reference_frame=world.root
    )
    robot_view.root.parent_connection.origin = (
        HomogeneousTransformationMatrix.from_xyz_rpy(
            0.3, -2.5, 0, reference_frame=world.root
        )
    )
    world.notify_state_change()

    plan = SequentialPlan(
        context,
        ParkArmsActionDescription(Arms.BOTH),
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
    milk_body.parent_connection.origin = HomogeneousTransformationMatrix.from_xyz_rpy(
        1, -2, 0.6, reference_frame=world.root
    )
    view.root.parent_connection.origin = HomogeneousTransformationMatrix.from_xyz_rpy(
        0.3, -2.5, 0, reference_frame=world.root
    )
    world.notify_state_change()

    plan = SequentialPlan(
        context,
        ParkArmsActionDescription(Arms.BOTH),
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

    assert len(plan.nodes) == len(plan.all_nodes)
    assert len(plan.edges) == len(plan.all_nodes) - 1


def test_place_multi(mutable_multiple_robot_apartment):
    world, view, context = mutable_multiple_robot_apartment

    left_arm = ViewManager.get_arm_view(Arms.LEFT, view)
    grasp_description = GraspDescription(
        ApproachDirection.FRONT,
        VerticalAlignment.NoAlignment,
        left_arm.manipulator,
    )

    milk_body = world.get_body_by_name("milk.stl")
    milk_body.parent_connection.origin = HomogeneousTransformationMatrix.from_xyz_rpy(
        1, -2, 0.6, reference_frame=world.root
    )
    view.root.parent_connection.origin = HomogeneousTransformationMatrix.from_xyz_rpy(
        0.3, -2.5, 0, reference_frame=world.root
    )
    world.notify_state_change()

    plan = SequentialPlan(
        context,
        ParkArmsActionDescription(Arms.BOTH),
        PickUpActionDescription(
            world.get_body_by_name("milk.stl"), Arms.LEFT, grasp_description
        ),
        PlaceActionDescription(
            world.get_body_by_name("milk.stl"),
            PoseStamped.from_list([1, -2.2, 0.6], frame=world.root),
            Arms.LEFT,
        ),
    )

    with simulated_robot:
        plan.perform()

    with pytest.raises(NoEdgeBetweenNodes):
        world.get_connection(
            left_arm.manipulator.tool_frame,
            world.get_body_by_name("milk.stl"),
        )

    milk_position = milk_body.global_pose.to_position().to_np()

    assert milk_position[:3] == pytest.approx([1, -2.2, 0.6], abs=0.01)

    assert len(plan.nodes) == len(plan.all_nodes)
    assert len(plan.edges) == len(plan.all_nodes) - 1


def test_look_at(immutable_multiple_robot_apartment):
    world, robot_view, context = immutable_multiple_robot_apartment
    description = LookAtActionDescription(
        [PoseStamped.from_list([3, 0, 1], frame=world.root)]
    )
    assert description.resolve().target == PoseStamped.from_list(
        [3, 0, 1], frame=world.root
    )

    plan = SequentialPlan(context, description)
    with simulated_robot:
        plan.perform()


def test_detect(immutable_multiple_robot_apartment):
    world, robot_view, context = immutable_multiple_robot_apartment
    milk_body = world.get_body_by_name("milk.stl")
    with world.modify_world():
        world.add_semantic_annotation(Milk(root=milk_body))

    robot_view.root.parent_connection.origin = (
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
        "cabinet10_drawer_middle_joint"
    ).position == pytest.approx(0.45, abs=0.1)


def test_close(immutable_multiple_robot_apartment):
    world, robot_view, context = immutable_multiple_robot_apartment

    world.get_connection_by_name("cabinet10_drawer_middle_joint").position = 0.45
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
    assert world.get_connection_by_name(
        "cabinet10_drawer_middle_joint"
    ).position == pytest.approx(0, abs=0.1)


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


def test_transport(mutable_multiple_robot_apartment):
    world, robot_view, context = mutable_multiple_robot_apartment

    description = TransportActionDescription(
        world.get_body_by_name("milk.stl"),
        [PoseStamped.from_list([3.1, 2.2, 0.95], [0.0, 0.0, 1.0, 0.0], world.root)],
        [Arms.RIGHT],
    )
    plan = SequentialPlan(
        context, MoveTorsoActionDescription([TorsoState.HIGH]), description
    )
    with simulated_robot:
        plan.perform()
    milk_position = world.get_body_by_name("milk.stl").global_pose.to_np()[:3, 3]
    dist = np.linalg.norm(milk_position - np.array([3.1, 2.2, 0.95]))
    assert dist <= 0.02

    assert len(plan.nodes) == len(plan.all_nodes)
    assert len(plan.edges) == len(plan.all_nodes) - 1
