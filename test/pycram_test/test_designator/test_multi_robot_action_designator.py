from copy import deepcopy

import numpy as np
import pytest
import rclpy
from rustworkx.rustworkx import NoEdgeBetweenNodes
from typing_extensions import Tuple, Generator

# The alternative mapping needs to be imported for the stretch to work properly
import pycram.alternative_motion_mappings.stretch_motion_mapping  # type: ignore
import pycram.alternative_motion_mappings.tiago_motion_mapping  # type: ignore
from giskardpy.utils.utils_for_tests import compare_axis_angle, compare_orientations
from pycram.datastructures.dataclasses import Context
from pycram.datastructures.enums import (
    Arms,
    AxisIdentifier,
    ApproachDirection,
    VerticalAlignment,
    DetectionTechnique,
)
from pycram.datastructures.grasp import GraspDescription
from pycram.datastructures.trajectory import PoseTrajectory

from pycram.motion_executor import simulated_robot
from pycram.plans.factories import sequential, execute_single
from pycram.robot_plans.actions.composite.facing import FaceAtAction
from pycram.robot_plans.actions.composite.transporting import TransportAction
from pycram.robot_plans.actions.core.container import OpenAction, CloseAction
from pycram.robot_plans.actions.core.misc import DetectAction
from pycram.robot_plans.actions.core.navigation import NavigateAction, LookAtAction
from pycram.robot_plans.actions.core.pick_up import (
    ReachAction,
    GraspingAction,
    PickUpAction,
)
from pycram.robot_plans.actions.core.placing import PlaceAction
from pycram.robot_plans.actions.core.robot_body import (
    MoveTorsoAction,
    SetGripperAction,
    ParkArmsAction,
    FollowToolCenterPointPathAction,
)

from pycram.view_manager import ViewManager
from semantic_digital_twin.adapters.ros.visualization.viz_marker import (
    VizMarkerPublisher,
)
from pycram.view_manager import ViewManager

from semantic_digital_twin.datastructures.definitions import (
    TorsoState,
    GripperState,
    JointStateType,
    StaticJointState,
)
from semantic_digital_twin.robots.abstract_robot import AbstractRobot
from semantic_digital_twin.robots.hsrb import HSRB
from semantic_digital_twin.robots.pr2 import PR2
from semantic_digital_twin.robots.stretch import Stretch
from semantic_digital_twin.robots.tiago import Tiago
from semantic_digital_twin.semantic_annotations.semantic_annotations import Milk
from semantic_digital_twin.spatial_types import (
    HomogeneousTransformationMatrix,
    Point3,
    Quaternion,
)
from semantic_digital_twin.spatial_types.spatial_types import Pose
from semantic_digital_twin.world import World


@pytest.fixture(scope="module", params=["hsrb", "stretch", "tiago", "pr2"])
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
        view = apartment_copy.get_semantic_annotations_by_type(HSRB)
        if not view:
            view = HSRB.from_world(apartment_copy)
        else:
            view = view[0]
        view.root.parent_connection.origin = (
            HomogeneousTransformationMatrix.from_xyz_rpy(1.5, 2, 0)
        )
        return apartment_copy, view
    elif request.param == "stretch":
        stretch_copy = deepcopy(stretch_world)
        apartment_copy.merge_world(
            stretch_copy,
        )
        view = apartment_copy.get_semantic_annotations_by_type(Stretch)
        if not view:
            view = Stretch.from_world(stretch_copy)
        else:
            view = view[0]
        view.root.parent_connection.origin = (
            HomogeneousTransformationMatrix.from_xyz_rpy(1.5, 2, 0)
        )
        return apartment_copy, view

    elif request.param == "tiago":
        tiago_copy = deepcopy(tiago_world)
        apartment_copy.merge_world(
            tiago_copy,
        )
        view = apartment_copy.get_semantic_annotations_by_type(Tiago)
        if not view:
            view = Tiago.from_world(tiago_copy)
        else:
            view = view[0]
        view.root.parent_connection.origin = (
            HomogeneousTransformationMatrix.from_xyz_rpy(1.5, 2, 0)
        )
        return apartment_copy, view

    elif request.param == "pr2":
        pr2_copy = deepcopy(pr2_world_setup)
        apartment_copy.merge_world(
            pr2_copy,
        )
        view = apartment_copy.get_semantic_annotations_by_type(PR2)
        if not view:
            view = PR2.from_world(pr2_copy)
        else:
            view = view[0]
        view.root.parent_connection.origin = (
            HomogeneousTransformationMatrix.from_xyz_rpy(1.5, 2, 0)
        )
        return apartment_copy, view


@pytest.fixture
def immutable_multiple_robot_apartment(
    setup_multi_robot_apartment,
) -> Generator[Tuple[World, AbstractRobot, Context]]:
    world, view = setup_multi_robot_apartment
    state = deepcopy(world.state._data)
    yield world, view, Context(world, view)
    world.state._data[:] = state
    world.notify_state_change()


@pytest.fixture
def mutable_multiple_robot_apartment(setup_multi_robot_apartment):
    world, view = setup_multi_robot_apartment
    copy_world: World = deepcopy(world)
    copy_view = copy_world.get_semantic_annotation_by_id(view.id)
    return copy_world, copy_view, Context(copy_world, copy_view)


def test_move_torso_multi(immutable_multiple_robot_apartment):
    world, view, context = immutable_multiple_robot_apartment
    plan = execute_single(MoveTorsoAction(TorsoState.HIGH), context=context)
    with simulated_robot:
        plan.perform()

    joint_state = view.torso.get_joint_state_by_type(TorsoState.HIGH)

    for connection, target in joint_state.items():
        assert connection.position == pytest.approx(target, abs=0.01)


def test_navigate_multi(immutable_multiple_robot_apartment):
    world, view, context = immutable_multiple_robot_apartment
    plan = execute_single(
        NavigateAction(
            Pose(Point3.from_iterable([1, 2, 0]), reference_frame=world.root)
        ),
        context=context,
    )

    with simulated_robot:
        plan.perform()

    robot_base_pose = view.root.global_transform
    robot_base_position = robot_base_pose.to_position().to_np()
    robot_base_orientation = robot_base_pose.to_quaternion().to_np()

    assert robot_base_position[:3] == pytest.approx([1, 2, 0], abs=0.01)
    assert robot_base_orientation == pytest.approx([0, 0, 0, 1], abs=0.01)


def test_move_gripper_multi(immutable_multiple_robot_apartment):
    world, view, context = immutable_multiple_robot_apartment

    plan = execute_single(SetGripperAction(Arms.LEFT, GripperState.OPEN), context)

    with simulated_robot:
        plan.perform()

    arm = view.arms[0]
    open_state = arm.manipulator.get_joint_state_by_type(GripperState.OPEN)
    close_state = arm.manipulator.get_joint_state_by_type(GripperState.CLOSE)

    for connection, target in open_state.items():
        assert connection.position == pytest.approx(target, abs=0.02)

    plan = execute_single(SetGripperAction(Arms.LEFT, GripperState.CLOSE), context)

    with simulated_robot:
        plan.perform()

    for connection, target in close_state.items():
        assert connection.position == pytest.approx(target, abs=0.02)


def test_park_arms_multi(immutable_multiple_robot_apartment):
    world, robot, context = immutable_multiple_robot_apartment
    description = ParkArmsAction(Arms.BOTH)
    plan = execute_single(description, context)
    assert description.arm == Arms.BOTH
    with simulated_robot:
        plan.perform()

    joints = []
    states = []
    for arm in robot.arms:
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
        0.3, -2.4, 0, reference_frame=world.root
    )
    world.notify_state_change()

    plan = sequential(
        [
            ParkArmsAction(Arms.BOTH),
            ReachAction(
                target_pose=Pose(
                    Point3.from_iterable([1, -2, 0.8]), reference_frame=world.root
                ),
                object_designator=milk_body,
                arm=Arms.LEFT,
                grasp_description=grasp_description,
            ),
        ],
        context=context,
    )

    with simulated_robot:
        plan.perform()

    manipulator_pose = left_arm.manipulator.tool_frame.global_transform
    manipulator_position = manipulator_pose.to_position().to_np()
    manipulator_orientation = manipulator_pose.to_quaternion().to_np()

    target_orientation = grasp_description.grasp_orientation()

    assert manipulator_position[:3] == pytest.approx([1, -2, 0.8], abs=0.01)
    compare_orientations(manipulator_orientation, target_orientation, decimal=2)


def test_follow_tcp_path_multi(immutable_multiple_robot_apartment):
    world, robot, context = immutable_multiple_robot_apartment

    if isinstance(robot, (Tiago)):
        # do not allow since
        robot.mobile_base.full_body_controlled = False
        robot.root.parent_connection.origin = (
            HomogeneousTransformationMatrix.from_xyz_rpy(
                1.7, 1.7, 0, reference_frame=world.root
            )
        )
        world.notify_state_change()

    if isinstance(robot, (Stretch)):
        # do not allow since
        robot.mobile_base.full_body_controlled = False
        robot.root.parent_connection.origin = (
            HomogeneousTransformationMatrix.from_xyz_rpy(
                2.12, 2.2, 0, reference_frame=world.root
            )
        )
        world.notify_state_change()
    # robot.full_body_controlled = True
    left_arm = ViewManager.get_arm_view(Arms.LEFT, robot)
    front_axis = tuple(
        int(v) for v in left_arm.manipulator.front_facing_axis.to_np()[:3]
    )
    grasp_axis = AxisIdentifier.from_tuple(front_axis)

    pose_T = world.get_body_by_name("milk.stl").global_transform
    pose = pose_T.to_pose()
    if grasp_axis == AxisIdentifier.X:
        target_pose = pose
    elif grasp_axis == AxisIdentifier.Z:
        offset_T = HomogeneousTransformationMatrix.from_xyz_axis_angle(
            axis=AxisIdentifier.Y.value,
            angle=np.pi / 2,
            reference_frame=world.root,
        )
        target_pose = (pose_T @ offset_T).to_pose()
    else:
        target_pose = pose

    waypoints = PoseTrajectory([target_pose])
    plan = sequential(
        [
            MoveTorsoAction(TorsoState.HIGH),
            ParkArmsAction(Arms.BOTH),
            FollowToolCenterPointPathAction(arm=Arms.LEFT, target_locations=waypoints),
        ],
        context,
    )
    with simulated_robot:
        plan.perform()

    tip_pose = left_arm.manipulator.tool_frame.global_transform
    dist = np.linalg.norm(tip_pose.to_position() - np.array(target_pose.to_position()))
    assert dist < 0.01


def test_grasping(immutable_multiple_robot_apartment):
    world, robot, context = immutable_multiple_robot_apartment
    left_arm = ViewManager.get_arm_view(Arms.LEFT, robot)

    grasp_description = GraspDescription(
        ApproachDirection.FRONT,
        VerticalAlignment.NoAlignment,
        left_arm.manipulator,
    )
    grasping_action = GraspingAction(
        world.get_body_by_name("milk.stl"), Arms.LEFT, grasp_description
    )

    milk_body = world.get_body_by_name("milk.stl")
    milk_body.parent_connection.origin = HomogeneousTransformationMatrix.from_xyz_rpy(
        1, -2, 0.8, reference_frame=world.root
    )
    robot.root.parent_connection.origin = HomogeneousTransformationMatrix.from_xyz_rpy(
        0.3, -2.4, 0, reference_frame=world.root
    )
    world.notify_state_change()

    plan = sequential(
        [
            ParkArmsAction(Arms.BOTH),
            grasping_action,
        ],
        context,
    )
    with simulated_robot:
        plan.perform()
    dist = np.linalg.norm(
        world.get_body_by_name("milk.stl").global_transform.to_np()[3, :3]
    )
    assert dist < 0.01


def test_pick_up_multi(mutable_multiple_robot_apartment):
    world, view, context = mutable_multiple_robot_apartment

    # VizMarkerPublisher(_world=world, node=rclpy_node).with_tf_publisher()
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
        0.3, -2.4, 0, reference_frame=world.root
    )
    world.notify_state_change()

    root = sequential(
        [
            ParkArmsAction(Arms.BOTH),
            PickUpAction(
                world.get_body_by_name("milk.stl"), Arms.LEFT, grasp_description
            ),
        ],
        context,
    )

    with simulated_robot:
        root.perform()

    assert (
        world.get_connection(
            left_arm.manipulator.tool_frame,
            world.get_body_by_name("milk.stl"),
        )
        is not None
    )

    assert len(root.plan.nodes) == len(root.plan.all_nodes)
    root.plan.validate()


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
        0.3, -2.4, 0, reference_frame=world.root
    )
    world.notify_state_change()

    root = sequential(
        [
            ParkArmsAction(Arms.BOTH),
            PickUpAction(
                world.get_body_by_name("milk.stl"), Arms.LEFT, grasp_description
            ),
            PlaceAction(
                world.get_body_by_name("milk.stl"),
                Pose(Point3.from_iterable([1, -2.2, 0.6]), reference_frame=world.root),
                Arms.LEFT,
            ),
        ],
        context,
    )

    with simulated_robot:
        root.perform()

    with pytest.raises(NoEdgeBetweenNodes):
        world.get_connection(
            left_arm.manipulator.tool_frame,
            world.get_body_by_name("milk.stl"),
        )

    milk_position = milk_body.global_transform.to_position().to_np()

    assert milk_position[:3] == pytest.approx([1, -2.2, 0.6], abs=0.01)

    root.plan.validate()


def test_look_at(immutable_multiple_robot_apartment):
    world, robot_view, context = immutable_multiple_robot_apartment
    description = LookAtAction(
        Pose(Point3.from_iterable([3, 0, 1]), reference_frame=world.root)
    )
    assert np.allclose(
        description.target.to_np(),
        Pose(Point3.from_iterable([3, 0, 1]), reference_frame=world.root).to_np(),
        atol=1e-3,
    )

    plan = execute_single(description, context)
    with simulated_robot:
        plan.perform()


def test_detect(immutable_multiple_robot_apartment):
    world, robot, context = immutable_multiple_robot_apartment
    milk_body = world.get_body_by_name("milk.stl")
    with world.modify_world():
        world.add_semantic_annotation(Milk(root=milk_body))

    robot.root.parent_connection.origin = HomogeneousTransformationMatrix.from_xyz_rpy(
        1.5, -2, 0
    )
    milk_body.parent_connection.origin = HomogeneousTransformationMatrix.from_xyz_rpy(
        2.5, -2, 1.2, reference_frame=world.root
    )

    description = DetectAction(
        technique=DetectionTechnique.TYPES,
        object_sem_annotation=Milk,
    )
    plan = execute_single(description, context)
    with simulated_robot:
        plan.perform()
    detected_objects = plan.result

    assert detected_objects[0].name.name == "milk.stl"
    assert detected_objects[0] is milk_body


def test_open(immutable_multiple_robot_apartment):
    world, robot, context = immutable_multiple_robot_apartment

    plan = sequential(
        [
            MoveTorsoAction(TorsoState.HIGH),
            ParkArmsAction(Arms.BOTH),
            NavigateAction(
                Pose(
                    Point3.from_iterable([1.6, 1.9, 0]),
                    Quaternion.from_iterable([0, 0, 0.3, 1]),
                    reference_frame=world.root,
                )
            ),
            OpenAction(world.get_body_by_name("handle_cab10_m"), Arms.LEFT),
        ],
        context,
    )
    with simulated_robot:
        plan.perform()
    assert world.get_connection_by_name(
        "cabinet10_drawer_middle_joint"
    ).position == pytest.approx(0.45, abs=0.1)


def test_close(immutable_multiple_robot_apartment):
    world, robot, context = immutable_multiple_robot_apartment

    world.get_connection_by_name("cabinet10_drawer_middle_joint").position = 0.3
    world.notify_state_change()

    plan = sequential(
        [
            MoveTorsoAction(TorsoState.HIGH),
            ParkArmsAction(Arms.BOTH),
            NavigateAction(
                Pose(
                    Point3.from_iterable([1.65, 2.0, 0]),
                    Quaternion.from_iterable([0, 0, 0.4, 1]),
                    reference_frame=world.root,
                )
            ),
            CloseAction(world.get_body_by_name("handle_cab10_m"), Arms.LEFT),
        ],
        context,
    )
    with simulated_robot:
        plan.perform()
    assert world.get_connection_by_name(
        "cabinet10_drawer_middle_joint"
    ).position == pytest.approx(0, abs=0.1)


def test_facing(immutable_multiple_robot_apartment):
    world, robot, context = immutable_multiple_robot_apartment

    with simulated_robot:
        milk_pose = world.get_body_by_name("milk.stl").global_pose
        plan = execute_single(FaceAtAction(milk_pose, True), context)
        plan.perform()
        milk_in_robot_frame = world.transform(
            world.get_body_by_name("milk.stl").global_transform,
            robot.root,
        )
        assert float(milk_in_robot_frame.to_position().y) == pytest.approx(
            0.0, abs=0.01
        )


def test_transport(mutable_multiple_robot_apartment, rclpy_node):
    world, robot, context = mutable_multiple_robot_apartment

    description = TransportAction(
        object_designator=world.get_body_by_name("milk.stl"),
        target_location=Pose(
            Point3.from_iterable([3.2, 2.2, 0.95]),
            Quaternion.from_iterable([0.0, 0.0, 1.0, 0.0]),
            reference_frame=world.root,
        ),
        arm=Arms.RIGHT,
    )
    plan = sequential([MoveTorsoAction(TorsoState.HIGH), description], context)
    with simulated_robot:
        plan.perform()
    milk_position = world.get_body_by_name("milk.stl").global_transform.to_np()[:3, 3]
    dist = np.linalg.norm(milk_position - np.array([3.2, 2.2, 0.95]))
    assert dist <= 0.02

    plan.plan.validate()
