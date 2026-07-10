from copy import deepcopy

import numpy as np
import pytest
from coraplex.alternative_motion_mappings.hsrb_motion_mapping import HSRBMoveMotion
from coraplex.alternative_motion_mappings.stretch_motion_mapping import (
    StretchMoveToolCenterPoint,
    StretchMoveSim,
    StretchMoveReal,
    StretchClose,
)
from coraplex.alternative_motion_mappings.tiago_motion_mapping import TiagoMoveSim
from coraplex.datastructures.dataclasses import Context
from coraplex.datastructures.enums import (
    Arms,
    AxisIdentifier,
    ApproachDirection,
    VerticalAlignment,
    DetectionTechnique,
)
from coraplex.datastructures.grasp import GraspDescription
from coraplex.datastructures.trajectory import PoseTrajectory
from coraplex.execution_environment import simulated_robot
from coraplex.plans.factories import sequential, execute_single
from coraplex.robot_plans.actions.composite.facing import FaceAtAction
from coraplex.robot_plans.actions.composite.transporting import TransportAction
from coraplex.robot_plans.actions.core.container import OpenAction, CloseAction
from coraplex.robot_plans.actions.core.misc import DetectAction, MoveToReach
from coraplex.robot_plans.actions.core.navigation import NavigateAction, LookAtAction
from coraplex.robot_plans.actions.core.pick_up import (
    ReachAction,
    GraspingAction,
    PickUpAction,
)
from coraplex.robot_plans.actions.core.placing import PlaceAction
from coraplex.robot_plans.actions.core.robot_body import (
    MoveTorsoAction,
    SetGripperAction,
    ParkArmsAction,
    FollowToolCenterPointPathAction,
)
from coraplex.view_manager import ViewManager
from giskardpy.utils.utils_for_tests import compare_axis_angle, compare_orientations
from rustworkx.rustworkx import NoEdgeBetweenNodes
from semantic_digital_twin.adapters.ros.visualization.viz_marker import (
    VizMarkerPublisher,
)
from semantic_digital_twin.datastructures.definitions import (
    TorsoState,
    GripperState,
    StaticJointState,
)
from semantic_digital_twin.robots.robot_part_mixins import HasMobileBase
from semantic_digital_twin.robots.robot_parts import AbstractRobot, EndEffector
from typing_extensions import Tuple, Generator

try:
    from semantic_digital_twin.robots.garmi import Garmi
except ImportError:
    Garmi = None
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
from semantic_digital_twin.spatial_types.spatial_types import Pose, Pose2D
from semantic_digital_twin.world import World

# The alternative motion mappings that should be available to the plans in this test module.
# Resolution filters by robot type and execution type, so passing the full set is always safe.
ALTERNATIVE_MOTION_MAPPINGS = [
    HSRBMoveMotion,
    StretchMoveToolCenterPoint,
    StretchMoveSim,
    StretchMoveReal,
    StretchClose,
    TiagoMoveSim,
]


@pytest.fixture(
    scope="session",
    params=[
        # TODO Garmi commented out until we get access to the robot description in CI
        # pytest.param(
        #     "garmi",
        #     marks=pytest.mark.skipif(
        #         Garmi is None,
        #         reason="GARMI semantic annotation not installed",
        #     ),
        # ),
        "hsrb",
        "stretch",
        "tiago",
        "pr2",
    ],
)
def setup_multi_robot_apartment(
    request,
    _hsr_world_setup,
    _stretch_world_setup,
    _tiago_world_setup,
    _pr2_world_setup,
    _apartment_world_setup,
):
    apartment_copy = deepcopy(_apartment_world_setup)

    if request.param == "hsrb":
        hsr_copy = deepcopy(_hsr_world_setup)
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
        stretch_copy = deepcopy(_stretch_world_setup)
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
        tiago_copy = deepcopy(_tiago_world_setup)
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
        pr2_copy = deepcopy(_pr2_world_setup)
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

    elif request.param == "garmi":
        if Garmi is None:
            pytest.skip("GARMI semantic annotation not installed")
        garmi_world_setup = request.getfixturevalue("garmi_world_setup")
        garmi_copy = deepcopy(garmi_world_setup)
        apartment_copy.merge_world(
            garmi_copy,
        )
        view = Garmi.from_world(apartment_copy)
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
    full_body_controlled = (
        view.mobile_base.full_body_controlled
        if isinstance(view, HasMobileBase)
        else False
    )
    yield world, view, Context(
        world, view, alternative_motion_mappings=ALTERNATIVE_MOTION_MAPPINGS
    )
    view.mobile_base.full_body_controlled = full_body_controlled
    world.state._data[:] = state
    world.notify_state_change()


@pytest.fixture
def mutable_multiple_robot_apartment(setup_multi_robot_apartment):
    world, view = setup_multi_robot_apartment
    copy_world: World = deepcopy(world)
    copy_view = copy_world.get_semantic_annotation_by_id(view.id)
    return (
        copy_world,
        copy_view,
        Context(
            copy_world,
            copy_view,
            alternative_motion_mappings=ALTERNATIVE_MOTION_MAPPINGS,
        ),
    )


def test_move_torso_multi(immutable_multiple_robot_apartment):
    world, view, context = immutable_multiple_robot_apartment
    plan = execute_single(MoveTorsoAction(TorsoState.HIGH), context=context)
    with simulated_robot:
        plan.perform()

    joint_state = view.get_torso().get_joint_state_by_type(TorsoState.HIGH)

    for connection, target in joint_state.items():
        assert connection.position == pytest.approx(target, abs=0.01)


def test_navigate_multi(immutable_multiple_robot_apartment, rclpy_node):
    world, view, context = immutable_multiple_robot_apartment
    target_position = [2, -2, 0]

    plan = execute_single(
        NavigateAction(
            Pose(Point3.from_iterable(target_position), reference_frame=world.root)
        ),
        context=context,
    )

    with simulated_robot:
        plan.perform()

    robot_base_pose = view.root.global_transform
    robot_base_position = robot_base_pose.to_position().to_np()
    robot_base_orientation = robot_base_pose.to_quaternion().to_np()

    assert robot_base_position[:3] == pytest.approx(target_position, abs=0.01)
    assert robot_base_orientation == pytest.approx([0, 0, 0, 1], abs=0.01)


def test_move_gripper_multi(immutable_multiple_robot_apartment):
    world, view, context = immutable_multiple_robot_apartment

    plan = execute_single(SetGripperAction(Arms.LEFT, GripperState.OPEN), context)

    with simulated_robot:
        plan.perform()

    arm = view.get_arms()[0]
    open_state = arm.end_effector.get_joint_state_by_type(GripperState.OPEN)
    close_state = arm.end_effector.get_joint_state_by_type(GripperState.CLOSE)

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
    for arm in robot.get_arms():
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
        left_arm.end_effector,
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

    end_effector_pose = left_arm.end_effector.tool_frame.global_transform
    end_effector_position = end_effector_pose.to_position().to_np()
    end_effector_orientation = end_effector_pose.to_quaternion().to_np()

    target_orientation = grasp_description.grasp_orientation()

    assert end_effector_position[:3] == pytest.approx([1, -2, 0.8], abs=0.01)
    compare_orientations(end_effector_orientation, target_orientation, decimal=2)


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
        int(v) for v in left_arm.end_effector.front_facing_axis.to_np()[:3]
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

    tip_pose = left_arm.end_effector.tool_frame.global_transform
    dist = np.linalg.norm(tip_pose.to_position() - np.array(target_pose.to_position()))
    assert dist < 0.01


def test_grasping(immutable_multiple_robot_apartment):
    world, robot, context = immutable_multiple_robot_apartment
    left_arm = ViewManager.get_arm_view(Arms.LEFT, robot)

    grasp_description = GraspDescription(
        ApproachDirection.FRONT,
        VerticalAlignment.NoAlignment,
        left_arm.end_effector,
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


def test_pick_up_multi(mutable_multiple_robot_apartment, rclpy_node):
    world, view, context = mutable_multiple_robot_apartment

    context.evaluate_conditions = False

    left_arm = ViewManager.get_arm_view(Arms.LEFT, view)
    grasp_description = GraspDescription(
        ApproachDirection.FRONT,
        VerticalAlignment.NoAlignment,
        left_arm.end_effector,
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
            left_arm.end_effector.tool_frame,
            world.get_body_by_name("milk.stl"),
        )
        is not None
    )

    assert np.allclose(
        world.get_body_by_name("milk.stl").global_pose.to_position().to_np(),
        left_arm.end_effector.tool_frame.global_pose.to_position().to_np(),
        atol=0.01,
    )

    assert len(root.plan.nodes) == len(root.plan.all_nodes)
    root.plan.validate()


def test_place_multi(mutable_multiple_robot_apartment):
    world, view, context = mutable_multiple_robot_apartment

    left_arm = ViewManager.get_arm_view(Arms.LEFT, view)
    grasp_description = GraspDescription(
        ApproachDirection.FRONT,
        VerticalAlignment.NoAlignment,
        left_arm.end_effector,
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
            left_arm.end_effector.tool_frame,
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

    # Detection no longer returns a value; it writes the result into the
    # world/belief state by marking the perceived annotation with its class label.
    milk_annotations = world.get_semantic_annotations_by_type(Milk)
    assert milk_annotations
    perceived = milk_annotations[0]
    assert perceived.class_label == "Milk"
    assert milk_body in perceived.bodies


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


def test_close(immutable_multiple_robot_apartment, rclpy_node):
    world, robot, context = immutable_multiple_robot_apartment

    world.get_connection_by_name("cabinet10_drawer_middle_joint").position = 0.3
    world.notify_state_change()

    navigate_position = [1.5, 1.85, 0] if isinstance(robot, Tiago) else [1.65, 2.0, 0]

    plan = sequential(
        [
            MoveTorsoAction(TorsoState.HIGH),
            ParkArmsAction(Arms.BOTH),
            NavigateAction(
                Pose(
                    Point3.from_iterable(navigate_position),
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
            Point3.from_iterable([3.1, 2.2, 0.95]),
            Quaternion.from_iterable([0.0, 0.0, 1.0, 0.0]),
            reference_frame=world.root,
        ),
        arm=Arms.RIGHT,
        grasp_description=GraspDescription(
            ApproachDirection.FRONT,
            VerticalAlignment.NoAlignment,
            ViewManager.get_end_effector_view(Arms.RIGHT, robot),
        ),
    )
    plan = sequential([MoveTorsoAction(TorsoState.HIGH), description], context)
    with simulated_robot:
        plan.perform()
    milk_position = world.get_body_by_name("milk.stl").global_transform.to_np()[:3, 3]
    dist = np.linalg.norm(milk_position - np.array([3.1, 2.2, 0.95]))
    assert dist <= 0.02

    plan.plan.validate()


def test_move_to_reach(immutable_multiple_robot_apartment, rclpy_node):
    world, robot, context = immutable_multiple_robot_apartment
    move_to_reach = MoveToReach(
        target_pose_offset_robot=Pose2D(0.2, -0.55),
        target_pose_end_effector=Pose.from_xyz_rpy(
            x=0.7, y=-1.3, z=0.9, reference_frame=world.root
        ),
        hip_rotation=0.0,
        grasp_description=GraspDescription(
            approach_direction=ApproachDirection.FRONT,
            vertical_alignment=VerticalAlignment.NoAlignment,
            rotate_gripper=False,
            end_effector=world.get_semantic_annotations_by_type(EndEffector)[0],
        ),
    )

    plan = execute_single(move_to_reach, context=context)
    with simulated_robot:
        plan.perform()


def test_transport_open_container(mutable_multiple_robot_apartment, rclpy_node):
    world, robot, context = mutable_multiple_robot_apartment

    VizMarkerPublisher(_world=world, node=rclpy_node).with_tf_publisher()
    context.ros_node = rclpy_node
    context.debug = True

    if isinstance(robot, HSRB):
        return
    description = TransportAction(
        object_designator=world.get_body_by_name("spoon.stl"),
        target_location=Pose.from_xyz_rpy(
            5.1, 3.3, 0.75, yaw=1.57, reference_frame=world.root
        ),
        arm=Arms.RIGHT,
        grasp_description=GraspDescription(
            ApproachDirection.FRONT,
            VerticalAlignment.TOP,
            ViewManager.get_end_effector_view(Arms.RIGHT, robot),
        ),
    )
    plan = sequential(
        [MoveTorsoAction(TorsoState.HIGH), ParkArmsAction(Arms.BOTH), description],
        context,
    )
    with simulated_robot:
        plan.perform()
    spoon_position = world.get_body_by_name("spoon.stl").global_transform.to_np()[:3, 3]
    dist = np.linalg.norm(spoon_position - np.array([5.1, 3.3, 0.75]))
    assert dist <= 0.02

    plan.plan.validate()
