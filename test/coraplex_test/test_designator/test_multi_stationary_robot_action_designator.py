import numpy as np
import pytest
from rustworkx import NoEdgeBetweenNodes

from giskardpy.utils.utils_for_tests import compare_axis_angle, compare_orientations
from coraplex.datastructures.dataclasses import Context
from coraplex.datastructures.trajectory import PoseTrajectory

from coraplex.execution_environment import simulated_robot
from coraplex.plans.factories import execute_single, sequential
from coraplex.robot_plans.actions.core.pick_up import (
    ReachAction,
    GraspingAction,
    PickUpAction,
)
from coraplex.robot_plans.actions.core.placing import PlaceAction
from coraplex.robot_plans.actions.core.robot_body import (
    ParkArmsAction,
    SetGripperAction,
    FollowToolCenterPointPathAction,
)
from coraplex.testing import _make_sine_scan_poses
from krrood.entity_query_language.factories import an, entity, variable

from semantic_digital_twin.datastructures.definitions import (
    GripperState,
    StaticJointState,
)
from semantic_digital_twin.datastructures.prefixed_name import PrefixedName
from semantic_digital_twin.robots.daisy import DAiSy
from semantic_digital_twin.robots.tracy import Tracy
from semantic_digital_twin.spatial_types import HomogeneousTransformationMatrix, Point3
from semantic_digital_twin.spatial_types.spatial_types import Pose
from semantic_digital_twin.world_description.connections import Connection6DoF
from semantic_digital_twin.world_description.geometry import Box, Scale
from semantic_digital_twin.world_description.shape_collection import ShapeCollection
from semantic_digital_twin.grasping.grasp_candidates import (
    GraspCandidate,
    CanBeGrasped,
)
from semantic_digital_twin.world import World
from semantic_digital_twin.world_description.world_entity import Body

from ...conftest import SAMPLING_SEED
from ..conftest import left_or_only_arm, right_or_only_arm
from ..world_snapshot import WorldSnapshot


@pytest.fixture(
    scope="session",
    params=[["tracy_world", Tracy], ["daisy_world", DAiSy]],
    ids=["Tracy", "DAiSy"],
)
def robot_setup(request):
    world = request.getfixturevalue(request.param[0])

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

    with world.modify_world():
        if request.param[1] == Tracy:
            box1_transformationmatrix = HomogeneousTransformationMatrix.from_xyz_rpy(
                0.8, 0.5, 0.93
            )
            box2_transformationmatrix = HomogeneousTransformationMatrix.from_xyz_rpy(
                0.8, -0.5, 0.93
            )
        elif request.param[1] == DAiSy:
            box1_transformationmatrix = HomogeneousTransformationMatrix.from_xyz_rpy(
                0.6, 1.0, 1.2
            )
            box2_transformationmatrix = HomogeneousTransformationMatrix.from_xyz_rpy(
                0.6, 0.1, 1.2
            )
        else:
            box1_transformationmatrix = HomogeneousTransformationMatrix.from_xyz_rpy()
            box2_transformationmatrix = HomogeneousTransformationMatrix.from_xyz_rpy()

        box1_connection = Connection6DoF.create_with_dofs(
            world=world,
            parent=world.root,
            child=box1,
            name=PrefixedName("box1_connection"),
            parent_T_connection_expression=box1_transformationmatrix,
        )

        box2_connection = Connection6DoF.create_with_dofs(
            world=world,
            parent=world.root,
            child=box2,
            name=PrefixedName("box2_connection"),
            parent_T_connection_expression=box2_transformationmatrix,
        )

        world.add_connection(box1_connection)
        world.add_connection(box2_connection)

        # The boxes stand in for any graspable object; the plans only need an annotation
        # to name them by, not a particular kind of object.
        world.add_semantic_annotations(
            [CanBeGrasped(root=box1), CanBeGrasped(root=box2)]
        )
    return world, request.param[1]


def graspable_annotation(world: World, body: Body) -> CanBeGrasped:
    """
    The annotation naming ``body`` for the actions that take one rather than a body.

    :param world: The world holding the annotations.
    :param body: The body the annotation is rooted at.
    :return: The annotation rooted at ``body``.
    """
    return an(
        entity(
            semantic_annotation := variable(
                CanBeGrasped, domain=world.semantic_annotations
            )
        ).where(semantic_annotation.root == body)
    ).first()


@pytest.fixture
def stationary_block_context(robot_setup):
    """
    The shared block world with one stationary robot, the robot and a context for both,
    returned to its initial model and state after the test.
    """
    block_world, robot_class = robot_setup
    snapshot = WorldSnapshot.capture(block_world)
    view = block_world.get_semantic_annotations_by_type(robot_class)[0]
    yield block_world, view, Context(block_world, view, sampling_seed=SAMPLING_SEED)
    snapshot.restore()


def test_park_arms_multi(stationary_block_context):
    world, view, context = stationary_block_context

    description = ParkArmsAction(context.robot.all_arms)
    plan = execute_single(description, context=context).plan
    with simulated_robot:
        plan.perform()

    joints = []
    states = []
    for arm in view.all_arms:
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


def test_reach_action_multi(stationary_block_context):
    world, view, context = stationary_block_context
    left_arm = left_or_only_arm(context.robot)

    box_body = world.get_body_by_name("box1")
    box = graspable_annotation(world, box_body)
    position = box_body.global_pose.position.to_np()
    grasp_pose = Pose.from_xyz_rpy(pitch=np.pi / 2, reference_frame=box_body)

    plan = sequential(
        [
            ParkArmsAction(context.robot.all_arms),
            ReachAction(
                grasp=GraspCandidate(box, grasp_pose),
                arm=left_or_only_arm(context.robot),
            ),
        ],
        context=context,
    ).plan

    with simulated_robot:
        plan.perform()

    end_effector_pose = left_arm.end_effector.tool_frame.global_transform
    end_effector_position = end_effector_pose.position.to_np()
    end_effector_orientation = end_effector_pose.quaternion.to_np()

    target_orientation = left_arm.end_effector.tool_frame_goal(grasp_pose).quaternion

    assert end_effector_position[:3] == pytest.approx(position[:3], abs=0.01)
    compare_orientations(
        end_effector_orientation, target_orientation.to_np(), decimal=2
    )


def test_move_gripper_multi(stationary_block_context):
    world, view, context = stationary_block_context

    plan = execute_single(
        SetGripperAction(
            end_effector=left_or_only_arm(context.robot).end_effector,
            motion=GripperState.OPEN,
        ),
        context=context,
    ).plan

    with simulated_robot:
        plan.perform()

    arm = view.all_arms[0]
    open_state = arm.end_effector.get_joint_state_by_type(GripperState.OPEN)
    close_state = arm.end_effector.get_joint_state_by_type(GripperState.CLOSE)

    for connection, target in open_state.items():
        assert connection.position == pytest.approx(target, abs=0.01)

    plan = execute_single(
        SetGripperAction(
            end_effector=left_or_only_arm(context.robot).end_effector,
            motion=GripperState.CLOSE,
        ),
        context=context,
    ).plan

    with simulated_robot:
        plan.perform()

    for connection, target in close_state.items():
        assert connection.position == pytest.approx(target, abs=0.01)


def test_grasping(stationary_block_context):
    world, robot_view, context = stationary_block_context
    left_arm = left_or_only_arm(context.robot)

    box_body = world.get_body_by_name("box1")
    description = GraspingAction(
        grasp=GraspCandidate(
            graspable_annotation(world, box_body),
            Pose.from_xyz_rpy(pitch=np.pi / 2, reference_frame=box_body),
        ),
        arm=left_or_only_arm(context.robot),
    )
    plan = sequential(
        [ParkArmsAction(context.robot.all_arms), description],
        context=context,
    ).plan
    with simulated_robot:
        plan.perform()

    # The grasp sits at the box's own origin, so that is where the tool frame ends up.
    assert np.allclose(
        box_body.global_pose.position.to_np(),
        left_arm.end_effector.tool_frame.global_pose.position.to_np(),
        atol=0.01,
    )


def test_pick_up_multi(stationary_block_context):
    world, view, context = stationary_block_context

    left_arm = left_or_only_arm(context.robot)
    box_body = world.get_body_by_name("box1")
    plan = sequential(
        [
            ParkArmsAction(context.robot.all_arms),
            PickUpAction(
                grasp=graspable_annotation(world, box_body).grasp_candidates()[0],
                arm=left_or_only_arm(context.robot),
            ),
        ],
        context=context,
    ).plan

    with simulated_robot:
        plan.perform()

    assert (
        world.get_connection(
            left_arm.end_effector.tool_frame,
            world.get_body_by_name("box1"),
        )
        is not None
    )

    plan.validate()


@pytest.fixture
def place_position(robot_setup) -> Point3:
    world, robot_class = robot_setup
    if robot_class == Tracy:
        return Point3(0.9, 0.0, 0.93, reference_frame=world.root)
    elif robot_class == DAiSy:
        return Point3(0.6, 1.2, 1.2, reference_frame=world.root)
    else:
        raise ValueError(f"Unsupported robot class: {robot_class}")


def test_place_multi(stationary_block_context, place_position):
    world, view, context = stationary_block_context

    left_arm = left_or_only_arm(context.robot)
    box_body = world.get_body_by_name("box1")

    plan = sequential(
        [
            ParkArmsAction(context.robot.all_arms),
            PickUpAction(
                grasp=graspable_annotation(world, box_body).grasp_candidates()[0],
                arm=left_or_only_arm(context.robot),
            ),
            PlaceAction(
                object_designator=graspable_annotation(world, box_body),
                target_location=Pose(place_position, reference_frame=world.root),
            ),
        ],
        context=context,
    ).plan

    with simulated_robot:
        plan.perform()

    with pytest.raises(NoEdgeBetweenNodes):
        world.get_connection(
            left_arm.end_effector.tool_frame,
            world.get_body_by_name("box1"),
        )
    box_body = world.get_body_by_name("box1")
    milk_position = box_body.global_transform.position.to_np()

    assert milk_position[:3] == pytest.approx(place_position.to_list()[:3], abs=0.01)
    plan.validate()


@pytest.fixture
def anchor_position(robot_setup) -> Point3:
    world, robot_class = robot_setup
    if robot_class == Tracy:
        return Point3(x=0.85, y=-0.25, z=0.95, reference_frame=world.root)
    elif robot_class == DAiSy:
        return Point3(x=0.0, y=0.0, z=0.95, reference_frame=world.root)
    else:
        raise ValueError(f"Unsupported robot class: {robot_class}")


def test_move_tcp_follows_sine_waypoints(stationary_block_context, anchor_position):
    world, view, context = stationary_block_context
    right_arm = right_or_only_arm(context.robot)
    anchor = Pose(anchor_position, reference_frame=world.root)
    anchor_T = anchor.homogeneous_matrix
    offset_T = HomogeneousTransformationMatrix.from_xyz_axis_angle(
        z=-0.03,
        axis=(0, 1, 0),
        angle=np.pi / 2,
        reference_frame=world.root,
    )
    target_pose = (anchor_T @ offset_T).pose
    waypoints = PoseTrajectory(_make_sine_scan_poses(target_pose, lane_axis="z"))

    plan = execute_single(
        FollowToolCenterPointPathAction(
            target_locations=waypoints, arm=right_or_only_arm(context.robot)
        ),
        context=context,
    )
    with simulated_robot:
        plan.perform()

    tip_pose = right_arm.end_effector.tool_frame.global_transform
    expected = waypoints.poses[-1]

    assert np.allclose(tip_pose.position, expected.position, atol=0.01)
    assert np.allclose(tip_pose.quaternion, expected.quaternion, atol=0.01)
