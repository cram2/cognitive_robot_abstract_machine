import mujoco
import numpy as np
import pytest
from rustworkx import NoEdgeBetweenNodes

from coraplex.datastructures.trajectory import PoseTrajectory
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
from cramph.composites import Sequence
from giskardpy.utils.utils_for_tests import compare_axis_angle, compare_orientations
from krrood.entity_query_language.factories import an, entity, variable
from semantic_digital_twin.adapters.multi_sim import MujocoSim
from semantic_digital_twin.datastructures.definitions import (
    GripperState,
    StaticJointState,
)
from semantic_digital_twin.datastructures.prefixed_name import PrefixedName
from semantic_digital_twin.grasping.grasp_candidates import GraspCandidate
from semantic_digital_twin.robots.daisy import DAiSy
from semantic_digital_twin.robots.tracy import Tracy
from semantic_digital_twin.semantic_annotations.mixins import HasGraspCandidates
from semantic_digital_twin.spatial_types import HomogeneousTransformationMatrix, Point3
from semantic_digital_twin.spatial_types.spatial_types import Pose
from semantic_digital_twin.world import World
from semantic_digital_twin.world_description.connections import Connection6DoF
from semantic_digital_twin.world_description.geometry import Box, Scale
from semantic_digital_twin.world_description.shape_collection import ShapeCollection
from semantic_digital_twin.world_description.world_entity import Body
from ..conftest import grasp_point_of, left_or_only_arm, right_or_only_arm
from ..world_snapshot import WorldSnapshot
from ...plan_running import perform_in_physics, run_plan, robot_extensions
from ...pytest_environment import runs_in_continuous_integration


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
                0.6, 1.0, 0.9
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
            [HasGraspCandidates(root=box1), HasGraspCandidates(root=box2)]
        )
    return world, request.param[1]


def graspable_annotation(world: World, body: Body) -> HasGraspCandidates:
    """
    The annotation naming ``body`` for the actions that take one rather than a body.

    :param world: The world holding the annotations.
    :param body: The body the annotation is rooted at.
    :return: The annotation rooted at ``body``.
    """
    return an(
        entity(
            semantic_annotation := variable(
                HasGraspCandidates, domain=world.semantic_annotations
            )
        ).where(semantic_annotation.root == body)
    ).first()


def top_down_grasp(
    world: World, body: Body, height_above_origin: float = 0.0
) -> GraspCandidate:
    """
    :param world: The world holding the annotation naming ``body``.
    :param body: The body to grasp.
    :param height_above_origin: How far above ``body``'s origin the grasp holds it.
    :return: A grasp reaching down onto ``body``, centred over its origin.
    """
    return GraspCandidate(
        graspable_annotation(world, body),
        Pose.from_xyz_rpy(z=height_above_origin, pitch=np.pi / 2, reference_frame=body),
    )


@pytest.fixture
def stationary_block_context(robot_setup):
    """
    The shared block world with one stationary robot, the robot and the context
    extensions of its plans, returned to its initial model and state after the test.
    """
    block_world, robot_class = robot_setup
    snapshot = WorldSnapshot.capture(block_world)
    view = block_world.get_semantic_annotations_by_type(robot_class)[0]
    yield block_world, view, robot_extensions(view)
    snapshot.restore()


def test_park_arms_multi(stationary_block_context):
    world, view, extensions = stationary_block_context

    description = ParkArmsAction(arms=view.all_arms)
    plan = description
    run_plan(plan, extensions)
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
    world, view, extensions = stationary_block_context
    left_arm = left_or_only_arm(view)

    box_body = world.get_body_by_name("box1")
    box = graspable_annotation(world, box_body)
    position = box_body.global_pose.position.to_np()
    grasp_pose = Pose.from_xyz_rpy(pitch=np.pi / 2, reference_frame=box_body)

    plan = Sequence(
        nodes=[
            ParkArmsAction(arms=view.all_arms),
            ReachAction(
                grasp=GraspCandidate(box, grasp_pose),
                arm=left_or_only_arm(view),
            ),
        ]
    )

    run_plan(plan, extensions)
    end_effector_pose = left_arm.end_effector.tool_frame.global_transform
    end_effector_orientation = end_effector_pose.quaternion.to_np()

    target_orientation = left_arm.end_effector.tool_frame_goal(grasp_pose).quaternion

    assert grasp_point_of(left_arm) == pytest.approx(position[:3], abs=0.01)
    compare_orientations(
        end_effector_orientation, target_orientation.to_np(), decimal=2
    )


def test_move_gripper_multi(stationary_block_context):
    world, view, extensions = stationary_block_context

    plan = SetGripperAction(
        configuration=left_or_only_arm(view).end_effector.default_configuration(
            GripperState.OPEN
        )
    )

    executor = run_plan(plan, extensions)
    arm = view.all_arms[0]
    open_state = arm.end_effector.get_joint_state_by_type(GripperState.OPEN)
    close_state = arm.end_effector.get_joint_state_by_type(GripperState.CLOSE)

    for connection, target in open_state.items():
        assert connection.position == pytest.approx(target, abs=0.01)

    plan = SetGripperAction(
        configuration=left_or_only_arm(view).end_effector.default_configuration(
            GripperState.CLOSE
        )
    )

    run_plan(plan, extensions)
    for connection, target in close_state.items():
        assert connection.position == pytest.approx(target, abs=0.01)


def test_grasping(stationary_block_context):
    world, robot_view, extensions = stationary_block_context
    left_arm = left_or_only_arm(robot_view)

    box_body = world.get_body_by_name("box1")
    description = GraspingAction(
        grasp=top_down_grasp(world, box_body), arm=left_or_only_arm(robot_view)
    )
    plan = Sequence(nodes=[ParkArmsAction(arms=robot_view.all_arms), description])
    run_plan(plan, extensions)
    # The grasp sits at the box's own origin, so that is where the fingers end up.
    assert np.allclose(
        box_body.global_pose.position.to_np().flatten()[:3],
        grasp_point_of(left_arm),
        atol=0.01,
    )


def test_pick_up_multi(stationary_block_context):
    world, view, extensions = stationary_block_context

    left_arm = left_or_only_arm(view)
    box_body = world.get_body_by_name("box1")
    plan = Sequence(
        nodes=[
            ParkArmsAction(arms=view.all_arms),
            PickUpAction(
                grasp=top_down_grasp(world, box_body), arm=left_or_only_arm(view)
            ),
        ]
    )

    run_plan(plan, extensions)
    assert (
        world.get_connection(
            left_arm.end_effector.tool_frame,
            world.get_body_by_name("box1"),
        )
        is not None
    )


@pytest.fixture
def place_position(robot_setup) -> Point3:
    world, robot_class = robot_setup
    if robot_class == Tracy:
        return Point3(0.9, 0.0, 0.93, reference_frame=world.root)
    elif robot_class == DAiSy:
        return Point3(0.6, 1.2, 0.9, reference_frame=world.root)
    else:
        raise ValueError(f"Unsupported robot class: {robot_class}")


def test_place_multi(stationary_block_context, place_position):
    world, view, extensions = stationary_block_context

    left_arm = left_or_only_arm(view)
    box_body = world.get_body_by_name("box1")

    plan = Sequence(
        nodes=[
            ParkArmsAction(arms=view.all_arms),
            PickUpAction(
                grasp=top_down_grasp(world, box_body), arm=left_or_only_arm(view)
            ),
            PlaceAction(
                object_designator=graspable_annotation(world, box_body),
                target_location=Pose(place_position, reference_frame=world.root),
            ),
        ]
    )

    run_plan(plan, extensions)
    with pytest.raises(NoEdgeBetweenNodes):
        world.get_connection(
            left_arm.end_effector.tool_frame,
            world.get_body_by_name("box1"),
        )
    box_body = world.get_body_by_name("box1")
    milk_position = box_body.global_transform.position.to_np()

    assert milk_position[:3] == pytest.approx(place_position.to_list()[:3], abs=0.01)


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
    world, view, extensions = stationary_block_context
    right_arm = right_or_only_arm(view)
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

    plan = FollowToolCenterPointPathAction(
        target_locations=waypoints, arm=right_or_only_arm(view)
    )
    run_plan(plan, extensions)
    tip_pose = right_arm.end_effector.tool_frame.global_transform
    expected = waypoints.poses[-1]

    assert np.allclose(tip_pose.position, expected.position, atol=0.01)
    assert np.allclose(tip_pose.quaternion, expected.quaternion, atol=0.01)


# %% picking and placing in physics


@pytest.fixture
def physically_simulated_block_context(stationary_block_context):
    """
    The block world with a box narrow enough for the gripper standing on the table, and
    its physics running, so a plan performed in it is carried out by the robot's own
    servos.
    """
    world, view, extensions = stationary_block_context
    if not isinstance(view, Tracy):
        pytest.skip("only Tracy is set up with the servos a physical simulation needs")
    # box1 and box2 are wider than the 85 mm Tracy's gripper opens.
    edge = 0.06
    narrow_box = Body(
        name=PrefixedName("narrow_box"),
        collision=ShapeCollection([Box(scale=Scale(edge, edge, edge))]),
        visual=ShapeCollection([Box(scale=Scale(edge, edge, edge))]),
    )
    with world.modify_world():
        world.add_connection(
            Connection6DoF.create_with_dofs(
                world=world,
                parent=world.root,
                child=narrow_box,
                name=PrefixedName("narrow_box_connection"),
                parent_T_connection_expression=HomogeneousTransformationMatrix.from_xyz_rpy(
                    0.8, 0.3, view.table.top_z + edge / 2
                ),
            )
        )
        world.add_semantic_annotations([HasGraspCandidates(root=narrow_box)])
    # Under the default pyramidal friction cone, a held box creeps down in the grip.
    simulation = MujocoSim(
        world=world,
        headless=True,
        cone=mujoco.mjtCone.mjCONE_ELLIPTIC,
        impratio=10.0,
    )
    simulation.start_stepped_simulation()
    yield world, view, extensions, simulation, narrow_box
    simulation.stop_simulation()


def grasp_near_the_top(
    world: World, box: Body, depth_below_top: float = 0.01
) -> GraspCandidate:
    """
    :param world: The world holding the annotation naming ``box``.
    :param box: The box to grasp.
    :param depth_below_top: How far below the top of ``box`` the grasp holds it.
    :return: A top-down grasp holding ``box`` just below its top, clear of the inner
        knuckles above Tracy's pads.
    """
    # Grasped deeper, the box reaches up into the inner knuckles, which push the gripper
    # up and sideways while it closes.
    top = box.collision.max_point.to_np()[2]
    return top_down_grasp(world, box, height_above_origin=top - depth_below_top)


@pytest.mark.skipif(
    not runs_in_continuous_integration(), reason="MuJoCo tests only run in CI"
)
def test_a_pick_in_physics_leaves_the_box_to_the_grip(
    physically_simulated_block_context,
):
    """
    Nothing puts the box under the gripper in the model, so whether it comes along at
    all is up to the pads holding it.
    """
    world, view, extensions, simulation, box = physically_simulated_block_context
    arm = left_or_only_arm(view)
    stood_on = box.parent_connection.parent
    resting_height = box.global_pose.position.to_np().flatten()[2]
    pick = PickUpAction(grasp=grasp_near_the_top(world, box), arm=arm)
    plan = Sequence(nodes=[ParkArmsAction(arms=view.all_arms), pick])

    perform_in_physics(plan, extensions, simulation)

    assert box.parent_connection.parent is stood_on
    with pytest.raises(NoEdgeBetweenNodes):
        world.get_connection(arm.end_effector.tool_frame, box)
    lifted_by = box.global_pose.position.to_np().flatten()[2] - resting_height
    assert lifted_by == pytest.approx(pick.retreat_distance, abs=0.01)


@pytest.mark.skipif(
    not runs_in_continuous_integration(), reason="MuJoCo tests only run in CI"
)
def test_a_place_in_physics_leaves_the_box_to_the_release(
    physically_simulated_block_context, place_position
):
    """
    Nothing puts the box back under the world in the model either, so where it ends up
    is where the pads opening leaves it.
    """
    world, view, extensions, simulation, box = physically_simulated_block_context
    arm = left_or_only_arm(view)
    stood_on = box.parent_connection.parent
    resting_height = box.global_pose.position.to_np().flatten()[2]
    target = Point3(
        *place_position.to_np().flatten()[:2],
        resting_height,
        reference_frame=world.root,
    )
    plan = Sequence(
        nodes=[
            ParkArmsAction(arms=view.all_arms),
            PickUpAction(grasp=grasp_near_the_top(world, box), arm=arm),
            PlaceAction(
                object_designator=graspable_annotation(world, box),
                target_location=Pose(target, reference_frame=world.root),
            ),
        ]
    )

    perform_in_physics(plan, extensions, simulation)

    assert box.parent_connection.parent is stood_on
    assert np.allclose(
        box.global_pose.position.to_np().flatten()[:3],
        target.to_np().flatten()[:3],
        atol=0.01,
    )
