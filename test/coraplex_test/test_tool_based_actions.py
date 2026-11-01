import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from coraplex.datastructures.enums import (
    CuttingTechnique,
    PouringSide,
    WipingTechnique,
)
from coraplex.exceptions import WipingTargetMissing
from coraplex.robot_plans.actions.base import ParallelAction
from coraplex.robot_plans.actions.core.navigation import NavigateAction
from coraplex.robot_plans.actions.composite.tool_based import (
    CuttingAction,
    MixingAction,
    PouringAction,
    WipingAction,
)
from giskardpy.motion_statechart.monitors.cartesian_monitors import PositionReached
from giskardpy.motion_statechart.tasks.align_planes import AlignPlanes
from giskardpy.motion_statechart.tasks.cartesian_tasks import (
    CartesianPositionTrajectory,
)
from cramph.composites import PausedUntilTrue, Sequence
from cramph.data_types import LifeCycleValues
from krrood.ormatic.data_access_objects.helper import to_dao
from semantic_digital_twin.datastructures.prefixed_name import PrefixedName
from semantic_digital_twin.semantic_annotations.semantic_annotations import (
    PouringCup,
    CuttingKnife,
    Sponge,
    Whisk,
)
from semantic_digital_twin.spatial_types import HomogeneousTransformationMatrix
from semantic_digital_twin.spatial_types.spatial_types import Pose
from semantic_digital_twin.world_description.connections import FixedConnection
from semantic_digital_twin.world_description.geometry import Box, Scale
from semantic_digital_twin.world_description.shape_collection import ShapeCollection
from semantic_digital_twin.world_description.world_entity import Body

from coraplex.robot_plans.robot_specific_plan_transformations import (
    KeepTheTorsoUprightWhileUsingATool,
)
from semantic_digital_twin.adapters.urdf import URDFParser
from semantic_digital_twin.robots.justin import Justin

from .conftest import expand
from ..plan_running import (
    robot_extensions,
    run_plan,
    simulated_executor,
    statechart_of,
)


def _add_box_body(world, name, size, position):
    shape_collection = ShapeCollection([Box(scale=Scale(*size))])
    body = Body(
        name=PrefixedName(name), collision=shape_collection, visual=shape_collection
    )
    with world.modify_world():
        world.add_kinematic_structure_entity(body)
        world.add_connection(
            FixedConnection(
                parent=world.root,
                child=body,
                parent_T_connection_expression=HomogeneousTransformationMatrix.from_xyz_rpy(
                    *position, reference_frame=world.root
                ),
            )
        )
    return body


@pytest.fixture
def tool_action_world(pr2_apartment_context):
    world, robot, extensions = pr2_apartment_context
    container = _add_box_body(
        world, "tool_test_container", (0.2, 0.2, 0.1), (2.4, 2.2, 1.0)
    )
    tool_body = _add_box_body(
        world, "tool_test_tool", (0.04, 0.04, 0.2), (1.0, 1.0, 1.0)
    )
    return world, robot, extensions, container, tool_body


def _tool_path_goals(action, extensions):
    """
    :return: The goals the action expanded into that follow its tool path, one per tool
        motion: the action itself where it runs its tool path side by side, and its
        steps otherwise.
    """
    expand(action, extensions)
    candidates = [action] if isinstance(action, ParallelAction) else action.nodes
    return [goal for goal in candidates if _trajectory_of(goal) is not None]


def _trajectory_of(goal):
    """
    :return: The trajectory task somewhere below `goal`, or None when it holds none.
    """
    for node in [goal, *goal.descendants]:
        if isinstance(node, CartesianPositionTrajectory):
            return node
    return None


def _alignments_of(goal):
    """
    :return: Every plane alignment held anywhere below `goal`.
    """
    return [node for node in [goal, *goal.descendants] if isinstance(node, AlignPlanes)]


def test_mixing_action_expands_to_aligned_motion(tool_action_world):
    world, robot, extensions, container, tool_body = tool_action_world
    whisk = Whisk(root=tool_body)

    action = MixingAction(container=container, arm=robot.right_arm, tool=whisk)
    goals = _tool_path_goals(action, extensions)

    assert len(goals) == 1
    trajectory = _trajectory_of(goals[0])
    assert len(trajectory.goal_points) > 0
    assert len(_alignments_of(goals[0])) == 1
    assert trajectory.tip_link == whisk.get_tool_frame()


def test_cutting_action_pointer_stride_reduces_waypoints(tool_action_world):
    world, robot, extensions, container, tool_body = tool_action_world
    knife = CuttingKnife(root=tool_body)

    dense_action = CuttingAction(
        object_to_cut=container,
        arm=robot.right_arm,
        tool=knife,
        technique=CuttingTechnique.SLICE,
    )
    strided_action = CuttingAction(
        object_to_cut=container,
        arm=robot.right_arm,
        tool=knife,
        technique=CuttingTechnique.SLICE,
        pointer_stride=10,
    )

    dense_goal = _tool_path_goals(dense_action, extensions)[0]
    strided_goal = _tool_path_goals(strided_action, extensions)[0]

    dense_points = _trajectory_of(dense_goal).goal_points
    strided_points = _trajectory_of(strided_goal).goal_points
    assert len(dense_points) > 0
    assert len(strided_points) == pytest.approx(len(dense_points) / 10, abs=1)
    assert len(_alignments_of(dense_goal)) == 2


def test_tool_motion_frees_the_manipulator_holding_the_tool(tool_action_world) -> None:
    """
    A tool works by touching what it is used on, so the manipulator holding it is freed
    from collision avoidance for the whole stroke.
    """
    world, robot, extensions, container, tool_body = tool_action_world
    whisk = Whisk(root=tool_body)

    action = MixingAction(container=container, arm=robot.right_arm, tool=whisk)
    expand(action, extensions)

    [rule] = action.temporary_collision_rules
    assert rule.end_effector is robot.right_arm.end_effector


def test_wiping_action_requires_container_or_target_pose(tool_action_world):
    world, robot, extensions, container, tool_body = tool_action_world
    sponge = Sponge(root=tool_body)

    with pytest.raises(WipingTargetMissing):
        WipingAction(arm=robot.right_arm, tool=sponge)


def test_wiping_action_around_target_pose(tool_action_world):
    world, robot, extensions, container, tool_body = tool_action_world
    sponge = Sponge(root=tool_body)

    action = WipingAction(
        arm=robot.right_arm,
        tool=sponge,
        target_pose=Pose.from_xyz_rpy(x=2.4, y=2.2, z=1.0, reference_frame=world.root),
    )
    goals = _tool_path_goals(action, extensions)

    assert len(goals) == 1
    assert len(_trajectory_of(goals[0]).goal_points) > 0
    assert len(_alignments_of(goals[0])) == 1


def test_pouring_action_poses_tilt_and_mirror(tool_action_world):
    world, robot, extensions, container, tool_body = tool_action_world
    cup = PouringCup(root=tool_body)

    right_action = PouringAction(
        target_container=container,
        source_container=cup,
        arm=robot.right_arm,
    )
    expand(right_action, extensions)
    right_pre_pose, right_pour_pose = right_action._pour_poses()

    pre_rotation = Rotation.from_quat(
        [float(value) for value in right_pre_pose.quaternion.to_np()]
    )
    pour_rotation = Rotation.from_quat(
        [float(value) for value in right_pour_pose.quaternion.to_np()]
    )
    tilt_magnitude = (pre_rotation.inv() * pour_rotation).magnitude()
    assert tilt_magnitude == pytest.approx(right_action.tilt_angle, abs=1e-6)
    assert float(right_pre_pose.x) == pytest.approx(float(right_pour_pose.x))
    assert float(right_pre_pose.y) == pytest.approx(float(right_pour_pose.y))

    left_action = PouringAction(
        target_container=container,
        source_container=cup,
        arm=robot.right_arm,
        pour_side=PouringSide.LEFT,
    )
    expand(left_action, extensions)
    left_pre_pose, _ = left_action._pour_poses()

    container_position = np.array(
        [
            float(container.global_pose.x),
            float(container.global_pose.y),
        ]
    )
    right_offset = (
        np.array([float(right_pre_pose.x), float(right_pre_pose.y)])
        - container_position
    )
    left_offset = (
        np.array([float(left_pre_pose.x), float(left_pre_pose.y)]) - container_position
    )
    np.testing.assert_allclose(left_offset, -right_offset, atol=1e-9)


@pytest.mark.parametrize(
    "arm_of, side",
    [
        (lambda robot: robot.right_arm, PouringSide.RIGHT),
        (lambda robot: robot.left_arm, PouringSide.LEFT),
    ],
    ids=["right-arm", "left-arm"],
)
def test_pouring_pours_to_the_side_of_its_arm_unless_told_otherwise(
    tool_action_world, arm_of, side
):
    world, robot, extensions, container, tool_body = tool_action_world
    action = PouringAction(
        target_container=container,
        source_container=PouringCup(root=tool_body),
        arm=arm_of(robot),
    )
    expand(action, extensions)

    assert action._effective_pour_side() is side


def _attach_box_to_gripper(world, robot, name, size, mount_z):
    shape_collection = ShapeCollection([Box(scale=Scale(*size))])
    body = Body(
        name=PrefixedName(name), collision=shape_collection, visual=shape_collection
    )
    tool_frame = robot.right_arm.end_effector.tool_frame
    with world.modify_world():
        world.add_kinematic_structure_entity(body)
        world.add_connection(
            FixedConnection(
                parent=tool_frame,
                child=body,
                parent_T_connection_expression=HomogeneousTransformationMatrix.from_xyz_rpy(
                    z=mount_z, reference_frame=tool_frame
                ),
            )
        )
    return body


def test_pouring_action_pour_point_lands_on_target_container_center(
    tool_action_world,
):
    world, robot, extensions, container, tool_body = tool_action_world
    held_source = _attach_box_to_gripper(
        world, robot, "held_pour_source", (0.04, 0.04, 0.2), -0.08
    )
    cup = PouringCup(root=held_source)

    action = PouringAction(
        target_container=container, source_container=cup, arm=robot.right_arm
    )
    expand(action, extensions)
    _, pour_pose = action._pour_poses()

    tool_frame = robot.right_arm.end_effector.tool_frame
    tool_frame_T_source = world.compute_forward_kinematics_np(tool_frame, held_source)
    mouth_in_tool_frame = tool_frame_T_source @ np.array([0.0, 0.0, 0.1, 1.0])
    mouth_in_world = pour_pose.homogeneous_matrix.to_np() @ mouth_in_tool_frame

    assert mouth_in_world[0] == pytest.approx(float(container.global_pose.x), abs=1e-6)
    assert mouth_in_world[1] == pytest.approx(float(container.global_pose.y), abs=1e-6)


def test_mixing_action_orm_roundtrip(tool_action_world, coraplex_testing_session):
    world, robot, extensions, container, tool_body = tool_action_world
    whisk = Whisk(root=tool_body)

    action = MixingAction(container=container, arm=robot.right_arm, tool=whisk)
    expand(action, extensions)

    dao = to_dao(action)
    coraplex_testing_session.add(dao)
    coraplex_testing_session.commit()

    assert dao.database_id is not None


# %% full body control


def test_a_tool_motion_moves_the_tool_relative_to_the_world(tool_action_world) -> None:
    """
    The base supports the arm during a tool motion, which only works if the tool path is
    expressed relative to the world rather than to the robot.
    """
    world, robot, extensions, container, tool_body = tool_action_world
    full_body_controlled = robot.mobile_base.full_body_controlled
    action = MixingAction(
        container=container, arm=robot.right_arm, tool=Whisk(root=tool_body)
    )

    goal = _tool_path_goals(action, extensions)[0]

    assert _trajectory_of(goal).root_link is world.root
    assert robot.mobile_base.full_body_controlled == full_body_controlled


# %% the tolerance a tool path is followed with


def test_a_tool_path_is_followed_within_the_position_threshold_of_its_action(
    tool_action_world,
) -> None:
    """
    A tool motion follows its path within its own position threshold, like every other
    tool center point motion.
    """
    world, robot, extensions, container, tool_body = tool_action_world
    action = MixingAction(
        container=container, arm=robot.right_arm, tool=Whisk(root=tool_body)
    )

    goal = _tool_path_goals(action, extensions)[0]

    assert _trajectory_of(goal).threshold == action.position_threshold


# %% wiping


def test_a_wipe_counts_as_done_once_the_tool_reached_its_final_waypoint(
    tool_action_world,
) -> None:
    world, robot, extensions, container, tool_body = tool_action_world
    sponge = Sponge(root=tool_body)
    action = WipingAction(
        arm=robot.right_arm,
        tool=sponge,
        target_pose=Pose.from_xyz_rpy(x=2.4, y=2.2, z=1.0, reference_frame=world.root),
    )

    expand(action, extensions)

    [final_waypoint_reached] = [
        node
        for node in [action, *action.descendants]
        if isinstance(node, PositionReached)
    ]
    assert final_waypoint_reached.tip_link is tool_body
    assert final_waypoint_reached.goal_point is action._waypoints[-1]
    assert final_waypoint_reached.threshold == action.final_waypoint_success_tolerance


def test_a_spread_wipe_is_done_only_once_the_tool_followed_its_path(
    tool_action_world,
) -> None:
    """
    A spread sweep ends where it starts, so the tool is at its final waypoint before the
    first stroke; that alone does not finish the wipe.
    """
    world, robot, extensions, container, tool_body = tool_action_world
    held_sponge = _attach_box_to_gripper(
        world, robot, "held_sponge", (0.05, 0.05, 0.05), 0.0
    )
    wiping = WipingAction(
        arm=robot.right_arm,
        tool=Sponge(root=held_sponge),
        technique=WipingTechnique.SPREAD,
        target_pose=Pose.from_xyz_rpy(x=2.4, y=2.2, z=1.1, reference_frame=world.root),
    )
    plan = Sequence(
        nodes=[
            NavigateAction(
                target_location=Pose.from_xyz_rpy(
                    x=1.85, y=2.2, reference_frame=world.root
                )
            ),
            wiping,
        ]
    )

    run_plan(plan, extensions)

    [tool_path] = [node for node in wiping.nodes if _trajectory_of(node) is not None]
    assert tool_path.life_cycle_state == LifeCycleValues.SUCCEEDED


def test_a_wipe_stalled_close_to_its_final_waypoint_is_done(
    tool_action_world,
) -> None:
    """
    A wipe whose tool gets stuck within the success tolerance of its final waypoint is
    accepted once the motion stalled.
    """
    world, robot, extensions, container, tool_body = tool_action_world
    held_sponge = _attach_box_to_gripper(
        world, robot, "held_sponge", (0.05, 0.05, 0.05), 0.0
    )
    # the sponge reaches up to about 1.58 m here, so it gets stuck 5 cm below the patch
    wiping = WipingAction(
        arm=robot.right_arm,
        tool=Sponge(root=held_sponge),
        technique=WipingTechnique.SPREAD,
        target_pose=Pose.from_xyz_rpy(x=2.4, y=2.2, z=1.63, reference_frame=world.root),
    )
    plan = Sequence(
        nodes=[
            NavigateAction(
                target_location=Pose.from_xyz_rpy(
                    x=1.85, y=2.2, reference_frame=world.root
                )
            ),
            wiping,
        ]
    )

    run_plan(plan, extensions)

    [final_waypoint_check] = [
        node for node in wiping.nodes if isinstance(node, PausedUntilTrue)
    ]
    assert final_waypoint_check.life_cycle_state == LifeCycleValues.SUCCEEDED


# %% keeping the torso upright


@pytest.fixture(scope="module")
def justin_world():
    world = URDFParser.from_file(Justin.get_ros_file_path()).parse()
    Justin.from_world(world)
    return world


def _rewritten_mixing(world, robot):
    """
    :return: A mixing action of `robot` in `world`, expanded and rewritten the way a
        simulated executor rewrites it.
    """
    container = _add_box_body(world, "mixed_container", (0.2, 0.2, 0.1), (1, 0, 1))
    tool_body = _add_box_body(world, "mixing_tool", (0.04, 0.04, 0.2), (1, 0.3, 1))
    action = MixingAction(
        container=container, arm=robot.all_arms[0], tool=Whisk(root=tool_body)
    )
    executor = simulated_executor(robot_extensions(robot))
    executor.prepare(statechart_of(executor, action))
    return action


def test_justin_keeps_its_torso_upright_while_it_uses_a_tool(justin_world) -> None:
    robot = justin_world.get_semantic_annotations_by_type(Justin)[0]

    action = _rewritten_mixing(justin_world, robot)

    torso_tip = robot.mobile_base.torso.tip
    torso_alignments = [
        alignment
        for alignment in _alignments_of(action)
        if alignment.tip_link is torso_tip
    ]
    assert len(torso_alignments) == 1


def test_a_robot_other_than_justin_keeps_its_torso_as_it_is(tool_action_world) -> None:
    world, robot, extensions, container, tool_body = tool_action_world
    action = MixingAction(
        container=container, arm=robot.right_arm, tool=Whisk(root=tool_body)
    )
    expand(action, extensions)

    assert not KeepTheTorsoUprightWhileUsingATool().is_applicable(action)
