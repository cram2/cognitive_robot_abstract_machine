from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass, field

import numpy as np
import pytest

from cramph.context import StatechartContext
from coraplex.plans.executors import PlanExecutor, KinematicallySimulatedPlanExecutor
from coraplex.plans.failures import MotionMadeNoProgress
from coraplex.robot_plans.actions.core.container import CloseAction
from coraplex.robot_plans.actions.core.misc import MoveToReach
from coraplex.robot_plans.actions.core.navigation import (
    NavigateAction,
    PathPlanningNavigateAction,
)
from coraplex.robot_plans.actions.core.pick_up import PickUpAction, ReachAction
from coraplex.robot_plans.actions.core.robot_body import (
    MoveManipulatorAction,
    SetGripperAction,
)
from cramph.composites import Parallel, Sequence, StatementNode
from cramph.context import ContextExtension
from cramph.data_types import LifeCycleValues
from cramph.executor import StatechartExecutor, TickingExecutorExtension
from cramph.node import StatechartNode
from giskardpy.motion_statechart.goals.base_pose_goals import (
    DifferentialDriveBaseGoal,
    HolonomicBaseGoal,
)
from giskardpy.motion_statechart.goals.grasped_mechanism import (
    ArticulateGraspedMechanism,
)
from giskardpy.motion_statechart.goals.gripper import MoveGripper
from giskardpy.motion_statechart.graph_node import ConvergingTask
from giskardpy.motion_statechart.monitors.overwrite_state_monitors import SetOdometry
from giskardpy.motion_statechart.tasks.align_planes import AlignPlanes
from giskardpy.motion_statechart.tasks.cartesian_tasks import (
    CartesianPose,
    CartesianPositionStraight,
    HoldOrientation,
    HoldPose,
)
from krrood.entity_query_language.factories import a, variable_from
from semantic_digital_twin.datastructures.definitions import GripperState
from semantic_digital_twin.grasping.grasp_candidates import GraspCandidate
from semantic_digital_twin.robots.garmi import (
    Garmi,
    GarmiLeftGripper,
)
from semantic_digital_twin.robots.robot_parts import AbstractRobot
from semantic_digital_twin.robots.tiago import Tiago
from semantic_digital_twin.semantic_annotations.semantic_annotations import (
    Handle,
    Milk,
)
from semantic_digital_twin.spatial_types import HomogeneousTransformationMatrix
from semantic_digital_twin.spatial_types.spatial_types import Pose, Pose2D
from semantic_digital_twin.world import World
from semantic_digital_twin.world_description.world_entity import Body

from ...plan_running import (
    robot_executor,
    robot_extensions,
    simulated_executor,
    statechart_of,
    world_of,
)

# %% recording a run


@dataclass
class OrientationWhileRunning(TickingExecutorExtension):
    """
    Records the orientation of a body after every tick in which a node is running.
    """

    body: Body = field(kw_only=True)
    """
    The body whose orientation is recorded.
    """

    node: StatechartNode | None = field(default=None, kw_only=True)
    """
    The node during whose run the orientation is recorded.
    """

    orientations: list[np.ndarray] = field(default_factory=list, init=False)
    """
    The rotation matrices of :attr:`body` relative to the world root, one per tick.
    """

    def after_tick(self, executor: StatechartExecutor) -> None:
        if self.node.life_cycle_state != LifeCycleValues.RUNNING:
            return
        self.orientations.append(self.body.global_pose.to_np()[:3, :3])


# %% preparing a plan for a robot


@pytest.fixture
def tiago_context(_tiago_world_setup):
    """
    A world holding a Tiago, the robot and the context extensions of its plans.
    """
    world = deepcopy(_tiago_world_setup)
    robot = world.get_semantic_annotations_by_type(Tiago)[0]
    return world, robot, robot_extensions(robot)


def prepared(
    plan: StatechartNode,
    executor: PlanExecutor,
) -> StatechartNode:
    """
    Expand and rewrite `plan` the way `executor` would before running it.

    :return: The plan, rewritten.
    """
    executor.prepare(statechart_of(executor, plan))
    return plan


def navigation_to(world: World, **parameters) -> NavigateAction:
    """
    :return: A drive to a pose two meters from the origin of `world`.
    """
    return NavigateAction(
        target_location=Pose.from_xyz_rpy(2.0, 2.0, 0.0, reference_frame=world.root),
        **parameters,
    )


def assert_drives_differentially_to(
    goal: StatechartNode, action: NavigateAction, robot: AbstractRobot
) -> None:
    """
    Check that `goal` drives the differential drive of `robot` to where `action` puts
    the base, within the threshold of `action`.
    """
    assert type(goal) is DifferentialDriveBaseGoal
    assert goal.mobile_base is robot.mobile_base
    assert goal.threshold == action.base_pose_threshold
    np.testing.assert_allclose(
        goal.goal_pose.to_np(),
        robot.mobile_base.pose_facing(action.target_location).to_np(),
    )
    assert goal.goal_pose.reference_frame is action.target_location.reference_frame


# %% driving a differential drive


@pytest.mark.parametrize("build_executor", [simulated_executor, robot_executor])
def test_stretch_drives_with_its_differential_drive(
    stretch_apartment_context, build_executor
) -> None:
    world, robot, extensions = stretch_apartment_context

    action = prepared(navigation_to(world), build_executor(extensions))

    assert_drives_differentially_to(action.goal, action, robot)


def test_stretch_drives_with_its_differential_drive_whatever_the_plan_adaptations(
    stretch_apartment_context,
) -> None:
    """
    The robot adaptations are kept apart from the executor's plan adaptations, so a
    caller replacing the latter keeps the robot driving the way it has to.
    """
    world, robot, extensions = stretch_apartment_context

    action = prepared(
        navigation_to(world), simulated_executor(extensions, plan_adaptations=[])
    )

    assert_drives_differentially_to(action.goal, action, robot)


def test_the_differential_drive_goal_takes_the_threshold_of_the_action(
    stretch_apartment_context,
) -> None:
    world, robot, extensions = stretch_apartment_context
    default_action = navigation_to(world)

    action = prepared(
        navigation_to(
            world, base_pose_threshold=10 * default_action.base_pose_threshold
        ),
        robot_executor(extensions),
    )

    assert action.base_pose_threshold != default_action.base_pose_threshold
    assert_drives_differentially_to(action.goal, action, robot)


def test_the_base_pose_threshold_defaults_to_the_one_of_the_differential_drive_goal(
    stretch_apartment_context,
) -> None:
    world, robot, extensions = stretch_apartment_context

    assert navigation_to(world).base_pose_threshold == (
        DifferentialDriveBaseGoal.threshold
    )


def test_stretch_drives_every_leg_of_a_path_with_its_differential_drive(
    stretch_apartment_context,
) -> None:
    world, robot, extensions = stretch_apartment_context
    target = Pose.from_xyz_rpy(2.0, 2.0, 0.0, reference_frame=world.root)
    unchanged = prepared(
        PathPlanningNavigateAction(target=target),
        simulated_executor(extensions, robot_adaptations=[]),
    )

    action = prepared(
        PathPlanningNavigateAction(target=target), simulated_executor(extensions)
    )

    legs = action.nodes
    assert [type(leg) for leg in legs] == [DifferentialDriveBaseGoal] * len(
        unchanged.nodes
    )
    assert all(leg.mobile_base is robot.mobile_base for leg in legs)
    assert all(leg.threshold == action.base_pose_threshold for leg in legs)
    np.testing.assert_allclose(legs[-1].goal_pose.to_np(), target.to_np())


@pytest.mark.parametrize("build_executor", [simulated_executor, robot_executor])
def test_a_tiago_drives_with_its_differential_drive(
    tiago_context, build_executor
) -> None:
    """
    A real Tiago cannot be put at a pose, so the drive is not tied to simulation.
    """
    world, robot, extensions = tiago_context

    action = prepared(navigation_to(world), build_executor(extensions))

    assert_drives_differentially_to(action.goal, action, robot)


def test_a_simulated_tiago_without_its_robot_adaptations_writes_the_odometry(
    tiago_context,
) -> None:
    world, robot, extensions = tiago_context

    action = prepared(
        navigation_to(world), simulated_executor(extensions, robot_adaptations=[])
    )

    assert type(action.goal) is SetOdometry


@pytest.mark.parametrize(
    "build_executor, goal_type",
    [(simulated_executor, SetOdometry), (robot_executor, HolonomicBaseGoal)],
)
def test_a_robot_of_another_type_keeps_its_drive_goal(
    pr2_apartment_context, build_executor, goal_type
) -> None:
    world, robot, extensions = pr2_apartment_context

    action = prepared(navigation_to(world), build_executor(extensions))

    assert type(action.goal) is goal_type


def test_stretch_keeps_its_drive_goal_without_its_robot_adaptations(
    stretch_apartment_context,
) -> None:
    world, robot, extensions = stretch_apartment_context

    action = prepared(
        navigation_to(world), robot_executor(extensions, robot_adaptations=[])
    )

    assert type(action.goal) is HolonomicBaseGoal


def test_a_simulated_stretch_arrives_driving_with_its_differential_drive(
    stretch_apartment_context,
) -> None:
    world, robot, extensions = stretch_apartment_context
    executor = simulated_executor(extensions)
    action = navigation_to(world)
    executor.compile(statechart_of(executor, action))

    executor.execute()

    np.testing.assert_allclose(
        robot.root.global_pose.to_np(),
        robot.mobile_base.pose_facing(action.target_location).to_np(),
        atol=2 * action.base_pose_threshold,
    )


def test_a_simulated_stretch_drives_differentially_to_a_pose_chosen_while_running(
    stretch_apartment_context,
) -> None:
    world, robot, extensions = stretch_apartment_context
    executor = simulated_executor(extensions)
    navigation = StatementNode(
        statement=a(NavigateAction)(
            target_location=variable_from([navigation_to(world).target_location])
        )
    )
    executor.compile(statechart_of(executor, navigation))

    executor.execute()

    [action] = navigation.grounded_nodes
    assert_drives_differentially_to(action.goal, action, robot)


def test_a_drive_chosen_while_running_is_watched_for_stalling_as_it_is_driven(
    stretch_apartment_context,
) -> None:
    """
    The transformation replaces the body of the chosen drive, so the monitor giving up
    on a stalled attempt has to watch the tasks of the body that runs.
    """
    world, robot, extensions = stretch_apartment_context
    executor = simulated_executor(extensions)
    navigation = StatementNode(
        statement=a(NavigateAction)(
            target_location=variable_from([navigation_to(world).target_location])
        )
    )
    executor.compile(statechart_of(executor, navigation))

    executor.execute()

    [steps] = navigation.chosen_children
    [stalled] = navigation.failure_monitors_of(steps)
    [action] = navigation.grounded_nodes
    assert stalled.monitored_tasks == [
        node for node in action.descendants if isinstance(node, ConvergingTask)
    ]


# %% closing a container


def close_action(world: World, robot: AbstractRobot) -> CloseAction:
    """
    :return: An action closing a container of the apartment with the arm of `robot`.
    """
    handle = world.get_semantic_annotations_by_type(Handle)[0]
    return CloseAction(grasp=handle.grasp_candidates()[0], arm=robot.all_arms[0])


def closing_step_of(action: CloseAction) -> StatechartNode:
    """
    :return: The step of an expanded close action between grasping the handle and
        letting go of it.
    """
    grasp, closing_step, release, back_off = action.nodes
    return closing_step


def test_a_simulated_stretch_stays_in_front_of_the_container_it_closes(
    stretch_apartment_context,
) -> None:
    world, robot, extensions = stretch_apartment_context
    unchanged = prepared(close_action(world, robot), robot_executor(extensions))
    default_close = closing_step_of(unchanged)
    executor = simulated_executor(extensions)

    action = prepared(close_action(world, robot), executor)

    closing_step = closing_step_of(action)
    tool_frame = action.arm.end_effector.tool_frame
    assert type(closing_step) is Parallel
    holding, facing, close = closing_step.nodes
    assert type(holding) is CartesianPose
    assert holding.root_link is action.grasp.graspable.root
    assert holding.tip_link is tool_frame
    np.testing.assert_allclose(holding.goal_pose.to_np(), Pose().to_np())
    assert holding.goal_pose.reference_frame is tool_frame
    assert type(facing) is AlignPlanes
    assert facing.root_link is world.root
    assert facing.tip_link is robot.root
    np.testing.assert_allclose(facing.goal_normal.to_np()[:3], [1, 0, 0])
    assert facing.goal_normal.reference_frame is action.grasp.graspable.root
    np.testing.assert_allclose(facing.tip_normal.to_np()[:3], [0, -1, 0])
    assert facing.tip_normal.reference_frame is robot.root
    assert type(close) is ArticulateGraspedMechanism
    assert close.tip_link is default_close.tip_link
    assert close.environment_link is default_close.environment_link
    assert close.goal_joint_state == default_close.goal_joint_state


def test_a_stretch_closes_a_container_with_its_base_free_to_turn(
    stretch_apartment_context,
) -> None:
    """
    The drawer travels further than the arm reaches, so the base has to turn to follow
    it, and a stand chosen for grasping the handle would not be one to close it from.
    """
    world, robot, extensions = stretch_apartment_context

    action = prepared(close_action(world, robot), simulated_executor(extensions))

    assert held_orientations_of(action) == []


def test_the_real_stretch_closes_a_container_unchanged(
    stretch_apartment_context,
) -> None:
    world, robot, extensions = stretch_apartment_context
    executor = robot_executor(extensions)

    action = prepared(close_action(world, robot), executor)

    assert type(closing_step_of(action)) is ArticulateGraspedMechanism


def test_a_robot_of_another_type_closes_a_container_unchanged(
    pr2_apartment_context,
) -> None:
    world, robot, extensions = pr2_apartment_context
    executor = simulated_executor(extensions)

    action = prepared(close_action(world, robot), executor)

    assert type(closing_step_of(action)) is ArticulateGraspedMechanism


# %% moving the tool center point in a straight line


def handle_grasp(world: World) -> GraspCandidate:
    """
    :return: A grasp on a handle of the apartment.
    """
    return world.get_semantic_annotations_by_type(Handle)[0].grasp_candidates()[0]


def reaching_goals_of(action: ReachAction) -> list[StatechartNode]:
    """
    :return: The goals of an expanded reach action bringing the tool center point to
        the pre-grasp pose and to the grasp pose. A pose goal is the step itself, holding
        what accompanies it; any other goal is the first node of its step.
    """
    return [
        step if isinstance(step, CartesianPose) else step.nodes[0]
        for step in action.nodes
    ]


def lifting_goal_of(action: PickUpAction) -> StatechartNode:
    """
    :return: The goal of an expanded pick up action lifting the grasped object, which
        runs in a parallel only where something accompanies it, such as the hold on the
        base of a robot that stands still.
    """
    lift = action.nodes[-1]
    return lift.nodes[0] if type(lift) is Parallel else lift


@pytest.mark.parametrize("build_executor", [simulated_executor, robot_executor])
def test_stretch_reaches_in_a_straight_line(
    stretch_apartment_context, build_executor
) -> None:
    world, robot, extensions = stretch_apartment_context
    unchanged = prepared(
        ReachAction(grasp=handle_grasp(world), arm=robot.all_arms[0]),
        build_executor(extensions, robot_adaptations=[]),
    )

    action = prepared(
        ReachAction(grasp=handle_grasp(world), arm=robot.all_arms[0]),
        build_executor(extensions),
    )

    for goal, default_goal in zip(
        reaching_goals_of(action), reaching_goals_of(unchanged), strict=True
    ):
        assert type(goal.nodes[0]) is CartesianPositionStraight
        assert goal.root_link is default_goal.root_link
        assert goal.tip_link is default_goal.tip_link
        np.testing.assert_allclose(
            goal.goal_pose.to_np(), default_goal.goal_pose.to_np()
        )
        assert goal.translation_threshold == default_goal.translation_threshold
        assert goal.orientation_threshold == default_goal.orientation_threshold
        assert goal.binding_policy == default_goal.binding_policy


@pytest.mark.parametrize("build_executor", [simulated_executor, robot_executor])
def test_stretch_lifts_in_a_straight_line(
    stretch_apartment_context, build_executor
) -> None:
    world, robot, extensions = stretch_apartment_context
    unchanged = lifting_goal_of(
        prepared(
            PickUpAction(grasp=handle_grasp(world), arm=robot.all_arms[0]),
            build_executor(extensions, robot_adaptations=[]),
        )
    )

    goal = lifting_goal_of(
        prepared(
            PickUpAction(grasp=handle_grasp(world), arm=robot.all_arms[0]),
            build_executor(extensions),
        )
    )

    assert type(goal.nodes[0]) is CartesianPositionStraight
    assert goal.root_link is unchanged.root_link
    assert goal.tip_link is unchanged.tip_link
    np.testing.assert_allclose(goal.goal_pose.to_np(), unchanged.goal_pose.to_np())
    assert goal.translation_threshold == unchanged.translation_threshold
    assert goal.orientation_threshold == unchanged.orientation_threshold


def held_orientations_of(action: StatechartNode) -> list[HoldOrientation]:
    """
    :return: The tasks holding the orientation of a link beside `action`, in the
        parallel running it, or below it.
    """
    beside = action.parent_node.nodes if type(action.parent_node) is Parallel else []
    return [
        node for node in [*beside, *action.descendants] if type(node) is HoldOrientation
    ]


@pytest.mark.parametrize("build_executor", [simulated_executor, robot_executor])
def test_stretch_holds_the_orientation_of_its_base_while_it_reaches(
    stretch_apartment_context, build_executor
) -> None:
    world, robot, extensions = stretch_apartment_context

    action = prepared(
        ReachAction(grasp=handle_grasp(world), arm=robot.all_arms[0]),
        build_executor(extensions),
    )

    [hold] = held_orientations_of(action)
    assert hold.parent_node is action.parent_node
    assert hold.root_link is world.root
    assert hold.tip_link is robot.root


def test_stretch_holds_the_orientation_of_its_base_once_for_nested_tool_motions(
    stretch_apartment_context,
) -> None:
    """
    Picking up reaches first, and both move the tool center point; the hold of the
    picking up already covers the reaching.
    """
    world, robot, extensions = stretch_apartment_context

    action = prepared(
        PickUpAction(grasp=handle_grasp(world), arm=robot.all_arms[0]),
        simulated_executor(extensions),
    )

    [hold] = held_orientations_of(action)
    assert hold.parent_node is action.parent_node


def test_stretch_holds_the_orientation_of_its_base_for_each_of_successive_tool_motions(
    stretch_apartment_context,
) -> None:
    """
    The hold of the first reach covers only that reach, not the one after it.
    """
    world, robot, extensions = stretch_apartment_context
    reaches = [
        ReachAction(grasp=handle_grasp(world), arm=robot.all_arms[0]) for _ in range(2)
    ]

    prepared(Sequence(nodes=list(reaches)), simulated_executor(extensions))

    for reach in reaches:
        [hold] = held_orientations_of(reach)
        assert hold.parent_node is reach.parent_node


def test_stretch_holds_the_orientation_of_its_base_beside_a_hold_of_something_else(
    stretch_apartment_context,
) -> None:
    world, robot, extensions = stretch_apartment_context
    reach = ReachAction(grasp=handle_grasp(world), arm=robot.all_arms[0])
    tool_frame = robot.all_arms[0].end_effector.tool_frame

    prepared(
        Parallel(nodes=[HoldPose(root_link=world.root, tip_link=tool_frame), reach]),
        simulated_executor(extensions),
    )

    [hold] = held_orientations_of(reach)
    assert hold.tip_link is robot.root


def test_stretch_holds_the_orientation_of_its_base_only_once_it_drove_there(
    stretch_apartment_context,
) -> None:
    """
    Moving to reach drives the base, which the held orientation would stop; the reach
    after the drive is held instead.
    """
    world, robot, extensions = stretch_apartment_context

    action = prepared(
        MoveToReach(
            target_pose_offset_robot=Pose2D.from_xy_yaw(0.2, -0.55),
            reference_T_grasp=Pose.from_xyz_rpy(
                x=0.7, y=-1.3, z=0.9, reference_frame=world.root
            ),
            hip_rotation=0.0,
            end_effector=robot.all_arms[0].end_effector,
        ),
        simulated_executor(extensions),
    )

    [hold] = held_orientations_of(action)
    [reaching] = [
        node for node in action.descendants if type(node) is MoveManipulatorAction
    ]
    assert hold.parent_node is reaching.parent_node


def stretch_reaching_milk_from(
    world: World, robot: AbstractRobot, extensions: list, root_T_base: Pose
) -> tuple[KinematicallySimulatedPlanExecutor, ReachAction, OrientationWhileRunning]:
    """
    Puts the base of a Stretch at `root_T_base`, and prepares a reach for the milk
    standing at a fixed place in a simulation that records the orientation of the base
    while it is held.

    :return: The executor, with the reach compiled but not run yet, the reach, and the
        recording.
    """
    milk = world.get_semantic_annotations_by_type(Milk)[0]
    milk.root.parent_connection.origin = HomogeneousTransformationMatrix.from_xyz_rpy(
        1, -2, 0.8, reference_frame=world.root
    )
    drive = robot.root.parent_connection
    drive.origin = world.transform(root_T_base, drive.parent).homogeneous_matrix
    recording = OrientationWhileRunning(body=robot.root)
    executor = KinematicallySimulatedPlanExecutor(
        StatechartContext.with_extensions(world, extensions), extensions=[recording]
    )
    action = ReachAction(
        grasp=GraspCandidate(milk, Pose(reference_frame=milk.root)),
        arm=robot.all_arms[0],
    )
    executor.compile(statechart_of(executor, action))
    [recording.node] = held_orientations_of(action)
    return executor, action, recording


def assert_kept_its_heading(
    recording: OrientationWhileRunning, tolerance: float = 0.01
) -> None:
    """
    Asserts that the recorded body turned at most `tolerance` radians away from where it
    faced when the recording started.
    """
    assert recording.orientations
    start = recording.orientations[0]
    turned = [
        np.arccos(np.clip((np.trace(start.T @ orientation) - 1) / 2, -1, 1))
        for orientation in recording.orientations
    ]
    assert max(turned) <= tolerance


@pytest.mark.parametrize(
    "root_T_base",
    [
        Pose.from_xyz_rpy(0.33, -1.98, 0, yaw=np.pi / 2),
        Pose.from_xyz_rpy(0.33, -2.2, 0, yaw=np.pi / 2),
    ],
    ids=["where its arm reaches the milk", "short of the milk along its heading"],
)
def test_a_simulated_stretch_reaches_without_turning_its_base(
    stretch_apartment_context, root_T_base
) -> None:
    """
    The Stretch stands with its arm towards the milk, and keeps its heading while it
    reaches, driving along it where it has to.
    """
    world, robot, extensions = stretch_apartment_context
    root_T_base.reference_frame = world.root

    executor, action, recording = stretch_reaching_milk_from(
        world, robot, extensions, root_T_base
    )

    executor.execute()

    assert action.life_cycle_state == LifeCycleValues.SUCCEEDED
    assert_kept_its_heading(recording)


def test_a_simulated_stretch_facing_away_fails_to_reach_rather_than_turning(
    stretch_apartment_context,
) -> None:
    """
    Turning to bring a target into reach sweeps the base through wide arcs, so a Stretch
    whose arm points away from the milk fails to reach it instead.

    Straining towards the milk bends the held heading by less than a degree, far from
    the turn reaching it would take.
    """
    world, robot, extensions = stretch_apartment_context
    executor, action, recording = stretch_reaching_milk_from(
        world,
        robot,
        extensions,
        Pose.from_xyz_rpy(0.4, -2.3, 0, reference_frame=world.root),
    )

    with pytest.raises(MotionMadeNoProgress):
        executor.execute()

    assert_kept_its_heading(recording, tolerance=np.deg2rad(2))


def test_a_robot_of_another_type_reaches_without_its_base_orientation_held(
    pr2_apartment_context,
) -> None:
    world, robot, extensions = pr2_apartment_context

    action = prepared(
        ReachAction(grasp=handle_grasp(world), arm=robot.all_arms[0]),
        simulated_executor(extensions),
    )

    assert held_orientations_of(action) == []


def test_a_robot_of_another_type_reaches_unchanged(pr2_apartment_context) -> None:
    world, robot, extensions = pr2_apartment_context

    action = prepared(
        ReachAction(grasp=handle_grasp(world), arm=robot.all_arms[0]),
        simulated_executor(extensions),
    )

    assert [type(goal) for goal in reaching_goals_of(action)] == [CartesianPose] * 2


def test_a_robot_of_another_type_lifts_unchanged(pr2_apartment_context) -> None:
    world, robot, extensions = pr2_apartment_context

    goal = lifting_goal_of(
        prepared(
            PickUpAction(grasp=handle_grasp(world), arm=robot.all_arms[0]),
            simulated_executor(extensions),
        )
    )

    assert type(goal) is CartesianPose


# %% moving GARMI's fingers


@pytest.fixture
def garmi_extensions(_garmi_world_setup) -> list[ContextExtension]:
    world = deepcopy(_garmi_world_setup)
    return robot_extensions(world.get_semantic_annotations_by_type(Garmi)[0])


def test_a_simulated_garmi_drives_its_fingers_with_a_giskard_goal(
    garmi_extensions,
) -> None:
    gripper = world_of(garmi_extensions).get_semantic_annotations_by_type(
        GarmiLeftGripper
    )[0]

    action = prepared(
        SetGripperAction(
            configuration=gripper.default_configuration(GripperState.OPEN)
        ),
        simulated_executor(garmi_extensions),
    )

    assert type(action.goal) is MoveGripper
