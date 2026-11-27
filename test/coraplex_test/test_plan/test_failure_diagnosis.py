"""
Experiment: a plan run by a simulated PR2 fails because of a node chosen by the test,
and the diagnosis read from the plan's causal graph has to name that node.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from datetime import timedelta

import pytest

from coraplex.datastructures.enums import InsertionPosition
from coraplex.exceptions import MotionDidNotFinish
from coraplex.plans.failure_diagnosis.causal_graph import RunCausalGraph
from coraplex.plans.failure_diagnosis.counterfactual import (
    StallCounterfactual,
    TaskWithJointsUnwound,
)
from giskardpy.motion_statechart.conflict_edges import ConflictEdge
from coraplex.plans.failure_diagnosis.diagnosis import FailureDiagnoser
from coraplex.plans.failure_diagnosis.exceptions import (
    FailureDiagnosis,
    NodeFailedByACondition,
    NodeFailedOnItsOwn,
    StatementRanOutOfCandidates,
    TaskStalled,
    Verdict,
)
from coraplex.plans.failures import EmptyUnderspecified, MotionMadeNoProgress
from coraplex.plans.plan_transformation import (
    PlanRewriting,
    RecoveryByInsertion,
    SequentialInsertion,
)
from coraplex.robot_plans.actions.base import Action
from coraplex.robot_plans.actions.core.navigation import NavigateAction
from coraplex.robot_plans.actions.core.pick_up import PickUpAction
from coraplex.robot_plans.actions.core.placing import PlaceAction
from coraplex.robot_plans.actions.core.robot_body import (
    MoveManipulatorAction,
    ParkArmsAction,
    SetGripperAction,
)
from cramph.causal_edges import WorldStateEdge
from cramph.composites import Parallel, Sequence, StatementNode
from cramph.data_types import LifeCycleValue
from cramph.failure_handling import HandledOnFailure, LimitedRecovery, RetryFailedNode
from cramph.node import FailsOnObservingFalse, StatechartNode, SucceedsOnObservingTrue
from cramph.statechart import Statechart
from cramph.world_modification_nodes import MoveBranch
from giskardpy.motion_statechart.monitors.joint_monitors import JointPositionReached
from giskardpy.motion_statechart.monitors.progress_monitors import BlockedPositionLimit
from giskardpy.motion_statechart.monitors.overwrite_state_monitors import SetOdometry
from giskardpy.motion_statechart.tasks.cartesian_tasks import (
    CartesianOrientation,
    CartesianPose,
)
from giskardpy.motion_statechart.tasks.joint_tasks import JointPositionList
from krrood.entity_query_language.factories import a, variable_from
from krrood_spatial_math.spatial_types import Pose, Pose2D, Vector3
from semantic_digital_twin.datastructures.definitions import GripperState
from semantic_digital_twin.datastructures.joint_state import JointState
from semantic_digital_twin.robots.robot_parts import EndEffector
from semantic_digital_twin.robots.tracy import Tracy, TracyJoint
from semantic_digital_twin.semantic_annotations.semantic_annotations import Milk
from semantic_digital_twin.world import World
from semantic_digital_twin.world_description.degree_of_freedom import PositionLimitSide
from semantic_digital_twin.world_description.kinematic_chain import KinematicChain
from .test_failure_recovery import GraspFailingItsFirstRuns
from coraplex.plans.executors import PhysicallySimulatedPlanExecutor
from cramph.context import StatechartContext
from semantic_digital_twin.adapters.multi_sim import MujocoSim
from ...plan_running import (
    robot_extensions,
    simulated_executor,
    statechart_of,
    world_of,
)
from ...dataset.wound_up_wrist import ShortWrist, WristWinding
from ...pytest_environment import runs_in_continuous_integration

# %% mimics


@dataclass(eq=False, repr=False, kw_only=True)
class GripperOpenCheck(
    SucceedsOnObservingTrue, FailsOnObservingFalse, JointPositionReached
):
    """
    A step that succeeds once a gripper joint stands at its open position and fails as
    soon as it does not.
    """


def gripper_open_check(end_effector: EndEffector) -> GripperOpenCheck:
    """
    :return: A check that the gripper of `end_effector` is open, on its driven joint.
    """
    open_state = end_effector.default_configuration(GripperState.OPEN).joint_state
    return GripperOpenCheck(
        connection=open_state.connections[0], position=open_state.target_values[0]
    )


def set_gripper(end_effector: EndEffector, state: GripperState) -> SetGripperAction:
    """
    :return: An action moving the gripper of `end_effector` to `state`.
    """
    return SetGripperAction(
        gripper_configuration=end_effector.default_configuration(state)
    )


@dataclass(kw_only=True)
class OpenTheGripperTheCheckReads(SequentialInsertion[GripperOpenCheck]):
    """
    Opens the gripper a :class:`GripperOpenCheck` reads before that check runs again.
    """

    end_effector: EndEffector
    """
    The end effector whose gripper is opened.
    """

    @property
    def position(self) -> InsertionPosition:
        return InsertionPosition.BEFORE

    def is_applicable(self, *, plan_node: GripperOpenCheck) -> bool:
        return True

    def anchor(self, *, plan_node: GripperOpenCheck) -> StatechartNode:
        return plan_node

    def nodes_to_insert(self, *, plan_node: GripperOpenCheck) -> list[StatechartNode]:
        return [set_gripper(self.end_effector, GripperState.OPEN)]


@dataclass(kw_only=True)
class RecoveryByRootCause(LimitedRecovery):
    """
    Diagnoses the failed node while the plan runs and recovers with the repair kept for
    the kind of action that caused the failure.
    """

    repairs: dict[type[Action], LimitedRecovery] = field(default_factory=dict)
    """
    The recovery to use when an action of a type caused the failure.
    """

    diagnoses: list[FailureDiagnosis] = field(default_factory=list)
    """
    Every diagnosis made, in order.
    """

    def recover(self, *, failed_node: StatechartNode) -> None:
        diagnosis = FailureDiagnoser.from_statechart(
            statechart=failed_node.statechart
        ).diagnose_node(node=failed_node)
        self.diagnoses.append(diagnosis)
        for root in diagnosis.root_causes:
            repair = self.repairs.get(
                type(Action.enclosing_action_of(node=root.causing_run.node))
            )
            if repair is not None:
                repair.recover(failed_node=failed_node)
                return


# %% helpers


def run_until_it_fails(
    plan: StatechartNode,
    extensions: list,
    failure: type[Exception],
    **executor_options,
) -> tuple[Statechart, Exception]:
    """
    :param executor_options: Further arguments of the simulated executor.
    :return: The statechart of `plan`, run until it failed with `failure`, and the
        failure.
    """
    executor = simulated_executor(extensions, **executor_options)
    executor.compile(statechart=statechart_of(executor, plan))
    with pytest.raises(failure) as raised:
        executor.execute()
    return executor.statechart, raised.value


def perform_in_physics_until_it_fails(
    plan: StatechartNode,
    extensions: list,
    simulation: MujocoSim,
    failure: type[Exception],
) -> tuple[Statechart, Exception]:
    """
    :return: The statechart of `plan`, performed in `simulation` until it failed with
        `failure`, and the failure.
    """
    executor = PhysicallySimulatedPlanExecutor(
        context=StatechartContext.with_extensions(
            world=world_of(extensions), extensions=extensions
        ),
        simulation=simulation,
    )
    executor.compile(statechart=statechart_of(executor, plan))
    with pytest.raises(failure) as raised:
        executor.execute()
    return executor.statechart, raised.value


def report(title: str, diagnoses: list[FailureDiagnosis]) -> None:
    """
    Print the diagnoses of one scenario, for the experiment's record.
    """
    print(f"\n===== {title}")
    for diagnosis in diagnoses:
        print(diagnosis)


def causing_actions(diagnosis: FailureDiagnosis) -> list[Action | None]:
    """
    :return: The action holding each root cause of `diagnosis`, in rank order.
    """
    return [
        Action.enclosing_action_of(node=root.causing_run.node)
        for root in diagnosis.root_causes
    ]


# %% a pick-up stalls because navigation left the base where the arm cannot reach


def test_a_stalled_reach_is_blamed_on_the_navigation_that_placed_the_base(
    pr2_apartment_context,
) -> None:
    """
    A reach that stops short of the milk is blamed on the navigation that left the base
    where the arm cannot reach, and trying the reach again free to move the base
    confirms it.
    """
    world, robot, extensions = pr2_apartment_context
    milk = world.get_semantic_annotations_by_type(Milk)[0]
    navigate = NavigateAction(
        planar_goal_pose=Pose2D.from_xy_yaw(
            x=1.5, y=2.4, yaw=math.pi, reference_frame=world.root
        )
    )
    plan = Sequence(
        nodes=[
            ParkArmsAction(arms=robot.all_arms),
            navigate,
            PickUpAction(
                grasp_candidate=milk.grasp_candidates()[0], arm=robot.right_arm
            ),
        ]
    )

    statechart, failure = run_until_it_fails(plan, extensions, MotionMadeNoProgress)
    diagnoses = FailureDiagnoser.from_statechart(
        statechart=statechart
    ).diagnose_failure(failure=failure)
    report("stalled reach after a bad navigation", diagnoses)

    for diagnosis in diagnoses:
        assert isinstance(diagnosis, TaskStalled)
        assert diagnosis.counterfactual == StallCounterfactual.REACHED
        assert diagnosis.verdict == Verdict.ONE_CAUSING_NODE
        [root] = diagnosis.root_causes
        assert isinstance(root.causing_run.node, SetOdometry)
        assert Action.enclosing_action_of(node=root.causing_run.node) is navigate


# %% a check fails because an earlier action closed the gripper it reads


def test_a_failed_gripper_check_is_blamed_on_the_action_that_closed_that_gripper(
    pr2_apartment_context,
) -> None:
    """
    A check that the gripper is open fails because an earlier action closed that
    gripper, and that action is named.
    """
    world, robot, extensions = pr2_apartment_context
    right, left = robot.right_arm.end_effector, robot.left_arm.end_effector
    closing = set_gripper(right, GripperState.CLOSE)
    check = gripper_open_check(right)
    plan = Sequence(
        nodes=[
            set_gripper(right, GripperState.OPEN),
            closing,
            set_gripper(left, GripperState.CLOSE),
            check,
        ]
    )

    statechart, failure = run_until_it_fails(plan, extensions, MotionDidNotFinish)
    diagnoses = FailureDiagnoser.from_statechart(
        statechart=statechart
    ).diagnose_failure(failure=failure)
    report("gripper check after closing that gripper", diagnoses)

    [diagnosis] = diagnoses
    assert diagnosis.origin_run.node is check
    assert isinstance(diagnosis, NodeFailedOnItsOwn)
    assert diagnosis.verdict == Verdict.ONE_CAUSING_NODE
    assert causing_actions(diagnosis) == [closing]


# %% two tasks drive the same joint at once


def test_a_task_stalled_by_another_driving_its_joint_is_blamed_on_it(
    pr2_apartment_context,
) -> None:
    """
    Two tasks of different goals that drive the torso at the same time: the one reported
    as stalled is blamed on the other.
    """
    world, robot, extensions = pr2_apartment_context
    torso = world.get_connection_by_name("torso_lift_joint")
    lower = JointPositionList(
        name="lower torso", goal_state=JointState.from_mapping({torso: 0.05})
    )
    raise_ = JointPositionList(
        name="raise torso", goal_state=JointState.from_mapping({torso: 0.3})
    )
    plan = Sequence(nodes=[Parallel(nodes=[lower, raise_])])

    statechart, failure = run_until_it_fails(plan, extensions, MotionMadeNoProgress)
    diagnoses = FailureDiagnoser.from_statechart(
        statechart=statechart
    ).diagnose_failure(failure=failure)
    report("two tasks on the torso at once", diagnoses)

    blamed = {
        diagnosis.origin_run.node: [
            root.causing_run.node for root in diagnosis.root_causes
        ]
        for diagnosis in diagnoses
    }
    assert blamed == {raise_: [lower]}


def test_tasks_on_different_joints_at_once_are_not_in_conflict(
    pr2_apartment_context,
) -> None:
    """
    Tasks that run at the same time on different joints are not in conflict.
    """
    world, robot, extensions = pr2_apartment_context
    torso = world.get_connection_by_name("torso_lift_joint")
    head = world.get_connection_by_name("head_pan_joint")
    plan = Sequence(
        nodes=[
            Parallel(
                nodes=[
                    JointPositionList(goal_state=JointState.from_mapping({torso: 0.2})),
                    JointPositionList(goal_state=JointState.from_mapping({head: 0.3})),
                ]
            )
        ]
    )
    executor = simulated_executor(extensions)
    executor.compile(statechart=statechart_of(executor, plan))
    executor.execute()

    graph = RunCausalGraph.from_statechart(statechart=executor.statechart)
    assert plan.life_cycle_state == LifeCycleValue.SUCCEEDED
    assert FailureDiagnoser(causal_graph=graph).diagnose_failure(failure=None) == []
    assert [edge for edge in graph.edges if isinstance(edge, ConflictEdge)] == []


# %% a task is failed by a condition reading a check of a gripper an action closed


def test_a_task_failed_by_a_condition_is_blamed_on_what_its_condition_read(
    pr2_apartment_context,
) -> None:
    """
    A task ended by a fail condition is blamed, through that condition, on what made the
    node it reads fail.
    """
    world, robot, extensions = pr2_apartment_context
    right = robot.right_arm.end_effector
    torso = world.get_connection_by_name("torso_lift_joint")
    closing = set_gripper(right, GripperState.CLOSE)
    check = JointPositionReached(
        name="gripper open",
        connection=gripper_open_check(right).connection,
        position=gripper_open_check(right).position,
    )
    lift = JointPositionList(
        name="lift torso", goal_state=JointState.from_mapping({torso: 0.3})
    )
    lift.fail_condition = check.observes_false
    plan = Sequence(nodes=[closing, Parallel(nodes=[check, lift])])

    statechart, failure = run_until_it_fails(plan, extensions, MotionDidNotFinish)
    diagnoses = FailureDiagnoser.from_statechart(
        statechart=statechart
    ).diagnose_failure(failure=failure)
    report("task failed by a condition reading a gripper check", diagnoses)

    [diagnosis] = diagnoses
    assert diagnosis.origin_run.node is lift
    assert isinstance(diagnosis, NodeFailedByACondition)
    assert causing_actions(diagnosis) == [closing]
    [root] = diagnosis.root_causes
    assert [edge.effect.node for edge in root.edges_to_origin] == [check, lift]


# %% a step that fails on its own is blamed on no other node


def test_a_step_failing_on_its_own_inside_a_handled_pick_up_is_its_own_origin(
    pr2_apartment_context,
) -> None:
    """
    A step that fails on its own inside a pick-up with failure handling is where the
    failure started, and no other node is blamed.
    """
    world, robot, extensions = pr2_apartment_context
    milk = world.get_semantic_annotations_by_type(Milk)[0]
    failing_grasp = GraspFailingItsFirstRuns(failing_runs=2)
    plan = Sequence(
        nodes=[
            ParkArmsAction(arms=robot.all_arms),
            NavigateAction(
                planar_goal_pose=Pose2D.from_xy_yaw(
                    x=1.5, y=2.4, reference_frame=world.root
                )
            ),
            HandledOnFailure(
                handled_node=PickUpAction(
                    grasp_candidate=milk.grasp_candidates()[0], arm=robot.right_arm
                ),
                failure_handling_strategy=RetryFailedNode(maximum_attempts=1),
            ),
        ]
    )

    statechart, failure = run_until_it_fails(
        plan,
        [*extensions, PlanRewriting(transformations=[failing_grasp])],
        MotionDidNotFinish,
    )
    diagnoses = FailureDiagnoser.from_statechart(
        statechart=statechart
    ).diagnose_failure(failure=failure)
    report("step failing on its own inside a retried pick-up", diagnoses)

    [inserted_step] = failing_grasp.inserted_steps
    [diagnosis] = diagnoses
    assert diagnosis.origin_run.node is inserted_step
    assert diagnosis.origin_run.sequence_number == 2
    assert diagnosis.verdict == Verdict.NO_OTHER_NODE


# %% the diagnosis runs inside the plan and picks the repair by the cause


def test_a_live_diagnosis_picks_the_repair_for_the_action_that_caused_the_failure(
    pr2_apartment_context,
) -> None:
    """
    Diagnosing a failure while the plan runs lets the failure handling repair the action
    that caused it rather than the one that failed.
    """
    world, robot, extensions = pr2_apartment_context
    right = robot.right_arm.end_effector
    closing = set_gripper(right, GripperState.CLOSE)
    check = gripper_open_check(right)
    recovery = RecoveryByRootCause(
        maximum_attempts=1,
        repairs={
            SetGripperAction: RecoveryByInsertion(
                maximum_attempts=1,
                transformation=OpenTheGripperTheCheckReads(end_effector=right),
            )
        },
    )
    plan = Sequence(
        nodes=[
            closing,
            HandledOnFailure(handled_node=check, failure_handling_strategy=recovery),
        ]
    )
    executor = simulated_executor(extensions)
    executor.compile(statechart=statechart_of(executor, plan))
    executor.execute()

    report("diagnosis inside the running plan", recovery.diagnoses)
    [diagnosis] = recovery.diagnoses
    assert diagnosis.origin_run.node is check
    assert causing_actions(diagnosis) == [closing]
    assert plan.life_cycle_state == LifeCycleValue.SUCCEEDED


# %% negative control: a goal out of reach from anywhere


def test_a_goal_out_of_reach_from_anywhere_is_blamed_on_no_other_node(
    pr2_apartment_context,
) -> None:
    """
    The stalled task reads the base pose an earlier node set, but tried again free to
    move the base it still stalls, so that node is not to blame.
    """
    world, robot, extensions = pr2_apartment_context
    plan = Sequence(
        nodes=[
            ParkArmsAction(arms=robot.all_arms),
            NavigateAction(
                planar_goal_pose=Pose2D.from_xy_yaw(
                    x=1.5, y=2.4, reference_frame=world.root
                )
            ),
            MoveManipulatorAction(
                arm=robot.right_arm,
                goal_pose=Pose.from_xyz_rpy(2.0, 2.4, 3.0, reference_frame=world.root),
            ),
        ]
    )

    statechart, failure = run_until_it_fails(plan, extensions, MotionMadeNoProgress)
    diagnoses = FailureDiagnoser.from_statechart(
        statechart=statechart
    ).diagnose_failure(failure=failure)
    report("goal out of reach from anywhere (negative control)", diagnoses)

    assert [
        (
            type(diagnosis),
            diagnosis.counterfactual,
            diagnosis.verdict,
            diagnosis.root_causes,
        )
        for diagnosis in diagnoses
    ] == [
        (
            TaskStalled,
            StallCounterfactual.STILL_STALLED,
            Verdict.NO_OTHER_NODE,
            [],
        )
    ]


# %% every candidate of an underspecified pick-up fails after a bad navigation


def test_a_pick_up_out_of_candidates_is_blamed_on_the_navigation_they_share(
    pr2_apartment_context,
) -> None:
    """
    An underspecified pick-up whose every candidate fails its trial is blamed on the
    navigation that all candidates started from.
    """
    world, robot, extensions = pr2_apartment_context
    milk = world.get_semantic_annotations_by_type(Milk)[0]
    navigate = NavigateAction(
        planar_goal_pose=Pose2D.from_xy_yaw(
            x=1.5, y=2.4, yaw=math.pi, reference_frame=world.root
        )
    )
    statement = StatementNode(
        statement=a(PickUpAction)(
            grasp_candidate=variable_from(milk.grasp_candidates()),
            arm=robot.right_arm,
        )
    )
    plan = Sequence(nodes=[ParkArmsAction(arms=robot.all_arms), navigate, statement])

    statechart, failure = run_until_it_fails(plan, extensions, EmptyUnderspecified)
    diagnoses = FailureDiagnoser.from_statechart(
        statechart=statechart
    ).diagnose_failure(failure=failure)
    report("underspecified pick-up out of candidates after a bad navigation", diagnoses)

    [diagnosis] = diagnoses
    assert isinstance(diagnosis, StatementRanOutOfCandidates)
    assert causing_actions(diagnosis) == [navigate]


# %% a place releases the grasp the pick-up before it left


def test_a_place_depends_on_the_pick_up_that_attached_its_object(
    pr2_apartment_context,
) -> None:
    """
    The causal graph holds an edge from the pick-up that attached the milk to the
    gripper to the place that releases it, through where the milk is attached.
    """
    world, robot, extensions = pr2_apartment_context
    milk = world.get_semantic_annotations_by_type(Milk)[0]
    pick_up = PickUpAction(
        grasp_candidate=milk.grasp_candidates()[0], arm=robot.right_arm
    )
    place = PlaceAction(
        arm=robot.right_arm,
        graspable=milk,
        goal_pose=Pose.from_xyz_rpy(2.4, 2.2, 1.0, reference_frame=world.root),
    )
    plan = Sequence(
        nodes=[
            ParkArmsAction(arms=robot.all_arms),
            NavigateAction(
                planar_goal_pose=Pose2D.from_xy_yaw(
                    x=1.5, y=2.4, reference_frame=world.root
                )
            ),
            pick_up,
            place,
        ]
    )
    executor = simulated_executor(extensions)
    executor.compile(statechart=statechart_of(executor, plan))
    executor.execute()

    graph = RunCausalGraph.from_statechart(statechart=executor.statechart)
    assert plan.life_cycle_state == LifeCycleValue.SUCCEEDED
    [attached] = [
        edge
        for edge in graph.edges
        if isinstance(edge, WorldStateEdge) and edge.effect.node is place
    ]
    assert isinstance(attached.cause.node, MoveBranch)
    assert Action.enclosing_action_of(node=attached.cause.node) is pick_up
    assert attached.frames == {milk.root}


# %% a wrist turn stalls at the limit an earlier node wound the wrist up to


def tracy_turning_its_tool(
    *,
    world: World,
    wrist_position: float,
    arm_pose: dict[TracyJoint, float],
    angle: float = math.pi / 2,
) -> tuple[list, CartesianPose]:
    """
    :param world: Tracy's world, whose right arm is put in `arm_pose`.
    :param wrist_position: Where the last wrist joint starts.
    :param arm_pose: The pose of the rest of the right arm.
    :param angle: How far to turn the tool about its own axis from where it starts.
    :return: The context extensions of Tracy's plans, and the turn of its right tool,
        to a goal fixed in the world.
    """
    JointState.from_str_dict(
        arm_pose | {TracyJoint.RIGHT_WRIST_3: wrist_position}, world=world
    ).apply_to(world)
    tip = world.get_kinematic_structure_entity_by_name("r_gripper_tool_frame")
    turn = CartesianPose(
        kinematic_chain=KinematicChain(root=world.root, tip=tip),
        goal_pose=world.transform(
            Pose.from_xyz_axis_angle(
                axis=Vector3(z=1), angle=angle, reference_frame=tip
            ),
            world.root,
        ),
    )
    tracy = world.get_semantic_annotations_by_type(Tracy)[0]
    return robot_extensions(tracy), turn


def orientation_task_of(turn: CartesianPose) -> CartesianOrientation:
    """
    :return: The task of `turn` that turns the tool, the one its wrist blocks.
    """
    [orientation] = [
        node for node in turn.nodes if isinstance(node, CartesianOrientation)
    ]
    return orientation


def test_a_turn_stalled_at_a_limit_is_blamed_on_the_node_that_wound_the_joint_up(
    tracy_world_copy: World, tracy_arm_pose: dict[TracyJoint, float]
) -> None:
    """
    A joint move turns the wrist a full turn, which leaves the tool where it was, but
    the tool's turn after it then stalls at the wrist's upper limit.

    Tried again with the wrist turned a full turn back, the turn reaches its goal, so
    the joint move is blamed.
    """
    extensions, turn = tracy_turning_its_tool(
        world=tracy_world_copy,
        wrist_position=WristWinding.UNWOUND.value,
        arm_pose=tracy_arm_pose,
    )
    wrist = tracy_world_copy.get_connection_by_name(TracyJoint.RIGHT_WRIST_3.value)
    wind_up = JointPositionList(
        name="wind up the wrist",
        goal_state=JointState.from_mapping({wrist: WristWinding.WOUND_UP.value}),
    )
    plan = Sequence(nodes=[wind_up, turn])

    # Winding the wrist a full turn takes most of the simulated time a plan of tasks
    # is given by default.
    statechart, failure = run_until_it_fails(
        plan,
        extensions,
        MotionMadeNoProgress,
        simulation_time_per_action=timedelta(seconds=80),
    )
    diagnoses = FailureDiagnoser.from_statechart(
        statechart=statechart
    ).diagnose_failure(failure=failure)
    report("tool turn after winding the wrist up", diagnoses)

    [diagnosis] = [
        diagnosis
        for diagnosis in diagnoses
        if diagnosis.origin_run.node is orientation_task_of(turn)
    ]
    assert (
        type(diagnosis),
        diagnosis.counterfactual,
        [root.causing_run.node for root in diagnosis.root_causes],
        [root.edges_to_origin[0].degrees_of_freedom for root in diagnosis.root_causes],
    ) == (
        TaskStalled,
        StallCounterfactual.REACHED_UNWOUND,
        [wind_up],
        [frozenset({wrist.raw_dof})],
    )


def test_a_turn_stalled_at_a_limit_the_joint_started_at_is_blamed_on_no_other_node(
    tracy_world_copy: World, tracy_arm_pose: dict[TracyJoint, float]
) -> None:
    """
    The wrist starts wound up, so the tool's turn stalls at its upper limit and reaches
    its goal with the wrist turned a full turn back, but no node of the plan wound it
    up.
    """
    extensions, turn = tracy_turning_its_tool(
        world=tracy_world_copy,
        wrist_position=WristWinding.WOUND_UP.value,
        arm_pose=tracy_arm_pose,
    )

    statechart, failure = run_until_it_fails(
        Sequence(nodes=[turn]), extensions, MotionMadeNoProgress
    )
    diagnoses = FailureDiagnoser.from_statechart(
        statechart=statechart
    ).diagnose_failure(failure=failure)
    report("tool turn from a wound up wrist", diagnoses)

    [diagnosis] = [
        diagnosis
        for diagnosis in diagnoses
        if diagnosis.origin_run.node is orientation_task_of(turn)
    ]
    assert (
        type(diagnosis),
        diagnosis.counterfactual,
        diagnosis.verdict,
    ) == (TaskStalled, StallCounterfactual.REACHED_UNWOUND, Verdict.NO_OTHER_NODE)


def test_a_turn_stalled_at_a_short_limit_is_blamed_on_the_node_that_moved_the_joint(
    tracy_world_copy: World, tracy_arm_pose: dict[TracyJoint, float]
) -> None:
    """
    The wrist turns less than a full turn, so it cannot be unwound.

    A joint move takes it close to its upper limit, and the tool's turn after it, taking
    the short way, stalls at that limit. Tried again with the wrist put back where the
    joint move found it, the turn takes the other way and reaches its goal, so the joint
    move is blamed.
    """
    wrist = tracy_world_copy.get_connection_by_name(TracyJoint.RIGHT_WRIST_3.value)
    wrist.raw_dof.limits.lower.position = ShortWrist.LOWER_LIMIT.value
    wrist.raw_dof.limits.upper.position = ShortWrist.UPPER_LIMIT.value
    extensions, turn = tracy_turning_its_tool(
        world=tracy_world_copy,
        wrist_position=ShortWrist.START.value,
        arm_pose=tracy_arm_pose,
        angle=ShortWrist.GOAL.value - math.tau - ShortWrist.START.value,
    )
    move_up = JointPositionList(
        name="move the wrist up",
        goal_state=JointState.from_mapping({wrist: ShortWrist.NEAR_UPPER_LIMIT.value}),
    )

    statechart, failure = run_until_it_fails(
        Sequence(nodes=[move_up, turn]),
        extensions,
        MotionMadeNoProgress,
        simulation_time_per_action=timedelta(seconds=80),
    )
    diagnoses = FailureDiagnoser.from_statechart(
        statechart=statechart
    ).diagnose_failure(failure=failure)
    report("tool turn after moving a short wrist up", diagnoses)

    [diagnosis] = [
        diagnosis
        for diagnosis in diagnoses
        if diagnosis.origin_run.node is orientation_task_of(turn)
    ]
    assert (
        type(diagnosis),
        diagnosis.counterfactual,
        [root.causing_run.node for root in diagnosis.root_causes],
        [root.edges_to_origin[0].degrees_of_freedom for root in diagnosis.root_causes],
    ) == (
        TaskStalled,
        StallCounterfactual.REACHED_FROM_EARLIER_POSITIONS,
        [move_up],
        [frozenset({wrist.raw_dof})],
    )


def test_only_a_revolute_joint_whose_range_spans_a_full_turn_is_unwound(
    pr2_apartment_context,
) -> None:
    """
    The PR2's torso slides and its elbow turns less than a full turn, so neither is
    turned a full turn back from a limit blocking a task.
    """
    world, robot, extensions = pr2_apartment_context
    torso = world.get_connection_by_name("torso_lift_joint")
    elbow = world.get_connection_by_name("r_elbow_flex_joint")
    task = JointPositionList(goal_state=JointState.from_mapping({torso: 0.2}))

    assert (
        TaskWithJointsUnwound.unwindable(
            world=world,
            blocked_position_limits=[
                BlockedPositionLimit(
                    degree_of_freedom=joint.raw_dof,
                    side=PositionLimitSide.UPPER,
                    task=task,
                )
                for joint in (torso, elbow)
            ],
        )
        == []
    )


# %% the same diagnoses in physics


@pytest.mark.skipif(
    not runs_in_continuous_integration(), reason="MuJoCo tests only run in CI"
)
def test_in_physics_a_gripper_check_is_blamed_on_the_action_that_closed_it(
    simulated_hsrb,
) -> None:
    """
    In MuJoCo, a gripper check that fails is blamed on the action that closed that
    gripper.
    """
    hsrb, simulation = simulated_hsrb
    [arm] = hsrb.all_arms
    closing = set_gripper(arm.end_effector, GripperState.CLOSE)
    check = gripper_open_check(arm.end_effector)
    plan = Sequence(nodes=[ParkArmsAction(arms=hsrb.all_arms), closing, check])

    statechart, failure = perform_in_physics_until_it_fails(
        plan, robot_extensions(hsrb), simulation, MotionDidNotFinish
    )
    diagnoses = FailureDiagnoser.from_statechart(
        statechart=statechart
    ).diagnose_failure(failure=failure)
    report("physics: gripper check after closing that gripper", diagnoses)

    [diagnosis] = diagnoses
    assert diagnosis.origin_run.node is check
    assert causing_actions(diagnosis) == [closing]


@pytest.mark.skipif(
    not runs_in_continuous_integration(), reason="MuJoCo tests only run in CI"
)
def test_in_physics_two_tasks_driving_one_joint_are_blamed_on_each_other(
    simulated_hsrb,
) -> None:
    """
    In MuJoCo, two tasks of different goals that drive one joint at the same time are
    each blamed on the other.
    """
    hsrb, simulation = simulated_hsrb
    lift = hsrb._world.get_connection_by_name("arm_lift_joint")
    lower = JointPositionList(
        name="lower lift", goal_state=JointState.from_mapping({lift: 0.05})
    )
    raise_ = JointPositionList(
        name="raise lift", goal_state=JointState.from_mapping({lift: 0.4})
    )
    plan = Sequence(nodes=[Parallel(nodes=[lower, raise_])])

    statechart, failure = perform_in_physics_until_it_fails(
        plan, robot_extensions(hsrb), simulation, MotionMadeNoProgress
    )
    diagnoses = FailureDiagnoser.from_statechart(
        statechart=statechart
    ).diagnose_failure(failure=failure)
    report("physics: two tasks on the lift joint at once", diagnoses)

    blamed = {
        d.origin_run.node: [root.causing_run.node for root in d.root_causes]
        for d in diagnoses
    }
    assert blamed == {lower: [raise_], raise_: [lower]}
