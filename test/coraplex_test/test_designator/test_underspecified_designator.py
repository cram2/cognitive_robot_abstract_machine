from dataclasses import dataclass
from uuid import UUID, uuid4

import pytest
from typing_extensions import Callable, Dict, Iterator, List

from krrood.entity_query_language.backends import (
    EntityQueryLanguageGenerativeBackend,
    ProbabilisticBackend,
)
from krrood.entity_query_language.factories import a, variable, variable_from
from cramph.composites import ChildChooserAccess, Sequence, StatementNode
from cramph.data_types import LifeCycleValues
from cramph.exceptions import MissingExecutorExtensionError
from cramph.node import StatechartNode

from coraplex.datastructures.enums import ActionTrialVisualization
from coraplex.plans.failures import (
    CandidateLimitReached,
    EmptyUnderspecified,
    MotionViolatedCollisionAvoidance,
    PlanFailure,
)
from giskardpy.motion_statechart.exceptions import CollisionViolatedError
from cramph.threaded_nodes import FunctionCall
from cramph.state_history import StateHistory, StateHistoryObserver
from coraplex.plans.underspecified import (
    ActionTrial,
    UnderspecifiedChildChooser,
)
from coraplex.robot_plans.actions.base import SequentialAction
from coraplex.robot_plans.actions.core.navigation import NavigateAction
from coraplex.robot_plans.actions.core.pick_up import PickUpAction, ReachAction
from cramph.visualization import RvizVisualization
from giskardpy.motion_statechart.monitors.overwrite_state_monitors import SetOdometry
from giskardpy.motion_statechart.tasks.cartesian_tasks import (
    CartesianPosition,
    CartesianPositionStraight,
)
from semantic_digital_twin.datastructures.prefixed_name import PrefixedName
from semantic_digital_twin.robots.robot_parts import Arm
from semantic_digital_twin.robots.stretch import Stretch
from semantic_digital_twin.semantic_annotations.semantic_annotations import (
    Handle,
    Milk,
)
from semantic_digital_twin.spatial_types.spatial_types import Pose
from semantic_digital_twin.world import World
from semantic_digital_twin.world_description.connections import FixedConnection
from semantic_digital_twin.world_description.world_entity import Body
from ...plan_running import run_plan, simulated_executor, statechart_of, with_grounding
from coraplex.plans.context_extensions import StatementGrounding
from coraplex.plans.executors import KinematicallySimulatedPlanExecutor

# %% mimics for testing candidate trials without depending on real motion physics


@dataclass
class TrialCall:
    """
    One recorded attempt of a :class:`RecordingAction`.
    """

    world: World
    """
    The world the attempt actually ran against, so a test can tell a trial copy
    from the real world by identity.
    """

    position_at_entry: float
    """
    The value of the probed degree of freedom when this attempt started, so a test can
    tell whether a later trial copy reflects an earlier real failure's state.
    """


class TrialProbe:
    """
    Records every attempt of a :class:`RecordingAction` across both trial and real
    execution.
    """

    def __init__(self):
        self.calls: list[TrialCall] = []


_registered_probes: dict[UUID, TrialProbe] = {}
"""
:class:`TrialProbe` instances, keyed by the id a :class:`RecordingAction` carries as
:attr:`~RecordingAction.probe_key`.

:meth:`~semantic_digital_twin.world.World.rebind_world_entities` deep-copies every value it does not recognize as a world entity, so a
rebound :class:`RecordingAction` cannot share a probe handed to it directly as a field - the
same isolation that makes a trial safe for real designators. Reaching the probe
out-of-band by an id, whose value survives copying even though its identity need not,
is what lets a test observe a candidate's trial and its later real attempt as one
sequence.
"""


@pytest.fixture(autouse=True)
def release_registered_probes() -> Iterator[None]:
    """
    Drop every probe a test registered once it has finished.

    A probe keeps the worlds its calls ran against, so a probe left registered would
    keep them in memory for the rest of the session.
    """
    yield
    _registered_probes.clear()


def register_probe() -> UUID:
    """
    Register a fresh :class:`TrialProbe` and return the id a :class:`RecordingAction`
    should carry to record into it.
    """
    key = uuid4()
    _registered_probes[key] = TrialProbe()
    return key


@dataclass(eq=False, repr=False, kw_only=True)
class RecordingAction(SequentialAction):
    """
    An action whose execution deterministically records itself and can be made to fail
    on a specific attempt, for testing the trial-then-real handling of an underspecified
    node's candidates without depending on real motion physics.
    """

    probe_key: UUID
    """
    Id of the :class:`TrialProbe` this action's attempts are recorded to.
    """

    dof_id: UUID
    """
    Id of the degree of freedom this action mutates on every attempt, to make world
    state changes observable.
    """

    fail_on_attempt_number: int | None = None
    """
    Report :attr:`reported_failure` once the probe has recorded this many calls; never
    fail if None.
    """

    reported_failure: Callable[[], PlanFailure] = PlanFailure
    """
    Builds the failure a failing attempt reports.
    """

    def create_steps(self) -> list[StatechartNode]:
        return [
            FunctionCall(function=self._record_attempt, failure_types=(PlanFailure,))
        ]

    def _record_attempt(self) -> None:
        """
        Record the world this attempt runs against and mutate the probed degree of
        freedom, failing on the configured attempt.
        """
        world = self.world
        probe = _registered_probes[self.probe_key]
        probe.calls.append(
            TrialCall(
                world=world,
                position_at_entry=world.state[self.dof_id].position,
            )
        )
        world.state[self.dof_id].position = len(probe.calls)
        world.notify_state_change()
        if len(probe.calls) == self.fail_on_attempt_number:
            raise self.reported_failure()


def test_underspecified_action(apartment_world_pr2_copy_with_context):
    """
    Test that an underspecified action resolves to a concrete candidate, which runs as
    the node's child.
    """
    world, robot, extensions = apartment_world_pr2_copy_with_context
    action = a(NavigateAction)(
        target_location=variable_from(
            [
                Pose.from_xyz_quaternion(1, -1, 0, reference_frame=world.root),
                Pose.from_xyz_quaternion(2, -1, 0, reference_frame=world.root),
            ]
        ),
    )

    plan = StatementNode(statement=action)
    run_plan(plan, extensions)

    assert plan.life_cycle_state == LifeCycleValues.SUCCEEDED
    assert isinstance(plan.grounded_nodes[-1], NavigateAction)


def test_underspecified_action_with_ellipsis(apartment_world_pr2_copy_with_context):
    """
    Test that an underspecified action resolves when a factory for a spatial type is
    used with ellipsis.
    """
    world, robot, extensions = apartment_world_pr2_copy_with_context
    extensions = with_grounding(extensions, query_backend=ProbabilisticBackend())
    action = a(NavigateAction)(
        target_location=a(Pose.from_xyz_rpy)(
            x=...,
            y=...,
            z=0.0,
            roll=0.0,
            pitch=0.0,
            yaw=0.0,
            reference_frame=robot.root,
        ),
    )

    plan = StatementNode(statement=action)
    run_plan(plan, extensions)

    assert plan.life_cycle_state == LifeCycleValues.SUCCEEDED
    assert isinstance(plan.grounded_nodes[-1], NavigateAction)


def test_underspecified_language(apartment_world_pr2_copy_with_context):
    """
    Test that entire plans can be underspecified.
    """
    world, robot, extensions = apartment_world_pr2_copy_with_context
    milk = world.get_semantic_annotations_by_type(Milk)[0]
    plan_generator = a(Sequence)(
        nodes=[
            a(NavigateAction)(
                target_location=(
                    target_locations := variable_from(
                        [
                            Pose.from_xyz_quaternion(
                                1, 0, 0, reference_frame=world.root
                            ),
                            Pose.from_xyz_quaternion(
                                2, 0, 0, reference_frame=world.root
                            ),
                        ]
                    )
                ),
            ),
            a(PickUpAction)(
                arm=variable(Arm, domain=robot.all_arms),
                grasp=milk.grasp_candidates()[0],
            ),
        ],
    )
    plans = list(EntityQueryLanguageGenerativeBackend().evaluate(plan_generator))
    assert len(plans) == len(list(target_locations._domain_)) * len(robot.all_arms)


# %% candidate trials


def test_isolation_rejected_candidate_never_touches_real_world(
    apartment_world_pr2_copy_with_context,
):
    """
    A candidate that only ever fails must be rejected during its trial, against a
    disposable copy of the world, and never attached to the plan or executed against the
    real world; only the candidate that survives its trial is executed for real.
    """
    world, robot, extensions = apartment_world_pr2_copy_with_context
    dof = world.degrees_of_freedom[0]
    probe_key = register_probe()

    action = a(RecordingAction)(
        probe_key=probe_key,
        dof_id=dof.id,
        fail_on_attempt_number=variable_from([1, None]),
    )
    plan = StatementNode(statement=action)
    run_plan(plan, extensions)

    assert plan.life_cycle_state == LifeCycleValues.SUCCEEDED
    assert len(plan.chosen_children) == 1
    assert plan.grounded_nodes[0].fail_on_attempt_number is None

    probe = _registered_probes[probe_key]
    assert len(probe.calls) == 3
    # candidate 1's trial: the only attempt it ever gets, and it is a copy.
    assert probe.calls[0].world is not world
    # candidate 2's trial: still a copy, restored to how candidate 1 found it.
    assert probe.calls[1].world is not world
    assert probe.calls[1].position_at_entry == probe.calls[0].position_at_entry
    # candidate 2's real attempt: the actual world.
    assert probe.calls[2].world is world

    # candidate 1's rejected trial only ever mutated its own throwaway copy.
    assert world.state[dof.id].position == 3


def test_rejected_candidates_are_tried_against_one_copy(
    apartment_world_pr2_copy_with_context,
):
    """
    Candidates that follow a rejected one are tried against the copy that rejection was
    made in, rather than each candidate copying the world again.

    Nothing has changed the real world between them, so the copy still describes it and
    rolling it back is enough to give the next candidate the same starting point.
    """
    world, robot, extensions = apartment_world_pr2_copy_with_context
    dof = world.degrees_of_freedom[0]
    probe_key = register_probe()

    action = a(RecordingAction)(
        probe_key=probe_key,
        dof_id=dof.id,
        fail_on_attempt_number=variable_from([1, 2, None]),
    )
    plan = StatementNode(statement=action)
    run_plan(plan, extensions)

    probe = _registered_probes[probe_key]
    # every call but the last is a trial; the last is the accepted candidate's real
    # attempt, which runs against the real world.
    trials = probe.calls[:-1]
    assert len(trials) == 3
    assert trials[0].world is not world
    assert trials[1].world is trials[0].world
    assert trials[2].world is trials[0].world


def test_real_failure_keeps_state_and_next_trial_reflects_it(
    apartment_world_pr2_copy_with_context,
):
    """
    A candidate that passes its trial but then fails for real must leave the real
    world's state exactly as the failed attempt left it; the next candidate's trial copy
    must be taken from that post-failure state, not the original one.
    """
    world, robot, extensions = apartment_world_pr2_copy_with_context
    dof = world.degrees_of_freedom[0]
    initial_position = world.state[dof.id].position
    probe_key = register_probe()

    action = a(RecordingAction)(
        probe_key=probe_key,
        dof_id=dof.id,
        fail_on_attempt_number=variable_from([2, None]),
    )
    plan = StatementNode(statement=action)
    run_plan(plan, extensions)

    assert plan.life_cycle_state == LifeCycleValues.SUCCEEDED
    # Both the failed and the accepted candidate are attached to the tree - a real
    # failure is not undone, only worked around by trying the next candidate.
    assert [action.fail_on_attempt_number for action in plan.grounded_nodes] == [
        2,
        None,
    ]

    probe = _registered_probes[probe_key]
    assert len(probe.calls) == 4
    # candidate 1's trial: a copy, starting from the untouched world.
    assert probe.calls[0].world is not world
    assert probe.calls[0].position_at_entry == initial_position
    # candidate 1's real attempt: the actual world, still untouched by the trial,
    # mutated and then failed.
    assert probe.calls[1].world is world
    assert probe.calls[1].position_at_entry == initial_position
    # candidate 2's trial: a fresh copy, taken *after* candidate 1's real failure -
    # it must already carry that mutation.
    assert probe.calls[2].world is not world
    assert probe.calls[2].position_at_entry == 2
    # candidate 2's real attempt: the actual world, still carrying candidate 1's
    # failed-attempt mutation, since nothing rolled it back.
    assert probe.calls[3].world is world
    assert probe.calls[3].position_at_entry == 2

    assert world.state[dof.id].position == 4


def test_the_underspecified_steps_of_one_plan_are_tried_against_one_copy(
    apartment_world_pr2_copy_with_context,
):
    """
    Every step of a plan is tried in the same copy, caught up with what the steps before
    it did for real.
    """
    world, robot, extensions = apartment_world_pr2_copy_with_context
    dof = world.degrees_of_freedom[0]
    probe_key = register_probe()
    plan = Sequence(
        nodes=[
            StatementNode(
                statement=a(RecordingAction)(
                    probe_key=probe_key,
                    dof_id=dof.id,
                    fail_on_attempt_number=variable_from([None]),
                )
            )
            for _ in range(2)
        ]
    )

    run_plan(plan, extensions)

    probe = _registered_probes[probe_key]
    first_trial, first_real_attempt, second_trial, second_real_attempt = probe.calls
    assert first_trial.world is not world
    assert second_trial.world is first_trial.world
    assert second_trial.position_at_entry == second_real_attempt.position_at_entry
    assert second_trial.position_at_entry != first_trial.position_at_entry


@dataclass
class _FailsWhenThePlanEnds(StateHistoryObserver):
    """
    An observer that fails once the root of the plan it observes ends.
    """

    plan: StatechartNode
    """
    The root of the observed plan.
    """

    def on_state_change(self, history: StateHistory) -> None:
        if self.plan in history.nodes_ended_in_latest_item():
            raise RuntimeError("observer failed at the end of the plan")


def _plan_of_two_underspecified_steps(world: World) -> Sequence:
    """
    :return: A plan of two underspecified steps that both succeed.
    """
    dof = world.degrees_of_freedom[0]
    probe_key = register_probe()
    return Sequence(
        nodes=[
            StatementNode(
                statement=a(RecordingAction)(
                    probe_key=probe_key,
                    dof_id=dof.id,
                    fail_on_attempt_number=variable_from([None]),
                )
            )
            for _ in range(2)
        ]
    )


def _child_chooser_of(plan: StatechartNode) -> UnderspecifiedChildChooser:
    """
    :return: The chooser grounding the underspecified steps of the statechart running
        `plan`.
    """
    [chooser] = plan.statechart.context.require_extension(ChildChooserAccess).choosers
    return chooser


def test_a_plan_releases_its_trial_copy_once_it_has_run(
    apartment_world_pr2_copy_with_context,
):
    """
    The copy of the world the steps were tried in is only needed while the plan runs.
    """
    world, robot, extensions = apartment_world_pr2_copy_with_context
    plan = _plan_of_two_underspecified_steps(world)

    run_plan(plan, extensions)

    assert _child_chooser_of(plan).trial._copied_world is None


def test_a_plan_releases_its_trial_copy_even_when_an_observer_fails(
    apartment_world_pr2_copy_with_context,
):
    """
    An observer failing as the plan ends must not keep the copy alive past the plan.
    """
    world, robot, extensions = apartment_world_pr2_copy_with_context
    plan = _plan_of_two_underspecified_steps(world)
    executor = simulated_executor(extensions)
    statechart = statechart_of(executor, plan)
    statechart.state_history.add_observer(_FailsWhenThePlanEnds(plan=plan))

    executor.compile(statechart)
    with pytest.raises(RuntimeError):
        executor.execute()

    assert _child_chooser_of(plan).trial._copied_world is None


def test_a_trial_catches_its_copy_up_with_what_the_world_gained(
    apartment_world_pr2_copy_with_context,
):
    """
    A body added after the copy was taken appears in the same copy once caught up.
    """
    world, robot, extensions = apartment_world_pr2_copy_with_context
    trial = ActionTrial(
        executor=simulated_executor(extensions),
        trial_executor_type=KinematicallySimulatedPlanExecutor,
    )
    copied = trial._copy()
    body = Body(name=PrefixedName("added_after_the_copy"))
    with world.modify_world():
        world.add_kinematic_structure_entity(body)
        world.add_connection(FixedConnection(parent=world.root, child=body))

    caught_up = trial._copy()

    assert caught_up is copied
    assert caught_up.get_kinematic_structure_entity_by_id(body.id).name == body.name
    trial.discard()


# %% how many candidates a step tries


def test_a_step_gives_up_after_as_many_candidates_as_the_context_allows(
    apartment_world_pr2_copy_with_context,
):
    world, robot, extensions = apartment_world_pr2_copy_with_context
    extensions = with_grounding(extensions, candidates_to_try=2)
    probe_key = register_probe()
    plan = StatementNode(
        statement=a(RecordingAction)(
            probe_key=probe_key,
            dof_id=world.degrees_of_freedom[0].id,
            fail_on_attempt_number=variable_from([1, 2, 3]),
        )
    )
    executor = simulated_executor(extensions)

    with pytest.raises(CandidateLimitReached) as failure:
        executor.compile(statechart_of(executor, plan))
        executor.execute()

    assert (
        failure.value.candidate_limit
        == next(
            e for e in extensions if isinstance(e, StatementGrounding)
        ).candidates_to_try
    )
    assert (
        len(_registered_probes[probe_key].calls)
        == next(
            e for e in extensions if isinstance(e, StatementGrounding)
        ).candidates_to_try
    )


def test_a_step_keeps_its_own_limit_over_the_contexts(
    apartment_world_pr2_copy_with_context,
):
    world, robot, extensions = apartment_world_pr2_copy_with_context
    extensions = with_grounding(extensions, candidates_to_try=2)
    plan = StatementNode(
        statement=a(RecordingAction)(
            probe_key=register_probe(),
            dof_id=world.degrees_of_freedom[0].id,
            fail_on_attempt_number=variable_from([1, 2, None]),
        ).limit(3)
    )
    executor = simulated_executor(extensions)

    executor.compile(statechart_of(executor, plan))
    executor.execute()

    assert plan.life_cycle_state == LifeCycleValues.SUCCEEDED


def test_a_step_that_runs_out_of_candidates_below_its_limit_says_it_is_empty(
    apartment_world_pr2_copy_with_context,
):
    world, robot, extensions = apartment_world_pr2_copy_with_context
    plan = StatementNode(
        statement=a(RecordingAction)(
            probe_key=register_probe(),
            dof_id=world.degrees_of_freedom[0].id,
            fail_on_attempt_number=variable_from([1, 2]),
        )
    )
    executor = simulated_executor(extensions)

    with pytest.raises(EmptyUnderspecified) as failure:
        executor.compile(statechart_of(executor, plan))
        executor.execute()

    assert type(failure.value) is EmptyUnderspecified


# %% a trial copy is published while debugging


@pytest.fixture
def debugging_executor(apartment_world_pr2_copy_with_context, rclpy_node):
    """
    The apartment with a PR2, its robot and an executor debugging its plans.
    """
    world, robot, extensions = apartment_world_pr2_copy_with_context
    yield world, robot, simulated_executor(extensions, ros_node=rclpy_node, debug=True)


def _compiled_trial_executor(
    trial: ActionTrial,
) -> tuple[KinematicallySimulatedPlanExecutor, World]:
    """
    :param trial: The trial whose executor is built.
    :return: An executor of `trial` with one candidate compiled in the copy, and that
        copy.
    """
    copied = trial._copy()
    executor = trial._trial_executor(copied)
    candidate = RecordingAction(
        probe_key=register_probe(), dof_id=copied.degrees_of_freedom[0].id
    )
    executor.compile(statechart_of(executor, Sequence(nodes=[candidate])))
    return executor, copied


def test_a_trial_publishes_its_copy_while_debugging(debugging_executor) -> None:
    """
    The candidates are tried in the copy, so a run being watched would otherwise show
    the robot standing still through every candidate it rejects.
    """
    world, robot, executor = debugging_executor
    trial = ActionTrial(
        executor=executor, trial_executor_type=KinematicallySimulatedPlanExecutor
    )

    trial_executor, copied = _compiled_trial_executor(trial)

    visualization = trial_executor.require_extension(RvizVisualization)
    assert visualization.is_rendering
    assert visualization.publisher._world is copied
    trial_executor.execute()
    trial.discard()


def test_a_trial_publishes_its_copy_apart_from_the_world_it_copies(
    debugging_executor,
) -> None:
    """
    The copy has the same frame names and markers as the world it was taken from, so it
    is published under a prefix and on a topic of its own rather than over that world.
    """
    world, robot, executor = debugging_executor
    trial = ActionTrial(
        executor=executor, trial_executor_type=KinematicallySimulatedPlanExecutor
    )

    trial_executor, copied = _compiled_trial_executor(trial)

    publisher = trial_executor.require_extension(RvizVisualization).publisher
    assert (
        publisher.tf_publisher.frame_names.prefix
        == ActionTrialVisualization.FRAME_PREFIX
    )
    assert publisher.topic_name == ActionTrialVisualization.MARKER_TOPIC
    trial_executor.execute()
    trial.discard()


def test_a_trial_copy_is_drawn_see_through(debugging_executor) -> None:
    """
    The copy is drawn translucent so that it can be told apart from the world it copies
    where the two overlap.
    """
    world, robot, executor = debugging_executor
    trial = ActionTrial(
        executor=executor, trial_executor_type=KinematicallySimulatedPlanExecutor
    )

    trial_executor, copied = _compiled_trial_executor(trial)

    publisher = trial_executor.require_extension(RvizVisualization).publisher
    assert publisher.alpha == trial.copy_marker_alpha
    trial_executor.execute()
    trial.discard()


def test_a_trial_publishes_nothing_without_debugging(
    apartment_world_pr2_copy_with_context,
):
    """
    Publishing the copy is a debugging aid, so a run that is not debugging publishes
    nothing.
    """
    world, robot, extensions = apartment_world_pr2_copy_with_context
    trial = ActionTrial(
        executor=simulated_executor(extensions),
        trial_executor_type=KinematicallySimulatedPlanExecutor,
    )

    trial_executor = trial._trial_executor(trial._copy())

    with pytest.raises(MissingExecutorExtensionError):
        trial_executor.require_extension(RvizVisualization)
    trial.discard()


def test_a_trial_adapts_a_candidate_to_the_robot_as_its_executor_does(
    apartment_world_pr2_copy_with_context,
) -> None:
    """
    A candidate is tried the way the robot will perform it, so the trial keeps the robot
    adaptations its executor was given, apart from how the base drives.
    """
    world, robot, extensions = apartment_world_pr2_copy_with_context
    executor = simulated_executor(extensions, robot_adaptations=[])
    trial = ActionTrial(
        executor=executor, trial_executor_type=KinematicallySimulatedPlanExecutor
    )

    trial_executor = trial._trial_executor(trial._copy())

    assert trial_executor.robot_adaptations == executor.robot_adaptations
    trial.discard()


def test_a_trial_keeps_how_its_executor_drives_a_differentially_driven_base(
    stretch_apartment_context,
) -> None:
    """
    The trial overrides how the base gets to its goal with a transformation of its own,
    so it keeps every robot adaptation of its executor.
    """
    world, robot, extensions = stretch_apartment_context
    executor = simulated_executor(extensions)
    trial = ActionTrial(
        executor=executor, trial_executor_type=KinematicallySimulatedPlanExecutor
    )

    trial_executor = trial._trial_executor(trial._copy())

    assert trial_executor.robot_adaptations == executor.robot_adaptations
    trial.discard()


def test_a_trial_puts_a_differentially_driven_base_at_its_goal(
    stretch_apartment_context,
) -> None:
    """
    A trial only decides whether a candidate can succeed, so it puts the base where the
    candidate drives it instead of spending the time driving there.
    """
    world, robot, extensions = stretch_apartment_context
    trial = ActionTrial(
        executor=simulated_executor(extensions),
        trial_executor_type=KinematicallySimulatedPlanExecutor,
    )
    copied = trial._copy()
    trial_executor = trial._trial_executor(copied)
    navigation = NavigateAction(
        target_location=Pose.from_xyz_rpy(2.0, 2.0, 0.0, reference_frame=copied.root)
    )

    trial_executor.prepare(statechart_of(trial_executor, navigation))

    assert type(navigation.goal) is SetOdometry
    trial.discard()


def test_a_trial_moves_the_tool_center_point_the_way_the_robot_does(
    stretch_apartment_context,
) -> None:
    world, robot, extensions = stretch_apartment_context
    trial = ActionTrial(
        executor=simulated_executor(extensions),
        trial_executor_type=KinematicallySimulatedPlanExecutor,
    )
    copied = trial._copy()
    trial_executor = trial._trial_executor(copied)
    reach = ReachAction(
        grasp=copied.get_semantic_annotations_by_type(Handle)[0].grasp_candidates()[0],
        arm=copied.get_semantic_annotations_by_type(Stretch)[0].all_arms[0],
    )

    trial_executor.prepare(statechart_of(trial_executor, reach))

    tool_frame = reach.arm.end_effector.tool_frame
    assert [
        type(node)
        for node in reach.descendants
        if isinstance(node, (CartesianPosition, CartesianPositionStraight))
        and node.tip_link is tool_frame
    ] == [CartesianPositionStraight] * 2
    trial.discard()


def test_a_trial_stops_publishing_its_copy_once_the_candidate_has_run(
    debugging_executor,
) -> None:
    """
    The copy is only published while a candidate runs in it.
    """
    world, robot, executor = debugging_executor
    trial = ActionTrial(
        executor=executor, trial_executor_type=KinematicallySimulatedPlanExecutor
    )
    trial_executor, copied = _compiled_trial_executor(trial)

    trial_executor.execute()

    assert not trial_executor.require_extension(RvizVisualization).is_rendering
    trial.discard()


def test_a_trial_tries_an_action_that_already_belongs_to_a_plan(
    debugging_executor,
) -> None:
    """
    An action in a statechart reaches that statechart's extensions, and with it the ROS
    node the run publishes through, which a trial must not try to copy.
    """
    world, robot, executor = debugging_executor
    stand_where_it_is = robot.root.global_pose
    action = NavigateAction(target_location=stand_where_it_is)
    statechart_of(executor, Sequence(nodes=[action]))
    trial = ActionTrial(
        executor=executor, trial_executor_type=KinematicallySimulatedPlanExecutor
    )

    assert trial.succeeds(action)
    trial.discard()


# %% a candidate whose motion could not keep its clearance


def a_collision_that_could_not_be_avoided() -> MotionViolatedCollisionAvoidance:
    """
    :return: The failure a motion reports when it could not keep the clearance it was
        given.
    """
    return MotionViolatedCollisionAvoidance(
        CollisionViolatedError(violated_collisions=[])
    )


def test_a_candidate_that_cannot_keep_its_clearance_is_discarded(
    apartment_world_pr2_copy_with_context,
):
    """
    A motion that could not keep its clearance says this candidate does not work here,
    not that the plan cannot go on, so grounding moves on to the next candidate instead
    of the failure escaping the trial.
    """
    world, robot, extensions = apartment_world_pr2_copy_with_context
    dof = world.degrees_of_freedom[0]
    probe_key = register_probe()

    action = a(RecordingAction)(
        probe_key=probe_key,
        dof_id=dof.id,
        fail_on_attempt_number=variable_from([1, None]),
        reported_failure=variable_from([a_collision_that_could_not_be_avoided]),
    )
    plan = StatementNode(statement=action)
    run_plan(plan, extensions)

    assert plan.life_cycle_state == LifeCycleValues.SUCCEEDED
    assert plan.grounded_nodes[-1].fail_on_attempt_number is None
