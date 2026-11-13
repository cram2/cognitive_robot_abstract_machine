import threading
from dataclasses import dataclass
from enum import Enum
from uuid import UUID, uuid4

import pytest
from typing_extensions import Callable, ClassVar, Dict, Iterator, List

from krrood.entity_query_language.backends import (
    EntityQueryLanguageGenerativeBackend,
    ProbabilisticBackend,
)
from krrood.entity_query_language.factories import a, variable, variable_from
from cramph.composites import (
    ChildChooserAccess,
    Parallel,
    Sequence,
    StatementNode,
)
from cramph.context import ContextExtension
from cramph.data_types import LifeCycleValue
from cramph.exceptions import MissingExecutorExtensionError
from cramph.monitors import CountTicks
from cramph.node import StatechartNode

from coraplex.datastructures.enums import ActionTrialVisualization
from coraplex.exceptions import MotionDidNotFinish
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
    CandidateHandOver,
    SubtreeRebuild,
    UnderspecifiedChildChooser,
)
from krrood.entity_query_language.query.match import Match
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
from krrood_spatial_math.spatial_types import Pose
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


# %% mimics claiming slots side by side


class SlotPosition(float, Enum):
    """
    The positions a degree of freedom standing for a slot is put at.
    """

    CLAIMED = 0.4242
    """
    The slot is taken, unlike at any position a degree of freedom of the robot starts
    at.
    """


@dataclass(eq=False, repr=False, kw_only=True)
class ClaimSlot(FunctionCall):
    """
    Claims one of the slots that degrees of freedom stand for, in the world of the
    statechart it runs in, and fails if one claimed it first.

    It records every attempt to its probe, if it has one, and fails after claiming on
    the attempt it is told to, as a grasp knocking the object over would.
    """

    _claiming: ClassVar[threading.Lock] = threading.Lock()
    """
    Makes looking at a slot and claiming it one step for every instance, since the
    nodes started on one tick run side by side in threads of their own.
    """

    slot: int
    """
    The index of the slot this node claims.
    """

    slot_dof_ids: tuple[UUID, ...]
    """
    Ids of the degrees of freedom that stand for the slots.
    """

    probe_key: UUID | None = None
    """
    Id of the :class:`TrialProbe` the attempts are recorded to, None for none.
    """

    fail_on_attempt_number: int | None = None
    """
    Fail after claiming once the probe has recorded this many attempts; never if None.
    """

    function: Callable[[], None] | None = None
    """
    Unused, since :meth:`run` claims the slot.
    """

    failure_types: tuple[type[BaseException], ...] = (PlanFailure,)

    def run(self) -> None:
        world = self.context.world
        state = world.state[self.slot_dof_ids[self.slot]]
        if self.probe_key is not None:
            probe = _registered_probes[self.probe_key]
            probe.calls.append(TrialCall(world=world, position_at_entry=state.position))
        with self._claiming:
            if state.position == SlotPosition.CLAIMED:
                raise PlanFailure()
            state.position = SlotPosition.CLAIMED
            world.notify_state_change()
        if (
            self.probe_key is not None
            and len(probe.calls) == self.fail_on_attempt_number
        ):
            raise PlanFailure()


@dataclass(eq=False, repr=False, kw_only=True)
class ClaimSlotAction(SequentialAction):
    """
    Claims one of the slots, after waiting a number of ticks, see :class:`ClaimSlot`.

    Two of them only succeed side by side when they claim different slots, although
    each succeeds on its own with any free slot.
    """

    slot: int
    """
    The index of the slot this action claims.
    """

    slot_dof_ids: tuple[UUID, ...]
    """
    Ids of the degrees of freedom that stand for the slots.
    """

    ticks_before_claiming: int = 0
    """
    How many ticks the action waits before it claims the slot.
    """

    probe_key: UUID | None = None
    """
    Id of the :class:`TrialProbe` the attempts are recorded to, None for none.
    """

    fail_on_attempt_number: int | None = None
    """
    Fail after claiming once the probe has recorded this many attempts; never if None.
    """

    def create_steps(self) -> list[StatechartNode]:
        waiting = (
            [CountTicks(ticks=self.ticks_before_claiming)]
            if self.ticks_before_claiming > 0
            else []
        )
        return [
            *waiting,
            ClaimSlot(
                slot=self.slot,
                slot_dof_ids=self.slot_dof_ids,
                probe_key=self.probe_key,
                fail_on_attempt_number=self.fail_on_attempt_number,
            ),
        ]


@dataclass(eq=False, repr=False, kw_only=True)
class ClaimSlotThroughAStatementAction(SequentialAction):
    """
    Claims a slot through a statement among its steps, as an action leaves part of
    what it does to be grounded once that part starts.
    """

    slot: int
    """
    The index of the slot this action claims.
    """

    slot_dof_ids: tuple[UUID, ...]
    """
    Ids of the degrees of freedom that stand for the slots.
    """

    def create_steps(self) -> list[StatechartNode]:
        return [
            StatementNode(
                statement=a(ClaimSlotAction)(
                    slot=self.slot, slot_dof_ids=self.slot_dof_ids
                )
            )
        ]


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

    assert plan.life_cycle_state == LifeCycleValue.SUCCEEDED
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

    assert plan.life_cycle_state == LifeCycleValue.SUCCEEDED
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
    plans = list(
        EntityQueryLanguageGenerativeBackend().evaluate(expression=plan_generator)
    )
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

    assert plan.life_cycle_state == LifeCycleValue.SUCCEEDED
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

    assert plan.life_cycle_state == LifeCycleValue.SUCCEEDED
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


@dataclass(kw_only=True)
class _FailsWhenThePlanEnds(StateHistoryObserver):
    """
    An observer that fails once the root of the plan it observes ends.
    """

    plan: StatechartNode
    """
    The root of the observed plan.
    """

    def on_state_change(self, *, history: StateHistory) -> None:
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
    [chooser] = [
        chooser
        for chooser in plan.statechart.context.require_extension(
            extension_type=ChildChooserAccess
        ).choosers
        if isinstance(chooser, UnderspecifiedChildChooser)
    ]
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
    statechart.state_history.add_observer(observer=_FailsWhenThePlanEnds(plan=plan))

    executor.compile(statechart=statechart)
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
        executor.compile(statechart=statechart_of(executor, plan))
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
        ).limit(n=3)
    )
    executor = simulated_executor(extensions)

    executor.compile(statechart=statechart_of(executor, plan))
    executor.execute()

    assert plan.life_cycle_state == LifeCycleValue.SUCCEEDED


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
        executor.compile(statechart=statechart_of(executor, plan))
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
    executor.compile(statechart=statechart_of(executor, Sequence(nodes=[candidate])))
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

    visualization = trial_executor.require_extension(extension_type=RvizVisualization)
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

    publisher = trial_executor.require_extension(
        extension_type=RvizVisualization
    ).publisher
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

    publisher = trial_executor.require_extension(
        extension_type=RvizVisualization
    ).publisher
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
        trial_executor.require_extension(extension_type=RvizVisualization)
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

    assert not trial_executor.require_extension(
        extension_type=RvizVisualization
    ).is_rendering
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

    assert trial.succeeds(SubtreeRebuild(root=action, candidates={}))
    trial.discard()


# %% a candidate whose motion could not keep its clearance


def a_collision_that_could_not_be_avoided() -> MotionViolatedCollisionAvoidance:
    """
    :return: The failure a motion reports when it could not keep the clearance it was
        given.
    """
    return MotionViolatedCollisionAvoidance(
        violation=CollisionViolatedError(violated_collisions=[])
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

    assert plan.life_cycle_state == LifeCycleValue.SUCCEEDED
    assert plan.grounded_nodes[-1].fail_on_attempt_number is None


# %% which statements of a parallel are grounded together


def _slot_dof_ids(world: World) -> tuple[UUID, ...]:
    """
    :return: Ids of the degrees of freedom that stand for four slots.
    """
    return tuple(dof.id for dof in world.degrees_of_freedom[:4])


def _statement_claiming_one_of(
    slots: list[int], slot_dof_ids: tuple[UUID, ...], **parameters
) -> Match[ClaimSlotAction]:
    """
    :return: A statement claiming one of `slots`, ranked in that order, with
        `parameters` fixed.
    """
    return a(ClaimSlotAction)(
        slot=variable_from(slots), slot_dof_ids=slot_dof_ids, **parameters
    )


def _prepared(
    plan: StatechartNode, extensions: list[ContextExtension]
) -> KinematicallySimulatedPlanExecutor:
    """
    :return: An executor that prepared `plan` without running it.
    """
    executor = simulated_executor(extensions)
    executor.prepare(statechart_of(executor, plan))
    return executor


def _grounded_with(statement: StatementNode) -> list[StatementNode]:
    """
    :return: The statements grounded together with `statement` once it asks for its
        child.
    """
    return UnderspecifiedChildChooser.statements_grounded_with(statement)


def test_the_statements_of_a_parallel_are_grounded_together(
    apartment_world_pr2_copy_with_context,
):
    world, robot, extensions = apartment_world_pr2_copy_with_context
    slot_dof_ids = _slot_dof_ids(world)
    side_by_side = Parallel(
        nodes=[_statement_claiming_one_of([0, 1], slot_dof_ids) for _ in range(2)]
    )
    first, second = side_by_side.nodes
    plan = Sequence(nodes=[side_by_side])

    _prepared(plan, extensions)

    assert plan.nodes == [side_by_side]
    assert _grounded_with(first) == [first, second]
    assert _grounded_with(second) == [first, second]


def test_the_statements_of_a_top_level_parallel_are_grounded_together(
    apartment_world_pr2_copy_with_context,
):
    world, robot, extensions = apartment_world_pr2_copy_with_context
    slot_dof_ids = _slot_dof_ids(world)
    side_by_side = Parallel(
        nodes=[_statement_claiming_one_of([0, 1], slot_dof_ids) for _ in range(2)]
    )

    executor = _prepared(side_by_side, extensions)

    assert executor.plan_nodes == [side_by_side]
    assert _grounded_with(side_by_side.nodes[0]) == side_by_side.nodes


def test_a_statement_later_in_a_branch_is_grounded_alone(
    apartment_world_pr2_copy_with_context,
):
    world, robot, extensions = apartment_world_pr2_copy_with_context
    slot_dof_ids = _slot_dof_ids(world)
    branch = Sequence(
        nodes=[
            ClaimSlotAction(slot=1, slot_dof_ids=slot_dof_ids),
            _statement_claiming_one_of([2], slot_dof_ids),
        ]
    )
    side_by_side = Parallel(
        nodes=[_statement_claiming_one_of([0], slot_dof_ids), branch]
    )
    statement, _ = side_by_side.nodes
    _, later = branch.nodes
    plan = Sequence(nodes=[side_by_side])

    _prepared(plan, extensions)

    assert _grounded_with(statement) == [statement]
    assert _grounded_with(later) == [later]


def test_a_branch_without_a_statement_starting_with_it_stays_in_the_parallel(
    apartment_world_pr2_copy_with_context,
):
    """
    Only the statements are grounded together, so the other branches start with the
    parallel, such as a node letting a gripper touch its surroundings meanwhile.
    """
    world, robot, extensions = apartment_world_pr2_copy_with_context
    slot_dof_ids = _slot_dof_ids(world)
    side_by_side = Parallel(
        nodes=[
            _statement_claiming_one_of([0], slot_dof_ids),
            ClaimSlotAction(slot=2, slot_dof_ids=slot_dof_ids),
            _statement_claiming_one_of([1], slot_dof_ids),
        ]
    )
    first, beside, second = side_by_side.nodes
    plan = Sequence(nodes=[side_by_side])

    _prepared(plan, extensions)

    assert plan.nodes == [side_by_side]
    assert side_by_side.nodes == [first, beside, second]
    assert _grounded_with(first) == [first, second]


def test_the_statements_of_a_parallel_needing_a_number_of_successes_are_grounded_together(
    apartment_world_pr2_copy_with_context,
):
    world, robot, extensions = apartment_world_pr2_copy_with_context
    slot_dof_ids = _slot_dof_ids(world)
    side_by_side = Parallel(
        nodes=[
            _statement_claiming_one_of([0], slot_dof_ids),
            ClaimSlotAction(slot=1, slot_dof_ids=slot_dof_ids),
            _statement_claiming_one_of([2], slot_dof_ids),
        ],
        minimum_success=1,
    )
    first, _, second = side_by_side.nodes
    plan = Sequence(nodes=[side_by_side])

    _prepared(plan, extensions)

    assert _grounded_with(first) == [first, second]


def test_the_statements_of_a_parallel_inside_a_parallel_are_grounded_with_the_outer_one(
    apartment_world_pr2_copy_with_context,
):
    world, robot, extensions = apartment_world_pr2_copy_with_context
    slot_dof_ids = _slot_dof_ids(world)
    inner = Parallel(nodes=[_statement_claiming_one_of([0], slot_dof_ids)])
    outer = Parallel(nodes=[inner, _statement_claiming_one_of([1], slot_dof_ids)])
    plan = Sequence(nodes=[outer])

    _prepared(plan, extensions)

    assert _grounded_with(inner.nodes[0]) == [inner.nodes[0], outer.nodes[1]]


def test_the_statements_of_a_parallel_later_in_a_branch_are_grounded_together(
    apartment_world_pr2_copy_with_context,
):
    world, robot, extensions = apartment_world_pr2_copy_with_context
    slot_dof_ids = _slot_dof_ids(world)
    inner = Parallel(
        nodes=[_statement_claiming_one_of([1], slot_dof_ids) for _ in range(2)]
    )
    outer = Parallel(
        nodes=[
            _statement_claiming_one_of([0], slot_dof_ids),
            Sequence(nodes=[ClaimSlotAction(slot=2, slot_dof_ids=slot_dof_ids), inner]),
        ]
    )
    plan = Sequence(nodes=[outer])

    _prepared(plan, extensions)

    assert _grounded_with(inner.nodes[0]) == inner.nodes


def test_a_statement_that_chose_is_grounded_alone_until_its_parallel_is_reset(
    apartment_world_pr2_copy_with_context,
):
    world, robot, extensions = apartment_world_pr2_copy_with_context
    slot_dof_ids = _slot_dof_ids(world)
    side_by_side = Parallel(
        nodes=[_statement_claiming_one_of([0, 1], slot_dof_ids) for _ in range(2)]
    )
    first, second = side_by_side.nodes
    plan = Sequence(nodes=[side_by_side])
    _prepared(plan, extensions)
    for slot, statement in enumerate(side_by_side.nodes):
        statement.choose_child_with(
            chooser=CandidateHandOver(
                candidate=ClaimSlotAction(slot=slot, slot_dof_ids=slot_dof_ids)
            )
        )
    grounded_before_the_reset = _grounded_with(first)

    for statement in side_by_side.nodes:
        statement.on_reset()

    assert grounded_before_the_reset == [first]
    assert _grounded_with(first) == [first, second]


# %% grounding the statements of a parallel together


def _statements_of(plan: Sequence) -> list[StatementNode]:
    """
    :return: The statements among the children of the one parallel `plan` runs, in
        their order.
    """
    [parallel] = plan.nodes
    return [node for node in parallel.nodes if isinstance(node, StatementNode)]


def test_statements_side_by_side_are_grounded_together(
    apartment_world_pr2_copy_with_context,
):
    """
    The slot each statement ranks first works on its own, but not beside the other, so
    they settle on different slots before either runs, rather than one of them failing
    for real first.
    """
    world, robot, extensions = apartment_world_pr2_copy_with_context
    slot_dof_ids = _slot_dof_ids(world)
    plan = Sequence(
        nodes=[
            Parallel(
                nodes=[
                    _statement_claiming_one_of([0, 1], slot_dof_ids) for _ in range(2)
                ]
            )
        ]
    )

    run_plan(plan, extensions)

    statements = _statements_of(plan)
    assert [statement.grounded_nodes[0].slot for statement in statements] == [0, 1]
    assert all(len(statement.chosen_children) == 1 for statement in statements)


def test_nodes_beside_statements_are_part_of_their_trial(
    apartment_world_pr2_copy_with_context,
):
    """
    A node that is not an action claims the slot the statement ranks first, so the
    statement settles on the other one before it runs.
    """
    world, robot, extensions = apartment_world_pr2_copy_with_context
    slot_dof_ids = _slot_dof_ids(world)
    plan = Sequence(
        nodes=[
            Parallel(
                nodes=[
                    _statement_claiming_one_of([0, 1], slot_dof_ids),
                    ClaimSlot(slot=0, slot_dof_ids=slot_dof_ids),
                ]
            )
        ]
    )

    run_plan(plan, extensions)

    [statement] = [
        node for node in _statements_of(plan) if isinstance(node, StatementNode)
    ]
    assert statement.grounded_nodes[0].slot == 1
    assert len(statement.chosen_children) == 1


def test_a_candidate_grounding_a_statement_of_its_own_is_tried_beside_the_others(
    apartment_world_pr2_copy_with_context,
):
    """
    The candidate leaves the slot it claims to a statement among its steps, which its
    trial grounds as running it does.
    """
    world, robot, extensions = apartment_world_pr2_copy_with_context
    slot_dof_ids = _slot_dof_ids(world)
    plan = Sequence(
        nodes=[
            Parallel(
                nodes=[
                    a(ClaimSlotThroughAStatementAction)(
                        slot=variable_from([0]), slot_dof_ids=slot_dof_ids
                    ),
                    ClaimSlot(slot=1, slot_dof_ids=slot_dof_ids),
                ]
            )
        ]
    )

    run_plan(plan, extensions)

    [statement] = [
        node for node in _statements_of(plan) if isinstance(node, StatementNode)
    ]
    assert len(statement.chosen_children) == 1
    assert world.state[slot_dof_ids[0]].position == SlotPosition.CLAIMED


# %% a branch failing while its parallel runs


def test_only_the_failed_statement_is_grounded_again_after_the_other_branch_succeeded(
    apartment_world_pr2_copy_with_context,
):
    """
    Opening a drawer fails after the pick-up beside it succeeded: the opening is
    grounded again, while the pick-up, which cannot be repeated, runs only once.
    """
    world, robot, extensions = apartment_world_pr2_copy_with_context
    slot_dof_ids = _slot_dof_ids(world)
    failing_probe = register_probe()
    succeeding_probe = register_probe()
    plan = Sequence(
        nodes=[
            Parallel(
                nodes=[
                    _statement_claiming_one_of(
                        [2, 3],
                        slot_dof_ids,
                        ticks_before_claiming=5,
                        probe_key=failing_probe,
                        fail_on_attempt_number=2,
                    ),
                    _statement_claiming_one_of(
                        [1], slot_dof_ids, probe_key=succeeding_probe
                    ),
                ]
            )
        ]
    )

    run_plan(plan, extensions)

    failed, succeeded = _statements_of(plan)
    assert [node.slot for node in failed.grounded_nodes] == [2, 3]
    assert len(succeeded.chosen_children) == 1
    trial, real = _registered_probes[succeeding_probe].calls
    assert trial.world is not world
    assert real.world is world


def test_a_statement_grounded_together_proposes_no_candidate_it_ran_again(
    apartment_world_pr2_copy_with_context,
):
    """
    The statement grounded together with the one beside it goes on from the candidate
    its parallel handed it once that one failed, as a statement grounding on its own
    goes on from the candidate it ran, so no attempt of its claims a slot it claimed
    already.
    """
    world, robot, extensions = apartment_world_pr2_copy_with_context
    slot_dof_ids = _slot_dof_ids(world)
    failing_probe = register_probe()
    plan = Sequence(
        nodes=[
            Parallel(
                nodes=[
                    _statement_claiming_one_of(
                        [2, 3],
                        slot_dof_ids,
                        ticks_before_claiming=5,
                        probe_key=failing_probe,
                        fail_on_attempt_number=2,
                    ),
                    _statement_claiming_one_of([1], slot_dof_ids),
                ]
            )
        ]
    )

    run_plan(plan, extensions)

    failed, _ = _statements_of(plan)
    assert [node.slot for node in failed.grounded_nodes] == [2, 3]
    assert [
        call.position_at_entry == SlotPosition.CLAIMED
        for call in _registered_probes[failing_probe].calls
    ] == [False, False, False, False]


def test_a_statement_grounded_again_is_tried_without_what_already_ended(
    apartment_world_pr2_copy_with_context,
):
    """
    The action beside the failed statement claimed slot 1 and ended, so it is not tried
    again beside the statement's next slot, where it would find its slot taken.
    """
    world, robot, extensions = apartment_world_pr2_copy_with_context
    slot_dof_ids = _slot_dof_ids(world)
    plan = Sequence(
        nodes=[
            Parallel(
                nodes=[
                    _statement_claiming_one_of(
                        [2, 3],
                        slot_dof_ids,
                        ticks_before_claiming=5,
                        probe_key=register_probe(),
                        fail_on_attempt_number=2,
                    ),
                    ClaimSlotAction(slot=1, slot_dof_ids=slot_dof_ids),
                ]
            )
        ]
    )

    run_plan(plan, extensions)

    [grounded] = _statements_of(plan)
    assert [node.slot for node in grounded.grounded_nodes] == [2, 3]


def test_a_statement_grounded_again_is_tried_needing_only_the_successes_still_missing(
    apartment_world_pr2_copy_with_context,
):
    """
    The action beside the failed statement succeeded and ended, so the statement's
    next slot is tried needing one success, not the two its parallel needed at first.
    """
    world, robot, extensions = apartment_world_pr2_copy_with_context
    slot_dof_ids = _slot_dof_ids(world)
    plan = Sequence(
        nodes=[
            Parallel(
                nodes=[
                    _statement_claiming_one_of(
                        [2, 3],
                        slot_dof_ids,
                        ticks_before_claiming=5,
                        probe_key=register_probe(),
                        fail_on_attempt_number=2,
                    ),
                    ClaimSlotAction(slot=1, slot_dof_ids=slot_dof_ids),
                ],
                minimum_success=2,
            )
        ]
    )

    run_plan(plan, extensions)

    [grounded] = _statements_of(plan)
    assert [node.slot for node in grounded.grounded_nodes] == [2, 3]


def test_a_statement_grounded_again_is_tried_beside_what_still_runs(
    apartment_world_pr2_copy_with_context,
):
    """
    The action beside the failed statement claims slot 1 later on, so the statement,
    whose first slot it left claimed, settles on slot 2 rather than on slot 1, which
    works on its own.
    """
    world, robot, extensions = apartment_world_pr2_copy_with_context
    slot_dof_ids = _slot_dof_ids(world)
    statement = _statement_claiming_one_of(
        [0, 1, 2],
        slot_dof_ids,
        probe_key=register_probe(),
        fail_on_attempt_number=2,
    )
    plan = Sequence(
        nodes=[
            Parallel(
                nodes=[
                    statement,
                    ClaimSlotAction(
                        slot=1, slot_dof_ids=slot_dof_ids, ticks_before_claiming=20
                    ),
                ]
            )
        ]
    )

    run_plan(plan, extensions)

    [grounded] = _statements_of(plan)
    assert [node.slot for node in grounded.grounded_nodes] == [0, 2]


def test_a_statement_starting_later_in_a_branch_is_tried_beside_what_still_runs(
    apartment_world_pr2_copy_with_context,
):
    """
    The action beside the statement claims slot 1 later on, so the statement settles on
    slot 2 rather than on slot 1, which works on its own.
    """
    world, robot, extensions = apartment_world_pr2_copy_with_context
    slot_dof_ids = _slot_dof_ids(world)
    branch = Sequence(
        nodes=[
            CountTicks(ticks=3),
            _statement_claiming_one_of([1, 2], slot_dof_ids),
        ]
    )
    plan = Sequence(
        nodes=[
            Parallel(
                nodes=[
                    branch,
                    ClaimSlotAction(
                        slot=1, slot_dof_ids=slot_dof_ids, ticks_before_claiming=20
                    ),
                ]
            )
        ]
    )

    run_plan(plan, extensions)

    [statement] = branch.nodes[1:]
    assert [node.slot for node in statement.grounded_nodes] == [2]


def test_a_statement_is_tried_beside_a_parallel_grounded_together_that_still_runs(
    apartment_world_pr2_copy_with_context,
):
    """
    The parallel beside the statement grounded its own statement to claim slot 1 later
    on, so the statement settles on slot 2 rather than on slot 1, which works on its
    own.
    """
    world, robot, extensions = apartment_world_pr2_copy_with_context
    slot_dof_ids = _slot_dof_ids(world)
    branch = Sequence(
        nodes=[
            CountTicks(ticks=3),
            _statement_claiming_one_of([1, 2], slot_dof_ids),
        ]
    )
    plan = Sequence(
        nodes=[
            Parallel(
                nodes=[
                    branch,
                    Sequence(
                        nodes=[
                            CountTicks(ticks=1),
                            Parallel(
                                nodes=[
                                    _statement_claiming_one_of(
                                        [1], slot_dof_ids, ticks_before_claiming=20
                                    )
                                ]
                            ),
                        ]
                    ),
                ]
            )
        ]
    )

    run_plan(plan, extensions)

    [statement] = branch.nodes[1:]
    assert [node.slot for node in statement.grounded_nodes] == [2]


def test_a_failed_statement_out_of_candidates_fails_its_parallel(
    apartment_world_pr2_copy_with_context,
):
    """
    The statement has no slot left once its only one is claimed, so the plan fails for
    that statement, and the branch beside it, which succeeded, is not run again.
    """
    world, robot, extensions = apartment_world_pr2_copy_with_context
    slot_dof_ids = _slot_dof_ids(world)
    succeeding_probe = register_probe()
    failing = _statement_claiming_one_of(
        [2],
        slot_dof_ids,
        ticks_before_claiming=5,
        probe_key=register_probe(),
        fail_on_attempt_number=2,
    )
    plan = Sequence(
        nodes=[
            Parallel(
                nodes=[
                    failing,
                    _statement_claiming_one_of(
                        [1], slot_dof_ids, probe_key=succeeding_probe
                    ),
                ]
            )
        ]
    )
    executor = simulated_executor(extensions)

    with pytest.raises(EmptyUnderspecified) as failure:
        executor.compile(statechart=statechart_of(executor, plan))
        executor.execute()

    assert type(failure.value) is EmptyUnderspecified
    assert failure.value.node.statement is failing
    [grounding] = plan.nodes
    assert grounding.life_cycle_state == LifeCycleValue.FAILED
    assert len(_registered_probes[succeeding_probe].calls) == 2


def test_a_failing_branch_without_a_statement_fails_its_parallel_without_running_it_again(
    apartment_world_pr2_copy_with_context,
):
    """
    The action beside the statement fails, and nothing grounds it anew, so the parallel
    fails, rather than being grounded again with the statement's next candidate and run
    once more.
    """
    world, robot, extensions = apartment_world_pr2_copy_with_context
    slot_dof_ids = _slot_dof_ids(world)
    statement_probe = register_probe()
    plan = Sequence(
        nodes=[
            Parallel(
                nodes=[
                    _statement_claiming_one_of(
                        [1, 2], slot_dof_ids, probe_key=statement_probe
                    ),
                    RecordingAction(
                        probe_key=register_probe(),
                        dof_id=world.degrees_of_freedom[5].id,
                        fail_on_attempt_number=2,
                    ),
                ]
            )
        ]
    )
    executor = simulated_executor(extensions)

    with pytest.raises(MotionDidNotFinish):
        executor.compile(statechart=statechart_of(executor, plan))
        executor.execute()

    [parallel] = plan.nodes
    [statement] = _statements_of(plan)
    assert parallel.life_cycle_state == LifeCycleValue.FAILED
    assert len(statement.chosen_children) == 1
    assert len(_registered_probes[statement_probe].calls) == 2


# %% a trial keeps the conditions of the plan


def test_a_trial_keeps_a_start_condition_wired_to_a_node_beside_it(
    apartment_world_pr2_copy_with_context,
):
    """
    The action beside the statement starts only once the statement failed, which it
    does not, so the statement's trial does not start it either.
    """
    world, robot, extensions = apartment_world_pr2_copy_with_context
    slot_dof_ids = _slot_dof_ids(world)
    fallback_probe = register_probe()
    statement = StatementNode(statement=_statement_claiming_one_of([0], slot_dof_ids))
    fallback = RecordingAction(
        probe_key=fallback_probe, dof_id=world.degrees_of_freedom[5].id
    )
    fallback.start_condition = statement.is_failed
    plan = Sequence(nodes=[Parallel(nodes=[statement, fallback], minimum_success=1)])

    run_plan(plan, extensions)

    assert statement.life_cycle_state == LifeCycleValue.SUCCEEDED
    assert _registered_probes[fallback_probe].calls == []
