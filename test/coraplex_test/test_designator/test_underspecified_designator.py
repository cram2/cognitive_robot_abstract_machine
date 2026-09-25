from dataclasses import dataclass, field

import pytest
from uuid import UUID, uuid4

from typing_extensions import Dict, List, Optional

from krrood.entity_query_language.backends import (
    EntityQueryLanguageGenerativeBackend,
    ProbabilisticBackend,
)
from krrood.entity_query_language.factories import a, an, variable, variable_from
from giskardpy.motion_statechart.data_types import LifeCycleValues

from coraplex.datastructures.enums import ActionTrialVisualization

from semantic_digital_twin.semantic_annotations.semantic_annotations import Milk
from coraplex.language import SequentialNode
from coraplex.execution_environment import simulated_robot
from coraplex.plans.executables import Executable
from coraplex.plans.factories import sequential, execute_single
from coraplex.plans.failures import PlanFailure
from coraplex.plans.plan_node import ExecutionBoundaryNode, PlanNode
from coraplex.plans.underspecified import ActionTrial
from coraplex.robot_plans.actions.base import ActionDescription
from coraplex.robot_plans.actions.core.navigation import NavigateAction
from coraplex.robot_plans.actions.core.pick_up import PickUpAction
from semantic_digital_twin.robots.robot_parts import AbstractRobot, Arm
from semantic_digital_twin.spatial_types.spatial_types import Pose
from semantic_digital_twin.world import World

# %% mimics for testing candidate trials without depending on real motion physics


@dataclass
class TrialCall:
    """
    One recorded attempt of a `RecordingAction`.
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
    Records every attempt of a `RecordingAction` across both trial and real execution.
    """

    def __init__(self):
        self.calls: List[TrialCall] = []


_registered_probes: Dict[UUID, TrialProbe] = {}
"""
`TrialProbe` instances, keyed by the id a `RecordingAction` carries as `probe_key`.

`World.rebind_world_entities` deep-copies every value it does not recognize as a world entity, so a
rebound `RecordingAction` cannot share a probe handed to it directly as a field - the
same isolation that makes a trial safe for real designators. Reaching the probe
out-of-band by an id, whose value survives copying even though its identity need not,
is what lets a test observe a candidate's trial and its later real attempt as one
sequence.
"""


def register_probe() -> UUID:
    """
    Register a fresh `TrialProbe` and return the id a `RecordingAction` should carry to
    record into it.
    """
    key = uuid4()
    _registered_probes[key] = TrialProbe()
    return key


@dataclass(eq=False, repr=False)
class RecordingExecutionNode(ExecutionBoundaryNode):
    """
    A leaf plan node whose parsed executable records the world it ran against and
    mutates a probed degree of freedom, failing once a configured number of attempts
    have been recorded.
    """

    probe_key: UUID = field(kw_only=True)
    """
    Id of the `TrialProbe` this node's attempts are recorded to.
    """

    dof_id: UUID = field(kw_only=True)
    """
    Id of the degree of freedom this node mutates on every attempt, to make world state
    changes observable.
    """

    fail_on_attempt_number: Optional[int] = field(kw_only=True, default=None)
    """
    Raise a `PlanFailure` once the probe has recorded this many calls; never raise if
    None.
    """

    def notify(self):
        pass

    def parse(self) -> Executable:
        return RecordingExecutable(
            context=self.context,
            probe_key=self.probe_key,
            dof_id=self.dof_id,
            fail_on_attempt_number=self.fail_on_attempt_number,
        )


@dataclass
class RecordingExecutable(Executable):
    """
    Executable half of `RecordingExecutionNode`; see its docstring.
    """

    probe_key: UUID = field(kw_only=True)
    dof_id: UUID = field(kw_only=True)
    fail_on_attempt_number: Optional[int] = field(kw_only=True)

    def execute(self) -> None:
        probe = _registered_probes[self.probe_key]
        probe.calls.append(
            TrialCall(
                world=self.context.world,
                position_at_entry=self.context.world.state[self.dof_id].position,
            )
        )
        self.context.world.state[self.dof_id].position = len(probe.calls)
        self.context.world.notify_state_change()
        if len(probe.calls) == self.fail_on_attempt_number:
            raise PlanFailure()


@dataclass
class RecordingAction(ActionDescription):
    """
    An action whose execution deterministically records itself and can be made to fail
    on a specific attempt, for testing `UnderspecifiedNode`'s trial-then-real candidate
    handling without depending on real motion physics.
    """

    probe_key: UUID = field(kw_only=True)
    dof_id: UUID = field(kw_only=True)
    fail_on_attempt_number: Optional[int] = field(kw_only=True, default=None)

    @property
    def _action_plan(self) -> PlanNode:
        return execute_single(
            RecordingExecutionNode(
                probe_key=self.probe_key,
                dof_id=self.dof_id,
                fail_on_attempt_number=self.fail_on_attempt_number,
            )
        )


def test_underspecified_action(apartment_world_pr2_copy_with_context):
    """
    Test that an underspecified action resolves to a concrete candidate and parses into
    an executable.

    Execution is deferred to parse().execute(), so performing the node only expands it;
    the resolved candidate is not performed here.
    """
    world, robot, context = apartment_world_pr2_copy_with_context
    action = a(NavigateAction)(
        target_location=variable_from(
            [
                Pose.from_xyz_quaternion(1, -1, 0, reference_frame=world.root),
                Pose.from_xyz_quaternion(2, -1, 0, reference_frame=world.root),
            ]
        ),
    )

    plan = execute_single(action_like=action, context=context).plan
    with simulated_robot:
        plan.perform()

    assert plan.root.status == LifeCycleValues.SUCCEEDED
    candidate = plan.root.children[0]
    assert isinstance(candidate.designator, NavigateAction)
    assert plan.root.parse() is not None
    assert plan.root._action_iterator is None, (
        "the action iterator must be released once grounding succeeds, so any resources a "
        "candidate generator only holds to validate against (for example a location's "
        "deep-copied test world) are not retained for the node's whole lifetime"
    )


def test_underspecified_action_with_ellipsis(apartment_world_pr2_copy_with_context):
    """
    Test that an underspecified action resolves and parses when a factory for a spatial
    type is used with ellipsis.

    Execution is deferred to parse().execute(), so performing the node only expands it;
    the resolved candidate is not performed here.
    """
    world, robot, context = apartment_world_pr2_copy_with_context
    context.query_backend = ProbabilisticBackend()
    action = a(NavigateAction)(
        target_location=a(Pose.from_xyz_rpy)(
            x=...,
            y=...,
            z=0.0,
            roll=0.0,
            pitch=0.0,
            yaw=0.0,
            reference_frame=context.robot.root,
        ),
    )

    plan = execute_single(action_like=action, context=context).plan
    with simulated_robot:
        plan.perform()

    assert plan.root.status == LifeCycleValues.SUCCEEDED
    candidate = plan.root.children[-1]
    assert isinstance(candidate.designator, NavigateAction)
    assert plan.root.parse() is not None


def test_underspecified_language(apartment_world_pr2_copy_with_context):
    """
    Test that entire plans can be underspecified.
    """
    world, robot, context = apartment_world_pr2_copy_with_context
    milk = world.get_semantic_annotations_by_type(Milk)[0]
    plan_generator = an(sequential, target_type=SequentialNode)(
        children=[
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
                arm=variable(Arm, domain=context.robot.get_arms()),
                grasp=milk.grasp_poses()[0],
            ),
        ],
        context=context,
    )
    plans = list(EntityQueryLanguageGenerativeBackend().evaluate(plan_generator))
    assert len(plans) == len(list(target_locations._domain_)) * len(
        context.robot.get_arms()
    )


# %% candidate trials


def test_isolation_rejected_candidate_never_touches_real_world(
    apartment_world_pr2_copy_with_context,
):
    """
    A candidate that only ever fails must be rejected during its trial, against a
    disposable copy of the world, and never attached to the plan or executed against the
    real world; only the candidate that survives its trial is executed for real.
    """
    world, robot, context = apartment_world_pr2_copy_with_context
    dof = world.degrees_of_freedom[0]
    probe_key = register_probe()

    action = a(RecordingAction)(
        probe_key=probe_key,
        dof_id=dof.id,
        fail_on_attempt_number=variable_from([1, None]),
    )
    plan = execute_single(action_like=action, context=context).plan
    with simulated_robot:
        plan.perform()

    assert plan.root.status == LifeCycleValues.SUCCEEDED
    assert len(plan.root.children) == 1
    assert plan.root.children[0].designator.fail_on_attempt_number is None

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
    world, robot, context = apartment_world_pr2_copy_with_context
    dof = world.degrees_of_freedom[0]
    probe_key = register_probe()

    action = a(RecordingAction)(
        probe_key=probe_key,
        dof_id=dof.id,
        fail_on_attempt_number=variable_from([1, 2, None]),
    )
    plan = execute_single(action_like=action, context=context).plan
    with simulated_robot:
        plan.perform()

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
    world, robot, context = apartment_world_pr2_copy_with_context
    dof = world.degrees_of_freedom[0]
    initial_position = world.state[dof.id].position
    probe_key = register_probe()

    action = a(RecordingAction)(
        probe_key=probe_key,
        dof_id=dof.id,
        fail_on_attempt_number=variable_from([2, None]),
    )
    plan = execute_single(action_like=action, context=context).plan
    with simulated_robot:
        plan.perform()

    assert plan.root.status == LifeCycleValues.SUCCEEDED
    # Both the failed and the accepted candidate are attached to the tree - a real
    # failure is not undone, only worked around by trying the next candidate.
    assert [
        child.designator.fail_on_attempt_number for child in plan.root.children
    ] == [
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


# %% a trial copy is published while debugging


@pytest.fixture
def debugging_context(apartment_world_pr2_copy_with_context, rclpy_node):
    """
    The apartment with a PR2, in a context that is debugging.
    """
    world, robot, context = apartment_world_pr2_copy_with_context
    context.ros_node = rclpy_node
    context.debug = True
    yield world, robot, context
    context.debug = False


def test_a_trial_publishes_its_copy_while_debugging(debugging_context):
    """
    The candidates are tried in the copy, so a run being watched would otherwise show
    the robot standing still through every candidate it rejects.
    """
    world, robot, context = debugging_context
    trial = ActionTrial(context=context)

    copied = trial._copy()

    assert trial._visualization.world is copied.world
    assert trial._visualization.is_rendering
    trial.discard()


def test_a_trial_publishes_its_copy_apart_from_the_world_it_copies(debugging_context):
    """
    The copy has the same frame names and markers as the world it was taken from, so it
    is published under a prefix and on a topic of its own rather than over that world.
    """
    world, robot, context = debugging_context
    trial = ActionTrial(context=context)

    trial._copy()

    publisher = trial._visualization.publisher
    assert (
        publisher.tf_publisher.frame_names.prefix
        == ActionTrialVisualization.FRAME_PREFIX
    )
    assert publisher.topic_name == ActionTrialVisualization.MARKER_TOPIC
    trial.discard()


def test_a_trial_copy_is_drawn_see_through(debugging_context):
    world, robot, context = debugging_context
    trial = ActionTrial(context=context)

    trial._copy()

    assert trial._visualization.publisher.alpha == trial.copy_marker_alpha
    trial.discard()


def test_a_trial_publishes_nothing_without_debugging(
    apartment_world_pr2_copy_with_context,
):
    world, robot, context = apartment_world_pr2_copy_with_context
    trial = ActionTrial(context=context)

    trial._copy()

    assert trial._visualization is None


def test_a_discarded_trial_stops_publishing_its_copy(debugging_context):
    world, robot, context = debugging_context
    trial = ActionTrial(context=context)
    trial._copy()
    visualization = trial._visualization

    trial.discard()

    assert not visualization.is_rendering
    assert trial._visualization is None


def test_a_replaced_copy_stops_being_published(debugging_context):
    """
    A copy that no longer matches the world is replaced, and only the one candidates are
    tried in is shown.
    """
    world, robot, context = debugging_context
    trial = ActionTrial(context=context)
    trial._copy()
    first = trial._visualization
    dof = world.degrees_of_freedom[0]
    world.state[dof.id].position = world.state[dof.id].position + 0.1
    world.notify_state_change()

    copied = trial._copy()

    assert not first.is_rendering
    assert trial._visualization.world is copied.world
    trial.discard()


def test_a_trial_tries_an_action_that_already_belongs_to_a_plan(debugging_context):
    """
    An action attached to a plan reaches that plan's context, and with it the ROS node
    the run publishes through, which a trial must not try to copy.
    """
    world, robot, context = debugging_context
    stand_where_it_is = robot.root.global_pose
    action = NavigateAction(stand_where_it_is)
    sequential([action], context)
    trial = ActionTrial(context=context)

    assert trial.succeeds(action)
    trial.discard()
