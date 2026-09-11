"""
Tests for ``Attempt`` (see ``giskardpy/motion_statechart/goals/templates.py``).

The template is exercised by compiling it into a real :class:`MotionStatechart` and
ticking the executor. ``ConstTrueNode`` / ``ConstFalseNode`` stand in for a motion that
is always / never at its goal, and ``CountControlCycles`` for a failure monitor that
fires after a known number of cycles, so none of these tests needs a world with a robot
in it.
"""

import pytest
from giskardpy.executor import Executor
from giskardpy.motion_statechart.context import MotionStatechartContext
from giskardpy.motion_statechart.data_types import (
    LifeCycleValues,
    ObservationStateValues,
)
from giskardpy.motion_statechart.goals.templates import Attempt, Sequence
from giskardpy.motion_statechart.graph_node import MotionStatechartNode
from giskardpy.motion_statechart.motion_statechart import MotionStatechart
from giskardpy.motion_statechart.monitors.payload_monitors import (
    CountControlCycles,
    Pulse,
)
from giskardpy.motion_statechart.nodes_for_testing.nodes_for_testing import (
    ConstFalseNode,
    ConstTrueNode,
    NodeObservingNothingYet,
)
from semantic_digital_twin.world import World

# Control cycles after which the attempts below have settled on a verdict.
SETTLE_CYCLES = 6

# Control cycles a failure monitor is given before it fires. Small enough that
# SETTLE_CYCLES still covers the cycles the attempt needs to react to it.
CYCLES_UNTIL_GIVING_UP = 2


def _compile(
    node: MotionStatechartNode,
) -> tuple[MotionStatechart, Executor]:
    """
    Add the node to a fresh statechart and compile it.

    :param node: The template under test.
    :return: The chart, so a caller can read its recorded history, and the executor.
    """
    motion_statechart = MotionStatechart()
    motion_statechart.add_node(node)
    executor = Executor(MotionStatechartContext(world=World()))
    executor.compile(motion_statechart=motion_statechart)
    return motion_statechart, executor


def _compile_and_tick(
    node: MotionStatechartNode, cycles: int = SETTLE_CYCLES
) -> tuple[MotionStatechart, Executor]:
    """
    Add the node to a fresh statechart, compile it and tick the executor.

    :param node: The template under test.
    :param cycles: Control cycles to run after compiling.
    :return: The chart and the executor, so a caller can keep ticking and inspect
        intermediate states.
    """
    motion_statechart, executor = _compile(node)
    for _ in range(cycles):
        executor.tick()
    return motion_statechart, executor


# %% reaching the goal


def test_an_attempt_succeeds_once_its_task_reaches_its_goal():
    """
    A motion at its goal is what this template is waiting for, and it says so itself
    rather than leaving the caller to read the task.

    The verdict is read rather than the observation behind it, because only the verdict
    outlasts the goal that reached it. The verdict belongs to the attempt: the task it
    held open is only taken down with it.
    """
    task = ConstTrueNode(name="task")
    attempt = Attempt(task=task, failure_monitors=[])

    _compile_and_tick(attempt)

    assert attempt.goal_reached_state == ObservationStateValues.TRUE
    assert attempt.life_cycle_state == LifeCycleValues.SUCCEEDED
    assert task.life_cycle_state == LifeCycleValues.INTERRUPTED


def test_an_attempt_ends_itself_without_anything_wiring_a_success_condition():
    """
    Supplying the ending a motion cannot produce is the whole point of the template, so
    it must not depend on a parent having wired one.
    """
    attempt = Attempt(task=ConstTrueNode(name="task"), failure_monitors=[])

    _compile_and_tick(attempt)

    assert attempt.life_cycle_state.is_terminal


# %% giving up


def test_an_attempt_fails_once_a_failure_monitor_fires():
    """
    A monitor that fires is the only thing that ends a motion short of its goal, and it
    is reported as a failure rather than as a motion still on its way.
    """
    task = ConstFalseNode(name="task")
    attempt = Attempt(
        task=task,
        failure_monitors=[
            CountControlCycles(control_cycles=CYCLES_UNTIL_GIVING_UP, name="gave_up")
        ],
    )

    _compile_and_tick(attempt)

    assert attempt.goal_reached_state == ObservationStateValues.FALSE
    assert attempt.life_cycle_state == LifeCycleValues.FAILED
    # The attempt is what gave up; the task was only taken down with it.
    assert task.life_cycle_state == LifeCycleValues.INTERRUPTED


def test_an_attempt_holds_its_task_open_until_it_is_decided():
    """
    A constraint that stops running stops being enforced, so the task keeps running for
    as long as neither outcome has been decided.
    """
    task = ConstFalseNode(name="task")
    attempt = Attempt(
        task=task,
        failure_monitors=[
            CountControlCycles(control_cycles=CYCLES_UNTIL_GIVING_UP, name="gave_up")
        ],
    )

    _compile_and_tick(attempt, cycles=CYCLES_UNTIL_GIVING_UP - 1)

    assert attempt.observation_state == ObservationStateValues.UNKNOWN
    assert task.life_cycle_state == LifeCycleValues.RUNNING


def test_giving_up_on_a_task_that_observed_nothing_interrupts_it():
    """
    Whether a task was judged or merely cut off is decided by what it observed, not by
    what gave up on it, so a task that never answered is interrupted rather than failed.

    The attempt itself fails either way: being given up on is a verdict about the
    attempt, not the absence of one.
    """
    task = NodeObservingNothingYet(name="task")
    attempt = Attempt(
        task=task,
        failure_monitors=[
            CountControlCycles(control_cycles=CYCLES_UNTIL_GIVING_UP, name="gave_up")
        ],
    )

    _compile_and_tick(attempt)

    assert task.life_cycle_state == LifeCycleValues.INTERRUPTED
    assert attempt.life_cycle_state == LifeCycleValues.FAILED


def test_an_attempt_observes_nothing_while_neither_outcome_has_happened():
    """
    A motion that has not arrived yet has not failed, so the observation stays undecided
    rather than reading as a failure the whole way there.
    """
    attempt = Attempt(
        task=ConstFalseNode(name="task"),
        failure_monitors=[CountControlCycles(control_cycles=99, name="gave_up")],
    )

    motion_statechart, _ = _compile_and_tick(attempt)

    assert set(motion_statechart.history.get_observation_history_of_node(attempt)) == {
        ObservationStateValues.UNKNOWN
    }


def test_an_attempt_without_failure_monitors_never_gives_up():
    """
    An empty list of failure monitors is the caller stating that this motion cannot
    fail, which leaves nothing that could end an attempt short of its goal.
    """
    attempt = Attempt(task=ConstFalseNode(name="task"), failure_monitors=[])

    _, executor = _compile(attempt)

    with pytest.raises(TimeoutError):
        executor.tick_until_end(timeout=SETTLE_CYCLES)


def test_reaching_the_goal_wins_over_a_failure_on_the_same_cycle():
    """
    A monitor firing on the cycle the motion arrives must not undo the arrival.
    """
    attempt = Attempt(
        task=ConstTrueNode(name="task"),
        failure_monitors=[ConstTrueNode(name="gave_up")],
    )

    _compile_and_tick(attempt)

    assert attempt.goal_reached_state == ObservationStateValues.TRUE
    assert attempt.life_cycle_state == LifeCycleValues.SUCCEEDED


# %% why an attempt was given up on


def test_failure_reasons_names_the_monitor_that_fired():
    """
    Which monitor ended an attempt is what turns a failure into a reason, so it has to
    survive the monitor being taken down along with the attempt.
    """
    fired = CountControlCycles(control_cycles=CYCLES_UNTIL_GIVING_UP, name="fired")
    stayed_quiet = CountControlCycles(control_cycles=99, name="stayed_quiet")
    attempt = Attempt(
        task=ConstFalseNode(name="task"), failure_monitors=[fired, stayed_quiet]
    )

    _compile_and_tick(attempt)

    assert attempt.failure_reasons == [fired]


def test_failure_reasons_is_empty_once_the_task_reached_its_goal():
    """
    Nothing gave up on an attempt that arrived, so there is no reason to report.
    """
    attempt = Attempt(
        task=ConstTrueNode(name="task"),
        failure_monitors=[CountControlCycles(control_cycles=99, name="gave_up")],
    )

    _compile_and_tick(attempt)

    assert attempt.failure_reasons == []


def test_failure_reasons_names_a_monitor_that_fired_for_a_single_cycle():
    """
    A goal reads its children a control cycle late, so a monitor that does not hold its
    reading is already back to observing nothing by the time it is asked why the attempt
    ended.
    """
    fired_briefly = Pulse(name="fired_briefly")
    attempt = Attempt(
        task=ConstFalseNode(name="task"), failure_monitors=[fired_briefly]
    )

    _compile_and_tick(attempt)

    assert attempt.life_cycle_state == LifeCycleValues.FAILED
    assert attempt.failure_reasons == [fired_briefly]


def test_failure_reasons_is_empty_when_a_monitor_fired_but_the_goal_was_reached():
    """
    Nothing was given up on when the task arrived anyway, so a monitor that fired on
    that same cycle is not a reason for anything.
    """
    attempt = Attempt(
        task=ConstTrueNode(name="task"),
        failure_monitors=[ConstTrueNode(name="fired")],
    )

    _compile_and_tick(attempt)

    assert attempt.life_cycle_state == LifeCycleValues.SUCCEEDED
    assert attempt.failure_reasons == []


def test_failure_reasons_lists_every_monitor_that_fired_at_once():
    """
    Two monitors can fire on the same control cycle, and neither of them is more the
    reason than the other.
    """
    first = CountControlCycles(control_cycles=CYCLES_UNTIL_GIVING_UP, name="first")
    second = CountControlCycles(control_cycles=CYCLES_UNTIL_GIVING_UP, name="second")
    attempt = Attempt(
        task=ConstFalseNode(name="task"), failure_monitors=[first, second]
    )

    _compile_and_tick(attempt)

    assert attempt.failure_reasons == [first, second]


# %% attempts as steps of an RPL style template


def test_an_attempt_is_a_usable_sequence_step():
    """
    A step that ends itself is what a sequence needs to move on, which is what wrapping
    a motion in an attempt buys.
    """
    first = Attempt(task=ConstTrueNode(name="first_task"), failure_monitors=[])
    second = Attempt(task=ConstTrueNode(name="second_task"), failure_monitors=[])

    _compile_and_tick(Sequence(nodes=[first, second]))

    assert first.life_cycle_state == LifeCycleValues.SUCCEEDED
    assert second.life_cycle_state == LifeCycleValues.SUCCEEDED


def test_a_failed_attempt_makes_its_sequence_report_a_failure():
    """
    A step that can never succeed used to leave a sequence waiting forever, because a
    motion short of its goal never ends.

    An attempt ends on its failure monitor instead, which is the verdict the sequence
    was already looking for.
    """
    failing_step = Attempt(
        task=ConstFalseNode(name="task"),
        failure_monitors=[
            CountControlCycles(control_cycles=CYCLES_UNTIL_GIVING_UP, name="gave_up")
        ],
    )
    sequence = Sequence(
        nodes=[
            failing_step,
            Attempt(task=ConstTrueNode(name="never_reached"), failure_monitors=[]),
        ]
    )

    _compile_and_tick(sequence)

    assert failing_step.life_cycle_state == LifeCycleValues.FAILED
    assert sequence.goal_reached_state == ObservationStateValues.FALSE
