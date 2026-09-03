"""
Tests for the repeating goal templates (see
``giskardpy/motion_statechart/goals/templates.py``).

The loop itself is exercised with a plain ``RepeatUntil`` whose failure monitor counts
control cycles, so it needs neither a world nor a converging task, and ``RepeatOnStall``
is exercised against a motion that really does stop making progress.
"""

from datetime import timedelta

from giskardpy.executor import Executor
from giskardpy.motion_statechart.context import MotionStatechartContext
from giskardpy.motion_statechart.data_types import (
    LifeCycleValues,
    ObservationStateValues,
)
from giskardpy.motion_statechart.goals.templates import RepeatOnStall, RepeatUntil
from giskardpy.motion_statechart.graph_node import EndMotion, MotionStatechartNode
from giskardpy.motion_statechart.motion_statechart import MotionStatechart
from giskardpy.motion_statechart.monitors.payload_monitors import (
    CountControlCycles,
    CountNodeResets,
)
from giskardpy.motion_statechart.nodes_for_testing.nodes_for_testing import (
    ConstFalseNode,
    ConstTrueNode,
)
from giskardpy.motion_statechart.tasks.cartesian_tasks import CartesianPosition
from semantic_digital_twin.spatial_types.spatial_types import Point3
from semantic_digital_twin.world import World

from .test_progress_monitors import unreachable_arm_goal

# Control cycles an attempt is given in the world free tests before it counts as failed.
ATTEMPT_CYCLES = 2

# Control cycles after which the world free loops below have settled on a verdict.
SETTLE_CYCLES = 12


def _repeat_on_timeout(
    task: MotionStatechartNode, target: int
) -> tuple[RepeatUntil, MotionStatechart, Executor]:
    """
    Build a compiled chart around a task that is retried until it has been reset
    `target` times.
    """
    loop = RepeatUntil(
        name="loop",
        task=task,
        stop_retry_monitor=CountNodeResets(name="counter", node=task, target=target),
        retry_trigger_monitor=CountControlCycles(
            name="timeout", control_cycles=ATTEMPT_CYCLES
        ),
    )
    motion_statechart = MotionStatechart()
    motion_statechart.add_node(loop)
    motion_statechart.add_node(EndMotion.when_true(loop))
    executor = Executor(MotionStatechartContext(world=World()))
    executor.compile(motion_statechart=motion_statechart)
    return loop, motion_statechart, executor


# %% the loop


def test_repeat_until_retries_until_the_monitor_gives_up():
    """
    A task that never succeeds is retried exactly as often as the monitor allows, and
    the goal then reports the failure rather than stalling on Unknown.
    """
    task = ConstFalseNode(name="task")
    loop, _, executor = _repeat_on_timeout(task, target=3)

    for _ in range(SETTLE_CYCLES):
        executor.tick()

    assert loop.stop_retry_monitor.resets == 3
    assert loop.observation_state == ObservationStateValues.FALSE


def test_repeat_until_succeeds_without_retrying():
    """
    A task that succeeds first time is never reset, so the loop exits on success instead
    of running to the monitor's bound.
    """
    task = ConstTrueNode(name="task")
    loop, motion_statechart, executor = _repeat_on_timeout(task, target=3)

    executor.tick_until_end(SETTLE_CYCLES)

    assert loop.stop_retry_monitor.resets == 0
    assert loop.observation_state == ObservationStateValues.TRUE
    assert motion_statechart.is_end_motion()


def test_repeat_until_puts_the_task_back_to_not_started():
    """
    Retrying really restarts the task rather than leaving it running, so a task that
    only behaves correctly from its start is safe to retry.

    Every reset but the last one starts a fresh run; the last is the one that trips the
    monitor, after which nothing starts again.
    """
    task = ConstFalseNode(name="task")
    loop, motion_statechart, executor = _repeat_on_timeout(task, target=3)

    for _ in range(SETTLE_CYCLES):
        executor.tick()

    life_cycles = motion_statechart.history.get_life_cycle_history_of_node(task)
    restarts = [
        index
        for index in range(1, len(life_cycles) - 1)
        if life_cycles[index - 1] == LifeCycleValues.RUNNING
        and life_cycles[index] == LifeCycleValues.NOT_STARTED
        and life_cycles[index + 1] == LifeCycleValues.RUNNING
    ]
    assert loop.stop_retry_monitor.resets == 3
    assert len(restarts) == loop.stop_retry_monitor.target - 1


def test_repeat_until_does_not_retry_after_giving_up():
    """
    Once the monitor has called the retrying off, no further attempt is started, so a
    finished loop stops consuming control cycles.
    """
    task = ConstFalseNode(name="task")
    loop, _, executor = _repeat_on_timeout(task, target=2)

    for _ in range(SETTLE_CYCLES):
        executor.tick()
    resets_when_given_up = loop.stop_retry_monitor.resets
    assert loop.observation_state == ObservationStateValues.FALSE

    for _ in range(SETTLE_CYCLES):
        executor.tick()

    assert loop.stop_retry_monitor.resets == resets_when_given_up
    assert task.life_cycle_state == LifeCycleValues.NOT_STARTED
    assert loop.observation_state == ObservationStateValues.FALSE


# %% the stall timeout


def _repeat_on_stall(**timeout_argument) -> RepeatOnStall:
    """
    Build a loop around a placeholder task, to read back how its stall timeout reached
    the progress monitor.
    """
    task = ConstFalseNode(name="task")
    return RepeatOnStall(
        name="loop",
        task=task,
        stop_retry_monitor=CountNodeResets(name="counter", node=task, target=1),
        **timeout_argument,
    )


def test_stall_timeout_is_measured_in_seconds():
    """
    A window longer than a day reaches the progress monitor whole, rather than losing
    its days on the way.
    """
    timeout = timedelta(days=1, seconds=30)

    loop = _repeat_on_stall(timeout=timeout)

    assert loop.retry_trigger_monitor.timeout == timeout.total_seconds()


def test_default_stall_timeout_leaves_an_attempt_time_to_converge():
    """
    The default window spans several seconds of simulated time, so an attempt is not
    declared stalled on the first control cycle in which nothing moves.
    """
    loop = _repeat_on_stall()

    assert loop.retry_trigger_monitor.timeout == timedelta(seconds=5).total_seconds()


# %% retrying a motion that stops making progress


def test_repeat_on_stall_retries_a_motion_that_stops_converging(
    pr2_world_state_reset: World,
):
    """
    An arm that has extended as far as it can stops closing on its goal, so the attempt
    is given up on and started again, until the monitor calls it off.
    """
    task = unreachable_arm_goal(pr2_world_state_reset)
    loop = RepeatOnStall(
        name="loop",
        task=task,
        stop_retry_monitor=CountNodeResets(name="counter", node=task, target=2),
        timeout=timedelta(seconds=1),
    )
    motion_statechart = MotionStatechart()
    motion_statechart.add_node(loop)
    motion_statechart.add_node(EndMotion.when_true(loop))

    executor = Executor(MotionStatechartContext(world=pr2_world_state_reset))
    executor.compile(motion_statechart=motion_statechart)
    for _ in range(2000):
        executor.tick()
        if loop.observation_state == ObservationStateValues.FALSE:
            break

    assert loop.stop_retry_monitor.resets == 2
    assert loop.observation_state == ObservationStateValues.FALSE


def test_repeat_on_stall_leaves_a_reachable_motion_alone(cylinder_bot_world: World):
    """
    A goal the robot converges on is never mistaken for a stalled attempt.
    """
    bot = cylinder_bot_world.get_kinematic_structure_entity_by_name("bot")
    task = CartesianPosition(
        root_link=cylinder_bot_world.root,
        tip_link=bot,
        goal_point=Point3(1, 0, 0, reference_frame=cylinder_bot_world.root),
    )
    loop = RepeatOnStall(
        name="loop",
        task=task,
        stop_retry_monitor=CountNodeResets(name="counter", node=task, target=1),
        timeout=timedelta(seconds=0.5),
    )
    motion_statechart = MotionStatechart()
    motion_statechart.add_node(loop)
    motion_statechart.add_node(EndMotion.when_true(loop))

    executor = Executor(MotionStatechartContext(world=cylinder_bot_world))
    executor.compile(motion_statechart=motion_statechart)
    executor.tick_until_end(2000)

    assert loop.stop_retry_monitor.resets == 0
    assert loop.observation_state == ObservationStateValues.TRUE
