"""
Tests for the monitored subtree templates (see ``cramph/src/cramph/composites.py``).

``build_artifacts`` reads nothing but the observations of the monitor and the monitored
node and the life cycle of the monitored node, all of which exist as variables from
construction, so each observation expression is evaluated by substituting values into it
rather than by ticking an executor. How the templates end and pause their monitored node
is tested by ticking an executor.
"""

import pytest

from cramph.executor import StatechartExecutor
from cramph.context import StatechartContext
from cramph.data_types import LifeCycleValues, ObservationStateValues
from cramph.node import DerivedConditionVariable
from cramph.node import StatechartNode
from cramph.composites import (
    MonitoredCompositeNode,
    PausedUntilTrue,
    PausedWhileTrue,
    Sequence,
    StoppedWhenTrue,
)
from cramph.monitors import CountTicks
from cramph.statechart import Statechart
from cramph.nodes_for_testing import (
    ConstFalseNode,
    ConstTrueNode,
    NodeFailingOnObservingFalse,
    NodeObservingNothingYet,
)
from krrood.symbolic_math.symbolic_math import Scalar
from semantic_digital_twin.world import World

SETTLE_TICKS = 6
"""
Ticks after which the templates ticked below have settled on an outcome.
"""

UNFINISHED_OBSERVATIONS = [ObservationStateValues.FALSE, ObservationStateValues.UNKNOWN]
"""
The observations of a monitored node that has not reached its goal.
"""

# %% evaluating a template's observation


def create_goal(
    goal_type: type[MonitoredCompositeNode],
) -> MonitoredCompositeNode:
    """
    :param goal_type: The template to instantiate.
    :return: A goal whose monitor and monitored node contribute nothing but their
        observation variables.
    """
    return goal_type(
        monitor=NodeObservingNothingYet(name="monitor"),
        monitored_node=NodeObservingNothingYet(name="monitored"),
    )


def observation_for(
    goal: MonitoredCompositeNode,
    monitored_observation: ObservationStateValues,
    monitor_observation: ObservationStateValues,
    monitored_life_cycle: LifeCycleValues = LifeCycleValues.RUNNING,
) -> ObservationStateValues:
    """
    Evaluate the observation a goal builds for one pair of input observations.

    An observation expression reads the observation a node took on the previous control
    tick, which is also its last observation, so the same value stands for both of a
    node's observation variables. The predicates over those variables and the life cycle
    state are replaced by what they stand for first, as compiling the observation does.

    :param goal: The goal whose observation expression is evaluated.
    :param monitored_observation: What the monitored node observed.
    :param monitor_observation: What the monitor observed.
    :param monitored_life_cycle: The life cycle state the monitored node is in.
    :return: What the goal observes.
    """
    artifacts = goal.build_artifacts()
    # A template may hand back a node's observation variable unwrapped, which cannot be
    # copied and therefore not substituted into.
    substituted = DerivedConditionVariable.substitute_in(
        Scalar(artifacts.observation)
    ).substitute(
        [
            goal.monitored_node.observation_variable,
            goal.monitored_node.last_observation,
            goal.monitored_node.life_cycle_variable,
            goal.monitor.observation_variable,
            goal.monitor.last_observation,
        ],
        [
            monitored_observation,
            monitored_observation,
            monitored_life_cycle,
            monitor_observation,
            monitor_observation,
        ],
    )
    return ObservationStateValues(float(substituted))


# %% pausing leaves the outcome to the monitored node


@pytest.mark.parametrize("goal_type", [PausedWhileTrue, PausedUntilTrue])
@pytest.mark.parametrize("monitored_observation", list(ObservationStateValues))
@pytest.mark.parametrize("monitor_observation", list(ObservationStateValues))
def test_pausing_templates_observe_the_monitored_node(
    goal_type: type[MonitoredCompositeNode],
    monitored_observation: ObservationStateValues,
    monitor_observation: ObservationStateValues,
) -> None:
    """
    A monitor that only pauses never changes the outcome, so the goal reports exactly
    what the monitored node observes.
    """
    goal = create_goal(goal_type)

    assert (
        observation_for(goal, monitored_observation, monitor_observation)
        is monitored_observation
    )


# %% stopping a monitored node


@pytest.mark.parametrize("monitor_observation", list(ObservationStateValues))
def test_stopped_when_true_succeeds_once_the_monitored_node_succeeded(
    monitor_observation: ObservationStateValues,
) -> None:
    """
    A monitored node that reached its goal makes the template succeed, whatever the
    monitor observes.
    """
    goal = create_goal(StoppedWhenTrue)

    assert (
        observation_for(goal, ObservationStateValues.TRUE, monitor_observation)
        is ObservationStateValues.TRUE
    )


@pytest.mark.parametrize("monitored_observation", UNFINISHED_OBSERVATIONS)
def test_stopped_when_true_fails_when_it_stopped_an_unfinished_node(
    monitored_observation: ObservationStateValues,
) -> None:
    """
    A monitor that fires before the monitored node reached its goal ends it as a
    failure.
    """
    goal = create_goal(StoppedWhenTrue)

    assert (
        observation_for(goal, monitored_observation, ObservationStateValues.TRUE)
        is ObservationStateValues.FALSE
    )


@pytest.mark.parametrize("monitor_observation", list(ObservationStateValues))
def test_stopped_when_true_keeps_succeeding_once_the_monitored_node_succeeded(
    monitor_observation: ObservationStateValues,
) -> None:
    """
    A monitored node that succeeded observes nothing any more, so its outcome is what
    makes the template succeed.
    """
    goal = create_goal(StoppedWhenTrue)

    assert (
        observation_for(
            goal,
            ObservationStateValues.UNKNOWN,
            monitor_observation,
            monitored_life_cycle=LifeCycleValues.SUCCEEDED,
        )
        is ObservationStateValues.TRUE
    )


def test_stopped_when_true_fails_when_it_stopped_a_node_observing_true() -> None:
    """
    The observation read on the tick after the monitor stopped the monitored node is
    still the True it took while running, which must not count as a success.
    """
    goal = create_goal(StoppedWhenTrue)

    assert (
        observation_for(
            goal,
            ObservationStateValues.TRUE,
            ObservationStateValues.TRUE,
            monitored_life_cycle=LifeCycleValues.INTERRUPTED,
        )
        is ObservationStateValues.FALSE
    )


@pytest.mark.parametrize("monitored_observation", UNFINISHED_OBSERVATIONS)
@pytest.mark.parametrize("monitor_observation", UNFINISHED_OBSERVATIONS)
def test_stopped_when_true_stays_unknown_while_the_monitored_node_runs(
    monitored_observation: ObservationStateValues,
    monitor_observation: ObservationStateValues,
) -> None:
    """
    Neither a monitored node that has not succeeded yet nor a monitor that has not fired
    decides anything, so the template observes Unknown.
    """
    goal = create_goal(StoppedWhenTrue)

    assert (
        observation_for(goal, monitored_observation, monitor_observation)
        is ObservationStateValues.UNKNOWN
    )


# %% a monitored node that ends on its own


def compile_chart(nodes: list[StatechartNode]) -> StatechartExecutor:
    """
    :param nodes: The top level nodes of a fresh statechart.
    :return: The executor, after compiling the statechart.
    """
    executor = StatechartExecutor(StatechartContext(world=World()))
    statechart = Statechart(context=executor.context)
    statechart.add_nodes(nodes)
    executor.compile(statechart=statechart)
    return executor


def tick_compiled(node: StatechartNode) -> StatechartExecutor:
    """
    :param node: The node to run as the only top level node of a fresh statechart.
    :return: The executor, after compiling the statechart and ticking it for
        :data:`SETTLE_TICKS` ticks.
    """
    executor = compile_chart([node])
    for _ in range(SETTLE_TICKS):
        executor.tick()
    return executor


@pytest.mark.parametrize(
    "goal_type, monitor_type",
    [
        (PausedWhileTrue, ConstFalseNode),
        (PausedUntilTrue, ConstTrueNode),
        (StoppedWhenTrue, ConstFalseNode),
    ],
)
def test_a_template_fails_once_its_monitored_node_failed_on_its_own(
    goal_type: type[MonitoredCompositeNode],
    monitor_type: type[StatechartNode],
) -> None:
    """
    A monitored node that failed never arrives, so the template must not keep its owner
    waiting for it.

    The monitor lets the monitored node run and never stops it.
    """
    monitored_node = NodeFailingOnObservingFalse(
        name="monitored", observation=ObservationStateValues.FALSE
    )
    goal = goal_type(
        monitor=monitor_type(name="monitor"), monitored_node=monitored_node
    )

    tick_compiled(goal)

    assert monitored_node.life_cycle_state == LifeCycleValues.FAILED
    assert goal.life_cycle_state == LifeCycleValues.FAILED


@pytest.mark.parametrize("goal_type", [PausedWhileTrue, StoppedWhenTrue])
def test_a_template_whose_monitored_node_failed_observing_true_fails_its_sequence(
    goal_type: type[MonitoredCompositeNode],
) -> None:
    """
    What a monitored node observed on the tick it failed is no arrival, so a sequence
    holding the template must not succeed on it.
    """
    monitored_node = ConstTrueNode(name="monitored")
    monitored_node.fail_condition = monitored_node.observes_true
    goal = goal_type(
        monitor=ConstFalseNode(name="monitor"), monitored_node=monitored_node
    )
    sequence = Sequence(nodes=[goal])

    tick_compiled(sequence)

    assert goal.life_cycle_state == LifeCycleValues.FAILED
    assert sequence.life_cycle_state == LifeCycleValues.FAILED


# %% pausing from the start


def test_paused_until_true_never_runs_its_node_before_the_monitor_observed_true() -> (
    None
):
    """
    A monitor that has not observed True yet holds the monitored node from the control
    tick the template starts in, so the node's constraints never act before that.
    """
    delay = CountTicks(name="delay", ticks=SETTLE_TICKS // 2)
    monitored_node = ConstTrueNode(name="monitored")
    goal = PausedUntilTrue(
        monitor=ConstFalseNode(name="monitor"), monitored_node=monitored_node
    )
    goal.start_condition = delay.observes_true
    executor = compile_chart([delay, goal])

    life_cycles = []
    for _ in range(SETTLE_TICKS):
        executor.tick()
        life_cycles.append(monitored_node.life_cycle_state)

    assert LifeCycleValues.PAUSED in life_cycles
    assert LifeCycleValues.RUNNING not in life_cycles
