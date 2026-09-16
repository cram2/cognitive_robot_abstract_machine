"""
Tests for the observation expressions built by the monitored subtree templates (see
``giskardpy/motion_statechart/monitors/templates.py``).

``build_artifacts`` reads nothing but the observations of the monitor and the monitored
node and the life cycle of the monitored node, all of which exist as variables from
construction, so each expression is evaluated by substituting values into it rather than
by ticking an executor.
"""

import pytest

from giskardpy.motion_statechart.context import MotionStatechartContext
from giskardpy.motion_statechart.data_types import (
    LifeCycleValues,
    ObservationStateValues,
)
from giskardpy.motion_statechart.graph_node import (
    DerivedConditionVariable,
    MotionStatechartNode,
)
from giskardpy.motion_statechart.monitors.templates import (
    MonitoredCompositeStatechartNode,
    PausedUntilTrue,
    PausedWhileTrue,
    StoppedWhenTrue,
)
from krrood.symbolic_math.symbolic_math import Scalar
from semantic_digital_twin.world import World

UNFINISHED_OBSERVATIONS = [ObservationStateValues.FALSE, ObservationStateValues.UNKNOWN]
"""
The observations of a monitored node that has not reached its goal.
"""

# %% evaluating a template's observation


def create_goal(
    goal_type: type[MonitoredCompositeStatechartNode],
) -> MonitoredCompositeStatechartNode:
    """
    :param goal_type: The template to instantiate.
    :return: A goal whose monitor and monitored node contribute nothing but their
        observation variables.
    """
    return goal_type(
        monitor=MotionStatechartNode(name="monitor"),
        monitored_node=MotionStatechartNode(name="monitored"),
    )


def observation_for(
    goal: MonitoredCompositeStatechartNode,
    monitored_observation: ObservationStateValues,
    monitor_observation: ObservationStateValues,
    monitored_life_cycle: LifeCycleValues = LifeCycleValues.RUNNING,
) -> ObservationStateValues:
    """
    Evaluate the observation a goal builds for one pair of input observations.

    An observation expression reads the observation a node took on the previous control
    cycle, which is also its last observation, so the same value stands for both of a
    node's observation variables. The predicates over those variables and the life cycle
    state are replaced by what they stand for first, as compiling the observation does.

    :param goal: The goal whose observation expression is evaluated.
    :param monitored_observation: What the monitored node observed.
    :param monitor_observation: What the monitor observed.
    :param monitored_life_cycle: The life cycle state the monitored node is in.
    :return: What the goal observes.
    """
    artifacts = goal.build_artifacts(MotionStatechartContext(world=World()))
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
    goal_type: type[MonitoredCompositeStatechartNode],
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
    A monitored node that succeeded observes nothing any more, so its verdict is what
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
    The observation read on the cycle after the monitor stopped the monitored node is
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
    is an outcome, so the template reports that it has none.
    """
    goal = create_goal(StoppedWhenTrue)

    assert (
        observation_for(goal, monitored_observation, monitor_observation)
        is ObservationStateValues.UNKNOWN
    )
