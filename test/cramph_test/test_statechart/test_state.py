"""
Tests of the state a statechart keeps for each of its nodes.
"""

from __future__ import annotations

from cramph.context import StatechartContext
from cramph.data_types import LifeCycleValues, ObservationStateValues
from cramph.monitors import CountTicks
from cramph.statechart import Statechart

# %% a state recorded before nodes joined


class TestStateCoveringNodesThatJoinedLater:
    """
    A state recorded before further nodes joined its statechart is completed with the
    value a node starts out with.
    """

    def test_nodes_that_joined_later_hold_the_default_value(
        self, statechart_context: StatechartContext
    ) -> None:
        statechart = Statechart(context=statechart_context)
        first_node = CountTicks(name="first", ticks=1)
        statechart.add_node(first_node)
        statechart.life_cycle_state[first_node] = LifeCycleValues.RUNNING
        statechart.observation_state[first_node] = ObservationStateValues.TRUE
        recorded_life_cycle = statechart.life_cycle_state.snapshot()
        recorded_observation = statechart.observation_state.snapshot()
        statechart.add_node(CountTicks(name="second", ticks=1))

        assert recorded_life_cycle.covering_every_node().data.tolist() == [
            LifeCycleValues.RUNNING,
            recorded_life_cycle.default_value,
        ]
        assert recorded_observation.covering_every_node().data.tolist() == [
            ObservationStateValues.TRUE,
            recorded_observation.default_value,
        ]
