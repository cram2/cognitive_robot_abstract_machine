"""
Tests for :class:`~cramph.reported_statechart.ReportedStatechart`, the copy of a
statechart rebuilt from the payloads a server reports it in.
"""

from __future__ import annotations

import json

from typing_extensions import Any

from cramph.composites import (
    ChildChooserAccess,
    CompositeNodeChoosingItsChild,
    Sequence,
)
from cramph.context import StatechartContext
from cramph.data_types import StatechartPayloadKey
from cramph.executor import StatechartExecutor
from cramph.monitors import CountTicks
from cramph.nodes_for_testing import ConstTrueNode
from cramph.reported_statechart import ReportedStatechart
from cramph.statechart import Statechart

from .test_composite_node_choosing_its_child import ChooserAnsweringInTurn

# %% helpers


def _states_payload(executor: StatechartExecutor) -> dict[str, Any]:
    """
    :return: The states of the statechart of `executor`, as a server reports them,
        after a trip through a JSON string.
    """
    statechart = executor.statechart
    payload = {
        StatechartPayloadKey.GOAL_ID: 0,
        StatechartPayloadKey.TICK_COUNT: executor.tick_count,
        StatechartPayloadKey.LIFE_CYCLE_STATE: statechart.life_cycle_state.to_json(),
        StatechartPayloadKey.OBSERVATION_STATE: statechart.observation_state.to_json(),
        StatechartPayloadKey.LAST_OBSERVATION_STATE: statechart.last_observation_state.to_json(),
    }
    return json.loads(json.dumps(payload))


# %% following the payloads


def test_states_reported_before_any_structure_are_ignored(
    statechart_context: StatechartContext,
    statechart_executor: StatechartExecutor,
) -> None:
    statechart = Statechart(context=statechart_context)
    statechart.add_node(ConstTrueNode(name="running"))
    statechart_executor.compile(statechart)
    reported_statechart = ReportedStatechart()

    reported_statechart.take_over(_states_payload(statechart_executor))

    assert reported_statechart.statechart is None


def test_the_copy_holds_the_children_chosen_while_the_statechart_runs(
    statechart_context: StatechartContext,
    statechart_executor: StatechartExecutor,
) -> None:
    statechart_context.add_extension(
        ChildChooserAccess(
            choosers=[ChooserAnsweringInTurn(children=[ConstTrueNode(name="chosen")])]
        )
    )
    statechart = Statechart(context=statechart_context)
    statechart.add_node(
        Sequence(
            nodes=[
                CountTicks(name="delay", ticks=2),
                CompositeNodeChoosingItsChild(name="statement"),
            ]
        )
    )
    statechart_executor.compile(statechart)
    reported_statechart = ReportedStatechart()
    reported_statechart.take_over(
        {
            **_states_payload(statechart_executor),
            StatechartPayloadKey.STATECHART: json.loads(
                json.dumps(statechart.create_structure_copy().to_json())
            ),
        }
    )
    reported_node_count = len(statechart.nodes)

    for _ in range(4):
        statechart_executor.tick()
    reported_statechart.take_over(
        {
            **_states_payload(statechart_executor),
            StatechartPayloadKey.JOINED_NODES: json.loads(
                json.dumps(
                    statechart.create_structure_copy().nodes_from_to_json(
                        first_node_index=reported_node_count
                    )
                )
            ),
        }
    )

    assert [node.name for node in reported_statechart.statechart.nodes] == [
        node.name for node in statechart.nodes
    ]
    assert (
        reported_statechart.statechart.life_cycle_state == statechart.life_cycle_state
    )
