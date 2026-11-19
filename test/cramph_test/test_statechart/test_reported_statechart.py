"""
Tests for :class:`~cramph.reported_statechart.ReportedStatechart`, the copy of a
statechart rebuilt from the payloads a server reports it in.
"""

from __future__ import annotations

import json
from dataclasses import dataclass

import pytest
from typing_extensions import Any

from cramph.composites import (
    ChildChooserAccess,
    CompositeNodeChoosingItsChild,
    Sequence,
)
from cramph.context import StatechartContext
from cramph.reported_statechart import ReportedStatechart, StatechartPayloadKey
from cramph.executor import StatechartExecutor
from cramph.monitors import CountTicks
from cramph.nodes_for_testing import ConstTrueNode
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
    statechart.add_node(node=ConstTrueNode(name="running"))
    statechart_executor.compile(statechart=statechart)
    reported_statechart = ReportedStatechart()

    reported_statechart.take_over(payload=_states_payload(statechart_executor))

    assert reported_statechart.statechart is None


@dataclass(kw_only=True)
class ChoiceReportedWhileRunning:
    """
    A statechart whose node chose its child while it ran, and the copy rebuilt from the
    structure reported at compile time and the change blocks reported afterwards.
    """

    statechart: Statechart
    """
    The statechart that ran.
    """

    choosing_node: CompositeNodeChoosingItsChild
    """
    The node of :attr:`statechart` that chose its child.
    """

    reported_statechart: ReportedStatechart
    """
    The copy rebuilt from the payloads.
    """


@pytest.fixture
def choice_reported_while_running(
    statechart_context: StatechartContext,
    statechart_executor: StatechartExecutor,
) -> ChoiceReportedWhileRunning:
    """
    A node that chooses its child once a delay ended, reported before and after it
    chose.
    """
    statechart_context.add_extension(
        extension=ChildChooserAccess(
            choosers=[ChooserAnsweringInTurn(children=[ConstTrueNode(name="chosen")])]
        )
    )
    statechart = Statechart(context=statechart_context)
    choosing_node = CompositeNodeChoosingItsChild(name="statement")
    statechart.add_node(
        node=Sequence(nodes=[CountTicks(name="delay", ticks=2), choosing_node])
    )
    statechart_executor.compile(statechart=statechart)
    reported_statechart = ReportedStatechart()
    reported_statechart.take_over(
        payload={
            **_states_payload(statechart_executor),
            StatechartPayloadKey.STATECHART: json.loads(
                json.dumps(statechart.create_structure_copy().to_json())
            ),
        }
    )
    reported_block_count = len(statechart.change_blocks)

    for _ in range(4):
        statechart_executor.tick()
    reported_statechart.take_over(
        payload={
            **_states_payload(statechart_executor),
            StatechartPayloadKey.CHANGE_BLOCKS: json.loads(
                json.dumps(
                    [
                        block.create_structure_copy().to_json()
                        for block in statechart.change_blocks[reported_block_count:]
                    ]
                )
            ),
        }
    )
    return ChoiceReportedWhileRunning(
        statechart=statechart,
        choosing_node=choosing_node,
        reported_statechart=reported_statechart,
    )


def test_the_copy_holds_the_children_chosen_while_the_statechart_runs(
    choice_reported_while_running: ChoiceReportedWhileRunning,
) -> None:
    """
    The child joined after the structure was reported, and its states follow.
    """
    statechart = choice_reported_while_running.statechart
    reported_statechart = choice_reported_while_running.reported_statechart

    assert [node.name for node in reported_statechart.statechart.nodes] == [
        node.name for node in statechart.nodes
    ]
    assert (
        reported_statechart.statechart.life_cycle_state == statechart.life_cycle_state
    )


def test_the_copy_reads_the_conditions_a_choice_gave_its_node(
    choice_reported_while_running: ChoiceReportedWhileRunning,
) -> None:
    """
    The node that chose succeeds once its child succeeds, although it joined the copy
    before it chose.
    """
    choosing_node = choice_reported_while_running.choosing_node
    reported_node = (
        choice_reported_while_running.reported_statechart.statechart.get_node_by_index(
            index=choosing_node.index
        )
    )

    assert [str(condition) for condition in reported_node.conditions] == [
        str(condition) for condition in choosing_node.conditions
    ]
