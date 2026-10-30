from __future__ import annotations

import json

from cramph.context import StatechartContext
from cramph.data_types import LifeCycleValues, ObservationStateValues, StatechartJSONKey
from cramph.executor import StatechartExecutor
from cramph.node import StatechartNode
from cramph.nodes_for_testing import NodeSucceedingOnObservingTrue
from cramph.statechart import Statechart
from semantic_digital_twin.world import World

# %% helpers


def _node_arriving_at_once(name: str) -> NodeSucceedingOnObservingTrue:
    """
    :return: A node that succeeds on the tick after it starts.
    """
    return NodeSucceedingOnObservingTrue(
        name=name, observation=ObservationStateValues.TRUE
    )


def _sent_through_json(data: dict) -> dict:
    """
    :return: `data` after a trip through a JSON string, as a receiver gets it.
    """
    return json.loads(json.dumps(data))


def _two_node_statechart() -> Statechart:
    """
    :return: A statechart in which `second` starts once `first` succeeded.
    """
    statechart = Statechart(context=StatechartContext(world=World()))
    statechart.add_node(first := _node_arriving_at_once("first"))
    statechart.add_node(second := _node_arriving_at_once("second"))
    second.start_condition = first.is_succeeded
    return statechart


# %% sending the nodes from an index on


def test_only_the_nodes_from_the_index_on_are_sent() -> None:
    statechart = _two_node_statechart()

    data = statechart.nodes_from_to_json(first_node_index=1)

    sent_names = [node_data["name"] for node_data in data[StatechartJSONKey.NODES]]
    assert sent_names == [statechart.nodes[1].name]


def test_only_the_conditions_of_the_sent_nodes_are_sent() -> None:
    statechart = _two_node_statechart()
    second = statechart.nodes[1]

    data = statechart.nodes_from_to_json(first_node_index=1)

    assert data[StatechartJSONKey.CONDITIONS] == [
        condition.to_json() for condition in second.conditions
    ]


# %% adding sent nodes to a running statechart


def test_sent_nodes_join_a_running_statechart_and_read_its_nodes(
    statechart_executor: StatechartExecutor,
) -> None:
    sender = _two_node_statechart()
    receiver = Statechart.from_json(
        _sent_through_json(sender.to_json()), context=statechart_executor.context
    )
    statechart_executor.compile(receiver)
    for _ in range(4):
        statechart_executor.tick()
    received_second = receiver.nodes[1]
    assert received_second.life_cycle_state == LifeCycleValues.SUCCEEDED
    third: StatechartNode = _node_arriving_at_once("third")
    sender.add_node(third)
    third.start_condition = sender.nodes[1].is_succeeded

    receiver.add_nodes_from_json(_sent_through_json(sender.nodes_from_to_json(2)))
    statechart_executor.tick()
    statechart_executor.tick()

    received_third = receiver.nodes[2]
    assert received_third.name == third.name
    assert received_third.life_cycle_state == LifeCycleValues.SUCCEEDED
    assert received_second.life_cycle_state == LifeCycleValues.SUCCEEDED
    assert (
        received_third.start_condition.free_variables()[0].statechart_node
        is received_second
    )
