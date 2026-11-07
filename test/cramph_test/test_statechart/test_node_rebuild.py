from __future__ import annotations

from dataclasses import dataclass, field

import cramph
import cramph.nodes_for_testing
from cramph.composites import Parallel, Sequence, StartedWhenTrue
from cramph.data_types import LifeCycleValues, ObservationStateValues
from cramph.executor import StatechartExecutor
from cramph.monitors import CountNodeResets
from cramph.node import (
    DerivedConditionVariable,
    EndStatechart,
    NodeRebuild,
    StatechartNode,
)
from cramph.nodes_for_testing import NodeSucceedingOnObservingTrue
from cramph.statechart import Statechart
from typing_extensions import Any

from .node_rebuild_audit import NodeRebuildAudit

# %% mimics


@dataclass
class RebuildLeavingNodesOut(NodeRebuild):
    """
    Builds every node anew except those it leaves out, and keeps every other value as it
    is.
    """

    left_out: list[StatechartNode] = field(default_factory=list)
    """
    The nodes it leaves out.
    """

    built: dict[StatechartNode, StatechartNode | None] = field(default_factory=dict)
    """
    The node each node was built anew as, None for one left out.
    """

    def rebuilt(self, node: StatechartNode) -> StatechartNode | None:
        if node not in self.built:
            self.built[node] = None if node in self.left_out else node.rebuilt_by(self)
        return self.built[node]

    def rebuilt_children(
        self, owner: StatechartNode, children: list[StatechartNode]
    ) -> list[StatechartNode]:
        return [
            rebuilt
            for child in children
            if (rebuilt := self.rebuilt(child)) is not None
        ]

    def rebound(self, value: Any) -> Any:
        return value


# %% helpers


def _succeeding(name: str) -> NodeSucceedingOnObservingTrue:
    """
    :return: A node that succeeds on the tick after it starts.
    """
    return NodeSucceedingOnObservingTrue(
        name=name, observation=ObservationStateValues.TRUE
    )


# %% building a node anew


def test_a_sequence_is_built_anew_holding_its_children_built_anew() -> None:
    first, second = _succeeding("first"), _succeeding("second")
    sequence = Sequence(name="sequence", nodes=[first, second])
    rebuild = RebuildLeavingNodesOut()

    rebuilt = rebuild.rebuilt(sequence)

    assert rebuilt is not sequence
    assert rebuilt.name == sequence.name
    assert rebuilt.nodes == [rebuild.built[first], rebuild.built[second]]
    assert all(rebuild.built[child] is not child for child in [first, second])


def test_a_sequence_holding_no_child_built_anew_is_left_out() -> None:
    child = _succeeding("child")
    rebuild = RebuildLeavingNodesOut(left_out=[child])

    assert rebuild.rebuilt(Sequence(nodes=[child])) is None


def test_a_parallel_built_anew_needs_as_many_successes() -> None:
    parallel = Parallel(nodes=[_succeeding("a"), _succeeding("b")], minimum_success=1)

    rebuilt = RebuildLeavingNodesOut().rebuilt(parallel)

    assert rebuilt.minimum_success == parallel.minimum_success


def test_a_monitored_node_is_built_anew_with_its_monitor_built_anew() -> None:
    monitor, monitored_node = _succeeding("monitor"), _succeeding("monitored")
    rebuild = RebuildLeavingNodesOut()

    rebuilt = rebuild.rebuilt(
        StartedWhenTrue(monitor=monitor, monitored_node=monitored_node)
    )

    assert rebuilt.monitor is rebuild.built[monitor]
    assert rebuilt.monitored_node is rebuild.built[monitored_node]


def test_a_monitored_node_without_its_monitored_node_is_left_out() -> None:
    monitored_node = _succeeding("monitored")
    rebuild = RebuildLeavingNodesOut(left_out=[monitored_node])

    rebuilt = rebuild.rebuilt(
        StartedWhenTrue(monitor=_succeeding("monitor"), monitored_node=monitored_node)
    )

    assert rebuilt is None


def test_a_monitor_counting_resets_counts_those_of_the_node_built_anew() -> None:
    counted = _succeeding("counted")
    rebuild = RebuildLeavingNodesOut()

    rebuilt = rebuild.rebuilt(CountNodeResets(node=counted, target=2))

    assert rebuilt.node is rebuild.built[counted]
    assert rebuilt.target == 2


def test_every_cramph_node_holding_nodes_builds_them_anew() -> None:
    audit = NodeRebuildAudit(
        modules=[cramph], ignored_modules=[cramph.nodes_for_testing]
    )
    assert audit.nodes_holding_nodes_they_do_not_rebuild() == []


# %% a condition reading the counterparts of the nodes it reads


def test_a_condition_reads_the_counterpart_of_a_node_it_reads() -> None:
    read, counterpart = _succeeding("read"), _succeeding("counterpart")

    carried_over = DerivedConditionVariable.reading_counterparts(
        read.is_succeeded, lambda node: counterpart
    )

    assert carried_over.free_variables() == [counterpart.is_succeeded]


def test_a_condition_reads_a_node_without_counterpart_as_it_is_now(
    statechart_executor: StatechartExecutor,
) -> None:
    statechart = Statechart(context=statechart_executor.context)
    ended = _succeeding("ended")
    statechart.add_node(ended)
    statechart.add_node(EndStatechart.when_true(ended))
    statechart_executor.compile(statechart)
    statechart_executor.tick_until_end(timeout=30)
    assert ended.life_cycle_state == LifeCycleValues.SUCCEEDED

    succeeded = DerivedConditionVariable.reading_counterparts(
        ended.is_succeeded, lambda node: None
    )
    failed = DerivedConditionVariable.reading_counterparts(
        ended.is_failed, lambda node: None
    )

    assert succeeded.is_constant_true()
    assert failed.is_constant_false()
