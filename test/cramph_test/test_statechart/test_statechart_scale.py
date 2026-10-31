import time

import pytest

from cramph.executor import StatechartExecutor
from cramph.context import StatechartContext
from cramph.data_types import LifeCycleValues
from cramph.composites import Sequence
from cramph.node import EndStatechart, StatechartNode
from cramph.statechart import Statechart
from cramph.nodes_for_testing import ConstTrueNode, ConstFalseNode
from semantic_digital_twin.world import World


def _build_chain(statechart: Statechart, length: int) -> list[StatechartNode]:
    """
    Builds a linear chain of ConstTrueNode instances directly on `statechart`, wired the
    same way :class:`Sequence` wires its children: each node starts once the previous
    node's observation is true, and ends on its own observation, so only one node in the
    chain is ever RUNNING at a time.
    """
    chain: list[StatechartNode] = []
    previous = None
    for _ in range(length):
        node = ConstTrueNode()
        statechart.add_node(node)
        if previous is not None:
            node.start_condition = previous.observes_true
        node.success_condition = node.observes_true
        chain.append(node)
        previous = node
    return chain


@pytest.mark.slow
@pytest.mark.parametrize("node_count", [100, 1_000, 10_000])
def test_long_sequence_scale(node_count: int):
    """
    Builds a single long Sequence of cheap ConstTrueNode instances, where exactly one
    node is RUNNING at any time, and measures compile/tick time as the graph grows.
    """
    executor = StatechartExecutor(StatechartContext(world=World()))
    msc = Statechart(context=executor.context)
    sequence = Sequence(nodes=[ConstTrueNode() for _ in range(node_count)])
    msc.add_node(sequence)
    msc.add_node(EndStatechart.when_true(sequence))

    t0 = time.perf_counter()
    executor.compile(statechart=msc)
    t_compile = time.perf_counter() - t0

    t0 = time.perf_counter()
    executor.tick_until_end(timeout=node_count + 10)
    t_tick = time.perf_counter() - t0

    print(
        f"[long_sequence] N={node_count} compile={t_compile:.4f}s tick={t_tick:.4f}s "
        f"({t_tick / node_count * 1e6:.2f} us/node)"
    )

    assert msc.is_ended()
    assert executor.tick_count == node_count + 2


@pytest.mark.slow
@pytest.mark.parametrize("branch_length", [10, 100, 1_000])
def test_many_alternative_branches_scale(branch_length: int):
    """
    Builds many branches of a linear ConstTrueNode chain where only the first branch is
    ever traveled; the remaining branches are gated off by a permanently-false condition
    and stay NOT_STARTED.

    Measures compile/tick time as the total, mostly dormant, graph grows.
    """
    branch_count = 10
    executor = StatechartExecutor(StatechartContext(world=World()))
    msc = Statechart(context=executor.context)

    gate = ConstFalseNode()
    msc.add_node(gate)

    branches = [_build_chain(msc, branch_length) for _ in range(branch_count)]
    active_branch, dead_branches = branches[0], branches[1:]

    for dead_branch in dead_branches:
        dead_branch[0].start_condition = gate.observes_true

    msc.add_node(EndStatechart.when_true(active_branch[-1]))

    total_nodes = branch_count * branch_length + 2  # + gate + EndStatechart

    t0 = time.perf_counter()
    executor.compile(statechart=msc)
    t_compile = time.perf_counter() - t0

    t0 = time.perf_counter()
    executor.tick_until_end(timeout=branch_length + 10)
    t_tick = time.perf_counter() - t0

    print(
        f"[alternative_branches] branches={branch_count} length={branch_length} "
        f"total_nodes={total_nodes} compile={t_compile:.4f}s tick={t_tick:.4f}s "
        f"({t_tick / total_nodes * 1e6:.2f} us/node)"
    )

    assert msc.is_ended()
    assert executor.tick_count == branch_length + 1
    for dead_branch in dead_branches:
        for node in dead_branch:
            assert node.life_cycle_state == LifeCycleValues.NOT_STARTED
