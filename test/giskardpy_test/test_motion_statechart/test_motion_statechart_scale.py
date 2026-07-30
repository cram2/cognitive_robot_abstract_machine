import time

import pytest

from giskardpy.executor import Executor
from giskardpy.motion_statechart.context import MotionStatechartContext
from giskardpy.motion_statechart.data_types import LifeCycleValues
from giskardpy.motion_statechart.goals.templates import Sequence
from giskardpy.motion_statechart.graph_node import EndMotion, MotionStatechartNode
from giskardpy.motion_statechart.motion_statechart import MotionStatechart
from giskardpy.motion_statechart.nodes_for_testing.nodes_for_testing import ConstTrueNode, ConstFalseNode
from semantic_digital_twin.world import World


def _build_chain(
    motion_statechart: MotionStatechart, length: int
) -> list[MotionStatechartNode]:
    """
    Builds a linear chain of ConstTrueNode instances directly on `motion_statechart`, wired the
    same way :class:`Sequence` wires its children: each node starts once the previous node's
    observation is true, and ends on its own observation, so only one node in the chain is ever
    RUNNING at a time.
    """
    chain: list[MotionStatechartNode] = []
    previous = None
    for _ in range(length):
        node = ConstTrueNode()
        motion_statechart.add_node(node)
        if previous is not None:
            node.start_condition = previous.observation_variable
        node.end_condition = node.observation_variable
        chain.append(node)
        previous = node
    return chain


@pytest.mark.slow
@pytest.mark.parametrize("node_count", [100, 1_000, 10_000])
def test_long_sequence_scale(node_count: int):
    """
    Builds a single long Sequence of cheap ConstTrueNode instances, where exactly one node is
    RUNNING at any time, and measures compile/tick time as the graph grows.
    """
    msc = MotionStatechart()
    sequence = Sequence(nodes=[ConstTrueNode() for _ in range(node_count)])
    msc.add_node(sequence)
    msc.add_node(EndMotion.when_true(sequence))

    executor = Executor(MotionStatechartContext(world=World()))

    t0 = time.perf_counter()
    executor.compile(motion_statechart=msc)
    t_compile = time.perf_counter() - t0

    t0 = time.perf_counter()
    executor.tick_until_end(timeout=node_count + 10)
    t_tick = time.perf_counter() - t0

    print(
        f"[long_sequence] N={node_count} compile={t_compile:.4f}s tick={t_tick:.4f}s "
        f"({t_tick / node_count * 1e6:.2f} us/node)"
    )

    assert msc.is_end_motion()
    assert executor.control_cycles == node_count + 2


@pytest.mark.slow
@pytest.mark.parametrize("branch_length", [10, 100, 1_000])
def test_many_alternative_branches_scale(branch_length: int):
    """
    Builds many branches of a linear ConstTrueNode chain where only the first branch is ever
    traveled; the remaining branches are gated off by a permanently-false condition and stay
    NOT_STARTED. Measures compile/tick time as the total, mostly dormant, graph grows.
    """
    branch_count = 10
    msc = MotionStatechart()

    gate = ConstFalseNode()
    msc.add_node(gate)

    branches = [_build_chain(msc, branch_length) for _ in range(branch_count)]
    active_branch, dead_branches = branches[0], branches[1:]

    for dead_branch in dead_branches:
        dead_branch[0].start_condition = gate.observation_variable

    msc.add_node(EndMotion.when_true(active_branch[-1]))

    executor = Executor(MotionStatechartContext(world=World()))
    total_nodes = branch_count * branch_length + 2  # + gate + EndMotion

    t0 = time.perf_counter()
    executor.compile(motion_statechart=msc)
    t_compile = time.perf_counter() - t0

    t0 = time.perf_counter()
    executor.tick_until_end(timeout=branch_length + 10)
    t_tick = time.perf_counter() - t0

    print(
        f"[alternative_branches] branches={branch_count} length={branch_length} "
        f"total_nodes={total_nodes} compile={t_compile:.4f}s tick={t_tick:.4f}s "
        f"({t_tick / total_nodes * 1e6:.2f} us/node)"
    )

    assert msc.is_end_motion()
    assert executor.control_cycles == branch_length + 1
    for dead_branch in dead_branches:
        for node in dead_branch:
            assert node.life_cycle_state == LifeCycleValues.NOT_STARTED
