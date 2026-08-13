"""
Tests for plan node callbacks: registered ``PlanCallback``s are notified when the
performed root starts and ends, and when each executed motion's giskard task transitions
through its life cycle during simulated execution.
"""

from dataclasses import dataclass, field

from typing_extensions import List, Tuple

from coraplex.execution_environment import simulated_robot
from coraplex.plans.factories import sequential
from coraplex.plans.plan_callbacks import PlanCallback
from coraplex.plans.plan_node import MotionNode, PlanNode
from coraplex.robot_plans.actions.core.robot_body import MoveTorsoAction
from semantic_digital_twin.datastructures.definitions import TorsoState

# %% recording callback


@dataclass
class StartEndRecorder(PlanCallback):
    """
    Records every start and end notification it receives, in order.
    """

    events: List[Tuple[str, PlanNode]] = field(default_factory=list)
    """
    The recorded (event, node) pairs in notification order.
    """

    def on_start(self, node: PlanNode):
        self.events.append(("start", node))

    def on_end(self, node: PlanNode):
        self.events.append(("end", node))


# %% perform notifies callbacks


def test_perform_notifies_root_and_motion_nodes(immutable_model_world):
    """
    Performing a plan notifies registered callbacks of the performed root's start and
    end, and of every executed motion node's start and end, each exactly once and in
    execution order.
    """
    world, robot_view, context = immutable_model_world
    plan = sequential([MoveTorsoAction(TorsoState.HIGH)], context=context).plan
    recorder = StartEndRecorder()
    plan.node_callbacks.append(recorder)

    with simulated_robot:
        plan.perform()

    assert recorder.events[0] == ("start", plan.root)
    assert recorder.events[-1] == ("end", plan.root)

    motion_nodes = [node for node in plan.all_nodes if isinstance(node, MotionNode)]
    assert len(motion_nodes) > 0
    started_motions = [
        node
        for event, node in recorder.events
        if event == "start" and isinstance(node, MotionNode)
    ]
    ended_motions = [
        node
        for event, node in recorder.events
        if event == "end" and isinstance(node, MotionNode)
    ]
    assert started_motions == motion_nodes
    assert ended_motions == motion_nodes
    for motion_node in motion_nodes:
        start_index = recorder.events.index(("start", motion_node))
        end_index = recorder.events.index(("end", motion_node))
        assert start_index < end_index


# %% simulated execution notifies motion ticks


@dataclass
class MotionTickRecorder(PlanCallback):
    """
    Records every statechart the motion executor reports a tick for.
    """

    statecharts: List[object] = field(default_factory=list)
    """
    The reported statecharts in notification order.
    """

    def on_motion_tick(self, statechart):
        self.statecharts.append(statechart)


def test_simulated_execution_notifies_every_motion_tick(immutable_model_world):
    """
    While the simulated executor ticks a plan's motions, every tick is reported with the
    statechart being executed.
    """
    world, robot_view, context = immutable_model_world
    plan = sequential([MoveTorsoAction(TorsoState.HIGH)], context=context).plan
    recorder = MotionTickRecorder()
    plan.node_callbacks.append(recorder)

    with simulated_robot:
        plan.perform()

    assert len(recorder.statecharts) > 0
    assert all(
        statechart is recorder.statecharts[0] for statechart in recorder.statecharts
    )
