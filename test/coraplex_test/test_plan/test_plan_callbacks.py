"""
Tests for plan node callbacks: registered ``PlanCallback``s are notified when the
performed root starts and ends, and when each executed motion's giskard task transitions
through its life cycle during simulated execution.
"""

from dataclasses import dataclass, field
from enum import StrEnum

from typing_extensions import List, Tuple

from coraplex.execution_environment import simulated_robot
from coraplex.plans.factories import sequential
from coraplex.plans.plan_callbacks import PlanCallback
from coraplex.plans.plan_node import MotionNode, PlanNode
from coraplex.robot_plans.actions.core.robot_body import MoveTorsoAction
from semantic_digital_twin.datastructures.definitions import TorsoState
from semantic_digital_twin.world import World

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


# %% the world may be written before each control cycle


class MotionTickMoment(StrEnum):
    """
    The two moments of one simulated tick that callbacks are notified at.
    """

    BEFORE_CONTROL_CYCLE = "before_control_cycle"
    AFTER_CONTROL_CYCLE = "after_control_cycle"


@dataclass
class MotionTickMomentRecorder(PlanCallback):
    """
    Records both notifications of every motion tick, with the world state version each
    of them arrived at.
    """

    world: World = field(kw_only=True, default=None)
    """
    The world whose state version is read at each notification.
    """

    moments: List[Tuple[MotionTickMoment, int]] = field(default_factory=list)
    """
    The recorded (moment, world state version) pairs in notification order.
    """

    def before_motion_tick(self, statechart):
        self.moments.append(
            (MotionTickMoment.BEFORE_CONTROL_CYCLE, self.world.state.version)
        )

    def on_motion_tick(self, statechart):
        self.moments.append(
            (MotionTickMoment.AFTER_CONTROL_CYCLE, self.world.state.version)
        )


def test_callbacks_are_notified_before_every_control_cycle(immutable_model_world):
    """
    Every simulated tick notifies its plan's callbacks before computing its control
    cycle, so a world write made from that notification is part of the state the cycle
    runs on.
    """
    world, robot_view, context = immutable_model_world
    plan = sequential([MoveTorsoAction(TorsoState.HIGH)], context=context).plan
    recorder = MotionTickMomentRecorder(world=world)
    plan.node_callbacks.append(recorder)

    with simulated_robot:
        plan.perform()

    tick_count = len(recorder.moments) // 2
    assert tick_count > 0
    assert [moment for moment, _ in recorder.moments] == [
        MotionTickMoment.BEFORE_CONTROL_CYCLE,
        MotionTickMoment.AFTER_CONTROL_CYCLE,
    ] * tick_count
    assert [
        (before_version, after_version)
        for (_, before_version), (_, after_version) in zip(
            recorder.moments[::2], recorder.moments[1::2]
        )
        if before_version >= after_version
    ] == []
