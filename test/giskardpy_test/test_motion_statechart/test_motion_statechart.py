import json
import logging
import time
from dataclasses import dataclass, field, fields

import numpy as np
import pytest
from giskardpy.data_types.exceptions import DuplicateNameException
from giskardpy.executor import Executor
from giskardpy.motion_statechart.context import MotionStatechartContext
from giskardpy.motion_statechart.data_types import (
    LifeCycleValues,
    ObservationStateValues,
    DefaultWeights,
    TransitionKind,
)
from giskardpy.motion_statechart.exceptions import (
    NotInMotionStatechartError,
    EndMotionInGoalError,
    InputNotExpressionError,
    SelfInStartConditionError,
    NonObservationVariableError,
    NodeAlreadyBelongsToDifferentNodeError,
    ConditionScopeError,
)
from giskardpy.motion_statechart.goals.templates import Sequence, Parallel
from giskardpy.motion_statechart.graph_node import (
    EndMotion,
    CancelMotion,
    Goal,
    MotionStatechartNode,
    NodeArtifacts,
    TrinaryCondition,
)
from giskardpy.motion_statechart.graph_node import ThreadPayloadMonitor
from giskardpy.motion_statechart.monitors.monitors import LocalMinimumReached
from giskardpy.motion_statechart.monitors.payload_monitors import (
    Print,
    Pulse,
    CountSeconds,
    CountControlCycles,
    CountSimulationTimeSeconds,
)
from giskardpy.motion_statechart.motion_statechart import (
    MotionStatechart,
)
from giskardpy.motion_statechart.tasks.align_planes import AlignPlanes
from giskardpy.motion_statechart.tasks.cartesian_tasks import (
    CartesianPose,
)
from giskardpy.motion_statechart.tasks.joint_tasks import JointPositionList, JointState
from giskardpy.motion_statechart.tasks.weight_scaling_goals import MaxManipulability
from giskardpy.motion_statechart.test_nodes.test_nodes import (
    ChangeStateOnEvents,
    ConstTrueNode,
    TestGoal,
    TestNestedGoal,
    ConstFalseNode,
    TestRunAfterStop,
    TestRunAfterStopFromPause,
    TestEndBeforeStart,
    TestUnpauseUnknownFromParentPause,
)
from giskardpy.motion_statechart.constraint_builders import GeometricConstraintBuilder
from giskardpy.qp.constraint import GiskardEqualityConstraint
from giskardpy.qp.enforcement_strategy import IntegralStrategy
from giskardpy.qp.constraint_collection import ConstraintCollection
import krrood.symbolic_math.symbolic_math as sm
from krrood.symbolic_math.symbolic_math import (
    trinary_logic_and,
    trinary_logic_not,
    trinary_logic_or,
    FloatVariable,
)
from semantic_digital_twin.spatial_types import (
    HomogeneousTransformationMatrix,
    Vector3,
    Point3,
)
from semantic_digital_twin.spatial_types.spatial_types import Pose
from semantic_digital_twin.world import World


def test_condition_to_str():
    msc = MotionStatechart()
    node1 = ConstTrueNode()
    msc.add_node(node1)
    node2 = ConstTrueNode()
    msc.add_node(node2)
    node3 = ConstTrueNode()
    msc.add_node(node3)
    end = EndMotion()
    msc.add_node(end)

    end.start_condition = trinary_logic_and(
        node1.observation_variable,
        trinary_logic_or(
            node2.observation_variable,
            trinary_logic_not(node3.observation_variable),
        ),
    )
    a = str(end._start_condition)
    assert a == '("ConstTrueNode#0" and ("ConstTrueNode#1" or not "ConstTrueNode#2"))'


def test_motion_statechart_to_dot(tmp_path):
    msc = MotionStatechart()
    node1 = ConstTrueNode()
    msc.add_node(node1)
    node2 = ConstTrueNode()
    msc.add_node(node2)
    end = EndMotion()
    msc.add_node(end)
    node1.end_condition = node2.observation_variable
    end.start_condition = trinary_logic_and(
        node1.observation_variable, node2.observation_variable
    )
    msc.draw(str(tmp_path / "muh.pdf"))


def test_print():
    msc = MotionStatechart()
    print_node1 = Print(name="cow", message="muh")
    msc.add_node(print_node1)
    print_node2 = Print(name="cow2", message="muh")
    msc.add_node(print_node2)

    node1 = ConstTrueNode()
    msc.add_node(node1)
    end = EndMotion()
    msc.add_node(end)

    node1.start_condition = print_node1.observation_variable
    print_node2.start_condition = node1.observation_variable
    end.start_condition = print_node2.observation_variable

    kin_sim = Executor(MotionStatechartContext(world=World()))
    kin_sim.compile(motion_statechart=msc)

    assert len(msc.nodes) == 4
    assert len(msc.edges) == 3

    assert print_node1.observation_state == ObservationStateValues.UNKNOWN
    assert node1.observation_state == ObservationStateValues.UNKNOWN
    assert print_node2.observation_state == ObservationStateValues.UNKNOWN
    assert end.observation_state == ObservationStateValues.UNKNOWN

    assert print_node1.life_cycle_state == LifeCycleValues.RUNNING
    assert node1.life_cycle_state == LifeCycleValues.NOT_STARTED
    assert print_node2.life_cycle_state == LifeCycleValues.NOT_STARTED
    assert end.life_cycle_state == LifeCycleValues.NOT_STARTED
    assert not msc.is_end_motion()

    kin_sim.tick()
    assert print_node1.observation_state == ObservationStateValues.TRUE
    assert node1.observation_state == ObservationStateValues.UNKNOWN
    assert print_node2.observation_state == ObservationStateValues.UNKNOWN
    assert end.observation_state == ObservationStateValues.UNKNOWN

    assert print_node1.life_cycle_state == LifeCycleValues.RUNNING
    assert node1.life_cycle_state == LifeCycleValues.RUNNING
    assert print_node2.life_cycle_state == LifeCycleValues.NOT_STARTED
    assert end.life_cycle_state == LifeCycleValues.NOT_STARTED
    assert not msc.is_end_motion()

    kin_sim.tick()
    assert print_node1.observation_state == ObservationStateValues.TRUE
    assert node1.observation_state == ObservationStateValues.TRUE
    assert print_node2.observation_state == ObservationStateValues.UNKNOWN
    assert end.observation_state == ObservationStateValues.UNKNOWN

    assert print_node1.life_cycle_state == LifeCycleValues.RUNNING
    assert node1.life_cycle_state == LifeCycleValues.RUNNING
    assert print_node2.life_cycle_state == LifeCycleValues.RUNNING
    assert end.life_cycle_state == LifeCycleValues.NOT_STARTED
    assert not msc.is_end_motion()

    kin_sim.tick()
    assert print_node1.observation_state == ObservationStateValues.TRUE
    assert node1.observation_state == ObservationStateValues.TRUE
    assert print_node2.observation_state == ObservationStateValues.TRUE
    assert end.observation_state == ObservationStateValues.UNKNOWN

    assert print_node1.life_cycle_state == LifeCycleValues.RUNNING
    assert node1.life_cycle_state == LifeCycleValues.RUNNING
    assert print_node2.life_cycle_state == LifeCycleValues.RUNNING
    assert end.life_cycle_state == LifeCycleValues.RUNNING
    assert not msc.is_end_motion()

    kin_sim.tick()
    assert print_node1.observation_state == ObservationStateValues.TRUE
    assert node1.observation_state == ObservationStateValues.TRUE
    assert print_node2.observation_state == ObservationStateValues.TRUE
    assert end.observation_state == ObservationStateValues.TRUE

    assert print_node1.life_cycle_state == LifeCycleValues.RUNNING
    assert node1.life_cycle_state == LifeCycleValues.RUNNING
    assert print_node2.life_cycle_state == LifeCycleValues.RUNNING
    assert end.life_cycle_state == LifeCycleValues.RUNNING
    assert msc.is_end_motion()


def test_draw_with_invisible_node(tmp_path):
    msc = MotionStatechart()
    msc.add_nodes(
        [
            sequence := Sequence(
                nodes=[s1n1 := ConstTrueNode(), s1n2 := ConstTrueNode()]
            ),
            sequence2 := Sequence(
                nodes=[s2n1 := ConstTrueNode(), s2n2 := ConstTrueNode()]
            ),
        ]
    )
    msc.add_node(EndMotion.when_all_true(msc.nodes))

    sequence.plot_specs.visible = False
    s1n2.plot_specs.visible = False
    s2n2.plot_specs.visible = False

    kin_sim = Executor(MotionStatechartContext(world=World()))
    kin_sim.compile(motion_statechart=msc)
    msc.draw(str(tmp_path / "muh.pdf"))


class TestConditions:
    def test_trinary_condition_default_expression_is_scalar(self):
        condition = TrinaryCondition(kind=TransitionKind.START)
        assert isinstance(condition.expression, sm.Scalar)

    def test_InvalidConditionError(self):
        node = ConstTrueNode()
        with pytest.raises(InputNotExpressionError):
            node.end_condition = node

    def test_nodes_cannot_have_themselves_as_start_condition(self):
        msc = MotionStatechart()
        node1 = ConstTrueNode()
        msc.add_node(node1)
        with pytest.raises(SelfInStartConditionError):
            node1.start_condition = node1.observation_variable

    def test_non_observation_variable_in_condition(self):
        msc = MotionStatechart()
        msc.add_node(node := ConstTrueNode())
        with pytest.raises(NonObservationVariableError):
            node.start_condition = FloatVariable(name="muh")

    def test_add_node_to_multiple_goals(self):
        msc = MotionStatechart()
        node = ConstTrueNode()
        msc.add_node(Sequence([node]))
        msc.add_node(Sequence([node]))

        kin_sim = Executor(
            MotionStatechartContext(
                world=World(),
            )
        )
        with pytest.raises(NodeAlreadyBelongsToDifferentNodeError):
            kin_sim.compile(motion_statechart=msc)

    def test_add_node_to_multiple_goals2(self):
        msc = MotionStatechart()
        node = ConstTrueNode()
        msc.add_node(node)
        msc.add_node(Sequence([node]))

        kin_sim = Executor(
            MotionStatechartContext(
                world=World(),
            )
        )
        with pytest.raises(NodeAlreadyBelongsToDifferentNodeError):
            kin_sim.compile(motion_statechart=msc)


@dataclass(eq=False, repr=False)
class _BuildCountingNode(MotionStatechartNode):
    """Node that records how often :meth:`build` is invoked."""

    build_count: int = field(default=0, init=False)
    """Number of times build() has run on this node."""

    def build(self, context: MotionStatechartContext) -> NodeArtifacts:
        self.build_count += 1
        return NodeArtifacts(observation=sm.Scalar.const_true())


@dataclass(eq=False, repr=False)
class _BuildCountingGoal(Goal):
    """Goal that records its own build calls and owns a counting child node."""

    build_count: int = field(default=0, init=False)
    """Number of times build() has run on this goal."""

    child: _BuildCountingNode = field(default=None, init=False)
    """The child node expanded by this goal."""

    def expand(self, context: MotionStatechartContext) -> None:
        self.child = _BuildCountingNode(name="counting_child")
        self.add_node(self.child)

    def build(self, context: MotionStatechartContext) -> NodeArtifacts:
        self.build_count += 1
        return NodeArtifacts(observation=self.child.observation_variable)


def _compile_msc(msc: MotionStatechart) -> Executor:
    executor = Executor(MotionStatechartContext(world=World()))
    executor.compile(motion_statechart=msc)
    return executor


def test_each_node_is_built_exactly_once():
    msc = MotionStatechart()
    goal = _BuildCountingGoal()
    msc.add_node(goal)
    msc.add_node(EndMotion.when_true(goal))

    _compile_msc(msc)

    assert goal.build_count == 1
    assert goal.child.build_count == 1


def test_state_iteration_yields_nodes():
    msc = MotionStatechart()
    node1 = ConstTrueNode()
    node2 = ConstTrueNode()
    msc.add_node(node1)
    msc.add_node(node2)

    assert list(iter(msc.life_cycle_state)) == msc.nodes
    assert dict(msc.observation_state).keys() == {node1, node2}


def test_two_goals(pr2_world_state_reset: World):
    torso_joint = pr2_world_state_reset.get_connection_by_name("torso_lift_joint")
    r_wrist_roll_joint = pr2_world_state_reset.get_connection_by_name(
        "r_wrist_roll_joint"
    )
    msc = MotionStatechart()
    msc.add_nodes(
        [
            JointPositionList(goal_state=JointState.from_mapping({torso_joint: 0.1})),
            local_min := LocalMinimumReached(),
        ]
    )
    msc.add_node(EndMotion.when_true(local_min))

    kin_sim = Executor(
        MotionStatechartContext(
            world=pr2_world_state_reset,
        )
    )
    kin_sim.compile(motion_statechart=msc)

    kin_sim.tick_until_end()
    assert np.isclose(torso_joint.position, 0.1, atol=1e-4)

    msc = MotionStatechart()
    msc.add_node(
        joint_goal := JointPositionList(
            goal_state=JointState.from_mapping({r_wrist_roll_joint: 1})
        )
    )
    msc.add_node(EndMotion.when_true(joint_goal))

    kin_sim = Executor(
        MotionStatechartContext(
            world=pr2_world_state_reset,
        )
    )
    kin_sim.compile(motion_statechart=msc)

    kin_sim.tick_until_end()
    assert np.isclose(torso_joint.position, 0.1, atol=1e-4)
    assert np.allclose(pr2_world_state_reset.state.velocities, 0)
    assert np.allclose(pr2_world_state_reset.state.accelerations, 0)
    assert np.allclose(pr2_world_state_reset.state.jerks, 0)


@dataclass(eq=False, repr=False)
class _TestThreadMonitor(ThreadPayloadMonitor):
    delay: float = 0.05
    return_value: float = ObservationStateValues.TRUE

    def _compute_observation(self):
        time.sleep(self.delay)
        return self.return_value


@dataclass(eq=False, repr=False)
class _RaisingThreadMonitor(ThreadPayloadMonitor):
    """Thread payload monitor whose observation computation always fails."""

    def _compute_observation(self) -> float:
        raise RuntimeError("observation failure")


@dataclass(eq=False, repr=False)
class _SucceedingThreadMonitor(ThreadPayloadMonitor):
    """Thread payload monitor whose observation computation succeeds."""

    def _compute_observation(self) -> float:
        return ObservationStateValues.TRUE


def test_thread_payload_monitor_non_blocking_and_caching():
    msc = MotionStatechart()
    mon = _TestThreadMonitor(
        delay=0.05,
        return_value=ObservationStateValues.TRUE,
    )
    msc.add_node(mon)
    # First call should be non-blocking and return Unknown until worker completes at least once
    start = time.perf_counter()
    val0 = mon.compute_observation()
    elapsed = time.perf_counter() - start
    assert elapsed < mon.delay / 4.0
    assert val0 == ObservationStateValues.UNKNOWN
    # Wait for worker to finish and cache
    time.sleep(mon.delay * 2)
    val1 = mon.compute_observation()
    assert val1 == ObservationStateValues.TRUE


def test_thread_payload_monitor_cleanup_stops_worker():
    monitor = _SucceedingThreadMonitor()
    assert monitor._thread.is_alive()

    monitor.cleanup(context=MotionStatechartContext.empty())

    monitor._thread.join(timeout=1.0)
    assert not monitor._thread.is_alive()


def test_thread_payload_monitor_surfaces_compute_exception():
    import giskardpy.motion_statechart.graph_node as graph_node_module

    records: list[logging.LogRecord] = []

    class _CapturingHandler(logging.Handler):
        def emit(self, record: logging.LogRecord) -> None:
            records.append(record)

    handler = _CapturingHandler(level=logging.ERROR)
    graph_node_module.logger.addHandler(handler)
    monitor = _RaisingThreadMonitor()
    try:
        monitor.compute_observation()
        for _ in range(100):
            if any(record.levelno >= logging.ERROR for record in records):
                break
            time.sleep(0.02)
        assert any(record.levelno >= logging.ERROR for record in records)
    finally:
        graph_node_module.logger.removeHandler(handler)
        monitor.cleanup(context=MotionStatechartContext.empty())


class TestMotionStatechartLogic:

    def test_transition_triggers(self, tmp_path):
        msc = MotionStatechart()

        changer = ChangeStateOnEvents()
        msc.add_node(changer)

        node1 = Pulse()
        msc.add_node(node1)

        node2 = Pulse()
        msc.add_node(node2)
        node2.start_condition = node1.observation_variable

        node3 = Pulse()
        msc.add_node(node3)
        node3.start_condition = trinary_logic_and(
            trinary_logic_not(node1.observation_variable),
            trinary_logic_not(node2.observation_variable),
        )

        node4 = Pulse()
        msc.add_node(node4)
        node4.start_condition = node3.observation_variable

        changer.start_condition = node1.observation_variable
        changer.pause_condition = node2.observation_variable
        changer.end_condition = node3.observation_variable
        changer.reset_condition = node4.observation_variable

        kin_sim = Executor(MotionStatechartContext(world=World()))
        kin_sim.compile(motion_statechart=msc)

        assert changer.state is None

        kin_sim.tick()
        msc.draw(str(tmp_path / "muh.pdf"))
        assert changer.life_cycle_state == LifeCycleValues.RUNNING
        assert changer.state == "on_start"

        kin_sim.tick()
        msc.draw(str(tmp_path / "muh.pdf"))
        assert changer.life_cycle_state == LifeCycleValues.PAUSED
        assert changer.state == "on_pause"

        kin_sim.tick()
        msc.draw(str(tmp_path / "muh.pdf"))
        assert changer.life_cycle_state == LifeCycleValues.RUNNING
        assert changer.state == "on_unpause"

        kin_sim.tick()
        msc.draw(str(tmp_path / "muh.pdf"))
        assert changer.life_cycle_state == LifeCycleValues.DONE
        assert changer.state == "on_end"

        kin_sim.tick()
        msc.draw(str(tmp_path / "muh.pdf"))
        assert changer.life_cycle_state == LifeCycleValues.NOT_STARTED
        assert changer.state == "on_reset"

    def test_live_state_requires_motion_statechart_membership(self):
        node = ConstTrueNode()
        # State variables and conditions are available before the node is added.
        assert node.observation_variable is not None
        assert node.life_cycle_variable is not None
        node.pause_condition = node.observation_variable
        node.end_condition = node.observation_variable
        node.reset_condition = node.observation_variable
        # Reading the live state still requires membership in a motion statechart.
        with pytest.raises(NotInMotionStatechartError):
            _ = node.motion_statechart
        with pytest.raises(NotInMotionStatechartError):
            _ = node.observation_state
        with pytest.raises(NotInMotionStatechartError):
            _ = node.life_cycle_state

    def test_cancel_motion(self, tmp_path):
        msc = MotionStatechart()
        node1 = ConstTrueNode()
        msc.add_node(node1)
        cancel = CancelMotion(exception=Exception("muh"))
        msc.add_node(cancel)
        cancel.start_condition = node1.observation_variable

        kin_sim = Executor(MotionStatechartContext(world=World()))
        kin_sim.compile(motion_statechart=msc)

        kin_sim.tick()  # first tick, cancel goes into running
        with pytest.raises(Exception):
            kin_sim.tick()  # second tick, cancel goes true and triggers
        msc.draw(str(tmp_path / "muh.pdf"))

    def test_motion_statechart(self):
        msc = MotionStatechart()

        node1 = ConstTrueNode()
        msc.add_node(node1)
        node2 = ConstTrueNode()
        msc.add_node(node2)
        node3 = ConstTrueNode()
        msc.add_node(node3)
        end = EndMotion()
        msc.add_node(end)

        node1.start_condition = trinary_logic_or(
            node3.observation_variable, node2.observation_variable
        )
        end.start_condition = node1.observation_variable

        kin_sim = Executor(MotionStatechartContext(world=World()))
        kin_sim.compile(motion_statechart=msc)

        assert len(msc.nodes) == 4
        assert len(msc.edges) == 3
        kin_sim.tick_until_end()

        assert len(msc.history) == 5
        # %% node1
        assert msc.history.get_life_cycle_history_of_node(node1) == [
            LifeCycleValues.NOT_STARTED,
            LifeCycleValues.NOT_STARTED,
            LifeCycleValues.RUNNING,
            LifeCycleValues.RUNNING,
            LifeCycleValues.RUNNING,
        ]
        assert msc.history.get_observation_history_of_node(node1) == [
            ObservationStateValues.UNKNOWN,
            ObservationStateValues.UNKNOWN,
            ObservationStateValues.UNKNOWN,
            ObservationStateValues.TRUE,
            ObservationStateValues.TRUE,
        ]
        # %% node2
        assert msc.history.get_life_cycle_history_of_node(node2) == [
            LifeCycleValues.NOT_STARTED,
            LifeCycleValues.RUNNING,
            LifeCycleValues.RUNNING,
            LifeCycleValues.RUNNING,
            LifeCycleValues.RUNNING,
        ]
        assert msc.history.get_observation_history_of_node(node2) == [
            ObservationStateValues.UNKNOWN,
            ObservationStateValues.UNKNOWN,
            ObservationStateValues.TRUE,
            ObservationStateValues.TRUE,
            ObservationStateValues.TRUE,
        ]
        # %% node3
        assert msc.history.get_life_cycle_history_of_node(node3) == [
            LifeCycleValues.NOT_STARTED,
            LifeCycleValues.RUNNING,
            LifeCycleValues.RUNNING,
            LifeCycleValues.RUNNING,
            LifeCycleValues.RUNNING,
        ]
        assert msc.history.get_observation_history_of_node(node3) == [
            ObservationStateValues.UNKNOWN,
            ObservationStateValues.UNKNOWN,
            ObservationStateValues.TRUE,
            ObservationStateValues.TRUE,
            ObservationStateValues.TRUE,
        ]
        # %% end
        assert msc.history.get_life_cycle_history_of_node(end) == [
            LifeCycleValues.NOT_STARTED,
            LifeCycleValues.NOT_STARTED,
            LifeCycleValues.NOT_STARTED,
            LifeCycleValues.RUNNING,
            LifeCycleValues.RUNNING,
        ]
        assert msc.history.get_observation_history_of_node(end) == [
            ObservationStateValues.UNKNOWN,
            ObservationStateValues.UNKNOWN,
            ObservationStateValues.UNKNOWN,
            ObservationStateValues.UNKNOWN,
            ObservationStateValues.TRUE,
        ]

    def test_goal(self, tmp_path):
        msc = MotionStatechart()

        node1 = ConstTrueNode()
        msc.add_node(node1)

        goal = TestGoal()
        msc.add_node(goal)

        goal.start_condition = node1.observation_variable

        end = EndMotion()
        msc.add_node(end)
        end.start_condition = goal.observation_variable

        kin_sim = Executor(MotionStatechartContext(world=World()))

        kin_sim.compile(motion_statechart=msc)
        kin_sim.tick_until_end()
        assert len(msc.history) == 7
        # %% goal
        assert msc.history.get_life_cycle_history_of_node(goal) == [
            LifeCycleValues.NOT_STARTED,
            LifeCycleValues.NOT_STARTED,
            LifeCycleValues.RUNNING,
            LifeCycleValues.RUNNING,
            LifeCycleValues.RUNNING,
            LifeCycleValues.RUNNING,
            LifeCycleValues.RUNNING,
        ]
        assert msc.history.get_observation_history_of_node(goal) == [
            ObservationStateValues.UNKNOWN,
            ObservationStateValues.UNKNOWN,
            ObservationStateValues.UNKNOWN,
            ObservationStateValues.UNKNOWN,
            ObservationStateValues.UNKNOWN,
            ObservationStateValues.TRUE,
            ObservationStateValues.TRUE,
        ]
        # %% node1
        assert msc.history.get_life_cycle_history_of_node(node1) == [
            LifeCycleValues.NOT_STARTED,
            LifeCycleValues.RUNNING,
            LifeCycleValues.RUNNING,
            LifeCycleValues.RUNNING,
            LifeCycleValues.RUNNING,
            LifeCycleValues.RUNNING,
            LifeCycleValues.RUNNING,
        ]
        assert msc.history.get_observation_history_of_node(node1) == [
            ObservationStateValues.UNKNOWN,
            ObservationStateValues.UNKNOWN,
            ObservationStateValues.TRUE,
            ObservationStateValues.TRUE,
            ObservationStateValues.TRUE,
            ObservationStateValues.TRUE,
            ObservationStateValues.TRUE,
        ]
        # %% sub_node1
        assert msc.history.get_life_cycle_history_of_node(goal.sub_node1) == [
            LifeCycleValues.NOT_STARTED,
            LifeCycleValues.NOT_STARTED,
            LifeCycleValues.RUNNING,
            LifeCycleValues.DONE,
            LifeCycleValues.DONE,
            LifeCycleValues.DONE,
            LifeCycleValues.DONE,
        ]
        assert msc.history.get_observation_history_of_node(goal.sub_node1) == [
            ObservationStateValues.UNKNOWN,
            ObservationStateValues.UNKNOWN,
            ObservationStateValues.UNKNOWN,
            ObservationStateValues.TRUE,
            ObservationStateValues.TRUE,
            ObservationStateValues.TRUE,
            ObservationStateValues.TRUE,
        ]
        # %% sub_node2
        assert msc.history.get_life_cycle_history_of_node(goal.sub_node2) == [
            LifeCycleValues.NOT_STARTED,
            LifeCycleValues.NOT_STARTED,
            LifeCycleValues.NOT_STARTED,
            LifeCycleValues.RUNNING,
            LifeCycleValues.RUNNING,
            LifeCycleValues.RUNNING,
            LifeCycleValues.RUNNING,
        ]
        assert msc.history.get_observation_history_of_node(goal.sub_node2) == [
            ObservationStateValues.UNKNOWN,
            ObservationStateValues.UNKNOWN,
            ObservationStateValues.UNKNOWN,
            ObservationStateValues.UNKNOWN,
            ObservationStateValues.TRUE,
            ObservationStateValues.TRUE,
            ObservationStateValues.TRUE,
        ]
        # %% sub_node2
        assert msc.history.get_life_cycle_history_of_node(end) == [
            LifeCycleValues.NOT_STARTED,
            LifeCycleValues.NOT_STARTED,
            LifeCycleValues.NOT_STARTED,
            LifeCycleValues.NOT_STARTED,
            LifeCycleValues.NOT_STARTED,
            LifeCycleValues.RUNNING,
            LifeCycleValues.RUNNING,
        ]
        assert msc.history.get_observation_history_of_node(end) == [
            ObservationStateValues.UNKNOWN,
            ObservationStateValues.UNKNOWN,
            ObservationStateValues.UNKNOWN,
            ObservationStateValues.UNKNOWN,
            ObservationStateValues.UNKNOWN,
            ObservationStateValues.UNKNOWN,
            ObservationStateValues.TRUE,
        ]
        msc.draw(str(tmp_path / "muh.pdf"))

    def test_reset(self, tmp_path):
        msc = MotionStatechart()
        node1 = ConstTrueNode()
        msc.add_node(node1)
        node2 = ConstTrueNode()
        msc.add_node(node2)
        node3 = ConstTrueNode()
        msc.add_node(node3)
        end = EndMotion()
        msc.add_node(end)
        node1.reset_condition = node2.observation_variable
        node2.start_condition = node1.observation_variable
        node3.start_condition = node2.observation_variable
        node2.end_condition = node2.observation_variable
        end.start_condition = trinary_logic_and(
            node1.observation_variable,
            node2.observation_variable,
            node3.observation_variable,
        )

        kin_sim = Executor(MotionStatechartContext(world=World()))
        kin_sim.compile(motion_statechart=msc)
        msc.draw(str(tmp_path / "muh.pdf"))

        kin_sim.tick()
        assert node1.observation_state == ObservationStateValues.TRUE
        assert node2.observation_state == ObservationStateValues.UNKNOWN
        assert end.observation_state == ObservationStateValues.UNKNOWN
        assert node1.life_cycle_state == LifeCycleValues.RUNNING
        assert node2.life_cycle_state == LifeCycleValues.RUNNING
        assert end.life_cycle_state == LifeCycleValues.NOT_STARTED
        assert not msc.is_end_motion()

        kin_sim.tick()
        assert node1.observation_state == ObservationStateValues.TRUE
        assert node2.observation_state == ObservationStateValues.TRUE
        assert end.observation_state == ObservationStateValues.UNKNOWN
        assert node1.life_cycle_state == LifeCycleValues.NOT_STARTED
        assert node2.life_cycle_state == LifeCycleValues.DONE
        assert node3.life_cycle_state == LifeCycleValues.RUNNING
        assert end.life_cycle_state == LifeCycleValues.NOT_STARTED
        assert not msc.is_end_motion()

        kin_sim.tick()
        assert node1.observation_state == ObservationStateValues.UNKNOWN
        assert node2.observation_state == ObservationStateValues.TRUE
        assert node3.observation_state == ObservationStateValues.TRUE
        assert end.observation_state == ObservationStateValues.UNKNOWN
        assert node1.life_cycle_state == LifeCycleValues.RUNNING
        assert node2.life_cycle_state == LifeCycleValues.DONE
        assert node3.life_cycle_state == LifeCycleValues.RUNNING
        assert end.life_cycle_state == LifeCycleValues.NOT_STARTED
        assert not msc.is_end_motion()

        kin_sim.tick()
        assert node1.observation_state == ObservationStateValues.TRUE
        assert node2.observation_state == ObservationStateValues.TRUE
        assert node3.observation_state == ObservationStateValues.TRUE
        assert end.observation_state == ObservationStateValues.UNKNOWN
        assert node1.life_cycle_state == LifeCycleValues.NOT_STARTED
        assert node2.life_cycle_state == LifeCycleValues.DONE
        assert node3.life_cycle_state == LifeCycleValues.RUNNING
        assert end.life_cycle_state == LifeCycleValues.RUNNING
        assert not msc.is_end_motion()

        kin_sim.tick()
        assert node1.observation_state == ObservationStateValues.UNKNOWN
        assert node2.observation_state == ObservationStateValues.TRUE
        assert node3.observation_state == ObservationStateValues.TRUE
        assert end.observation_state == ObservationStateValues.TRUE
        assert node1.life_cycle_state == LifeCycleValues.RUNNING
        assert node2.life_cycle_state == LifeCycleValues.DONE
        assert node3.life_cycle_state == LifeCycleValues.RUNNING
        assert end.life_cycle_state == LifeCycleValues.RUNNING
        assert msc.is_end_motion()

    def test_nested_goals(self):
        msc = MotionStatechart()

        node1 = ConstTrueNode(name="w")
        msc.add_node(node1)

        outer = TestNestedGoal()
        msc.add_node(outer)
        outer.start_condition = node1.observation_variable

        end = EndMotion()
        msc.add_node(end)
        end.start_condition = outer.observation_variable

        json_data = msc.to_json()
        json_str = json.dumps(json_data)
        new_json_data = json.loads(json_str)
        msc_copy = MotionStatechart.from_json(new_json_data)

        for node in msc.nodes:
            assert node.index == msc_copy.get_node_by_index(node.index).index

        kin_sim = Executor(MotionStatechartContext(world=World()))
        node1 = msc_copy.get_nodes_by_type(ConstTrueNode)[0]
        outer = msc_copy.get_nodes_by_type(TestNestedGoal)[0]
        end = msc_copy.get_nodes_by_type(EndMotion)[0]
        kin_sim.compile(motion_statechart=msc_copy)

        assert node1.depth == 0
        assert outer.depth == 0
        assert end.depth == 0
        assert outer.inner.depth == 1
        assert outer.inner.sub_node1.depth == 2
        assert outer.inner.sub_node2.depth == 2

        assert node1.observation_state == ObservationStateValues.UNKNOWN
        assert outer.inner.sub_node1.observation_state == ObservationStateValues.UNKNOWN
        assert outer.inner.sub_node2.observation_state == ObservationStateValues.UNKNOWN
        assert outer.inner.observation_state == ObservationStateValues.UNKNOWN
        assert outer.observation_state == ObservationStateValues.UNKNOWN
        assert end.observation_state == ObservationStateValues.UNKNOWN

        assert node1.life_cycle_state == LifeCycleValues.RUNNING
        assert outer.inner.sub_node1.life_cycle_state == LifeCycleValues.NOT_STARTED
        assert outer.inner.sub_node2.life_cycle_state == LifeCycleValues.NOT_STARTED
        assert outer.inner.life_cycle_state == LifeCycleValues.NOT_STARTED
        assert outer.life_cycle_state == LifeCycleValues.NOT_STARTED
        assert end.life_cycle_state == LifeCycleValues.NOT_STARTED
        assert not msc_copy.is_end_motion()

        kin_sim.tick()
        assert node1.observation_state == ObservationStateValues.TRUE
        assert outer.inner.sub_node1.observation_state == ObservationStateValues.UNKNOWN
        assert outer.inner.sub_node2.observation_state == ObservationStateValues.UNKNOWN
        assert outer.inner.observation_state == ObservationStateValues.UNKNOWN
        assert outer.observation_state == ObservationStateValues.UNKNOWN
        assert end.observation_state == ObservationStateValues.UNKNOWN

        assert node1.life_cycle_state == LifeCycleValues.RUNNING
        assert outer.inner.sub_node1.life_cycle_state == LifeCycleValues.RUNNING
        assert outer.inner.sub_node2.life_cycle_state == LifeCycleValues.NOT_STARTED
        assert outer.inner.life_cycle_state == LifeCycleValues.RUNNING
        assert outer.life_cycle_state == LifeCycleValues.RUNNING
        assert end.life_cycle_state == LifeCycleValues.NOT_STARTED
        assert not msc_copy.is_end_motion()

        kin_sim.tick()
        assert node1.observation_state == ObservationStateValues.TRUE
        assert outer.inner.sub_node1.observation_state == ObservationStateValues.TRUE
        assert outer.inner.sub_node2.observation_state == ObservationStateValues.UNKNOWN
        assert outer.inner.observation_state == ObservationStateValues.UNKNOWN
        assert outer.observation_state == ObservationStateValues.UNKNOWN
        assert end.observation_state == ObservationStateValues.UNKNOWN

        assert node1.life_cycle_state == LifeCycleValues.RUNNING
        assert outer.inner.sub_node1.life_cycle_state == LifeCycleValues.DONE
        assert outer.inner.sub_node2.life_cycle_state == LifeCycleValues.RUNNING
        assert outer.inner.life_cycle_state == LifeCycleValues.RUNNING
        assert outer.life_cycle_state == LifeCycleValues.RUNNING
        assert end.life_cycle_state == LifeCycleValues.NOT_STARTED
        assert not msc_copy.is_end_motion()

        kin_sim.tick()
        assert node1.observation_state == ObservationStateValues.TRUE
        assert outer.inner.sub_node1.observation_state == ObservationStateValues.TRUE
        assert outer.inner.sub_node2.observation_state == ObservationStateValues.TRUE
        assert outer.inner.observation_state == ObservationStateValues.UNKNOWN
        assert outer.observation_state == ObservationStateValues.UNKNOWN
        assert end.observation_state == ObservationStateValues.UNKNOWN

        assert node1.life_cycle_state == LifeCycleValues.RUNNING
        assert outer.inner.sub_node1.life_cycle_state == LifeCycleValues.DONE
        assert outer.inner.sub_node2.life_cycle_state == LifeCycleValues.RUNNING
        assert outer.inner.life_cycle_state == LifeCycleValues.RUNNING
        assert outer.life_cycle_state == LifeCycleValues.RUNNING
        assert end.life_cycle_state == LifeCycleValues.NOT_STARTED
        assert not msc_copy.is_end_motion()

        kin_sim.tick()
        assert node1.observation_state == ObservationStateValues.TRUE
        assert outer.inner.sub_node1.observation_state == ObservationStateValues.TRUE
        assert outer.inner.sub_node2.observation_state == ObservationStateValues.TRUE
        assert outer.inner.observation_state == ObservationStateValues.TRUE
        assert outer.observation_state == ObservationStateValues.UNKNOWN
        assert end.observation_state == ObservationStateValues.UNKNOWN

        assert node1.life_cycle_state == LifeCycleValues.RUNNING
        assert outer.inner.sub_node1.life_cycle_state == LifeCycleValues.DONE
        assert outer.inner.sub_node2.life_cycle_state == LifeCycleValues.RUNNING
        assert outer.inner.life_cycle_state == LifeCycleValues.RUNNING
        assert outer.life_cycle_state == LifeCycleValues.RUNNING
        assert end.life_cycle_state == LifeCycleValues.NOT_STARTED
        assert not msc_copy.is_end_motion()

        kin_sim.tick()
        assert node1.observation_state == ObservationStateValues.TRUE
        assert outer.inner.sub_node1.observation_state == ObservationStateValues.TRUE
        assert outer.inner.sub_node2.observation_state == ObservationStateValues.TRUE
        assert outer.inner.observation_state == ObservationStateValues.TRUE
        assert outer.observation_state == ObservationStateValues.TRUE
        assert end.observation_state == ObservationStateValues.UNKNOWN

        assert node1.life_cycle_state == LifeCycleValues.RUNNING
        assert outer.inner.sub_node1.life_cycle_state == LifeCycleValues.DONE
        assert outer.inner.sub_node2.life_cycle_state == LifeCycleValues.RUNNING
        assert outer.inner.life_cycle_state == LifeCycleValues.RUNNING
        assert outer.life_cycle_state == LifeCycleValues.RUNNING
        assert end.life_cycle_state == LifeCycleValues.RUNNING
        assert not msc_copy.is_end_motion()

        kin_sim.tick()
        assert node1.observation_state == ObservationStateValues.TRUE
        assert outer.inner.sub_node1.observation_state == ObservationStateValues.TRUE
        assert outer.inner.sub_node2.observation_state == ObservationStateValues.TRUE
        assert outer.inner.observation_state == ObservationStateValues.TRUE
        assert outer.observation_state == ObservationStateValues.TRUE
        assert end.observation_state == ObservationStateValues.TRUE

        assert node1.life_cycle_state == LifeCycleValues.RUNNING
        assert outer.inner.sub_node1.life_cycle_state == LifeCycleValues.DONE
        assert outer.inner.sub_node2.life_cycle_state == LifeCycleValues.RUNNING
        assert outer.inner.life_cycle_state == LifeCycleValues.RUNNING
        assert outer.life_cycle_state == LifeCycleValues.RUNNING
        assert end.life_cycle_state == LifeCycleValues.RUNNING
        assert msc_copy.is_end_motion()


def test_long_goal(pr2_world_state_reset: World):
    msc = MotionStatechart()
    msc.add_nodes(
        [
            cart_goal := CartesianPose(
                root_link=pr2_world_state_reset.root,
                tip_link=pr2_world_state_reset.get_kinematic_structure_entity_by_name(
                    "base_footprint"
                ),
                goal_pose=HomogeneousTransformationMatrix.from_xyz_rpy(
                    x=50, reference_frame=pr2_world_state_reset.root
                ),
            ),
            JointPositionList(
                goal_state=JointState.from_str_dict(
                    {
                        "torso_lift_joint": 0.2999225173357618,
                        "head_pan_joint": 0.042,
                        "head_tilt_joint": -0.37,
                        "r_upper_arm_roll_joint": -0.9487714747527726,
                        "r_shoulder_pan_joint": -1.0047307505973626,
                        "r_shoulder_lift_joint": 0.48736790658811985,
                        "r_forearm_roll_joint": -14.895833882874182,
                        "r_elbow_flex_joint": -1.392377908925028,
                        "r_wrist_flex_joint": -0.4548695149411013,
                        "r_wrist_roll_joint": 0.11426798984097819,
                        "l_upper_arm_roll_joint": 1.7383062350263658,
                        "l_shoulder_pan_joint": 1.8799810286792007,
                        "l_shoulder_lift_joint": 0.011627231224188975,
                        "l_forearm_roll_joint": 312.67276414458695,
                        "l_elbow_flex_joint": -2.0300928925694675,
                        "l_wrist_flex_joint": -0.1,
                        "l_wrist_roll_joint": -6.062015047706399,
                    },
                    world=pr2_world_state_reset,
                )
            ),
        ]
    )
    msc.add_node(EndMotion.when_true(cart_goal))

    kin_sim = Executor(
        MotionStatechartContext(
            world=pr2_world_state_reset,
        )
    )
    kin_sim.compile(motion_statechart=msc)
    t = time.perf_counter()
    kin_sim.tick_until_end(1_000_000)
    after = time.perf_counter()
    diff = after - t
    print(diff / kin_sim.control_cycles)


def test_counting():
    class FakeClock:
        def __init__(self):
            self.t = 0.0

        def time(self):
            return self.t

        def advance(self, dt):
            self.t += dt

    clock = FakeClock()

    msc = MotionStatechart()
    seconds = 1
    msc.add_nodes(
        [counter := CountSeconds(seconds=seconds, _now=clock.time), pulse := Pulse()]
    )

    pulse.start_condition = counter.observation_variable
    counter.reset_condition = pulse.observation_variable

    msc.add_node(end := EndMotion())

    end.start_condition = trinary_logic_and(
        counter.observation_variable, trinary_logic_not(pulse.observation_variable)
    )

    kin_sim = Executor(
        MotionStatechartContext(
            world=World(),
        )
    )
    kin_sim.compile(motion_statechart=msc)

    # Advance fake time deterministically without wall-clock sleeps
    step = 0.1
    while not msc.is_end_motion():
        kin_sim.tick()
        clock.advance(step)
        if kin_sim.control_cycles > 1000:
            raise TimeoutError("test stuck")

    # it takes 2 * seconds to finish the counters
    # + 1 for pulse to trigger
    # + 1 for reset
    # + 1 for EndMotion to transition to RUNNING
    # + 1 for EndMotion to observe True
    assert np.allclose(seconds * 2 + 0.4, clock.time())


def test_count_ticks():
    msc = MotionStatechart()
    msc.add_node(counter := CountControlCycles(control_cycles=3))
    msc.add_node(EndMotion.when_true(counter))
    kin_sim = Executor(MotionStatechartContext(world=World()))
    kin_sim.compile(motion_statechart=msc)
    kin_sim.tick_until_end()
    # ending tacks 4 ticks, one to turn EndMotion to true
    assert kin_sim.control_cycles == 3 + 1


def test_count_control_cycles_returns_false_until_target():
    node = CountControlCycles(control_cycles=3)
    context = MotionStatechartContext(world=World())
    node.on_start(context)
    assert node.on_tick(context) == ObservationStateValues.FALSE
    assert node.on_tick(context) == ObservationStateValues.FALSE
    assert node.on_tick(context) == ObservationStateValues.TRUE


def test_count_simulation_time_seconds_reaches_target_on_exact_tick():
    context = MotionStatechartContext(world=World())
    ticks_until_true = 4
    seconds = context.qp_controller_config.control_dt * ticks_until_true
    node = CountSimulationTimeSeconds(seconds=seconds)
    node.on_start(context)
    for _ in range(ticks_until_true - 1):
        assert node.on_tick(context) == ObservationStateValues.FALSE
    assert node.on_tick(context) == ObservationStateValues.TRUE


def test_count_simulation_time_seconds_on_start_resets_counter():
    context = MotionStatechartContext(world=World())
    seconds = context.qp_controller_config.control_dt * 2
    node = CountSimulationTimeSeconds(seconds=seconds)
    node.on_start(context)
    node.on_tick(context)
    node.on_tick(context)
    node.on_start(context)
    assert node.on_tick(context) == ObservationStateValues.FALSE


def test_count_simulation_time_seconds_with_executor():
    context = MotionStatechartContext(world=World())
    ticks_until_true = 20
    seconds = context.qp_controller_config.control_dt * ticks_until_true
    msc = MotionStatechart()
    msc.add_node(counter := CountSimulationTimeSeconds(seconds=seconds))
    msc.add_node(EndMotion.when_true(counter))
    kin_sim = Executor(context)
    kin_sim.compile(motion_statechart=msc)
    kin_sim.tick_until_end()
    # +1 for EndMotion to turn True, as in test_count_ticks
    assert kin_sim.control_cycles == ticks_until_true + 1


class TestEndMotion:
    def test_end_motion_when_all_done1(self, tmp_path):
        msc = MotionStatechart()
        msc.add_nodes(
            [
                ConstTrueNode(),
                ConstTrueNode(),
            ]
        )
        end = EndMotion.when_all_true(msc.nodes)
        msc.add_node(end)

        kin_sim = Executor(
            MotionStatechartContext(
                world=World(),
            )
        )
        kin_sim.compile(motion_statechart=msc)
        kin_sim.tick_until_end()
        msc.draw(str(tmp_path / "muh.pdf"))
        assert end.life_cycle_state == LifeCycleValues.RUNNING

    def test_end_motion_when_all_done2(self, tmp_path):
        msc = MotionStatechart()
        msc.add_nodes(
            [
                ConstTrueNode(),
                ConstFalseNode(),
            ]
        )
        end = EndMotion.when_all_true(msc.nodes)
        msc.add_node(end)

        kin_sim = Executor(
            MotionStatechartContext(
                world=World(),
            )
        )
        kin_sim.compile(motion_statechart=msc)
        with pytest.raises(TimeoutError):
            kin_sim.tick_until_end()
        msc.draw(str(tmp_path / "muh.pdf"))
        assert end.life_cycle_state == LifeCycleValues.NOT_STARTED

    def test_end_motion_when_any_done1(self, tmp_path):
        msc = MotionStatechart()
        msc.add_nodes(
            [
                ConstTrueNode(),
                ConstFalseNode(),
            ]
        )
        end = EndMotion.when_any_true(msc.nodes)
        msc.add_node(end)

        kin_sim = Executor(
            MotionStatechartContext(
                world=World(),
            )
        )
        kin_sim.compile(motion_statechart=msc)
        kin_sim.tick_until_end()
        msc.draw(str(tmp_path / "muh.pdf"))
        assert end.life_cycle_state == LifeCycleValues.RUNNING

    def test_end_motion_when_any_done2(self, tmp_path):
        msc = MotionStatechart()
        msc.add_nodes(
            [
                ConstFalseNode(),
                ConstFalseNode(),
            ]
        )
        end = EndMotion.when_any_true(msc.nodes)
        msc.add_node(end)

        kin_sim = Executor(
            MotionStatechartContext(
                world=World(),
            )
        )
        kin_sim.compile(motion_statechart=msc)
        with pytest.raises(TimeoutError):
            kin_sim.tick_until_end()
        msc.draw(str(tmp_path / "muh.pdf"))
        assert end.life_cycle_state == LifeCycleValues.NOT_STARTED

    def test_goals_cannot_have_end_motion(self):
        msc = MotionStatechart()
        msc.add_node(Sequence([ConstTrueNode(), EndMotion()]))
        with pytest.raises(EndMotionInGoalError):
            kin_sim = Executor(
                MotionStatechartContext(
                    world=World(),
                )
            )
            kin_sim.compile(motion_statechart=msc)


class TestTemplates:

    def test_sequence_goal(self, tmp_path):
        msc = MotionStatechart()
        node = Sequence(
            nodes=[
                ConstTrueNode(),
                ConstTrueNode(),
                ConstTrueNode(),
                ConstTrueNode(),
            ]
        )
        msc.add_node(node)
        msc.add_node(EndMotion.when_true(node))

        kin_sim = Executor(MotionStatechartContext(world=World()))
        kin_sim.compile(motion_statechart=msc)
        kin_sim.tick_until_end()
        msc.draw(str(tmp_path / "muh.pdf"))
        assert kin_sim.control_cycles == 6
        assert msc.nodes[1].life_cycle_state == LifeCycleValues.RUNNING
        assert msc.nodes[2].life_cycle_state == LifeCycleValues.DONE
        assert msc.nodes[3].life_cycle_state == LifeCycleValues.DONE
        assert msc.nodes[4].life_cycle_state == LifeCycleValues.DONE
        assert msc.nodes[5].life_cycle_state == LifeCycleValues.DONE

    def test_parallel(self):
        msc = MotionStatechart()
        msc.add_nodes(
            [
                parallel := Parallel(
                    [
                        CountControlCycles(control_cycles=3),
                        CountControlCycles(control_cycles=5),
                    ]
                ),
            ]
        )
        msc.add_node(EndMotion.when_true(parallel))

        kin_sim = Executor(
            MotionStatechartContext(
                world=World(),
            )
        )
        kin_sim.compile(motion_statechart=msc)
        kin_sim.tick_until_end()
        # 5 (longest ticker) + 1 (for parallel to turn True) + 1 (for end to trigger)
        assert kin_sim.control_cycles == 7

    def test_parallel_with_tasks(self, pr2_world_state_reset: World):
        map = pr2_world_state_reset.root
        r_tip = pr2_world_state_reset.get_kinematic_structure_entity_by_name(
            "r_gripper_tool_frame"
        )
        msc = MotionStatechart()
        msc.add_node(
            parallel := Parallel(
                [
                    AlignPlanes(
                        root_link=map,
                        tip_link=r_tip,
                        tip_normal=Vector3.X(reference_frame=r_tip),
                        goal_normal=Vector3.X(reference_frame=map),
                    ),
                    AlignPlanes(
                        root_link=map,
                        tip_link=r_tip,
                        tip_normal=Vector3.Y(reference_frame=r_tip),
                        goal_normal=Vector3.Z(reference_frame=map),
                    ),
                ]
            )
        )
        msc.add_node(EndMotion.when_true(parallel))

        kin_sim = Executor(
            MotionStatechartContext(
                world=pr2_world_state_reset,
            )
        )
        kin_sim.compile(motion_statechart=msc)
        kin_sim.tick_until_end()

    def test_parallel_minimum_success(self):
        """Test that Parallel completes when minimum_success nodes are True"""
        msc = MotionStatechart()
        msc.add_nodes(
            [
                parallel := Parallel(
                    [
                        CountControlCycles(control_cycles=2),
                        CountControlCycles(control_cycles=4),
                        CountControlCycles(control_cycles=6),
                    ],
                    minimum_success=2,
                ),
            ]
        )
        msc.add_node(EndMotion.when_true(parallel))

        kin_sim = Executor(
            MotionStatechartContext(world=World()),
        )
        kin_sim.compile(motion_statechart=msc)
        kin_sim.tick_until_end()
        # 4 (second ticker completes) + 1 (for parallel to turn True) + 1 (for end to trigger)
        assert kin_sim.control_cycles == 6

    def test_parallel_minimum_success_zero(self):
        """Test that Parallel completes when no node is True"""
        msc = MotionStatechart()
        msc.add_nodes(
            [
                parallel := Parallel(
                    [
                        CountControlCycles(control_cycles=3),
                        CountControlCycles(control_cycles=5),
                        CountControlCycles(control_cycles=7),
                    ],
                    minimum_success=0,
                ),
            ]
        )
        msc.add_node(EndMotion.when_true(parallel))

        kin_sim = Executor(
            MotionStatechartContext(world=World()),
        )
        kin_sim.compile(motion_statechart=msc)
        kin_sim.tick_until_end()
        # 0 (no ticker completes) + 1 (for parallel to turn True) + 1 (for end to trigger)
        assert kin_sim.control_cycles == 2


def test_constraint_collection(pr2_world_state_reset: World):
    """
    Test the constraint collection naming behavior. Expected behavior is:
    - Not naming constraints should result in automatically generated unique names
    - Manually naming constraints the same name should result in an Exception
    - Merging constraint collections should handle duplicates via prefix if they are in different collections
    - Merge raises an Exception if a collection contains duplicates in itself
    """
    col = ConstraintCollection()
    tip = pr2_world_state_reset.get_kinematic_structure_entity_by_name(
        "r_gripper_tool_frame"
    )
    root = pr2_world_state_reset.get_kinematic_structure_entity_by_name("odom_combined")

    expr = Vector3.X(tip).angle_between(Vector3.Y(root))

    GeometricConstraintBuilder(col).add_point_goal_constraints(
        frame_P_current=Point3(0, 0, 0, reference_frame=tip),
        frame_P_goal=Point3(0, 0, 0, reference_frame=tip),
        reference_velocity=0.1,
        quadratic_weight=DefaultWeights.WEIGHT_BELOW_CA,
    )
    assert len(col.equality_constraints) >= 3

    for i in range(3):
        col.add_equality_constraint(
            reference_velocity=0.1 * i,
            equality_bound=0.0,
            quadratic_weight=DefaultWeights.WEIGHT_BELOW_CA,
            task_expression=expr,
        )

    col.add_inequality_constraint(
        name="same_name",
        reference_velocity=0.2,
        quadratic_weight=DefaultWeights.WEIGHT_BELOW_CA,
        task_expression=expr,
        lower_error=0.1,
        upper_error=0.2,
    )

    with pytest.raises(DuplicateNameException):
        col.add_equality_constraint(
            name="same_name",
            reference_velocity=0.2,
            equality_bound=0.0,
            quadratic_weight=DefaultWeights.WEIGHT_BELOW_CA,
            task_expression=expr,
        )

    col2 = ConstraintCollection()
    col2.add_equality_constraint(
        name="same_name",
        reference_velocity=0.2,
        equality_bound=0.0,
        quadratic_weight=DefaultWeights.WEIGHT_BELOW_CA,
        task_expression=expr,
    )

    col.merge("prefix", col2)
    assert any(c.name.startswith("prefix/") for c in col._constraints)

    with pytest.raises(DuplicateNameException):
        col.merge("", col2)

    col3 = ConstraintCollection()
    col3.add_equality_constraint(
        name="same_name",
        reference_velocity=0.2,
        equality_bound=0.0,
        quadratic_weight=DefaultWeights.WEIGHT_BELOW_CA,
        task_expression=expr,
    )
    constraint = GiskardEqualityConstraint(
        name="same_name",
        expression=expr,
        normalization_factor=0.1,
        quadratic_weight=DefaultWeights.WEIGHT_BELOW_CA,
        lower_slack_limit=-float("inf"),
        upper_slack_limit=float("inf"),
        linear_weight=0,
        enforcement_strategy=IntegralStrategy,
        bound=0.0,
    )
    col3._constraints.append(constraint)

    with pytest.raises(DuplicateNameException):
        col3._are_names_unique()

    with pytest.raises(DuplicateNameException):
        col2.merge("", col3)


class TestLifeCycleTransitions:
    """
    Tests the LifeCycle transitions of nodes in various edge cases and intended behavior.
    """

    def test_run_after_stop(self):
        """
        Test for node to run after the parent node already stopped.
        """
        msc = MotionStatechart()

        msc.add_node(
            sequence := Sequence(
                [
                    ConstTrueNode(),
                    TestRunAfterStop(),
                    CountControlCycles(name="delay endmotion", control_cycles=5),
                ]
            )
        )
        msc.add_node(EndMotion.when_true(sequence))

        kin_sim = Executor(MotionStatechartContext(world=World()))
        kin_sim.compile(motion_statechart=msc)
        kin_sim.tick_until_end()

        assert sequence.nodes[1].cancel.life_cycle_state == LifeCycleValues.NOT_STARTED
        assert sequence.nodes[1].ticking1.life_cycle_state == LifeCycleValues.DONE
        assert sequence.nodes[1].ticking2.life_cycle_state == LifeCycleValues.DONE
        assert sequence.nodes[1].life_cycle_state == LifeCycleValues.DONE

    def test_run_after_stop_from_pause(self):
        """
        Test for node to run from paused while the parent node already stopped.
        """
        msc = MotionStatechart()

        msc.add_node(
            sequence := Sequence(
                [
                    ConstTrueNode(),
                    TestRunAfterStopFromPause(),
                    CountControlCycles(name="delay endmotion", control_cycles=5),
                ]
            )
        )
        msc.add_node(EndMotion.when_true(sequence))

        kin_sim = Executor(MotionStatechartContext(world=World()))
        kin_sim.compile(motion_statechart=msc)
        kin_sim.tick_until_end()

        assert sequence.nodes[1].cancel.life_cycle_state == LifeCycleValues.NOT_STARTED
        assert sequence.nodes[1].ticking1.life_cycle_state == LifeCycleValues.DONE
        assert sequence.nodes[1].ticking2.life_cycle_state == LifeCycleValues.DONE
        assert sequence.nodes[1].ticking3.life_cycle_state == LifeCycleValues.DONE
        assert sequence.nodes[1].pulse.life_cycle_state == LifeCycleValues.DONE
        assert sequence.nodes[1].life_cycle_state == LifeCycleValues.DONE

    def test_end_before_start(self):
        """
        Test for node to start even if it's end condition is met before start condition.
        Node3 should start and run for 1 tick before ending, instead of never starting.
        """
        msc = MotionStatechart()

        node1 = CountControlCycles(control_cycles=1)
        node2 = ConstTrueNode()
        node3 = ConstTrueNode()

        msc.add_nodes(nodes=[node1, node2, node3])
        msc.add_node(EndMotion.when_true(node3))

        node3.start_condition = node1.observation_variable
        node3.end_condition = node2.observation_variable

        kin_sim = Executor(MotionStatechartContext(world=World()))
        kin_sim.compile(motion_statechart=msc)
        kin_sim.tick()

        assert node3.life_cycle_state == LifeCycleValues.RUNNING
        assert node3.observation_state == ObservationStateValues.UNKNOWN

        kin_sim.tick()

        assert node3.life_cycle_state == LifeCycleValues.DONE
        assert node3.observation_state == ObservationStateValues.TRUE

    def test_end_before_start_in_template(self):
        """
        Test for node to start even if it's end condition is met before start condition,
        when the nodes are inside a template.
        """
        msc = MotionStatechart()

        node = TestEndBeforeStart()
        msc.add_node(node)

        kin_sim = Executor(MotionStatechartContext(world=World()))
        kin_sim.compile(motion_statechart=msc)
        kin_sim.tick()

        assert node.node3.life_cycle_state == LifeCycleValues.RUNNING
        assert node.node3.observation_state == ObservationStateValues.UNKNOWN

        kin_sim.tick()

        assert node.node3.life_cycle_state == LifeCycleValues.DONE
        assert node.node3.observation_state == ObservationStateValues.TRUE

    def test_intended_transitions(self):
        """
        Test for intended LifeCycle transitions of nodes.
        """
        msc = MotionStatechart()

        count_node1 = CountControlCycles(control_cycles=1, name="node1")
        count_node2 = CountControlCycles(control_cycles=2, name="node2")
        end_count_node1 = CountControlCycles(control_cycles=11, name="end_node1")
        pulse_node1 = Pulse(name="pulse1")
        pulse_node2 = Pulse(name="pulse2")

        msc.add_nodes(
            nodes=[
                count_node1,
                count_node2,
                end_count_node1,
                pulse_node1,
                pulse_node2,
            ]
        )
        msc.add_node(end_node := EndMotion.when_true(end_count_node1))

        pulse_node1.start_condition = count_node1.observation_variable
        pulse_node2.start_condition = count_node2.observation_variable
        count_node2.start_condition = pulse_node1.observation_variable

        count_node1.pause_condition = pulse_node1.observation_variable

        count_node1.end_condition = count_node2.observation_variable
        pulse_node1.end_condition = count_node2.observation_variable

        count_node1.reset_condition = pulse_node2.observation_variable
        count_node2.reset_condition = pulse_node2.observation_variable
        pulse_node1.reset_condition = pulse_node2.observation_variable

        kin_sim = Executor(MotionStatechartContext(world=World()))
        kin_sim.compile(motion_statechart=msc)
        kin_sim.tick_until_end()

        assert len(msc.history) == 14
        # %% count_node1 history
        assert msc.history.get_life_cycle_history_of_node(count_node1) == [
            LifeCycleValues.NOT_STARTED,
            LifeCycleValues.RUNNING,
            LifeCycleValues.RUNNING,
            LifeCycleValues.PAUSED,
            LifeCycleValues.RUNNING,
            LifeCycleValues.DONE,
            LifeCycleValues.NOT_STARTED,
            LifeCycleValues.RUNNING,
            LifeCycleValues.RUNNING,
            LifeCycleValues.PAUSED,
            LifeCycleValues.RUNNING,
            LifeCycleValues.DONE,
            LifeCycleValues.DONE,
            LifeCycleValues.DONE,
        ]
        assert msc.history.get_observation_history_of_node(count_node1) == [
            ObservationStateValues.UNKNOWN,
            ObservationStateValues.UNKNOWN,
            ObservationStateValues.TRUE,
            ObservationStateValues.TRUE,
            ObservationStateValues.TRUE,
            ObservationStateValues.TRUE,
            ObservationStateValues.TRUE,
            ObservationStateValues.UNKNOWN,
            ObservationStateValues.TRUE,
            ObservationStateValues.TRUE,
            ObservationStateValues.TRUE,
            ObservationStateValues.TRUE,
            ObservationStateValues.TRUE,
            ObservationStateValues.TRUE,
        ]

        # %% count_node2 history
        assert msc.history.get_life_cycle_history_of_node(count_node2) == [
            LifeCycleValues.NOT_STARTED,
            LifeCycleValues.NOT_STARTED,
            LifeCycleValues.NOT_STARTED,
            LifeCycleValues.RUNNING,
            LifeCycleValues.RUNNING,
            LifeCycleValues.RUNNING,
            LifeCycleValues.NOT_STARTED,
            LifeCycleValues.NOT_STARTED,
            LifeCycleValues.NOT_STARTED,
            LifeCycleValues.RUNNING,
            LifeCycleValues.RUNNING,
            LifeCycleValues.RUNNING,
            LifeCycleValues.RUNNING,
            LifeCycleValues.RUNNING,
        ]
        assert msc.history.get_observation_history_of_node(count_node2) == [
            ObservationStateValues.UNKNOWN,
            ObservationStateValues.UNKNOWN,
            ObservationStateValues.UNKNOWN,
            ObservationStateValues.UNKNOWN,
            ObservationStateValues.FALSE,
            ObservationStateValues.TRUE,
            ObservationStateValues.TRUE,
            ObservationStateValues.UNKNOWN,
            ObservationStateValues.UNKNOWN,
            ObservationStateValues.UNKNOWN,
            ObservationStateValues.FALSE,
            ObservationStateValues.TRUE,
            ObservationStateValues.TRUE,
            ObservationStateValues.TRUE,
        ]

        # %% end_count_node1 history
        assert msc.history.get_life_cycle_history_of_node(end_count_node1) == [
            LifeCycleValues.NOT_STARTED,
            LifeCycleValues.RUNNING,
            LifeCycleValues.RUNNING,
            LifeCycleValues.RUNNING,
            LifeCycleValues.RUNNING,
            LifeCycleValues.RUNNING,
            LifeCycleValues.RUNNING,
            LifeCycleValues.RUNNING,
            LifeCycleValues.RUNNING,
            LifeCycleValues.RUNNING,
            LifeCycleValues.RUNNING,
            LifeCycleValues.RUNNING,
            LifeCycleValues.RUNNING,
            LifeCycleValues.RUNNING,
        ]
        assert msc.history.get_observation_history_of_node(end_count_node1) == [
            ObservationStateValues.UNKNOWN,
            ObservationStateValues.UNKNOWN,
            ObservationStateValues.FALSE,
            ObservationStateValues.FALSE,
            ObservationStateValues.FALSE,
            ObservationStateValues.FALSE,
            ObservationStateValues.FALSE,
            ObservationStateValues.FALSE,
            ObservationStateValues.FALSE,
            ObservationStateValues.FALSE,
            ObservationStateValues.FALSE,
            ObservationStateValues.FALSE,
            ObservationStateValues.TRUE,
            ObservationStateValues.TRUE,
        ]

        # %% end_node history
        assert msc.history.get_life_cycle_history_of_node(end_node) == [
            LifeCycleValues.NOT_STARTED,
            LifeCycleValues.NOT_STARTED,
            LifeCycleValues.NOT_STARTED,
            LifeCycleValues.NOT_STARTED,
            LifeCycleValues.NOT_STARTED,
            LifeCycleValues.NOT_STARTED,
            LifeCycleValues.NOT_STARTED,
            LifeCycleValues.NOT_STARTED,
            LifeCycleValues.NOT_STARTED,
            LifeCycleValues.NOT_STARTED,
            LifeCycleValues.NOT_STARTED,
            LifeCycleValues.NOT_STARTED,
            LifeCycleValues.RUNNING,
            LifeCycleValues.RUNNING,
        ]
        assert msc.history.get_observation_history_of_node(end_node) == [
            ObservationStateValues.UNKNOWN,
            ObservationStateValues.UNKNOWN,
            ObservationStateValues.UNKNOWN,
            ObservationStateValues.UNKNOWN,
            ObservationStateValues.UNKNOWN,
            ObservationStateValues.UNKNOWN,
            ObservationStateValues.UNKNOWN,
            ObservationStateValues.UNKNOWN,
            ObservationStateValues.UNKNOWN,
            ObservationStateValues.UNKNOWN,
            ObservationStateValues.UNKNOWN,
            ObservationStateValues.UNKNOWN,
            ObservationStateValues.UNKNOWN,
            ObservationStateValues.TRUE,
        ]

        # %% pulse_node1 history
        assert msc.history.get_life_cycle_history_of_node(pulse_node1) == [
            LifeCycleValues.NOT_STARTED,
            LifeCycleValues.NOT_STARTED,
            LifeCycleValues.RUNNING,
            LifeCycleValues.RUNNING,
            LifeCycleValues.RUNNING,
            LifeCycleValues.DONE,
            LifeCycleValues.NOT_STARTED,
            LifeCycleValues.NOT_STARTED,
            LifeCycleValues.RUNNING,
            LifeCycleValues.RUNNING,
            LifeCycleValues.RUNNING,
            LifeCycleValues.DONE,
            LifeCycleValues.DONE,
            LifeCycleValues.DONE,
        ]
        assert msc.history.get_observation_history_of_node(pulse_node1) == [
            ObservationStateValues.UNKNOWN,
            ObservationStateValues.UNKNOWN,
            ObservationStateValues.UNKNOWN,
            ObservationStateValues.TRUE,
            ObservationStateValues.FALSE,
            ObservationStateValues.FALSE,
            ObservationStateValues.FALSE,
            ObservationStateValues.UNKNOWN,
            ObservationStateValues.UNKNOWN,
            ObservationStateValues.TRUE,
            ObservationStateValues.FALSE,
            ObservationStateValues.FALSE,
            ObservationStateValues.FALSE,
            ObservationStateValues.FALSE,
        ]

        # %% pulse_node2 history
        assert msc.history.get_life_cycle_history_of_node(pulse_node2) == [
            LifeCycleValues.NOT_STARTED,
            LifeCycleValues.NOT_STARTED,
            LifeCycleValues.NOT_STARTED,
            LifeCycleValues.NOT_STARTED,
            LifeCycleValues.NOT_STARTED,
            LifeCycleValues.RUNNING,
            LifeCycleValues.RUNNING,
            LifeCycleValues.RUNNING,
            LifeCycleValues.RUNNING,
            LifeCycleValues.RUNNING,
            LifeCycleValues.RUNNING,
            LifeCycleValues.RUNNING,
            LifeCycleValues.RUNNING,
            LifeCycleValues.RUNNING,
        ]
        assert msc.history.get_observation_history_of_node(pulse_node2) == [
            ObservationStateValues.UNKNOWN,
            ObservationStateValues.UNKNOWN,
            ObservationStateValues.UNKNOWN,
            ObservationStateValues.UNKNOWN,
            ObservationStateValues.UNKNOWN,
            ObservationStateValues.UNKNOWN,
            ObservationStateValues.TRUE,
            ObservationStateValues.FALSE,
            ObservationStateValues.FALSE,
            ObservationStateValues.FALSE,
            ObservationStateValues.FALSE,
            ObservationStateValues.FALSE,
            ObservationStateValues.FALSE,
            ObservationStateValues.FALSE,
        ]

    def test_unpause_unknown_from_parent_pause(self):
        """
        Test for child node to unpause when parent node unpauses.
        Child node pause condition is unknown.
        """

        msc = MotionStatechart()

        pulse = Pulse()
        unpause = TestUnpauseUnknownFromParentPause()

        msc.add_nodes(nodes=[pulse, unpause])
        msc.add_node(EndMotion.when_true(unpause))

        unpause.pause_condition = pulse.observation_variable

        kin_sim = Executor(MotionStatechartContext(world=World()))
        kin_sim.compile(motion_statechart=msc)
        kin_sim.tick_until_end()

        assert unpause.count_ticks1.life_cycle_state == LifeCycleValues.DONE
        assert unpause.cancel.life_cycle_state == LifeCycleValues.NOT_STARTED

        assert unpause.count_ticks1.observation_state == ObservationStateValues.TRUE
        assert unpause.observation_state == ObservationStateValues.TRUE

    def test_long_pause(self):
        msc = MotionStatechart()
        msc.add_nodes(
            [
                node1 := Parallel([ConstTrueNode(), ConstFalseNode()]),
                pulse := Pulse(length=5),
            ]
        )
        node1.pause_condition = pulse.observation_variable
        msc.add_node(EndMotion.when_false(pulse))

        kin_sim = Executor(MotionStatechartContext(world=World()))
        kin_sim.compile(motion_statechart=msc)
        kin_sim.tick_until_end()
        msc.plot_gantt_chart()

        assert len(msc.history) == 5


class TestMaxManipulability:
    def test_MaxManipulability(self, pr2_world_state_reset: World):
        root = pr2_world_state_reset.get_body_by_name("base_footprint")
        tip = pr2_world_state_reset.get_body_by_name("r_gripper_tool_frame")

        goal_pose = Pose.from_xyz_rpy(
            x=0.8, y=-0.3, z=1.0, reference_frame=pr2_world_state_reset.root
        )
        msc = MotionStatechart()
        cart_goal = CartesianPose(
            root_link=pr2_world_state_reset.root,
            tip_link=tip,
            goal_pose=goal_pose,
        )
        msc.add_nodes([cart_goal, MaxManipulability(root_link=root, tip_link=tip)])
        msc.add_node(EndMotion.when_true(cart_goal))

        kin_sim = Executor(MotionStatechartContext(world=pr2_world_state_reset))
        kin_sim.compile(motion_statechart=msc)
        kin_sim.tick_until_end()

        fk = pr2_world_state_reset.compute_forward_kinematics_np(
            pr2_world_state_reset.root, tip
        )
        assert np.allclose(fk, goal_pose.to_np(), atol=cart_goal.threshold)


class TestEagerStateVariables:
    """
    A node's observation and life cycle variables are available right after construction,
    before it is added to a motion statechart, so conditions can be wired on nested nodes.
    """

    def test_state_variables_available_before_adding_to_statechart(self):
        node = ConstTrueNode()
        assert node.observation_variable is node.observation_variable
        assert node.life_cycle_variable is node.life_cycle_variable

    def test_nested_self_referential_end_condition_before_compile(self):
        msc = MotionStatechart()
        msc.add_node(
            Sequence(
                [
                    ConstTrueNode(),
                    barrier := Parallel(
                        [ConstTrueNode(), ConstFalseNode()], minimum_success=1
                    ),
                ]
            )
        )
        barrier.end_condition = barrier.observation_variable
        msc._expand_goals(MotionStatechartContext.empty())
        msc._add_transitions()
        assert barrier in barrier._end_condition.node_dependencies

    def test_nested_end_condition_survives_json_round_trip(self):
        msc = MotionStatechart()
        msc.add_node(
            sequence := Sequence(
                [
                    ConstTrueNode(),
                    barrier := Parallel(
                        [ConstTrueNode(), ConstFalseNode()], minimum_success=1
                    ),
                ]
            )
        )
        barrier.end_condition = barrier.observation_variable
        msc.add_node(EndMotion.when_true(sequence))

        msc._expand_goals(MotionStatechartContext.empty())
        json_data = msc.create_structure_copy().to_json()
        new_json_data = json.loads(json.dumps(json_data))
        msc_copy = MotionStatechart.from_json(new_json_data)
        msc_copy._add_transitions()

        barrier_copy = msc_copy.get_node_by_index(barrier.index)
        assert barrier_copy in barrier_copy._end_condition.node_dependencies
        assert barrier_copy.unique_name in str(barrier_copy._end_condition)

    def test_nodes_with_same_name_have_distinct_variable_names(self):
        first = ConstTrueNode(name="same")
        second = ConstTrueNode(name="same")
        assert first.observation_variable.name != second.observation_variable.name

    def test_self_referential_start_condition_raises_before_add(self):
        node = ConstTrueNode()
        with pytest.raises(SelfInStartConditionError):
            node.start_condition = node.observation_variable


class TestConditionScoping:
    """
    A condition may only reference the node itself or nodes sharing the same parent.
    References across template levels raise :class:`ConditionScopeError` during compilation.
    """

    def test_outside_node_cannot_reference_node_inside_template(self):
        msc = MotionStatechart()
        child = ConstTrueNode()
        msc.add_node(Sequence([child]))
        msc.add_node(EndMotion.when_true(child))

        kin_sim = Executor(MotionStatechartContext(world=World()))
        with pytest.raises(ConditionScopeError):
            kin_sim.compile(motion_statechart=msc)

    def test_template_node_cannot_reference_node_in_sibling_template(self):
        msc = MotionStatechart()
        node_a = ConstTrueNode()
        node_b = ConstTrueNode()
        msc.add_node(first := Parallel([node_a]))
        msc.add_node(Parallel([node_b]))
        node_b.start_condition = node_a.observation_variable
        msc.add_node(EndMotion.when_true(first))

        kin_sim = Executor(MotionStatechartContext(world=World()))
        with pytest.raises(ConditionScopeError):
            kin_sim.compile(motion_statechart=msc)

    def test_nested_template_node_cannot_reference_outer_node(self):
        msc = MotionStatechart()
        node_a = ConstTrueNode()
        node_b = ConstTrueNode()
        msc.add_node(sequence := Sequence([node_a, Parallel([node_b])]))
        node_b.pause_condition = node_a.observation_variable
        msc.add_node(EndMotion.when_true(sequence))

        kin_sim = Executor(MotionStatechartContext(world=World()))
        with pytest.raises(ConditionScopeError):
            kin_sim.compile(motion_statechart=msc)

    def test_parent_cannot_reference_child(self):
        msc = MotionStatechart()
        child = ConstTrueNode()
        parallel = Parallel([child])
        parallel.end_condition = child.observation_variable
        msc.add_node(parallel)
        msc.add_node(EndMotion.when_true(parallel))

        kin_sim = Executor(MotionStatechartContext(world=World()))
        with pytest.raises(ConditionScopeError):
            kin_sim.compile(motion_statechart=msc)

    def test_child_cannot_reference_parent(self):
        msc = MotionStatechart()
        child = ConstTrueNode()
        parallel = Parallel([child])
        child.pause_condition = parallel.observation_variable
        msc.add_node(parallel)
        msc.add_node(EndMotion.when_true(parallel))

        kin_sim = Executor(MotionStatechartContext(world=World()))
        with pytest.raises(ConditionScopeError):
            kin_sim.compile(motion_statechart=msc)

    def test_siblings_inside_template_can_reference_each_other(self):
        msc = MotionStatechart()
        node_a = ConstTrueNode()
        node_b = ConstTrueNode()
        node_b.start_condition = node_a.observation_variable
        msc.add_node(parallel := Parallel([node_a, node_b]))
        msc.add_node(EndMotion.when_true(parallel))

        kin_sim = Executor(MotionStatechartContext(world=World()))
        kin_sim.compile(motion_statechart=msc)
        kin_sim.tick_until_end()

    def test_self_referential_end_condition_inside_template_compiles(self):
        msc = MotionStatechart()
        msc.add_node(
            sequence := Sequence(
                [
                    ConstTrueNode(),
                    barrier := Parallel(
                        [ConstTrueNode(), ConstFalseNode()], minimum_success=1
                    ),
                ]
            )
        )
        barrier.end_condition = barrier.observation_variable
        msc.add_node(EndMotion.when_true(sequence))

        kin_sim = Executor(MotionStatechartContext(world=World()))
        kin_sim.compile(motion_statechart=msc)
        kin_sim.tick_until_end()
