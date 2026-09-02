import json
import logging
import threading
import time
from dataclasses import dataclass, field

import numpy as np
import pytest

import giskardpy.motion_statechart.graph_node as graph_node_module
import krrood.symbolic_math.symbolic_math as sm
from giskardpy.data_types.exceptions import DuplicateNameException
from giskardpy.executor import Executor, SimulationPacer
from giskardpy.motion_statechart.constraint_builders import GeometricConstraintBuilder
from giskardpy.motion_statechart.context import MotionStatechartContext
from giskardpy.motion_statechart.data_types import (
    LifeCycleValues,
    LifeCyclePredicate,
    ObservationStateValues,
    DefaultWeights,
    TransitionKind,
)
from giskardpy.motion_statechart.exceptions import (
    NotInMotionStatechartError,
    EndMotionInGoalError,
    GoalWithoutChildrenError,
    InputNotExpressionError,
    SelfInStartConditionError,
    UnsupportedConditionVariableError,
    NodeAlreadyBelongsToDifferentNodeError,
    ConditionScopeError,
    TerminalNodeInConditionError,
    EmptyDegreesOfFreedomError,
    MissingErrorSignalError,
    CyclicPredicateDependencyError,
    UnsupportedObservationVariableError,
)
from giskardpy.motion_statechart.goals.templates import Sequence, Parallel
from giskardpy.motion_statechart.graph_node import (
    ConvergingTask,
    EndMotion,
    CancelMotion,
    Goal,
    MotionStatechartNode,
    NodeArtifacts,
    TerminalNode,
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
    ThreadedPredicateMonitor,
    CheckControlCycleCount,
)
from giskardpy.motion_statechart.motion_statechart import (
    MotionStatechart,
)
from giskardpy.motion_statechart.nodes_for_testing.nodes_for_testing import (
    ChangeStateOnEvents,
    GoalCuttingOffItsChild,
    GoalCuttingOffItsChildAtItsGoal,
    GoalWithChildFailingOnItsOwn,
    GoalWithChildStartingLate,
    NodeObservingAPredicate,
    NodeObservingGoalReached,
    NodeObservingNothingYet,
    ConstTrueNode,
    TestGoal,
    TestNestedGoal,
    ConstFalseNode,
    TestRunAfterStop,
    TestRunAfterStopFromPause,
    TestEndBeforeStart,
    TestUnpauseUnknownFromParentPause,
)
from giskardpy.motion_statechart.tasks.align_planes import AlignPlanes
from giskardpy.motion_statechart.tasks.cartesian_tasks import (
    CartesianPose,
)
from giskardpy.motion_statechart.tasks.joint_tasks import JointPositionList, JointState
from giskardpy.motion_statechart.tasks.weight_scaling_goals import MaxManipulability
from giskardpy.qp.constraint import GiskardEqualityConstraint
from giskardpy.qp.constraint_collection import ConstraintCollection
from giskardpy.qp.enforcement_strategy import IntegralStrategy
from krrood.symbolic_math.symbolic_math import (
    trinary_logic_and,
    trinary_logic_not,
    trinary_logic_or,
    FloatVariable,
)
from semantic_digital_twin.adapters.ros.visualization.viz_marker import (
    VizMarkerPublisher,
)
from semantic_digital_twin.datastructures.prefixed_name import PrefixedName
from semantic_digital_twin.spatial_types import (
    HomogeneousTransformationMatrix,
    Quaternion,
    RotationMatrix,
    Vector3,
    Point3,
)
from semantic_digital_twin.spatial_types.spatial_types import Pose
from semantic_digital_twin.world import World
from semantic_digital_twin.world_description.connections import FixedConnection
from semantic_digital_twin.world_description.geometry import Box, Color, Scale
from semantic_digital_twin.world_description.shape_collection import ShapeCollection
from semantic_digital_twin.world_description.world_entity import Body
from semantic_digital_twin.robots.pr2 import PR2Joint

from ...semantic_digital_twin_test.test_orm.test_orm import hsr_world_state_reset

# %% a clock the test advances instead of waiting


@dataclass
class FakeClock:
    """
    Stands in for :func:`time.monotonic` so tests can advance time without sleeping.
    """

    seconds: float = 0.0
    """
    The current time, in seconds.
    """

    def time(self) -> float:
        """
        :return: The current time, in the shape a node's clock is called in.
        """
        return self.seconds

    def advance(self, seconds: float) -> None:
        """
        :param seconds: How far to move the clock forward.
        """
        self.seconds += seconds


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

    sequence.plot_specifications.visible = False
    s1n2.plot_specifications.visible = False
    s2n2.plot_specifications.visible = False

    kin_sim = Executor(MotionStatechartContext(world=World()))
    kin_sim.compile(motion_statechart=msc)
    msc.draw(str(tmp_path / "muh.pdf"))


@dataclass(eq=False, repr=False)
class _NodeThatEndsTheMotion(TerminalNode):
    """
    A terminal node other than the two the statechart ships with.
    """


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

    def test_unsupported_variable_in_condition(self):
        msc = MotionStatechart()
        msc.add_node(node := ConstTrueNode())
        with pytest.raises(UnsupportedConditionVariableError):
            node.start_condition = FloatVariable(name="muh")

    def test_end_motion_cannot_gate_another_node(self):
        """
        The motion is over once an EndMotion is true, so no transition can depend on it.
        """
        msc = MotionStatechart()
        msc.add_nodes([node := ConstTrueNode(), end := EndMotion()])
        with pytest.raises(TerminalNodeInConditionError) as exception_info:
            node.start_condition = end.observation_variable

        assert exception_info.value.terminal_node is end

    def test_cancel_motion_cannot_gate_another_node(self):
        """
        A CancelMotion ends the motion just like an EndMotion does.
        """
        msc = MotionStatechart()
        cancel = CancelMotion(exception=Exception("cancelled"))
        msc.add_nodes([node := ConstTrueNode(), cancel])
        with pytest.raises(TerminalNodeInConditionError) as exception_info:
            node.start_condition = cancel.observation_variable

        assert exception_info.value.terminal_node is cancel

    def test_terminal_nodes_are_rejected_in_every_condition_kind(self):
        """
        No transition of any kind can happen after the motion has ended.
        """
        msc = MotionStatechart()
        msc.add_nodes([node := ConstTrueNode(), end := EndMotion()])
        with pytest.raises(TerminalNodeInConditionError):
            node.pause_condition = end.observation_variable
        with pytest.raises(TerminalNodeInConditionError):
            node.end_condition = end.observation_variable
        with pytest.raises(TerminalNodeInConditionError):
            node.reset_condition = end.observation_variable

    def test_any_terminal_node_cannot_gate_another_node(self):
        """
        The rule follows from ending the motion, not from being one of the two nodes
        that happen to do so today.
        """
        msc = MotionStatechart()
        msc.add_nodes([node := ConstTrueNode(), terminal := _NodeThatEndsTheMotion()])
        with pytest.raises(TerminalNodeInConditionError) as exception_info:
            node.start_condition = terminal.observation_variable

        assert exception_info.value.terminal_node is terminal

    def test_a_terminal_node_cannot_reference_itself(self):
        """
        A terminal node's own transitions are as unreachable as everyone else's.
        """
        msc = MotionStatechart()
        msc.add_node(end := EndMotion())
        with pytest.raises(TerminalNodeInConditionError):
            end.end_condition = end.observation_variable

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
    """
    Node that records how often :meth:`build` is invoked.
    """

    build_count: int = field(default=0, init=False)
    """
    Number of times build() has run on this node.
    """

    def build_artifacts(self, context: MotionStatechartContext) -> NodeArtifacts:
        self.build_count += 1
        return NodeArtifacts(observation=sm.Scalar.const_true())


@dataclass(eq=False, repr=False)
class _BuildCountingGoal(Goal):
    """
    Goal that records its own build calls and owns a counting child node.
    """

    build_count: int = field(default=0, init=False)
    """
    Number of times build() has run on this goal.
    """

    child: _BuildCountingNode = field(default=None, init=False)
    """
    The child node expanded by this goal.
    """

    def expand(self, context: MotionStatechartContext) -> None:
        self.child = _BuildCountingNode(name="counting_child")
        self.add_node(self.child)

    def build_artifacts(self, context: MotionStatechartContext) -> NodeArtifacts:
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


# %% goals populated before compile


def test_adding_the_same_node_to_a_goal_twice_registers_it_once():
    """
    A goal that already owns a node must not register it a second time, so a chart can
    be populated before compile without the template's expand duplicating its children.
    """
    msc = MotionStatechart()
    goal = Sequence()
    msc.add_node(goal)
    node = ConstTrueNode()

    goal.add_node(node)
    index_after_first_add = node.index
    goal.add_node(node)

    assert goal.nodes == [node]
    assert msc.nodes == [goal, node]
    assert node.index == index_after_first_add


def test_goal_populated_before_compile_matches_one_populated_by_expand():
    """
    Adding a sequence's children up front yields the same children, wiring and
    observation as passing them to the template's constructor and letting expand add
    them.

    .. note:: The charts' node order differs, because children populated up front are
        registered before the following top level nodes rather than during compilation.
    """
    populated_before_compile = MotionStatechart()
    goal = Sequence()
    populated_before_compile.add_node(goal)
    goal.add_node(ConstTrueNode(name="a"))
    goal.add_node(ConstTrueNode(name="b"))
    populated_before_compile.add_node(EndMotion.when_true(goal))

    populated_by_expand = MotionStatechart()
    expanded_goal = Sequence(nodes=[ConstTrueNode(name="a"), ConstTrueNode(name="b")])
    populated_by_expand.add_node(expanded_goal)
    populated_by_expand.add_node(EndMotion.when_true(expanded_goal))

    for msc in (populated_before_compile, populated_by_expand):
        executor = _compile_msc(msc)
        while not msc.is_end_motion():
            executor.tick()

    assert [node.name for node in goal.nodes] == ["a", "b"]
    assert [node.name for node in expanded_goal.nodes] == ["a", "b"]
    assert sorted(node.name for node in populated_before_compile.nodes) == sorted(
        node.name for node in populated_by_expand.nodes
    )
    # expand still wires the sequence: the second child waits for the first to succeed
    for sequence in (goal, expanded_goal):
        assert sequence.nodes[1].start_condition.free_variables() == [
            sequence.nodes[0].is_succeeded
        ]
    assert goal.observation_state == expanded_goal.observation_state
    assert populated_before_compile.is_end_motion()
    assert populated_by_expand.is_end_motion()


# %% build orchestration and artifact production


@dataclass(eq=False, repr=False)
class _SetupThenArtifactsNode(MotionStatechartNode):
    """
    Node that performs setup in :meth:`build` and describes itself in
    :meth:`build_artifacts`.
    """

    hook_calls: list[str] = field(default_factory=list, init=False)
    """
    Names of the build hooks that ran, in the order they ran.
    """

    def build(self, context: MotionStatechartContext) -> NodeArtifacts:
        self.hook_calls.append("build")
        return super().build(context)

    def build_artifacts(self, context: MotionStatechartContext) -> NodeArtifacts:
        self.hook_calls.append("build_artifacts")
        return NodeArtifacts(observation=sm.Scalar.const_true())


@dataclass(eq=False, repr=False)
class _ConvergingTaskWithoutErrorSignal(ConvergingTask):
    """
    Converging task whose artifacts leave the error unset.
    """

    def build_artifacts(self, context: MotionStatechartContext) -> NodeArtifacts:
        return NodeArtifacts()


def test_build_delegates_to_build_artifacts():
    msc = MotionStatechart()
    node = _SetupThenArtifactsNode()
    msc.add_node(node)
    msc.add_node(EndMotion.when_true(node))

    executor = _compile_msc(msc)

    assert node.hook_calls == ["build", "build_artifacts"]
    executor.tick()
    assert node.observation_state == ObservationStateValues.TRUE


def test_converging_task_without_error_signal_is_rejected():
    msc = MotionStatechart()
    task = _ConvergingTaskWithoutErrorSignal()
    msc.add_node(task)
    msc.add_node(EndMotion.when_true(task))

    with pytest.raises(MissingErrorSignalError):
        _compile_msc(msc)


def test_state_iteration_yields_nodes():
    msc = MotionStatechart()
    node1 = ConstTrueNode()
    node2 = ConstTrueNode()
    msc.add_node(node1)
    msc.add_node(node2)

    assert list(iter(msc.life_cycle_state)) == msc.nodes
    assert dict(msc.observation_state).keys() == {node1, node2}


def test_two_goals(pr2_world_state_reset: World):
    torso_joint = pr2_world_state_reset.get_connection_by_name(PR2Joint.TORSO_LIFT)
    r_wrist_roll_joint = pr2_world_state_reset.get_connection_by_name(
        PR2Joint.RIGHT_WRIST_ROLL
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


def test_parallel_local_minimum_reached_tolerates_stall(pr2_world_state_reset: World):
    """
    A :class:`JointPositionList` goal and a :class:`LocalMinimumReached` monitor
    combined via ``Parallel(..., minimum_success=1)`` must finish once the commanded
    joint's velocity has settled near zero, even though the goal task itself never
    reaches its nominal target -- simulating a joint that got physically stopped (e.g.
    a gripper finger against a grasped object) before arriving. Capping the joint's own
    velocity limit to a tiny value makes it provably unable to traverse the requested
    distance within the tick budget, while its tracked velocity still settles below the
    stall threshold almost immediately.

    This is the pattern that replaces baking stall-tolerance into a task's own
    observation: the goal's observation still means "goal reached", nothing else, and
    the ``Parallel`` node is what tolerates the stall.
    """
    torso_joint = pr2_world_state_reset.get_connection_by_name(PR2Joint.TORSO_LIFT)
    torso_joint.raw_dof.limits.lower.velocity = -1e-3
    torso_joint.raw_dof.limits.upper.velocity = 1e-3

    msc = MotionStatechart()
    msc.add_node(
        combined := Parallel(
            [
                joint_goal := JointPositionList(
                    goal_state=JointState.from_mapping({torso_joint: 1.0}),
                ),
                LocalMinimumReached(
                    degrees_of_freedom=[torso_joint.raw_dof],
                    minimum_time=0.2,
                    measure_from_own_start=True,
                ),
            ],
            minimum_success=1,
        )
    )
    msc.add_node(EndMotion.when_true(combined))

    kin_sim = Executor(MotionStatechartContext(world=pr2_world_state_reset))
    kin_sim.compile(motion_statechart=msc)
    kin_sim.tick_until_end(timeout=1000)

    assert not np.isclose(torso_joint.position, 1.0, atol=1e-2), (
        "the joint should not have reached its nominal target -- its velocity "
        "was capped far too low to traverse the distance within the timeout, "
        "this test is only meaningful if it stayed far away"
    )
    assert msc.observation_state[joint_goal] == ObservationStateValues.FALSE, (
        "the goal task's own observation must still mean 'goal reached' -- it must "
        "not be the thing that turned true here, only the surrounding Parallel"
    )


def test_joint_position_list_alone_times_out_on_stall(
    pr2_world_state_reset: World,
):
    """
    Regression control for test_parallel_local_minimum_reached_tolerates_stall: a bare
    :class:`JointPositionList`, without the surrounding ``Parallel`` +
    :class:`LocalMinimumReached`, must never reach EndMotion in the same stalled
    scenario -- proving the monitor is what unblocks it, not some unrelated change.
    """
    torso_joint = pr2_world_state_reset.get_connection_by_name(PR2Joint.TORSO_LIFT)
    torso_joint.raw_dof.limits.lower.velocity = -1e-3
    torso_joint.raw_dof.limits.upper.velocity = 1e-3

    msc = MotionStatechart()
    msc.add_node(
        joint_goal := JointPositionList(
            goal_state=JointState.from_mapping({torso_joint: 1.0}),
        )
    )
    msc.add_node(EndMotion.when_true(joint_goal))

    kin_sim = Executor(MotionStatechartContext(world=pr2_world_state_reset))
    kin_sim.compile(motion_statechart=msc)

    with pytest.raises(TimeoutError):
        kin_sim.tick_until_end(timeout=200)


def test_local_minimum_reached_only_depends_on_given_degrees_of_freedom(
    pr2_world_state_reset: World,
):
    """
    LocalMinimumReached(degrees_of_freedom=[...]) must only depend on the given subset
    of degrees of freedom, not every active one in the world -- otherwise passing a
    subset to tolerate a stall on one joint could be defeated by unrelated motion
    elsewhere in the robot.
    """
    torso_joint = pr2_world_state_reset.get_connection_by_name(PR2Joint.TORSO_LIFT)
    moving_joint = pr2_world_state_reset.get_connection_by_name(
        PR2Joint.RIGHT_WRIST_ROLL
    )

    msc = MotionStatechart()
    msc.add_nodes(
        [
            JointPositionList(goal_state=JointState.from_mapping({moving_joint: 2.0})),
            local_min := LocalMinimumReached(
                degrees_of_freedom=[torso_joint.raw_dof], minimum_time=0.1
            ),
        ]
    )
    msc.add_node(EndMotion.when_true(local_min))

    kin_sim = Executor(MotionStatechartContext(world=pr2_world_state_reset))
    kin_sim.compile(motion_statechart=msc)

    # Checked directly against local_min's own observation state, not against
    # is_end_motion()/tick_until_end(): EndMotion additionally waits for every active
    # DOF's velocity to settle before actually ending the whole motion (see
    # test_end_motion_waits_for_convergence), so it is no longer a reliable proxy for
    # "when did this specific monitor become true".
    for _ in range(10):
        kin_sim.tick()
    assert msc.observation_state[local_min] == ObservationStateValues.TRUE, (
        "the scoped monitor should have settled almost immediately -- torso_joint "
        "never moves"
    )
    assert not np.isclose(moving_joint.position, 2.0, atol=1e-2), (
        "r_wrist_roll_joint should still be mid-motion at this point -- otherwise "
        "this test doesn't prove the monitor ignored it"
    )


def test_local_minimum_reached_measures_minimum_time_from_own_start_by_default():
    """
    measure_from_own_start must default to True: LocalMinimumReached is normally used to
    detect a stall on one specific, possibly late-starting motion (e.g. wrapped in a
    Parallel around it), so minimum_time should count from when the monitor itself
    started running, not from the start of the whole motion chart, even if the caller
    never sets the flag explicitly.
    """
    assert LocalMinimumReached().measure_from_own_start is True


def test_local_minimum_reached_raises_on_explicitly_empty_degrees_of_freedom(
    pr2_world_state_reset,
):
    """
    Passing an explicit empty degrees_of_freedom list is a caller misconfiguration (e.g.
    an empty set of connections upstream) and must raise, rather than silently turning
    the monitor into a constant-true observation that could mask the bug.
    """
    monitor = LocalMinimumReached(degrees_of_freedom=[])
    context = MotionStatechartContext(world=pr2_world_state_reset)

    with pytest.raises(EmptyDegreesOfFreedomError):
        monitor.build(context)


@dataclass(eq=False, repr=False)
class _TestThreadMonitor(ThreadPayloadMonitor):
    delay: float = 0.05
    return_value: float = ObservationStateValues.TRUE

    def _compute_observation(self):
        time.sleep(self.delay)
        return self.return_value


@dataclass(eq=False, repr=False)
class _RaisingThreadMonitor(ThreadPayloadMonitor):
    """
    Thread payload monitor whose observation computation always fails.
    """

    def _compute_observation(self) -> float:
        raise RuntimeError("observation failure")


@dataclass(eq=False, repr=False)
class _SucceedingThreadMonitor(ThreadPayloadMonitor):
    """
    Thread payload monitor whose observation computation succeeds.
    """

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


def _tick_until(sim, predicate, timeout=2.0):
    deadline = time.time() + timeout
    while time.time() < deadline:
        sim.tick()
        if predicate():
            return
        time.sleep(0.005)
    raise AssertionError("condition not reached within timeout")


def test_threaded_predicate_monitor_unknown_then_true():
    gate = threading.Event()
    msc = MotionStatechart()
    # predicate blocks on the gate, so we can observe the UNKNOWN phase
    mon = ThreadedPredicateMonitor(predicate=lambda: gate.wait(2.0), name="cond")
    msc.add_node(mon)
    end = EndMotion.when_true(mon)
    msc.add_node(end)

    sim = Executor(MotionStatechartContext(world=World()))
    sim.compile(motion_statechart=msc)

    # while the predicate is blocked, the monitor stays UNKNOWN and ticking
    # never blocks on the (slow) evaluation
    for _ in range(3):
        t0 = time.perf_counter()
        sim.tick()
        assert time.perf_counter() - t0 < 0.5
        assert mon.observation_state == ObservationStateValues.UNKNOWN
        assert not msc.is_end_motion()

    gate.set()
    _tick_until(sim, lambda: mon.observation_state == ObservationStateValues.TRUE)
    assert mon.observation_state == ObservationStateValues.TRUE
    sim.tick()
    assert msc.is_end_motion()


def test_threaded_predicate_monitor_false():
    msc = MotionStatechart()
    mon = ThreadedPredicateMonitor(predicate=lambda: False, name="cond")
    msc.add_node(mon)
    end = EndMotion.when_true(mon)
    msc.add_node(end)

    sim = Executor(MotionStatechartContext(world=World()))
    sim.compile(motion_statechart=msc)

    _tick_until(sim, lambda: mon.observation_state == ObservationStateValues.FALSE)
    assert mon.observation_state == ObservationStateValues.FALSE
    assert not msc.is_end_motion()


def test_threaded_predicate_monitor_false_triggers_cancel():
    msc = MotionStatechart()
    mon = ThreadedPredicateMonitor(predicate=lambda: False, name="cond")
    msc.add_node(mon)
    cancel = CancelMotion(exception=Exception("condition is false"))
    cancel.start_condition = trinary_logic_not(mon.observation_variable)
    msc.add_node(cancel)

    sim = Executor(MotionStatechartContext(world=World()))
    sim.compile(motion_statechart=msc)

    with pytest.raises(Exception, match="condition is false"):
        _tick_until(sim, lambda: False)


def test_threaded_predicate_monitor_exception_is_false():
    def boom():
        raise RuntimeError("query failed")

    msc = MotionStatechart()
    mon = ThreadedPredicateMonitor(predicate=boom, name="cond")
    msc.add_node(mon)
    end = EndMotion.when_true(mon)
    msc.add_node(end)

    sim = Executor(MotionStatechartContext(world=World()))
    sim.compile(motion_statechart=msc)

    # a raising predicate must not crash the control loop; it reports FALSE
    try:
        _tick_until(sim, lambda: mon.observation_state == ObservationStateValues.FALSE)
    except RuntimeError:
        pass
    assert mon.observation_state == ObservationStateValues.UNKNOWN


def test_thread_payload_monitor_cleanup_stops_worker():
    monitor = _SucceedingThreadMonitor()
    assert monitor._thread.is_alive()

    monitor.cleanup(context=MotionStatechartContext.empty())

    monitor._thread.join(timeout=1.0)
    assert not monitor._thread.is_alive()


def test_thread_payload_monitor_surfaces_compute_exception():
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
        # A node that only records callbacks never decides what it observes, so ending
        # it cannot judge it.
        assert changer.life_cycle_state == LifeCycleValues.INTERRUPTED
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
            LifeCycleValues.SUCCEEDED,
            LifeCycleValues.SUCCEEDED,
            LifeCycleValues.SUCCEEDED,
            LifeCycleValues.SUCCEEDED,
        ]
        assert msc.history.get_observation_history_of_node(goal.sub_node1) == [
            ObservationStateValues.UNKNOWN,
            ObservationStateValues.UNKNOWN,
            ObservationStateValues.UNKNOWN,
            ObservationStateValues.TRUE,
            ObservationStateValues.UNKNOWN,
            ObservationStateValues.UNKNOWN,
            ObservationStateValues.UNKNOWN,
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
        """
        A reset returns a node to NOT_STARTED, from where it starts and observes again.

        The node that triggers the reset is ended by its own goal, so what outlives it
        is its verdict, and both the reset trigger and the end of the motion read that.
        """
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
        node2.end_condition = node2.observation_variable
        node3.start_condition = node2.goal_reached
        end.start_condition = trinary_logic_and(
            node1.observation_variable,
            node2.goal_reached,
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

        # node2 reaches its goal, which both ends it and resets node1.
        kin_sim.tick()
        assert node1.observation_state == ObservationStateValues.TRUE
        assert node2.observation_state == ObservationStateValues.TRUE
        assert end.observation_state == ObservationStateValues.UNKNOWN
        assert node1.life_cycle_state == LifeCycleValues.NOT_STARTED
        assert node2.life_cycle_state == LifeCycleValues.SUCCEEDED
        assert node3.life_cycle_state == LifeCycleValues.RUNNING
        assert end.life_cycle_state == LifeCycleValues.NOT_STARTED
        assert not msc.is_end_motion()

        # node1 starts over with nothing observed yet, and node2 stops observing.
        kin_sim.tick()
        assert node1.observation_state == ObservationStateValues.UNKNOWN
        assert node2.observation_state == ObservationStateValues.UNKNOWN
        assert node3.observation_state == ObservationStateValues.TRUE
        assert end.observation_state == ObservationStateValues.UNKNOWN
        assert node1.life_cycle_state == LifeCycleValues.RUNNING
        assert node2.life_cycle_state == LifeCycleValues.SUCCEEDED
        assert node3.life_cycle_state == LifeCycleValues.RUNNING
        assert end.life_cycle_state == LifeCycleValues.NOT_STARTED
        assert not msc.is_end_motion()

        kin_sim.tick()
        assert node1.observation_state == ObservationStateValues.TRUE
        assert node2.observation_state == ObservationStateValues.UNKNOWN
        assert node3.observation_state == ObservationStateValues.TRUE
        assert end.observation_state == ObservationStateValues.UNKNOWN
        assert node1.life_cycle_state == LifeCycleValues.RUNNING
        assert node2.life_cycle_state == LifeCycleValues.SUCCEEDED
        assert node3.life_cycle_state == LifeCycleValues.RUNNING
        assert end.life_cycle_state == LifeCycleValues.RUNNING
        assert not msc.is_end_motion()

        kin_sim.tick()
        assert node1.observation_state == ObservationStateValues.TRUE
        assert node2.observation_state == ObservationStateValues.UNKNOWN
        assert node3.observation_state == ObservationStateValues.TRUE
        assert end.observation_state == ObservationStateValues.TRUE
        assert node1.life_cycle_state == LifeCycleValues.RUNNING
        assert node2.life_cycle_state == LifeCycleValues.SUCCEEDED
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
        assert outer.inner.sub_node1.life_cycle_state == LifeCycleValues.SUCCEEDED
        assert outer.inner.sub_node2.life_cycle_state == LifeCycleValues.RUNNING
        assert outer.inner.life_cycle_state == LifeCycleValues.RUNNING
        assert outer.life_cycle_state == LifeCycleValues.RUNNING
        assert end.life_cycle_state == LifeCycleValues.NOT_STARTED
        assert not msc_copy.is_end_motion()

        kin_sim.tick()
        assert node1.observation_state == ObservationStateValues.TRUE
        assert outer.inner.sub_node1.observation_state == ObservationStateValues.UNKNOWN
        assert outer.inner.sub_node2.observation_state == ObservationStateValues.TRUE
        assert outer.inner.observation_state == ObservationStateValues.UNKNOWN
        assert outer.observation_state == ObservationStateValues.UNKNOWN
        assert end.observation_state == ObservationStateValues.UNKNOWN

        assert node1.life_cycle_state == LifeCycleValues.RUNNING
        assert outer.inner.sub_node1.life_cycle_state == LifeCycleValues.SUCCEEDED
        assert outer.inner.sub_node2.life_cycle_state == LifeCycleValues.RUNNING
        assert outer.inner.life_cycle_state == LifeCycleValues.RUNNING
        assert outer.life_cycle_state == LifeCycleValues.RUNNING
        assert end.life_cycle_state == LifeCycleValues.NOT_STARTED
        assert not msc_copy.is_end_motion()

        kin_sim.tick()
        assert node1.observation_state == ObservationStateValues.TRUE
        assert outer.inner.sub_node1.observation_state == ObservationStateValues.UNKNOWN
        assert outer.inner.sub_node2.observation_state == ObservationStateValues.TRUE
        assert outer.inner.observation_state == ObservationStateValues.TRUE
        assert outer.observation_state == ObservationStateValues.UNKNOWN
        assert end.observation_state == ObservationStateValues.UNKNOWN

        assert node1.life_cycle_state == LifeCycleValues.RUNNING
        assert outer.inner.sub_node1.life_cycle_state == LifeCycleValues.SUCCEEDED
        assert outer.inner.sub_node2.life_cycle_state == LifeCycleValues.RUNNING
        assert outer.inner.life_cycle_state == LifeCycleValues.RUNNING
        assert outer.life_cycle_state == LifeCycleValues.RUNNING
        assert end.life_cycle_state == LifeCycleValues.NOT_STARTED
        assert not msc_copy.is_end_motion()

        kin_sim.tick()
        assert node1.observation_state == ObservationStateValues.TRUE
        assert outer.inner.sub_node1.observation_state == ObservationStateValues.UNKNOWN
        assert outer.inner.sub_node2.observation_state == ObservationStateValues.TRUE
        assert outer.inner.observation_state == ObservationStateValues.TRUE
        assert outer.observation_state == ObservationStateValues.TRUE
        assert end.observation_state == ObservationStateValues.UNKNOWN

        assert node1.life_cycle_state == LifeCycleValues.RUNNING
        assert outer.inner.sub_node1.life_cycle_state == LifeCycleValues.SUCCEEDED
        assert outer.inner.sub_node2.life_cycle_state == LifeCycleValues.RUNNING
        assert outer.inner.life_cycle_state == LifeCycleValues.RUNNING
        assert outer.life_cycle_state == LifeCycleValues.RUNNING
        assert end.life_cycle_state == LifeCycleValues.RUNNING
        assert not msc_copy.is_end_motion()

        kin_sim.tick()
        assert node1.observation_state == ObservationStateValues.TRUE
        assert outer.inner.sub_node1.observation_state == ObservationStateValues.UNKNOWN
        assert outer.inner.sub_node2.observation_state == ObservationStateValues.TRUE
        assert outer.inner.observation_state == ObservationStateValues.TRUE
        assert outer.observation_state == ObservationStateValues.TRUE
        assert end.observation_state == ObservationStateValues.TRUE

        assert node1.life_cycle_state == LifeCycleValues.RUNNING
        assert outer.inner.sub_node1.life_cycle_state == LifeCycleValues.SUCCEEDED
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
                        PR2Joint.TORSO_LIFT: 0.2999225173357618,
                        PR2Joint.HEAD_PAN: 0.042,
                        PR2Joint.HEAD_TILT: -0.37,
                        PR2Joint.RIGHT_UPPER_ARM_ROLL: -0.9487714747527726,
                        PR2Joint.RIGHT_SHOULDER_PAN: -1.0047307505973626,
                        PR2Joint.RIGHT_SHOULDER_LIFT: 0.48736790658811985,
                        PR2Joint.RIGHT_FOREARM_ROLL: -14.895833882874182,
                        PR2Joint.RIGHT_ELBOW_FLEX: -1.392377908925028,
                        PR2Joint.RIGHT_WRIST_FLEX: -0.4548695149411013,
                        PR2Joint.RIGHT_WRIST_ROLL: 0.11426798984097819,
                        PR2Joint.LEFT_UPPER_ARM_ROLL: 1.7383062350263658,
                        PR2Joint.LEFT_SHOULDER_PAN: 1.8799810286792007,
                        PR2Joint.LEFT_SHOULDER_LIFT: 0.011627231224188975,
                        PR2Joint.LEFT_FOREARM_ROLL: 312.67276414458695,
                        PR2Joint.LEFT_ELBOW_FLEX: -2.0300928925694675,
                        PR2Joint.LEFT_WRIST_FLEX: -0.1,
                        PR2Joint.LEFT_WRIST_ROLL: -6.062015047706399,
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

    def test_end_motion_when_all_true_accepts_a_single_node(self):
        """
        A list of one is valid input, so combining it must not depend on there being
        something to combine it with.
        """
        msc = MotionStatechart()
        msc.add_node(ConstTrueNode())
        msc.add_node(end := EndMotion.when_all_true(msc.nodes))

        kin_sim = Executor(MotionStatechartContext(world=World()))
        kin_sim.compile(motion_statechart=msc)
        kin_sim.tick_until_end()

        assert end.life_cycle_state == LifeCycleValues.RUNNING

    def test_end_motion_when_any_true_accepts_a_single_node(self):
        msc = MotionStatechart()
        msc.add_node(ConstTrueNode())
        msc.add_node(end := EndMotion.when_any_true(msc.nodes))

        kin_sim = Executor(MotionStatechartContext(world=World()))
        kin_sim.compile(motion_statechart=msc)
        kin_sim.tick_until_end()

        assert end.life_cycle_state == LifeCycleValues.RUNNING

    def test_cancel_motion_when_all_true_accepts_a_single_node(self):
        msc = MotionStatechart()
        msc.add_node(ConstTrueNode())
        cancelled = Exception("cancelled")
        msc.add_node(CancelMotion.when_all_true(msc.nodes, exception=cancelled))

        kin_sim = Executor(MotionStatechartContext(world=World()))
        kin_sim.compile(motion_statechart=msc)
        with pytest.raises(type(cancelled)) as error:
            kin_sim.tick_until_end()

        assert error.value is cancelled

    def test_cancel_motion_when_any_true_accepts_a_single_node(self):
        msc = MotionStatechart()
        msc.add_node(ConstTrueNode())
        cancelled = Exception("cancelled")
        msc.add_node(CancelMotion.when_any_true(msc.nodes, exception=cancelled))

        kin_sim = Executor(MotionStatechartContext(world=World()))
        kin_sim.compile(motion_statechart=msc)
        with pytest.raises(type(cancelled)) as error:
            kin_sim.tick_until_end()

        assert error.value is cancelled

    def test_end_motion_when_failed_waits_for_the_node_to_end(self):
        """
        Being short of its goal is not yet a failure: the node has to have been ended
        while it was.
        """
        msc = MotionStatechart()
        msc.add_nodes(
            [
                trigger := CountControlCycles(control_cycles=2),
                falling_short := ConstFalseNode(),
                end := EndMotion.when_failed(falling_short),
            ]
        )
        falling_short.end_condition = trigger.observation_variable

        executor = Executor(MotionStatechartContext(world=World()))
        executor.compile(motion_statechart=msc)
        executor.tick()

        assert falling_short.life_cycle_state == LifeCycleValues.RUNNING
        assert end.life_cycle_state == LifeCycleValues.NOT_STARTED

        executor.tick()

        assert falling_short.life_cycle_state == LifeCycleValues.FAILED
        assert end.life_cycle_state == LifeCycleValues.RUNNING

    def test_cancel_motion_when_failed_raises_once_the_node_fails(self):
        msc = MotionStatechart()
        msc.add_nodes([trigger := ConstTrueNode(), falling_short := ConstFalseNode()])
        falling_short.end_condition = trigger.observation_variable
        cancelled = Exception("cancelled")
        msc.add_node(CancelMotion.when_failed(falling_short, exception=cancelled))

        executor = Executor(MotionStatechartContext(world=World()))
        executor.compile(motion_statechart=msc)
        with pytest.raises(type(cancelled)) as error:
            executor.tick_until_end()

        assert error.value is cancelled

    @pytest.mark.parametrize(
        "factory",
        [CancelMotion.when_true, EndMotion.when_true],
    )
    def test_when_true_reads_the_goal_rather_than_the_observation(self, factory):
        """
        The observation behind a verdict is gone once the node ends, so a terminal node
        built from one would stop arming exactly when the verdict it waits for arrives.
        """
        msc = MotionStatechart()
        msc.add_node(watched := ConstTrueNode())

        terminal_node = factory(watched)

        assert terminal_node._start_condition.expression.free_variables() == [
            watched.goal_reached
        ]

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
    def test_hsr_cutting(self, hsr_world_state_reset: World, rclpy_node):
        """
        The HSR cuts a loaf with a knife: down, up, then a sideways shift, repeated
        until five seconds have passed and paused while a human is close.
        """
        map_link = hsr_world_state_reset.root
        gripper = hsr_world_state_reset.get_body_by_name("hand_gripper_tool_frame")
        with hsr_world_state_reset.modify_world():
            knife = Body(
                name=PrefixedName("knife"),
                visual=ShapeCollection(
                    [
                        Box(
                            scale=Scale(0.05, 0.01, 0.15),
                            color=Color(R=0.0, G=0.588, B=0.784),
                        )
                    ]
                ),
            )
            hsr_world_state_reset.add_connection(
                FixedConnection(
                    parent=gripper,
                    child=knife,
                    parent_T_connection_expression=HomogeneousTransformationMatrix.from_xyz_rpy(
                        z=0.06, reference_frame=gripper
                    ),
                )
            )
            loaf = Body(
                name=PrefixedName("loaf"),
                visual=ShapeCollection(
                    [
                        Box(
                            scale=Scale(0.1, 0.2, 0.06),
                            color=Color(R=0.784, G=0.588, B=0.0),
                        )
                    ]
                ),
            )
            hsr_world_state_reset.add_connection(
                FixedConnection(
                    parent=map_link,
                    child=loaf,
                    parent_T_connection_expression=HomogeneousTransformationMatrix.from_xyz_rpy(
                        x=0.91, y=0.25, z=0.62, reference_frame=map_link
                    ),
                )
            )

        depth = 0.1
        right_shift = -0.1
        # The knife's x axis points up in map, so its -x stroke cuts downwards.
        pre_cut_pose = Pose(
            position=Point3(x=0.85, y=0.2, z=0.75, reference_frame=map_link),
            orientation=Quaternion.from_rotation_matrix(
                RotationMatrix.from_vectors(
                    x=Vector3(z=1, reference_frame=map_link),
                    y=Vector3(y=-1, reference_frame=map_link),
                )
            ),
            reference_frame=map_link,
        )

        msc = MotionStatechart()
        position_knife = CartesianPose(
            name="Position Knife",
            root_link=map_link,
            tip_link=knife,
            goal_pose=pre_cut_pose,
        )
        cut = Sequence(
            [
                CartesianPose(
                    name="Down",
                    root_link=map_link,
                    tip_link=knife,
                    goal_pose=Pose(
                        position=Point3(x=-depth, reference_frame=knife),
                        reference_frame=knife,
                    ),
                ),
                CartesianPose(
                    name="Up",
                    root_link=map_link,
                    tip_link=knife,
                    goal_pose=Pose(
                        position=Point3(x=depth, reference_frame=knife),
                        reference_frame=knife,
                    ),
                ),
                CartesianPose(
                    name="Move Right",
                    root_link=map_link,
                    tip_link=knife,
                    goal_pose=Pose(
                        position=Point3(y=right_shift, reference_frame=knife),
                        reference_frame=knife,
                    ),
                ),
            ],
            name="Cut",
        )
        # A human blocks the cut 50 cycles after the knife is in place, for 50 cycles.
        wait_for_human = CountControlCycles(name="Human Approaching", control_cycles=30)
        human_close = Pulse(name="Human Close?", length=50)
        done = CheckControlCycleCount(name="Done?", threshold=150)
        msc.add_nodes([position_knife, cut, wait_for_human, human_close, done])

        position_knife.end_condition = position_knife.observation_variable
        cut.start_condition = position_knife.goal_reached
        cut.end_condition = cut.goal_reached
        wait_for_human.start_condition = position_knife.goal_reached
        human_close.start_condition = wait_for_human.observation_variable
        human_close.end_condition = done.goal_reached
        done.start_condition = cut.goal_reached
        done.reset_condition = trinary_logic_not(done.goal_reached)
        cut.pause_condition = human_close.observation_variable
        # Each finished pass restarts the cut, until the five seconds are up.
        cut.reset_condition = trinary_logic_not(done.goal_reached)
        msc.add_node(EndMotion.when_true(done))

        executor = Executor(MotionStatechartContext(world=hsr_world_state_reset))
        executor.compile(motion_statechart=msc)

        executor.tick_until_end()
        msc.draw("/tmp/muh.pdf")

        assert done.observation_state == ObservationStateValues.TRUE
        cut_life_cycle = msc.history.get_life_cycle_history_of_node(cut)
        assert LifeCycleValues.PAUSED in cut_life_cycle
        restarts = sum(
            1
            for previous, current in zip(cut_life_cycle, cut_life_cycle[1:])
            if current == LifeCycleValues.NOT_STARTED
            and previous != LifeCycleValues.NOT_STARTED
        )
        assert restarts >= 1

    def test_sequence_goal(self, tmp_path):
        """
        Every step but the first starts on the cycle its predecessor succeeds.
        """
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

        cycles_to_run_the_steps = 6
        assert kin_sim.control_cycles == cycles_to_run_the_steps
        assert msc.nodes[1].life_cycle_state == LifeCycleValues.RUNNING
        assert msc.nodes[2].life_cycle_state == LifeCycleValues.SUCCEEDED
        assert msc.nodes[3].life_cycle_state == LifeCycleValues.SUCCEEDED
        assert msc.nodes[4].life_cycle_state == LifeCycleValues.SUCCEEDED
        assert msc.nodes[5].life_cycle_state == LifeCycleValues.SUCCEEDED

    def test_a_sequence_without_steps_is_rejected(self):
        msc = MotionStatechart()
        msc.add_node(Sequence(nodes=[]))

        kin_sim = Executor(MotionStatechartContext(world=World()))
        with pytest.raises(GoalWithoutChildrenError):
            kin_sim.compile(motion_statechart=msc)

    def test_a_parallel_without_nodes_is_rejected(self):
        msc = MotionStatechart()
        msc.add_node(Parallel(nodes=[]))

        kin_sim = Executor(MotionStatechartContext(world=World()))
        with pytest.raises(GoalWithoutChildrenError):
            kin_sim.compile(motion_statechart=msc)

    def test_sequence_gives_a_terminal_step_no_end_condition(self):
        """
        A sequence ends each step by its own observation, but a step that ends the whole
        motion has nothing left to transition to.
        """
        msc = MotionStatechart()
        cancel = CancelMotion(exception=Exception("cancelled"))
        msc.add_node(
            sequence := Sequence(nodes=[CountControlCycles(control_cycles=3), cancel])
        )
        msc.add_node(EndMotion.when_true(sequence))

        kin_sim = Executor(MotionStatechartContext(world=World()))
        kin_sim.compile(motion_statechart=msc)

        assert cancel.end_condition.free_variables() == []

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
        """
        Test that Parallel completes when minimum_success nodes are True.
        """
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
        """
        Test that Parallel completes when no node is True.
        """
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
    Test the constraint collection naming behavior.

    Expected behavior is: - Not naming constraints should result in automatically generated unique names
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
        quadratic_weight=DefaultWeights.WEIGHT_BELOW_COLLISION_AVOIDANCE,
    )
    assert len(col.equality_constraints) >= 3

    for i in range(3):
        col.add_equality_constraint(
            reference_velocity=0.1 * i,
            equality_bound=0.0,
            quadratic_weight=DefaultWeights.WEIGHT_BELOW_COLLISION_AVOIDANCE,
            task_expression=expr,
        )

    col.add_inequality_constraint(
        name="same_name",
        reference_velocity=0.2,
        quadratic_weight=DefaultWeights.WEIGHT_BELOW_COLLISION_AVOIDANCE,
        task_expression=expr,
        lower_error=0.1,
        upper_error=0.2,
    )

    with pytest.raises(DuplicateNameException):
        col.add_equality_constraint(
            name="same_name",
            reference_velocity=0.2,
            equality_bound=0.0,
            quadratic_weight=DefaultWeights.WEIGHT_BELOW_COLLISION_AVOIDANCE,
            task_expression=expr,
        )

    col2 = ConstraintCollection()
    col2.add_equality_constraint(
        name="same_name",
        reference_velocity=0.2,
        equality_bound=0.0,
        quadratic_weight=DefaultWeights.WEIGHT_BELOW_COLLISION_AVOIDANCE,
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
        quadratic_weight=DefaultWeights.WEIGHT_BELOW_COLLISION_AVOIDANCE,
        task_expression=expr,
    )
    constraint = GiskardEqualityConstraint(
        name="same_name",
        expression=expr,
        normalization_factor=0.1,
        quadratic_weight=DefaultWeights.WEIGHT_BELOW_COLLISION_AVOIDANCE,
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
    Tests the LifeCycle transitions of nodes in various edge cases and intended
    behavior.
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
        assert (
            sequence.nodes[1].ticking1.life_cycle_state == LifeCycleValues.INTERRUPTED
        )
        assert (
            sequence.nodes[1].ticking2.life_cycle_state == LifeCycleValues.INTERRUPTED
        )
        assert sequence.nodes[1].life_cycle_state == LifeCycleValues.SUCCEEDED

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
        assert (
            sequence.nodes[1].ticking1.life_cycle_state == LifeCycleValues.INTERRUPTED
        )
        assert (
            sequence.nodes[1].ticking2.life_cycle_state == LifeCycleValues.INTERRUPTED
        )
        assert (
            sequence.nodes[1].ticking3.life_cycle_state == LifeCycleValues.INTERRUPTED
        )
        assert sequence.nodes[1].pulse.life_cycle_state == LifeCycleValues.INTERRUPTED
        assert sequence.nodes[1].life_cycle_state == LifeCycleValues.SUCCEEDED

    def test_end_before_start(self):
        """
        Test for node to start even if its success condition is met before its start
        condition.

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

        assert node3.life_cycle_state == LifeCycleValues.SUCCEEDED
        assert node3.observation_state == ObservationStateValues.TRUE

    def test_end_before_start_in_template(self):
        """
        Test for node to start even if its success condition is met before its start
        condition, when the nodes are inside a template.
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

        assert node.node3.life_cycle_state == LifeCycleValues.SUCCEEDED
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
            LifeCycleValues.SUCCEEDED,
            LifeCycleValues.NOT_STARTED,
            LifeCycleValues.RUNNING,
            LifeCycleValues.RUNNING,
            LifeCycleValues.PAUSED,
            LifeCycleValues.RUNNING,
            LifeCycleValues.SUCCEEDED,
            LifeCycleValues.SUCCEEDED,
            LifeCycleValues.SUCCEEDED,
        ]
        assert msc.history.get_observation_history_of_node(count_node1) == [
            ObservationStateValues.UNKNOWN,
            ObservationStateValues.UNKNOWN,
            ObservationStateValues.TRUE,
            ObservationStateValues.TRUE,
            ObservationStateValues.TRUE,
            ObservationStateValues.TRUE,
            ObservationStateValues.UNKNOWN,
            ObservationStateValues.UNKNOWN,
            ObservationStateValues.TRUE,
            ObservationStateValues.TRUE,
            ObservationStateValues.TRUE,
            ObservationStateValues.TRUE,
            ObservationStateValues.UNKNOWN,
            ObservationStateValues.UNKNOWN,
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
        # A pulse is judged on whether it was still pulsing when it was ended.
        pulse_node1_observations = msc.history.get_observation_history_of_node(
            pulse_node1
        )
        assert pulse_node1_observations == [
            ObservationStateValues.UNKNOWN,
            ObservationStateValues.UNKNOWN,
            ObservationStateValues.UNKNOWN,
            ObservationStateValues.TRUE,
            ObservationStateValues.FALSE,
            ObservationStateValues.FALSE,
            ObservationStateValues.UNKNOWN,
            ObservationStateValues.UNKNOWN,
            ObservationStateValues.UNKNOWN,
            ObservationStateValues.TRUE,
            ObservationStateValues.FALSE,
            ObservationStateValues.FALSE,
            ObservationStateValues.UNKNOWN,
            ObservationStateValues.UNKNOWN,
        ]
        # The two control cycles it is ended on. Each verdict comes from the
        # observation of that same cycle and is then latched.
        first_end, second_end = 5, 11
        assert msc.history.get_life_cycle_history_of_node(pulse_node1) == [
            LifeCycleValues.NOT_STARTED,
            LifeCycleValues.NOT_STARTED,
            LifeCycleValues.RUNNING,
            LifeCycleValues.RUNNING,
            LifeCycleValues.RUNNING,
            LifeCycleValues.verdict_for(pulse_node1_observations[first_end]),
            LifeCycleValues.NOT_STARTED,
            LifeCycleValues.NOT_STARTED,
            LifeCycleValues.RUNNING,
            LifeCycleValues.RUNNING,
            LifeCycleValues.RUNNING,
            LifeCycleValues.verdict_for(pulse_node1_observations[second_end]),
            LifeCycleValues.verdict_for(pulse_node1_observations[second_end]),
            LifeCycleValues.verdict_for(pulse_node1_observations[second_end]),
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

        assert unpause.count_ticks1.life_cycle_state == LifeCycleValues.SUCCEEDED
        assert unpause.cancel.life_cycle_state == LifeCycleValues.NOT_STARTED

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

    def test_a_child_starts_while_its_parent_end_condition_has_no_answer(self):
        """
        Only an end condition that is true ends a node, so a parent whose end condition
        is still undecided is not ending and does not hold its child back.
        """
        msc = MotionStatechart()
        msc.add_nodes(
            [
                undecided := NodeObservingNothingYet(),
                goal := GoalWithChildStartingLate(delay_in_control_cycles=2),
            ]
        )
        goal.end_condition = undecided.observation_variable

        kin_sim = Executor(MotionStatechartContext(world=World()))
        kin_sim.compile(motion_statechart=msc)
        for _ in range(3):
            kin_sim.tick()

        assert undecided.observation_state == ObservationStateValues.UNKNOWN
        assert goal.life_cycle_state == LifeCycleValues.RUNNING
        assert goal.child.life_cycle_state == LifeCycleValues.RUNNING

    def test_a_reset_outranks_a_start(self):
        """
        A reset outranks every other transition, so a node whose reset is held true does
        not start while it is.
        """
        msc = MotionStatechart()
        msc.add_nodes([trigger := ConstTrueNode(), node := ConstTrueNode()])
        node.start_condition = trigger.observation_variable
        node.reset_condition = trigger.observation_variable

        executor = _compile_msc(msc)
        executor.tick()
        assert node.life_cycle_state == LifeCycleValues.NOT_STARTED

        executor.tick()
        assert node.life_cycle_state == LifeCycleValues.NOT_STARTED

    def test_a_node_starts_once_its_reset_drops(self):
        """
        A reset holds a node back only for as long as it is true, so a start condition
        that outlives it still starts the node.
        """
        msc = MotionStatechart()
        msc.add_nodes(
            [
                trigger := ConstTrueNode(),
                reset := Pulse(),
                node := ConstTrueNode(),
            ]
        )
        node.start_condition = trigger.observation_variable
        node.reset_condition = reset.observation_variable

        executor = _compile_msc(msc)
        executor.tick()
        assert node.life_cycle_state == LifeCycleValues.NOT_STARTED

        executor.tick()
        assert node.life_cycle_state == LifeCycleValues.RUNNING

    def test_a_resetting_ancestor_holds_back_a_child_that_would_start(self):
        """
        An ancestor resets everything beneath it, so a child whose start condition turns
        true on the tick its ancestor is reset stays where it is.
        """
        msc = MotionStatechart()
        msc.add_nodes(
            [
                reset := CountControlCycles(control_cycles=2),
                goal := GoalWithChildStartingLate(delay_in_control_cycles=2),
            ]
        )
        goal.reset_condition = reset.observation_variable

        executor = _compile_msc(msc)
        executor.tick()
        executor.tick()

        assert goal.life_cycle_state == LifeCycleValues.NOT_STARTED
        assert goal.child.life_cycle_state == LifeCycleValues.NOT_STARTED


# %% life cycle verdicts


class TestLifeCycleVerdicts:
    """
    Tests which terminal state a node ends in, depending on the condition that ended it.
    """

    @staticmethod
    def _compile(msc: MotionStatechart) -> Executor:
        """
        :param msc: The motion statechart to compile.
        :return: An executor ready to tick `msc`.
        """
        executor = Executor(MotionStatechartContext(world=World()))
        executor.compile(motion_statechart=msc)
        return executor

    def test_ending_a_node_at_its_goal_succeeds_it(self):
        msc = MotionStatechart()
        msc.add_nodes([trigger := ConstTrueNode(), node := ConstTrueNode()])
        node.end_condition = trigger.observation_variable

        self._compile(msc).tick()

        assert node.life_cycle_state == LifeCycleValues.SUCCEEDED

    def test_ending_a_node_short_of_its_goal_fails_it(self):
        msc = MotionStatechart()
        msc.add_nodes([trigger := ConstTrueNode(), node := ConstFalseNode()])
        node.end_condition = trigger.observation_variable

        self._compile(msc).tick()

        assert node.life_cycle_state == LifeCycleValues.FAILED

    def test_ending_a_counter_at_what_it_counts_succeeds_it(self):
        """
        Reaching what it counts is what succeeding means for a counter.
        """
        clock = FakeClock()
        counted_seconds = 1.0
        msc = MotionStatechart()
        msc.add_nodes(
            [
                trigger := ConstTrueNode(),
                node := CountSeconds(seconds=counted_seconds, _now=clock.time),
            ]
        )
        node.end_condition = trigger.observation_variable

        executor = self._compile(msc)
        clock.advance(counted_seconds)
        executor.tick()

        assert node.observation_state == ObservationStateValues.TRUE
        assert node.life_cycle_state == LifeCycleValues.SUCCEEDED

    def test_ending_a_counter_short_of_what_it_counts_fails_it(self):
        """
        The same rule reads the other way: a counter ended before it counted far enough
        did not reach what it counts.
        """
        msc = MotionStatechart()
        msc.add_nodes(
            [
                trigger := ConstTrueNode(),
                node := CountControlCycles(control_cycles=100),
            ]
        )
        node.end_condition = trigger.observation_variable

        self._compile(msc).tick()

        assert node.observation_state == ObservationStateValues.FALSE
        assert node.life_cycle_state == LifeCycleValues.FAILED

    def test_ending_a_node_that_has_observed_nothing_interrupts_it(self):
        """
        An observation that is unknown while the node runs is no basis for a verdict.
        """
        msc = MotionStatechart()
        msc.add_nodes([trigger := ConstTrueNode(), node := NodeObservingNothingYet()])
        node.end_condition = trigger.observation_variable

        self._compile(msc).tick()

        assert node.observation_state == ObservationStateValues.UNKNOWN
        assert node.life_cycle_state == LifeCycleValues.INTERRUPTED

    def test_an_ending_ancestor_interrupts_its_child(self):
        msc = MotionStatechart()
        msc.add_nodes([trigger := ConstTrueNode(), goal := GoalCuttingOffItsChild()])
        goal.end_condition = trigger.observation_variable

        self._compile(msc).tick()

        assert goal.life_cycle_state == LifeCycleValues.SUCCEEDED
        assert goal.child.life_cycle_state == LifeCycleValues.INTERRUPTED

    def test_being_cut_off_interrupts_a_child_that_reached_its_goal(self):
        """
        Collateral shutdown is never a judgement, so a child sitting at its goal is
        interrupted rather than succeeded when its parent takes it down.
        """
        msc = MotionStatechart()
        msc.add_nodes(
            [trigger := ConstTrueNode(), goal := GoalCuttingOffItsChildAtItsGoal()]
        )
        goal.end_condition = trigger.observation_variable

        self._compile(msc).tick()

        assert goal.child.observation_state == ObservationStateValues.TRUE
        assert goal.child.life_cycle_state == LifeCycleValues.INTERRUPTED

    def test_being_cut_off_leaves_the_goal_of_a_child_unanswered(self):
        """
        A child that was never judged cannot report that it missed its goal, even though
        the reading it was at is gone.
        """
        msc = MotionStatechart()
        msc.add_nodes(
            [trigger := ConstTrueNode(), goal := GoalCuttingOffItsChildAtItsGoal()]
        )
        goal.end_condition = trigger.observation_variable

        self._compile(msc).tick()

        assert goal.child.goal_reached_state == ObservationStateValues.UNKNOWN

    def test_being_ended_outranks_being_cut_off(self):
        """
        A node something asked to end is judged on its own terms, even if its parent
        ends on the same tick.
        """
        msc = MotionStatechart()
        msc.add_nodes(
            [trigger := ConstTrueNode(), goal := GoalWithChildFailingOnItsOwn()]
        )
        goal.end_condition = trigger.observation_variable

        self._compile(msc).tick()

        assert goal.life_cycle_state == LifeCycleValues.SUCCEEDED
        assert goal.child.life_cycle_state == LifeCycleValues.FAILED

    def test_reset_leaves_succeeded(self):
        msc = MotionStatechart()
        msc.add_nodes(
            [
                trigger := ConstTrueNode(),
                reset := CountControlCycles(control_cycles=2),
                node := ConstTrueNode(),
            ]
        )
        node.end_condition = trigger.observation_variable
        node.reset_condition = reset.observation_variable

        executor = self._compile(msc)
        executor.tick()
        assert node.life_cycle_state == LifeCycleValues.SUCCEEDED

        executor.tick()
        assert node.life_cycle_state == LifeCycleValues.NOT_STARTED

    def test_reset_leaves_failed(self):
        msc = MotionStatechart()
        msc.add_nodes(
            [
                trigger := ConstTrueNode(),
                reset := CountControlCycles(control_cycles=2),
                node := ConstFalseNode(),
            ]
        )
        node.end_condition = trigger.observation_variable
        node.reset_condition = reset.observation_variable

        executor = self._compile(msc)
        executor.tick()
        assert node.life_cycle_state == LifeCycleValues.FAILED

        executor.tick()
        assert node.life_cycle_state == LifeCycleValues.NOT_STARTED

    def test_reset_leaves_interrupted(self):
        msc = MotionStatechart()
        msc.add_nodes(
            [
                trigger := ConstTrueNode(),
                reset := CountControlCycles(control_cycles=2),
                goal := GoalCuttingOffItsChild(),
            ]
        )
        goal.end_condition = trigger.observation_variable
        goal.reset_condition = reset.observation_variable

        executor = self._compile(msc)
        executor.tick()
        assert goal.child.life_cycle_state == LifeCycleValues.INTERRUPTED

        executor.tick()
        assert goal.child.life_cycle_state == LifeCycleValues.NOT_STARTED

    def test_an_ended_node_observes_nothing_and_keeps_its_verdict(self):
        """
        A node that is no longer running is no longer observing, so its observation says
        so and only its verdict still answers for it.
        """
        msc = MotionStatechart()
        msc.add_nodes([trigger := ConstTrueNode(), node := ConstFalseNode()])
        node.end_condition = trigger.observation_variable

        executor = self._compile(msc)
        executor.tick()
        assert node.observation_state == ObservationStateValues.FALSE

        executor.tick()
        assert node.life_cycle_state == LifeCycleValues.FAILED
        assert node.observation_state == ObservationStateValues.UNKNOWN

    def test_a_paused_node_keeps_the_observation_it_made(self):
        """
        A paused node resumes and observes again, so the reading it was interrupted on
        is kept rather than discarded.
        """
        msc = MotionStatechart()
        msc.add_nodes([pause_trigger := ConstTrueNode(), node := ConstTrueNode()])
        node.pause_condition = pause_trigger.observation_variable

        executor = self._compile(msc)
        executor.tick()
        assert node.observation_state == ObservationStateValues.TRUE

        executor.tick()
        assert node.life_cycle_state == LifeCycleValues.PAUSED
        assert node.observation_state == ObservationStateValues.TRUE

    def test_only_the_verdict_of_an_ended_node_still_starts_a_later_node(self):
        """
        A condition that outlives the node it reads has to read the verdict, since the
        observation behind it is gone by the time the condition is asked again.
        """
        msc = MotionStatechart()
        msc.add_nodes(
            [
                trigger := ConstTrueNode(),
                finished := ConstTrueNode(),
                later := CountControlCycles(control_cycles=3),
                on_verdict := ConstTrueNode(),
                on_observation := ConstTrueNode(),
            ]
        )
        finished.end_condition = trigger.observation_variable
        on_verdict.start_condition = sm.trinary_logic_and(
            finished.is_succeeded, later.observation_variable
        )
        on_observation.start_condition = sm.trinary_logic_and(
            finished.observation_variable, later.observation_variable
        )

        executor = self._compile(msc)
        for _ in range(5):
            executor.tick()

        assert finished.life_cycle_state == LifeCycleValues.SUCCEEDED
        assert later.observation_state == ObservationStateValues.TRUE
        assert on_verdict.life_cycle_state == LifeCycleValues.RUNNING
        assert on_observation.life_cycle_state == LifeCycleValues.NOT_STARTED


# %% the verdict rule itself


@pytest.mark.parametrize(
    "observation, expected_verdict",
    [
        (ObservationStateValues.TRUE, LifeCycleValues.SUCCEEDED),
        (ObservationStateValues.FALSE, LifeCycleValues.FAILED),
        (ObservationStateValues.UNKNOWN, LifeCycleValues.INTERRUPTED),
    ],
)
def test_verdict_for_covers_every_observation_value(
    observation: ObservationStateValues, expected_verdict: LifeCycleValues
):
    assert LifeCycleValues.verdict_for(observation) is expected_verdict


@pytest.mark.parametrize("observation", list(ObservationStateValues))
def test_every_verdict_is_terminal(observation: ObservationStateValues):
    assert LifeCycleValues.verdict_for(observation).is_terminal


# %% what a composite goal reads from its children


class TestGoalReached:
    """
    Tests what a node reports about reaching its goal: the observation it is taking
    while it runs, and the verdict it earned once it has ended.
    """

    def test_a_node_that_has_not_started_reads_unknown(self):
        """
        A node that has not begun has reached nothing and observed nothing.
        """
        msc = MotionStatechart()
        msc.add_nodes([blocker := ConstFalseNode(), node := ConstTrueNode()])
        node.start_condition = blocker.observation_variable

        executor = _compile_msc(msc)
        executor.tick()

        assert node.life_cycle_state == LifeCycleValues.NOT_STARTED
        assert node.goal_reached_state == ObservationStateValues.UNKNOWN

    @pytest.mark.parametrize(
        "node_type, expected",
        [
            (ConstTrueNode, ObservationStateValues.TRUE),
            (ConstFalseNode, ObservationStateValues.FALSE),
        ],
    )
    def test_a_running_node_reads_what_it_observes(self, node_type, expected):
        """
        Nothing has ended, so the live observation is the only answer there is.
        """
        msc = MotionStatechart()
        msc.add_node(node := node_type())

        executor = _compile_msc(msc)
        executor.tick()

        assert node.life_cycle_state == LifeCycleValues.RUNNING
        assert node.goal_reached_state == expected

    def test_a_node_that_ended_well_reads_true(self):
        msc = MotionStatechart()
        msc.add_nodes([trigger := ConstTrueNode(), node := ConstTrueNode()])
        node.end_condition = trigger.observation_variable

        executor = _compile_msc(msc)
        for _ in range(3):
            executor.tick()

        assert node.life_cycle_state == LifeCycleValues.SUCCEEDED
        assert node.observation_state == ObservationStateValues.UNKNOWN
        assert node.goal_reached_state == ObservationStateValues.TRUE

    def test_a_node_that_ended_badly_reads_false(self):
        """
        A node that ended without reaching its goal says so, rather than falling back on
        the observation it no longer takes.
        """
        msc = MotionStatechart()
        msc.add_nodes([trigger := ConstTrueNode(), node := ConstFalseNode()])
        node.end_condition = trigger.observation_variable

        executor = _compile_msc(msc)
        for _ in range(3):
            executor.tick()

        assert node.life_cycle_state == LifeCycleValues.FAILED
        assert node.observation_state == ObservationStateValues.UNKNOWN
        assert node.goal_reached_state == ObservationStateValues.FALSE

    def test_every_node_is_read_against_its_own_life_cycle_state(self):
        """
        One node per life cycle state, so a node reading another node's row would show
        up here.
        """
        msc = MotionStatechart()
        msc.add_nodes(
            [
                trigger := ConstTrueNode(),
                blocker := ConstFalseNode(),
                not_started := ConstTrueNode(),
                running := ConstTrueNode(),
                succeeded := ConstTrueNode(),
                failed := ConstFalseNode(),
                interrupted := NodeObservingNothingYet(),
            ]
        )
        not_started.start_condition = blocker.observation_variable
        for node in (succeeded, failed, interrupted):
            node.end_condition = trigger.observation_variable

        executor = _compile_msc(msc)
        for _ in range(3):
            executor.tick()

        assert {
            node: node.life_cycle_state
            for node in (not_started, running, succeeded, failed, interrupted)
        } == {
            not_started: LifeCycleValues.NOT_STARTED,
            running: LifeCycleValues.RUNNING,
            succeeded: LifeCycleValues.SUCCEEDED,
            failed: LifeCycleValues.FAILED,
            interrupted: LifeCycleValues.INTERRUPTED,
        }
        assert {
            node: node.goal_reached_state
            for node in (not_started, running, succeeded, failed, interrupted)
        } == {
            not_started: ObservationStateValues.UNKNOWN,
            running: ObservationStateValues.TRUE,
            succeeded: ObservationStateValues.TRUE,
            failed: ObservationStateValues.FALSE,
            interrupted: ObservationStateValues.UNKNOWN,
        }

    def test_it_renders_as_one_variable(self):
        msc = MotionStatechart()
        msc.add_nodes([finished := ConstTrueNode(), later := ConstTrueNode()])
        later.start_condition = finished.goal_reached

        assert str(later._start_condition) == f'"{finished.goal_reached.display_name}"'

    def test_it_survives_a_json_round_trip(self):
        msc = MotionStatechart()
        msc.add_nodes([finished := ConstTrueNode(), later := ConstTrueNode()])
        later.start_condition = finished.goal_reached

        msc_copy = MotionStatechart.from_json(
            json.loads(json.dumps(msc.create_structure_copy().to_json()))
        )
        msc_copy._add_transitions()

        later_copy = msc_copy.get_node_by_index(later.index)
        finished_copy = msc_copy.get_node_by_index(finished.index)
        assert later_copy._start_condition.expression.free_variables() == [
            finished_copy.goal_reached
        ]

    def test_a_sequence_survives_losing_its_last_step_observation(self):
        """
        A finished step's observation is only kept because a terminal state freezes it.

        Reading the verdict instead makes the sequence independent of that.
        """
        msc = MotionStatechart()
        msc.add_node(sequence := Sequence(nodes=[ConstTrueNode(), ConstTrueNode()]))
        executor = Executor(MotionStatechartContext(world=World()))
        executor.compile(motion_statechart=msc)
        for _ in range(4):
            executor.tick()
        assert sequence.observation_state == ObservationStateValues.TRUE

        last_step = sequence.nodes[-1]
        assert last_step.life_cycle_state == LifeCycleValues.SUCCEEDED
        msc.observation_state[last_step] = ObservationStateValues.UNKNOWN
        executor.tick()

        assert sequence.observation_state == ObservationStateValues.TRUE

    def test_a_parallel_counts_an_ended_child_and_a_running_one(self):
        """
        A parallel ends none of its children, so a child that keeps running is judged by
        what it observes now and a child something else ended by its verdict.
        """
        msc = MotionStatechart()
        msc.add_node(
            parallel := Parallel(
                nodes=[ended := Pulse(), still_running := ConstTrueNode()]
            )
        )
        ended.end_condition = ended.observation_variable

        executor = _compile_msc(msc)
        for _ in range(4):
            executor.tick()

        assert ended.life_cycle_state == LifeCycleValues.SUCCEEDED
        assert still_running.life_cycle_state == LifeCycleValues.RUNNING
        assert parallel.observation_state == ObservationStateValues.TRUE

    def test_a_parallel_is_not_satisfied_by_children_true_at_different_times(self):
        """
        A parallel asks whether its children reached their goals at the same time, so a
        child that reached its goal and drifted away again stops counting towards it.
        """
        msc = MotionStatechart()
        msc.add_node(
            parallel := Parallel(
                nodes=[
                    drifting := Pulse(),
                    late := CountControlCycles(control_cycles=3),
                ]
            )
        )

        executor = _compile_msc(msc)
        for _ in range(6):
            executor.tick()

        assert (
            ObservationStateValues.TRUE
            in msc.history.get_observation_history_of_node(drifting)
        )
        assert (
            ObservationStateValues.TRUE
            in msc.history.get_observation_history_of_node(late)
        )
        assert set(msc.history.get_observation_history_of_node(parallel)) == {
            ObservationStateValues.UNKNOWN,
            ObservationStateValues.FALSE,
        }

    def test_an_observation_expression_reads_the_previous_control_cycle(self):
        """
        The observation update runs before whether a node reached its goal is derived,
        so an observation expression reads what the previous control cycle left behind.
        """
        msc = MotionStatechart()
        msc.add_nodes(
            [
                watched := CountControlCycles(control_cycles=2),
                observer := NodeObservingGoalReached(watched_node=watched),
            ]
        )

        executor = _compile_msc(msc)
        for _ in range(2):
            executor.tick()

        assert watched.goal_reached_state == ObservationStateValues.TRUE
        assert observer.observation_state == ObservationStateValues.FALSE

        executor.tick()

        assert observer.observation_state == ObservationStateValues.TRUE

    def test_a_condition_reads_the_current_control_cycle(self):
        """
        Whether a node reached its goal is derived before the life cycle update, so a
        transition condition acts on it on the cycle the goal is reached.
        """
        msc = MotionStatechart()
        msc.add_nodes(
            [
                watched := CountControlCycles(control_cycles=2),
                waiting := ConstFalseNode(),
            ]
        )
        waiting.start_condition = watched.goal_reached

        executor = _compile_msc(msc)
        for _ in range(2):
            executor.tick()

        assert watched.goal_reached_state == ObservationStateValues.TRUE
        assert waiting.life_cycle_state == LifeCycleValues.RUNNING


# %% life cycle predicates


class TestLifeCyclePredicates:
    """
    Tests the truth tables of the life cycle predicates and their use in conditions.
    """

    @pytest.mark.parametrize(
        "predicate",
        [
            LifeCyclePredicate.IS_SUCCEEDED,
            LifeCyclePredicate.IS_FAILED,
            LifeCyclePredicate.IS_INTERRUPTED,
        ],
    )
    @pytest.mark.parametrize(
        "life_cycle_state",
        sorted(set(LifeCycleValues) - LifeCycleValues.terminal_states()),
    )
    def test_verdict_predicate_is_unknown_before_a_node_ends(
        self, predicate, life_cycle_state
    ):
        """
        How a node ended has no answer until it ends.
        """
        assert (
            predicate.value.truth_value(life_cycle_state)
            == ObservationStateValues.UNKNOWN
        )

    @pytest.mark.parametrize(
        "predicate, verdict",
        [
            (LifeCyclePredicate.IS_SUCCEEDED, LifeCycleValues.SUCCEEDED),
            (LifeCyclePredicate.IS_FAILED, LifeCycleValues.FAILED),
        ],
    )
    def test_a_judged_node_answers_whether_it_succeeded(self, predicate, verdict):
        """
        Succeeding and failing are the two ways of being judged, so each answers the
        other.
        """
        for life_cycle_state in LifeCycleValues.judged_states():
            expected = (
                ObservationStateValues.TRUE
                if life_cycle_state is verdict
                else ObservationStateValues.FALSE
            )
            assert predicate.value.truth_value(life_cycle_state) == expected

    @pytest.mark.parametrize(
        "predicate",
        [LifeCyclePredicate.IS_SUCCEEDED, LifeCyclePredicate.IS_FAILED],
    )
    def test_an_interrupted_node_was_never_judged(self, predicate):
        """
        Being cut off is no verdict, so how the node would have been judged stays as
        open as it was while the node was running.
        """
        assert (
            predicate.value.truth_value(LifeCycleValues.INTERRUPTED)
            == ObservationStateValues.UNKNOWN
        )

    def test_being_interrupted_is_definite_once_a_node_ends(self):
        """
        Unlike the two verdicts, whether a node was cut off is answered by every way of
        ending.
        """
        for life_cycle_state in LifeCycleValues.terminal_states():
            expected = (
                ObservationStateValues.TRUE
                if life_cycle_state is LifeCycleValues.INTERRUPTED
                else ObservationStateValues.FALSE
            )
            assert (
                LifeCyclePredicate.IS_INTERRUPTED.value.truth_value(life_cycle_state)
                == expected
            )

    @pytest.mark.parametrize(
        "predicate, phase",
        [
            (LifeCyclePredicate.IS_NOT_STARTED, LifeCycleValues.NOT_STARTED),
            (LifeCyclePredicate.IS_RUNNING, LifeCycleValues.RUNNING),
            (LifeCyclePredicate.IS_PAUSED, LifeCycleValues.PAUSED),
        ],
    )
    def test_phase_predicate_is_binary_in_every_state(self, predicate, phase):
        """
        Where a node is right now always has an answer, so a phase predicate is never
        unknown.
        """
        for life_cycle_state in LifeCycleValues:
            expected = (
                ObservationStateValues.TRUE
                if life_cycle_state is phase
                else ObservationStateValues.FALSE
            )
            assert predicate.value.truth_value(life_cycle_state) == expected

    @pytest.mark.parametrize("life_cycle_state", list(LifeCycleValues))
    def test_is_terminated_matches_the_terminal_states(self, life_cycle_state):
        expected = (
            ObservationStateValues.TRUE
            if life_cycle_state.is_terminal
            else ObservationStateValues.FALSE
        )
        assert (
            LifeCyclePredicate.IS_TERMINATED.value.truth_value(life_cycle_state)
            == expected
        )

    def test_a_condition_starts_a_node_on_the_cycle_a_verdict_is_reached(self):
        """
        A predicate reads the life cycle its node reaches in the same step, so a node
        reacting to a verdict starts on the cycle that verdict is reached.
        """
        msc = MotionStatechart()
        msc.add_nodes(
            [
                trigger := ConstTrueNode(),
                first := ConstTrueNode(),
                second := ConstFalseNode(),
            ]
        )
        first.end_condition = trigger.observation_variable
        second.start_condition = first.is_succeeded

        executor = Executor(MotionStatechartContext(world=World()))
        executor.compile(motion_statechart=msc)
        assert second.life_cycle_state == LifeCycleValues.NOT_STARTED

        executor.tick()
        assert first.life_cycle_state == LifeCycleValues.SUCCEEDED
        assert second.life_cycle_state == LifeCycleValues.RUNNING

    def test_a_predicate_follows_the_life_cycle_state_of_its_node(self):
        msc = MotionStatechart()
        msc.add_nodes([trigger := ConstTrueNode(), node := ConstFalseNode()])
        node.end_condition = trigger.observation_variable

        executor = Executor(MotionStatechartContext(world=World()))
        executor.compile(motion_statechart=msc)
        assert node.is_failed.resolve() == ObservationStateValues.UNKNOWN

        executor.tick()
        assert node.life_cycle_state == LifeCycleValues.FAILED
        assert node.is_failed.resolve() == ObservationStateValues.TRUE

    def test_nodes_reading_each_others_verdicts_in_a_cycle_are_rejected(self):
        """
        Neither next state can be computed before the other, so there is no order in
        which the step could be evaluated.
        """
        msc = MotionStatechart()
        msc.add_nodes([first := ConstTrueNode(), second := ConstFalseNode()])
        first.start_condition = second.is_failed
        second.start_condition = first.is_failed

        with pytest.raises(CyclicPredicateDependencyError) as exception_info:
            _compile_msc(msc)

        assert set(exception_info.value.cycle) == {first, second}

    def test_a_node_reading_its_own_verdict_reads_the_state_it_entered_with(self):
        """
        A node cannot react to the state the current step gives it, so a predicate it
        reads about itself is the one it started the step in.
        """
        msc = MotionStatechart()
        msc.add_nodes([trigger := ConstTrueNode(), node := ConstFalseNode()])
        node.end_condition = trigger.observation_variable
        node.reset_condition = node.is_failed

        executor = _compile_msc(msc)

        executor.tick()
        assert node.life_cycle_state == LifeCycleValues.FAILED

        executor.tick()
        assert node.life_cycle_state == LifeCycleValues.NOT_STARTED

    def test_a_predicate_in_an_observation_expression_is_rejected(self):
        """
        Observations are computed before the life cycle update, so there is no next
        state for them to read.
        """
        msc = MotionStatechart()
        msc.add_nodes(
            [watched := ConstTrueNode(), watcher := NodeObservingAPredicate()]
        )
        watcher.watched_node = watched

        with pytest.raises(UnsupportedObservationVariableError) as exception_info:
            _compile_msc(msc)

        assert exception_info.value.unsupported_variable is watched.is_succeeded

    def test_a_predicate_variable_is_created_once_per_node(self):
        node = ConstTrueNode()
        assert node.is_failed is node.is_failed
        assert node.is_failed is not node.is_succeeded

    def test_a_condition_renders_a_predicate_by_name(self):
        msc = MotionStatechart()
        msc.add_nodes([first := ConstTrueNode(), second := ConstFalseNode()])
        second.start_condition = first.is_failed

        assert (
            str(second._start_condition)
            == f'"{first.unique_name}.{LifeCyclePredicate.IS_FAILED.attribute_name}"'
        )

    def test_a_condition_with_a_predicate_survives_a_json_round_trip(self):
        msc = MotionStatechart()
        msc.add_nodes([first := ConstTrueNode(), second := ConstFalseNode()])
        second.start_condition = sm.trinary_logic_and(
            first.is_failed, first.observation_variable
        )

        condition_copy = TrinaryCondition.from_json(
            json.loads(json.dumps(second._start_condition.to_json())),
            motion_statechart=msc,
        )

        assert condition_copy == second._start_condition

    def test_a_predicate_makes_its_node_a_dependency_of_the_condition(self):
        msc = MotionStatechart()
        msc.add_nodes([first := ConstTrueNode(), second := ConstFalseNode()])
        second.start_condition = first.is_failed

        assert second._start_condition.node_dependencies == [first]

    def test_a_raw_life_cycle_variable_is_rejected_in_a_condition(self):
        """
        The raw life cycle value cannot be rendered back into a condition string, so
        only the predicates may be read.
        """
        msc = MotionStatechart()
        msc.add_nodes([first := ConstTrueNode(), second := ConstFalseNode()])

        with pytest.raises(UnsupportedConditionVariableError):
            second.start_condition = first.life_cycle_variable

    def test_a_start_condition_may_not_read_its_own_verdict(self):
        msc = MotionStatechart()
        msc.add_node(node := ConstTrueNode())

        with pytest.raises(SelfInStartConditionError):
            node.start_condition = node.is_failed


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
        msc.add_nodes(
            [
                cart_goal,
                manipulability := MaxManipulability(root_link=root, tip_link=tip),
            ]
        )
        manipulability.end_condition = cart_goal.observation_variable
        msc.add_node(EndMotion.when_true(cart_goal))

        kin_sim = Executor(MotionStatechartContext(world=pr2_world_state_reset))
        kin_sim.compile(motion_statechart=msc)
        kin_sim.tick_until_end()

        fk = pr2_world_state_reset.compute_forward_kinematics_np(
            pr2_world_state_reset.root, tip
        )
        assert np.allclose(fk, goal_pose.to_np(), atol=cart_goal.translation_threshold)


class TestEagerStateVariables:
    """
    A node's observation and life cycle variables are available right after
    construction, before it is added to a motion statechart, so conditions can be wired
    on nested nodes.
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

    References across template levels raise :class:`ConditionScopeError` during
    compilation.
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
