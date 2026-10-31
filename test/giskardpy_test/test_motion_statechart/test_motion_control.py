from __future__ import annotations

from dataclasses import dataclass, field

import pytest

from cramph.composites import Sequence
from cramph.context import StatechartContext
from cramph.data_types import LifeCycleValues
from cramph.exceptions import (
    ConflictingTickDurationError,
    NodesMissingContextExtensionsError,
)
from cramph.executor import StatechartExecutor, TickingExecutorExtension
from cramph.nodes_for_testing import ConstTrueNode
from cramph.statechart import Statechart
from giskardpy.motion_control import MotionControl
from giskardpy.motion_statechart.context import MotionControlContext
from giskardpy.motion_statechart.graph_node import EndMotion
from giskardpy.motion_statechart.monitors.monitors import LocalMinimumReached
from giskardpy.motion_statechart.tasks.cartesian_tasks import CartesianPosition
from giskardpy.motion_statechart.tasks.joint_tasks import JointPositionList
from semantic_digital_twin.datastructures.joint_state import JointState
from semantic_digital_twin.spatial_types import Point3
from semantic_digital_twin.world import World
from semantic_digital_twin.world_description.connections import PrismaticConnection
from semantic_digital_twin.world_description.degree_of_freedom import DegreeOfFreedom

# %% context


def test_motion_control_ticks_at_the_control_rate() -> None:
    motion_control = MotionControl()
    context = StatechartContext(world=World())

    motion_control.extend_context(context)

    assert (
        context.require_tick_duration()
        == motion_control.qp_controller_config.control_time_step.total_seconds()
    )


def test_motion_control_hands_its_controller_configuration_to_the_nodes() -> None:
    motion_control = MotionControl()
    context = StatechartContext(world=World())

    motion_control.extend_context(context)

    assert (
        context.require_extension(MotionControlContext).qp_controller_config
        is motion_control.qp_controller_config
    )


def test_motion_control_rejects_a_context_ticking_at_another_rate() -> None:
    motion_control = MotionControl()
    context = StatechartContext(
        world=World(),
        tick_duration=motion_control.qp_controller_config.control_time_step.total_seconds()
        * 2,
    )

    with pytest.raises(ConflictingTickDurationError):
        motion_control.extend_context(context)


def test_a_node_reading_the_controller_configuration_needs_motion_control() -> None:
    statechart = Statechart(context=StatechartContext(world=World()))
    statechart.add_node(node := LocalMinimumReached())

    with pytest.raises(NodesMissingContextExtensionsError) as raised:
        statechart.compiler.compile()

    assert raised.value.nodes_by_missing_extension == {MotionControlContext: [node]}


# %% sharing a statechart with other modules


@dataclass
class ExtensionCountingTicks(TickingExecutorExtension):
    """
    An executor extension of another module, counting the ticks it was called after.
    """

    ticks: int = 0
    """
    How many ticks this extension was called after.
    """

    def after_tick(self, executor: StatechartExecutor) -> None:
        self.ticks += 1


def test_motion_nodes_and_nodes_of_other_modules_run_in_one_statechart(
    cylinder_bot_world: World,
) -> None:
    root = cylinder_bot_world.root
    tip = cylinder_bot_world.get_kinematic_structure_entity_by_name("bot")
    goal = CartesianPosition(
        root_link=root,
        tip_link=tip,
        goal_point=Point3(x=0.1, reference_frame=root),
    )
    other_node = ConstTrueNode()
    tick_counter = ExtensionCountingTicks()
    executor = StatechartExecutor(
        context=StatechartContext(world=cylinder_bot_world),
        extensions=[MotionControl(), tick_counter],
    )
    statechart = Statechart(context=executor.context)
    statechart.add_nodes(
        [goal, other_node, EndMotion.when_all_true([goal, other_node])]
    )

    executor.compile(statechart)
    executor.tick_until_end()

    assert statechart.is_ended()
    assert goal.life_cycle_state == LifeCycleValues.RUNNING
    assert other_node.life_cycle_state == LifeCycleValues.RUNNING
    assert tick_counter.ticks == executor.tick_count
    assert executor.require_extension(MotionControl).qp_controller is not None


# %% degrees of freedom no running node moves


@dataclass
class StateRecordedWhileNodeRuns(TickingExecutorExtension):
    """
    Records the state of a degree of freedom after every tick in which a node runs.

    .. note:: List it after :class:`MotionControl`, so it records the state after the
        commands of the tick were applied.
    """

    degree_of_freedom: DegreeOfFreedom
    """
    The degree of freedom whose state is recorded.
    """

    node: JointPositionList
    """
    The node during whose run the state is recorded.
    """

    positions: list[float] = field(default_factory=list)
    """
    The position after every tick in which :attr:`node` ran.
    """

    velocities: list[float] = field(default_factory=list)
    """
    The velocity after every tick in which :attr:`node` ran.
    """

    def after_tick(self, executor: StatechartExecutor) -> None:
        if self.node.life_cycle_state != LifeCycleValues.RUNNING:
            return
        state = executor.context.world.state[self.degree_of_freedom.id]
        self.positions.append(float(state.position))
        self.velocities.append(float(state.velocity))


def run_one_joint_goal_after_another(
    world: World,
    first: JointPositionList,
    second: JointPositionList,
    recorded: PrismaticConnection,
    while_running: JointPositionList,
) -> StateRecordedWhileNodeRuns:
    """
    :param world: The world the goals move.
    :param first: The goal that runs first.
    :param second: The goal that runs once `first` ended.
    :param recorded: The connection whose state is recorded.
    :param while_running: The goal during whose run the state is recorded.
    :return: The state of `recorded` during the run of `while_running`.
    """
    recording = StateRecordedWhileNodeRuns(
        degree_of_freedom=recorded.raw_dof, node=while_running
    )
    executor = StatechartExecutor(
        context=StatechartContext(world=world),
        extensions=[MotionControl(), recording],
    )
    statechart = Statechart(context=executor.context)
    statechart.add_node(sequence := Sequence(nodes=[first, second]))
    statechart.add_node(EndMotion.when_true(sequence))
    executor.compile(statechart)
    executor.tick_until_end()
    return recording


def test_a_joint_at_its_limit_stays_put_until_its_node_runs(
    prismatic_bot2: World,
) -> None:
    moved_first, waiting = prismatic_bot2.controlled_connections
    lower_limit = waiting.raw_dof.limits.lower.position
    waiting.position = lower_limit
    first = JointPositionList(goal_state=JointState.from_mapping({moved_first: 0.8}))
    second = JointPositionList(goal_state=JointState.from_mapping({waiting: 0.0}))

    recording = run_one_joint_goal_after_another(
        prismatic_bot2, first, second, recorded=waiting, while_running=first
    )

    assert recording.positions == [lower_limit] * len(recording.positions)


def test_a_joint_is_at_rest_once_its_node_ended(prismatic_bot2: World) -> None:
    moved_second, moved_first = prismatic_bot2.controlled_connections
    upper_limit = moved_first.raw_dof.limits.upper.position
    first = JointPositionList(
        goal_state=JointState.from_mapping({moved_first: upper_limit})
    )
    second = JointPositionList(goal_state=JointState.from_mapping({moved_second: 0.8}))

    recording = run_one_joint_goal_after_another(
        prismatic_bot2, first, second, recorded=moved_first, while_running=second
    )

    # Stopping leaves nothing but the rounding error of the last integration step.
    assert recording.velocities[-1] == pytest.approx(0.0, abs=1e-12)
