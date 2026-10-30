from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from cramph.context import StatechartContext
from cramph.executor import StatechartExecutor
from cramph.nodes_for_testing import ConstTrueNode
from cramph.statechart import Statechart
from giskardpy.motion_control import (
    DebugExpressionRecording,
    MotionControl,
    WorldStateTrajectoryRecording,
)
from giskardpy.motion_statechart.debug_expression_trajectory import (
    DebugExpressionTrajectory,
    RecordedDebugExpression,
)
from giskardpy.motion_statechart.tasks.cartesian_tasks import CartesianPosition
from semantic_digital_twin.datastructures.prefixed_name import PrefixedName
from semantic_digital_twin.spatial_types import (
    HomogeneousTransformationMatrix,
    Point3,
)
from semantic_digital_twin.world import World
from semantic_digital_twin.world_description.connections import FixedConnection
from semantic_digital_twin.world_description.world_entity import Body
from semantic_digital_twin.world_description.world_state import WorldStateTrajectory

# %% helpers


@dataclass
class TicksAroundARecompile:
    """
    How long a test ticks a statechart before and after it compiles again.
    """

    before: int = 3
    """
    The ticks run before what makes the statechart compile again.
    """

    after: int = 2
    """
    The ticks run after what makes the statechart compile again; a moving robot comes to
    rest first, which takes some of them.
    """

    @property
    def recorded_before(self) -> int:
        """
        :return: How many points are recorded before the statechart compiles again: the
            one of the compile and one per tick.
        """
        return 1 + self.before

    def recorded_times(self, executor: StatechartExecutor) -> list[float]:
        """
        :param executor: The executor that ran the ticks.
        :return: The time of every point recorded over the whole run.
        """
        tick_duration = executor.context.require_tick_duration()
        return [
            tick * tick_duration for tick in range(self.recorded_before + self.after)
        ]


def _position_goal(world: World, tip: Body, name: str) -> CartesianPosition:
    """
    :return: A goal moving `tip` to a point too far away to reach during a test.
    """
    return CartesianPosition(
        root_link=world.root,
        tip_link=tip,
        goal_point=Point3(x=5, reference_frame=world.root),
        name=name,
    )


def _compile_recorded(world: World, tips: list[Body]) -> StatechartExecutor:
    """
    :return: An executor recording the world state and the debug expressions, whose
        compiled statechart moves each of `tips`, with a goal named after it, towards a
        far away point.
    """
    executor = StatechartExecutor(
        context=StatechartContext(world=world),
        extensions=[
            MotionControl(),
            WorldStateTrajectoryRecording(),
            DebugExpressionRecording(),
        ],
    )
    statechart = Statechart(context=executor.context)
    for tip in tips:
        statechart.add_node(_position_goal(world, tip, name=str(tip.name)))
    executor.compile(statechart)
    return executor


def _compile_a_recorded_drive(world: World) -> StatechartExecutor:
    """
    :return: An executor recording the world state and the debug expressions, whose
        compiled statechart drives the robot of `world` towards a far away point.
    """
    return _compile_recorded(world, [_robot(world)])


def _robot(world: World) -> Body:
    """
    :return: The body of the robot of `world`.
    """
    return world.get_kinematic_structure_entity_by_name("bot")


def _position(world: World, body: Body) -> np.ndarray:
    """
    :return: The homogeneous position of `body` in the root of `world`.
    """
    return world.compute_forward_kinematics(world.root, body).to_np()[:, 3]


def _tick(executor: StatechartExecutor, ticks: int) -> None:
    """
    Tick `executor` `ticks` times.
    """
    for _ in range(ticks):
        executor.tick()


def _world_state_trajectory(executor: StatechartExecutor) -> WorldStateTrajectory:
    """
    :return: The world state trajectory `executor` recorded.
    """
    return executor.require_extension(
        WorldStateTrajectoryRecording
    ).plotter.world_state_trajectory


def _debug_expression_trajectory(
    executor: StatechartExecutor,
) -> DebugExpressionTrajectory:
    """
    :return: The debug expression trajectory `executor` recorded.
    """
    return executor.require_extension(
        DebugExpressionRecording
    ).plotter.debug_expression_trajectory


def _recorded(executor: StatechartExecutor, name: str) -> RecordedDebugExpression:
    """
    :return: The only debug expression called `name` that `executor` recorded.
    """
    [recorded] = [
        recorded
        for recorded in _debug_expression_trajectory(
            executor
        ).recorded_debug_expressions
        if recorded.name == name
    ]
    return recorded


def _add_box_next_to_the_robot(world: World) -> Body:
    """
    :return: A box fixed to the root of `world`, half a metre in front of the origin.
    """
    box = Body(name=PrefixedName("box"))
    with world.modify_world():
        world.add_connection(
            FixedConnection(
                parent=world.root,
                child=box,
                parent_T_connection_expression=HomogeneousTransformationMatrix.from_xyz_rpy(
                    x=0.5
                ),
            )
        )
    return box


# %% a statechart that takes a node while it runs


def test_the_world_state_recorded_before_a_node_joined_is_kept(
    cylinder_bot_world: World,
) -> None:
    ticks = TicksAroundARecompile()
    executor = _compile_a_recorded_drive(cylinder_bot_world)
    state_at_compile = cylinder_bot_world.state._data.copy()
    _tick(executor, ticks.before)

    executor.statechart.add_node(ConstTrueNode())
    _tick(executor, ticks.after)

    trajectory = _world_state_trajectory(executor)
    assert np.allclose(trajectory.times, ticks.recorded_times(executor))
    assert np.array_equal(trajectory.data[0], state_at_compile)


def test_the_debug_expressions_recorded_before_a_node_joined_are_kept(
    cylinder_bot_world: World,
) -> None:
    ticks = TicksAroundARecompile()
    executor = _compile_a_recorded_drive(cylinder_bot_world)
    position_at_compile = (
        _recorded(executor, f"{_robot(cylinder_bot_world).name}/current")
        .values[0]
        .copy()
    )
    _tick(executor, ticks.before)

    executor.statechart.add_node(ConstTrueNode())
    _tick(executor, ticks.after)

    trajectory = _debug_expression_trajectory(executor)
    current = _recorded(executor, f"{_robot(cylinder_bot_world).name}/current")
    assert np.allclose(trajectory.times, ticks.recorded_times(executor))
    assert len(current.values) == len(trajectory.times)
    assert np.array_equal(current.values[0], position_at_compile)


def test_a_debug_expression_of_a_joined_node_has_no_value_before_it_joined(
    cylinder_bot_world: World,
) -> None:
    """
    A node added while the robot moves joins once the robot came to rest, so its debug
    expressions have no value up to that recompile and follow the robot after it.
    """
    ticks = TicksAroundARecompile(after=60)
    world = cylinder_bot_world
    executor = _compile_a_recorded_drive(world)
    _tick(executor, ticks.before)

    executor.statechart.add_node(_position_goal(world, _robot(world), name="joined"))
    _tick(executor, ticks.after)

    joined = _recorded(executor, "joined/current")
    driven = _recorded(executor, f"{_robot(cylinder_bot_world).name}/current")
    joined_from = next(
        index
        for index, value in enumerate(joined.values)
        if not np.all(np.isnan(value))
    )
    assert joined_from > ticks.recorded_before
    assert np.all(np.isnan(joined.values[:joined_from]))
    assert np.array_equal(joined.values[joined_from:], driven.values[joined_from:])


# %% a kinematic structure that changes while the statechart runs


def test_the_world_state_recorded_before_a_branch_moved_is_kept(
    cylinder_bot_world: World,
) -> None:
    ticks = TicksAroundARecompile(after=60)
    world = cylinder_bot_world
    box = _add_box_next_to_the_robot(world)
    executor = _compile_a_recorded_drive(world)
    state_at_compile = world.state._data.copy()
    _tick(executor, ticks.before)

    world.move_branch(box, _robot(world))
    _tick(executor, ticks.after)

    trajectory = _world_state_trajectory(executor)
    assert np.allclose(trajectory.times, ticks.recorded_times(executor))
    assert np.array_equal(trajectory.data[0], state_at_compile)


def test_a_debug_expression_is_recorded_along_the_moved_branch(
    cylinder_bot_world: World,
) -> None:
    ticks = TicksAroundARecompile(after=60)
    world = cylinder_bot_world
    box = _add_box_next_to_the_robot(world)
    executor = _compile_recorded(world, [_robot(world), box])
    position_at_compile = _position(world, box)
    _tick(executor, ticks.before)

    world.move_branch(box, _robot(world))
    _tick(executor, ticks.after)

    carried = _recorded(executor, f"{box.name}/current")
    assert len(carried.values) == ticks.recorded_before + ticks.after
    assert np.allclose(carried.values[0], position_at_compile)
    assert np.allclose(carried.values[-1], _position(world, box))
    assert not np.allclose(position_at_compile, _position(world, box))
