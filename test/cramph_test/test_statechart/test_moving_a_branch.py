from __future__ import annotations

from dataclasses import dataclass, field

import pytest

from cramph.node import EndedByOwner
from cramph.context import StatechartContext
from cramph.data_types import LifeCycleValues, ObservationStateValues
from cramph.executor import StatechartExecutor, TickingExecutorExtension
from cramph.node import EndStatechart, NodeArtifacts, StatechartNode
from cramph.statechart import Statechart
from cramph.world_modification_nodes import MoveBranch
from semantic_digital_twin.datastructures.prefixed_name import PrefixedName
from semantic_digital_twin.spatial_types import Vector3
from semantic_digital_twin.world import World
from semantic_digital_twin.world_description.connections import (
    FixedConnection,
    PrismaticConnection,
)
from semantic_digital_twin.world_description.world_entity import Body

# %% the world whose structure changes


@dataclass
class SlidingWorld:
    """
    A world with a box resting at the origin and a slider that moves along x, so that
    the box moves with the slider only once it hangs below it.
    """

    world: World
    """
    The world holding every body.
    """

    slider: Body
    """
    The body moved along x by :attr:`slider_connection`.
    """

    box: Body
    """
    The body fixed to the root, which is moved below :attr:`slider`.
    """

    slider_connection: PrismaticConnection
    """
    The joint moving :attr:`slider` along x.
    """


@pytest.fixture()
def sliding_world() -> SlidingWorld:
    world = World()
    root = Body(name=PrefixedName("root"))
    slider = Body(name=PrefixedName("slider"))
    box = Body(name=PrefixedName("box"))
    with world.modify_world():
        world.add_kinematic_structure_entity(root)
        slider_connection = PrismaticConnection.create_with_dofs(
            world=world, parent=root, child=slider, axis=Vector3.X()
        )
        world.add_connection(slider_connection)
        world.add_connection(FixedConnection(parent=root, child=box))
    return SlidingWorld(
        world=world, slider=slider, box=box, slider_connection=slider_connection
    )


@pytest.fixture()
def sliding_executor(sliding_world: SlidingWorld) -> StatechartExecutor:
    return StatechartExecutor(context=StatechartContext(world=sliding_world.world))


@pytest.fixture()
def recording() -> ExtensionRecordingCompiles:
    """
    :return: An extension recording in which order it was told about compiling.
    """
    return ExtensionRecordingCompiles()


@pytest.fixture()
def recording_executor(
    sliding_world: SlidingWorld, recording: ExtensionRecordingCompiles
) -> StatechartExecutor:
    """
    :return: An executor on the sliding world that tells :func:`recording`.
    """
    return StatechartExecutor(
        context=StatechartContext(world=sliding_world.world), extensions=[recording]
    )


# %% nodes and extensions recording what happens to them


@dataclass(eq=False, repr=False, kw_only=True)
class NodeCountingItsBuilds(EndedByOwner, StatechartNode):
    """
    A node that keeps running and counts how often it was set up and built.
    """

    set_up_count: int = field(default=0, init=False)
    """
    How often :meth:`set_up` ran.
    """

    build_count: int = field(default=0, init=False)
    """
    How often :meth:`build_artifacts` ran.
    """

    def set_up(self) -> None:
        self.set_up_count += 1

    def build_artifacts(self) -> NodeArtifacts:
        self.build_count += 1
        return NodeArtifacts()


@dataclass(eq=False, repr=False, kw_only=True)
class NodeObservingABodyPastAPosition(EndedByOwner, StatechartNode):
    """
    A node that keeps running and observes whether a body lies at or beyond a position
    along the x axis of the root.
    """

    body: Body
    """
    The body whose position is observed.
    """

    position: float
    """
    The x coordinate the body has to reach.
    """

    def build_artifacts(self) -> NodeArtifacts:
        root_T_body = self.context.world.compose_forward_kinematics_expression(
            self.context.world.root, self.body
        )
        return NodeArtifacts(observation=root_T_body.position.x >= self.position)


@dataclass
class ExtensionRecordingCompiles(TickingExecutorExtension):
    """
    An executor extension that records in which order it was told about compiling.
    """

    events: list[str] = field(default_factory=list)
    """
    The hooks that ran, in order, each by its method name.
    """

    def before_recompile(self, executor: StatechartExecutor) -> bool:
        self.events.append(self.before_recompile.__name__)
        return True

    def after_compile(self, executor: StatechartExecutor) -> None:
        self.events.append(self.after_compile.__name__)


def _compile(executor: StatechartExecutor, *nodes: StatechartNode) -> Statechart:
    """
    :return: A statechart holding `nodes`, compiled by `executor`.
    """
    statechart = Statechart(context=executor.context)
    for node in nodes:
        statechart.add_node(node)
    executor.compile(statechart)
    return statechart


# %% moving a branch


def test_moving_a_branch_puts_the_body_below_its_new_parent(
    sliding_world: SlidingWorld, sliding_executor: StatechartExecutor
) -> None:
    move = MoveBranch(branch_root=sliding_world.box, new_parent=sliding_world.slider)
    _compile(sliding_executor, move, EndStatechart.when_true(move))

    sliding_executor.tick_until_end(timeout=10)

    assert sliding_world.box.parent_connection.parent is sliding_world.slider
    assert move.life_cycle_state == LifeCycleValues.SUCCEEDED


def test_a_node_moving_a_branch_has_the_chart_rebuilt_on_the_next_tick(
    sliding_world: SlidingWorld, sliding_executor: StatechartExecutor
) -> None:
    counting = NodeCountingItsBuilds(name="counting")
    move = MoveBranch(branch_root=sliding_world.box, new_parent=sliding_world.slider)
    _compile(sliding_executor, counting, move)
    builds_in_the_moving_tick = counting.build_count

    sliding_executor.tick()

    assert (builds_in_the_moving_tick, counting.build_count) == (1, 2)


# %% following the world's structure


def test_a_node_reading_a_moved_body_reads_it_below_its_new_parent(
    sliding_world: SlidingWorld, sliding_executor: StatechartExecutor
) -> None:
    observer = NodeObservingABodyPastAPosition(
        name="observer", body=sliding_world.box, position=0.5
    )
    _compile(sliding_executor, observer)

    sliding_world.world.move_branch(sliding_world.box, sliding_world.slider)
    sliding_world.slider_connection.position = 1.0
    sliding_executor.tick()

    assert observer.observation_state == ObservationStateValues.TRUE


def test_every_node_is_built_again_once_the_world_structure_changed(
    sliding_world: SlidingWorld, sliding_executor: StatechartExecutor
) -> None:
    counting = NodeCountingItsBuilds(name="counting")
    _compile(sliding_executor, counting)

    sliding_world.world.move_branch(sliding_world.box, sliding_world.slider)
    sliding_executor.tick()

    assert counting.build_count == 2


def test_a_node_is_set_up_once_however_often_it_is_built(
    sliding_world: SlidingWorld, sliding_executor: StatechartExecutor
) -> None:
    counting = NodeCountingItsBuilds(name="counting")
    _compile(sliding_executor, counting)

    sliding_world.world.move_branch(sliding_world.box, sliding_world.slider)
    sliding_executor.tick()

    assert counting.set_up_count == 1


def test_a_running_node_keeps_running_and_the_history_goes_on_across_a_rebuild(
    sliding_world: SlidingWorld, sliding_executor: StatechartExecutor
) -> None:
    counting = NodeCountingItsBuilds(name="counting")
    statechart = _compile(sliding_executor, counting)
    sliding_executor.tick()
    ticks_before = sliding_executor.tick_count
    history_before = list(statechart.state_history.items)

    sliding_world.world.move_branch(sliding_world.box, sliding_world.slider)
    sliding_executor.tick()

    assert counting.life_cycle_state == LifeCycleValues.RUNNING
    assert sliding_executor.tick_count == ticks_before + 1
    assert statechart.state_history.items[: len(history_before)] == history_before


def test_extensions_are_told_before_and_after_a_rebuild(
    sliding_world: SlidingWorld,
    recording_executor: StatechartExecutor,
    recording: ExtensionRecordingCompiles,
) -> None:
    _compile(recording_executor, NodeCountingItsBuilds(name="counting"))
    recording.events.clear()

    sliding_world.world.move_branch(sliding_world.box, sliding_world.slider)
    recording_executor.tick()

    assert recording.events == [
        ExtensionRecordingCompiles.before_recompile.__name__,
        ExtensionRecordingCompiles.after_compile.__name__,
    ]


def test_a_tick_in_an_unchanged_world_does_not_compile_again(
    recording_executor: StatechartExecutor, recording: ExtensionRecordingCompiles
) -> None:
    _compile(recording_executor, NodeCountingItsBuilds(name="counting"))
    recording.events.clear()

    recording_executor.tick()

    assert recording.events == []


def test_nodes_joining_after_the_world_structure_changed_have_every_node_built_again(
    sliding_world: SlidingWorld, sliding_executor: StatechartExecutor
) -> None:
    counting = NodeCountingItsBuilds(name="counting")
    statechart = _compile(sliding_executor, counting)

    sliding_world.world.move_branch(sliding_world.box, sliding_world.slider)
    with statechart.modify():
        statechart.add_node(NodeCountingItsBuilds(name="joining"))

    assert counting.build_count == 2
