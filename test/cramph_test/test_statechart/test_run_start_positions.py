"""
Tests of recording where the degrees of freedom a node moves were when each of its runs
started.
"""

from __future__ import annotations

from dataclasses import dataclass

from cramph.composites import Sequence
from cramph.context import StatechartContext
from cramph.executor import StatechartExecutor
from cramph.node import (
    EndStatechart,
    NodeArtifacts,
    StatechartNode,
    SucceedsOnObservingTrue,
)
from cramph.node_runs import StatechartRuns
from cramph.run_start_positions import RunStartPositionRecording, RunStartPositions
from cramph.statechart import Statechart
from cramph.world_access import WorldAccess
from krrood.symbolic_math.symbolic_math import Scalar
from semantic_digital_twin.world_description.connections import ScrewAxisConnection

from ...dataset.sliding_world import SlidingWorld

# %% a node that sets the slider the moment it starts


@dataclass(eq=False, repr=False, kw_only=True)
class SetsTheSliderOnStart(SucceedsOnObservingTrue, StatechartNode):
    """
    Puts the slider at :attr:`target` when it starts, within the tick it starts on, and
    succeeds right away.
    """

    slider_connection: ScrewAxisConnection
    """
    The joint moving the slider.
    """

    target: float
    """
    Where the slider is put.
    """

    def build_artifacts(self) -> NodeArtifacts:
        return NodeArtifacts(observation=Scalar.const_true())

    def on_start(self) -> None:
        self.slider_connection.position = self.target

    def compute_world_access(self) -> WorldAccess:
        """
        :return: That the node moves the slider's degree of freedom.
        """
        return WorldAccess(
            moved_degrees_of_freedom=frozenset(self.slider_connection.dofs)
        )


# %% where each run found the slider


def test_each_run_records_where_it_found_what_it_moves(
    sliding_world: SlidingWorld,
) -> None:
    """
    Two steps set the slider one after the other, each the moment it starts: each run
    records where the slider was before the tick it started on, not where it put it.
    """
    executor = StatechartExecutor(
        context=StatechartContext(world=sliding_world.world),
        extensions=[RunStartPositionRecording()],
    )
    first = SetsTheSliderOnStart(
        slider_connection=sliding_world.slider_connection, target=0.3
    )
    second = SetsTheSliderOnStart(
        slider_connection=sliding_world.slider_connection, target=0.7
    )
    sequence = Sequence(nodes=[first, second])
    statechart = Statechart(context=executor.context)
    statechart.add_node(node=sequence)
    statechart.add_node(node=EndStatechart.when_true(node=sequence))
    executor.compile(statechart=statechart)
    executor.tick_until_end()

    runs = StatechartRuns(statechart=statechart)
    record = executor.context.require_extension(extension_type=RunStartPositions)
    assert [
        record.position_at_start_of(
            run=runs.latest_run_of_node(node=node),
            degree_of_freedom=sliding_world.slider_connection.raw_dof,
        )
        for node in (first, second)
    ] == [0.0, first.target]
