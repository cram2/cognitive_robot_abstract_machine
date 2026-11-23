"""
The causal graph whose vertices are the runs of a plan's nodes: the edges
:mod:`cramph.causal_edges` reads from the statechart, the conflicts between motion
tasks, and the results the plan's actions hand over.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import cached_property

from random_events.set import Set
from random_events.variable import Symbolic
from typing_extensions import Iterable

from coraplex.robot_plans.actions.base import Action
from cramph.causal_edges import (
    CausalEdgeSource,
    ChangesOfTheWorld,
    ContainmentEdges,
    ControlEdges,
    RunEdge,
    WorldAccessIndex,
    WorldEdges,
)
from cramph.data_types import LifeCycleValue
from cramph.node import StatechartNode
from cramph.node_runs import NodeRun, StatechartRuns
from cramph.statechart import Statechart
from giskardpy.motion_statechart.conflict_edges import ConflictEdge, ConflictEdges
from probabilistic_model.causal_structure.graph import (
    CausalGraph,
    GraphEdge,
    GraphNode,
    NodeLevel,
)

# %% results actions hand over


@dataclass(frozen=True, kw_only=True)
class HandOverEdge(RunEdge):
    """
    The effect takes over the result of the cause, see
    :meth:`~coraplex.robot_plans.actions.base.Action.handed_over_from`.
    """

    def describe(self) -> str:
        return "hands its result over to"


@dataclass(kw_only=True)
class HandOverEdges(CausalEdgeSource):
    """
    Each run of an action is acted on by the latest runs of the nodes whose result it
    takes over.
    """

    def edges(self, *, runs: StatechartRuns, access: WorldAccessIndex) -> list[RunEdge]:
        edges = []
        for node, node_runs in runs.runs_of_node.items():
            if not isinstance(node, Action) or not node_runs:
                continue
            for source in node.handed_over_from():
                for run in node_runs:
                    source_run = runs.latest_run_started_by(
                        node=source, tick=run.start_tick
                    )
                    if source_run is not None:
                        edges.append(HandOverEdge(cause=source_run, effect=run))
        return edges


# %% conflicts between the tasks of different actions


@dataclass(kw_only=True)
class PlanConflictEdges(ConflictEdges):
    """
    :class:`~giskardpy.motion_statechart.conflict_edges.ConflictEdges` in a plan, where
    a goal ends below the action that holds it.
    """

    def is_part_of_a_goal(self, *, node: StatechartNode) -> bool:
        """
        Differs from :meth:`~giskardpy.motion_statechart.conflict_edges.ConflictEdges.is
        _part_of_a_goal` in that an action is never part of a goal, even one that takes
        a goal pose.
        """
        return not isinstance(node, Action) and super().is_part_of_a_goal(node=node)


# %% the causal graph of a plan


@dataclass(kw_only=True)
class NodeCausalGraph:
    """
    The causal graph whose vertices are the runs of a statechart's nodes, with the edges
    that say how the outcome of one run can decide that of another.
    """

    runs: StatechartRuns
    """
    The runs the vertices stand for.
    """

    access: WorldAccessIndex
    """
    What each node that ran does to the world.
    """

    edges: list[RunEdge]
    """
    Every edge found, those kept beside :attr:`graph` included.
    """

    @classmethod
    def from_statechart(
        cls,
        *,
        statechart: Statechart,
        sources: Iterable[CausalEdgeSource] | None = None,
    ) -> NodeCausalGraph:
        """
        :param statechart: A statechart that ran, at least in part.
        :param sources: The sources of edges, every kind of edge if None.
        :return: Its causal graph.
        """
        runs = StatechartRuns(statechart=statechart)
        access = WorldAccessIndex(runs=runs)
        sources = (
            [
                ContainmentEdges(),
                ControlEdges(),
                HandOverEdges(),
                WorldEdges(),
                PlanConflictEdges(),
            ]
            if sources is None
            else sources
        )
        edges = [
            edge
            for source in sources
            for edge in source.edges(runs=runs, access=access)
        ]
        return cls(runs=runs, access=access, edges=edges)

    @cached_property
    def changes_of_the_world(self) -> ChangesOfTheWorld:
        """
        :return: The changes the ended runs made to the world.
        """
        return ChangesOfTheWorld.from_runs(runs=self.runs, access=self.access)

    @cached_property
    def vertex_of_run(self) -> dict[NodeRun, GraphNode]:
        """
        :return: The vertex standing for each run.
        """
        outcomes = Set.from_iterable(LifeCycleValue)
        return {
            run: GraphNode(
                variable=Symbolic(run.variable_name, domain=outcomes),
                level=NodeLevel.EXAMPLE_ATTRIBUTE,
            )
            for run in self.runs.all_runs
        }

    @cached_property
    def run_of_vertex(self) -> dict[GraphNode, NodeRun]:
        """
        :return: The run each vertex stands for.
        """
        return {vertex: run for run, vertex in self.vertex_of_run.items()}

    @cached_property
    def graph(self) -> CausalGraph:
        """
        :return: The causal graph over the runs, holding every edge but the conflicts.
        """
        pairs = {
            (edge.cause, edge.effect)
            for edge in self.edges
            if not isinstance(edge, ConflictEdge)
        }
        return CausalGraph.from_edges(
            list(self.vertex_of_run.values()),
            [
                GraphEdge.directed_edge(
                    self.vertex_of_run[cause], self.vertex_of_run[effect]
                )
                for cause, effect in pairs
            ],
        )

    def edges_into(self, *, run: NodeRun) -> list[RunEdge]:
        """
        :param run: A run.
        :return: The edges whose effect is `run`.
        """
        return [edge for edge in self.edges if edge.effect == run]

    def possible_causes(self, *, run: NodeRun) -> list[NodeRun]:
        """
        :param run: A run.
        :return: The runs with a directed path to `run` in :attr:`graph`.
        """
        return [
            self.run_of_vertex[vertex]
            for vertex in self.graph.possible_ancestors(self.vertex_of_run[run])
        ]
