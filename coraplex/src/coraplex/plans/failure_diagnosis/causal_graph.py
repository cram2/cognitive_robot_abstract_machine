"""
The causal graph whose vertices are the runs of a plan's nodes: the edges
:mod:`cramph.causal_edges` reads from the statechart and the conflicts between motion
tasks.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import cached_property

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
from cramph.node import StatechartNode
from cramph.node_runs import StatechartRuns
from cramph.state_history import NodeRun
from cramph.statechart import Statechart
from giskardpy.motion_statechart.conflict_edges import ConflictEdges

# %% conflicts between the tasks of different actions


@dataclass(kw_only=True)
class PlanConflictEdges(ConflictEdges):
    """
    :class:`~giskardpy.motion_statechart.conflict_edges.ConflictEdges` in a plan, where
    a goal ends below the action that holds it.
    """

    def is_part_of_a_goal(self, *, node: StatechartNode) -> bool:
        """
        Differs from :meth:`ConflictEdges.is_part_of_a_goal` in that an action is never
        part of a goal, even one that takes a goal pose.
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

    def edges_into(self, *, run: NodeRun) -> list[RunEdge]:
        """
        :param run: A run.
        :return: The edges whose effect is `run`.
        """
        return [edge for edge in self.edges if edge.effect == run]


# %% a path to the run where a failure started


@dataclass(frozen=True, kw_only=True)
class RootCause:
    """
    A run that caused a failure, and the edges from it to the run where the failure
    started.
    """

    run: NodeRun
    """
    The run that caused the failure.
    """

    chain: tuple[RunEdge, ...]
    """
    The edges from :attr:`run` to the run where the failure started, in order.
    """
