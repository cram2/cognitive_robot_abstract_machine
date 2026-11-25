"""
Causal edges between the runs of a statechart's nodes, read from the statechart itself:

its tree, the conditions of its nodes, and what its nodes change and read in the world.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from functools import cached_property
from uuid import UUID

from typing_extensions import Iterable

from cramph.data_types import TransitionKind
from cramph.node import StatechartNode
from cramph.node_runs import StatechartRuns
from cramph.state_history import NodeRun
from cramph.world_access import WorldAccess
from semantic_digital_twin.world_description.degree_of_freedom import DegreeOfFreedom
from semantic_digital_twin.world_description.world_entity import (
    KinematicStructureEntity,
)

# %% what each node does to the world


@dataclass(kw_only=True)
class WorldAccessOfNodes:
    """
    What each node that ran does to the world, as it declares it.
    """

    runs: StatechartRuns
    """
    The runs of the statechart whose nodes are read.
    """

    @cached_property
    def world_access_of_each_node(self) -> dict[StatechartNode, WorldAccess]:
        """
        :return: What each node that ran does to the world.
        """
        return {
            node: node.compute_world_access()
            for node, runs in self.runs.runs_by_node.items()
            if runs
        }

    def __getitem__(self, node: StatechartNode) -> WorldAccess:
        """
        :param node: A node of the statechart.
        :return: What it does to the world, nothing for a node that never ran.
        """
        return self.world_access_of_each_node.get(node, WorldAccess())


# %% causal edges between runs


@dataclass(frozen=True, kw_only=True)
class CausalEdge(ABC):
    """
    Says that the outcome of :attr:`effect` depends on the outcome of :attr:`cause`.
    """

    cause: NodeRun
    """
    The run whose outcome the outcome of :attr:`effect` depends on.
    """

    effect: NodeRun
    """
    The run whose outcome depends on the outcome of :attr:`cause`.
    """

    @abstractmethod
    def describe_relation(self) -> str:
        """
        :return: How the cause acts on the effect, as the verb phrase between them.
        """

    @staticmethod
    def join_sorted_names(
        *, entities: Iterable[DegreeOfFreedom | KinematicStructureEntity]
    ) -> str:
        """
        :return: The names of `entities`, sorted and joined.
        """
        return ", ".join(sorted(str(entity.name.name) for entity in entities))


@dataclass(frozen=True, kw_only=True)
class ContainmentEdge(CausalEdge):
    """
    The effect is a composite node whose outcome is made from that of its child, the
    cause.
    """

    def describe_relation(self) -> str:
        return "is part of"


@dataclass(frozen=True, kw_only=True)
class ConditionEdge(CausalEdge):
    """
    A transition condition of the effect reads the cause.
    """

    transition_kind: TransitionKind
    """
    The kind of the condition of the effect that reads the cause.
    """

    def describe_relation(self) -> str:
        return f"is read by the {self.transition_kind.name.lower()} condition of"


@dataclass(frozen=True, kw_only=True)
class WorldStateEdge(CausalEdge):
    """
    The cause was the last run to change degrees of freedom or frames the effect reads,
    before the effect started.
    """

    degrees_of_freedom: frozenset[DegreeOfFreedom] = frozenset()
    """
    The degrees of freedom the cause left where the effect read them.
    """

    frames: frozenset[KinematicStructureEntity] = frozenset()
    """
    The frames the cause attached where the effect read them.
    """

    def describe_relation(self) -> str:
        changed = self.join_sorted_names(
            entities=[*self.degrees_of_freedom, *self.frames]
        )
        return f"left ({changed}) where they were read by"


class CausalEdgeFinder(ABC):
    """
    Finds one kind of causal edge between the runs of a statechart.
    """

    @abstractmethod
    def find_edges(
        self, *, runs: StatechartRuns, access: WorldAccessOfNodes
    ) -> list[CausalEdge]:
        """
        :param runs: The runs of the statechart.
        :param access: What each node that ran does to the world.
        :return: The edges of this kind.
        """


@dataclass(kw_only=True)
class ContainmentEdges(CausalEdgeFinder):
    """
    Each run of a child acts on the run of its parent that held it.
    """

    def find_edges(
        self, *, runs: StatechartRuns, access: WorldAccessOfNodes
    ) -> list[CausalEdge]:
        edges = []
        for run in runs.all_runs:
            parent = run.node.parent_node
            if parent is None:
                continue
            parent_run = runs.latest_run_started_at_or_before(
                node=parent, tick=run.start_tick
            )
            if parent_run is not None:
                edges.append(ContainmentEdge(cause=run, effect=parent_run))
        return edges


@dataclass(kw_only=True)
class ConditionEdges(CausalEdgeFinder):
    """
    Each run of a node is acted on by the runs of the other nodes its written conditions
    read, except the nodes it holds or is held by: for its start condition the runs that
    had started when it started, for any other condition those that had started when it
    ended.
    """

    def find_edges(
        self, *, runs: StatechartRuns, access: WorldAccessOfNodes
    ) -> list[CausalEdge]:
        edges = []
        for node, node_runs in runs.runs_by_node.items():
            for condition in node.conditions:
                for read_node in set(condition.node_dependencies):
                    if node.is_ancestor_or_descendant_of(other=read_node):
                        continue
                    for run in node_runs:
                        read_run = runs.latest_run_started_at_or_before(
                            node=read_node,
                            tick=(
                                run.start_tick
                                if condition.kind is TransitionKind.START
                                else run.last_tick
                            ),
                        )
                        if read_run is not None:
                            edges.append(
                                ConditionEdge(
                                    cause=read_run,
                                    effect=run,
                                    transition_kind=condition.kind,
                                )
                            )
        return edges


@dataclass(kw_only=True)
class WorldStateEdges(CausalEdgeFinder):
    """
    Each run reading a degree of freedom or frame is acted on by the last run that
    changed it and ended before the reading run started.
    """

    def find_edges(
        self, *, runs: StatechartRuns, access: WorldAccessOfNodes
    ) -> list[CausalEdge]:
        movers = WorldChanges.from_runs(runs=runs, access=access)
        edges = []
        for reader in runs.all_runs:
            reader_access = access[reader.node]
            left_by: dict[NodeRun, tuple[set, set]] = {}
            for degree_of_freedom in reader_access.read_degrees_of_freedom:
                last = movers.last_run_that_moved(
                    degree_of_freedom_id=degree_of_freedom.id, reader=reader
                )
                if last is not None:
                    left_by.setdefault(last, (set(), set()))[0].add(degree_of_freedom)
            for frame in reader_access.read_frames:
                last = movers.last_run_that_reparented(frame=frame, reader=reader)
                if last is not None:
                    left_by.setdefault(last, (set(), set()))[1].add(frame)
            edges.extend(
                WorldStateEdge(
                    cause=mover,
                    effect=reader,
                    degrees_of_freedom=frozenset(degrees_of_freedom),
                    frames=frozenset(frames),
                )
                for mover, (degrees_of_freedom, frames) in left_by.items()
            )
        return edges


@dataclass(kw_only=True)
class WorldChange:
    """
    What one ended run changed in the world.
    """

    run: NodeRun
    """
    The run.
    """

    degrees_of_freedom: list[DegreeOfFreedom]
    """
    The degrees of freedom it moved.
    """

    frames: list[KinematicStructureEntity]
    """
    The frames it attached to a new parent.
    """


@dataclass(kw_only=True)
class WorldChanges:
    """
    The changes the ended runs of a statechart made to the world.
    """

    changes: list[WorldChange]
    """
    What each ended run that changed the world changed, in the order of the runs.
    """

    @classmethod
    def from_runs(
        cls, *, runs: StatechartRuns, access: WorldAccessOfNodes
    ) -> WorldChanges:
        """
        :return: The changes the ended runs of `runs` made.
        """
        changes = [
            WorldChange(
                run=run,
                degrees_of_freedom=list(access[run.node].moved_degrees_of_freedom),
                frames=list(access[run.node].reparented_frames),
            )
            for run in runs.all_runs
            if run.end_tick is not None
        ]
        return cls(
            changes=[
                change
                for change in changes
                if change.degrees_of_freedom or change.frames
            ]
        )

    def last_run_that_moved(
        self, *, degree_of_freedom_id: UUID, reader: NodeRun
    ) -> NodeRun | None:
        """
        :return: The last run unrelated to `reader` that moved the degree of freedom
            with `degree_of_freedom_id` and ended before `reader` started.
        """
        return self.latest_unrelated_run_ended_before(
            changers=[
                change.run
                for change in self.changes
                if any(
                    moved.id == degree_of_freedom_id
                    for moved in change.degrees_of_freedom
                )
            ],
            reader=reader,
        )

    def last_run_that_reparented(
        self, *, frame: KinematicStructureEntity, reader: NodeRun
    ) -> NodeRun | None:
        """
        :return: The last run unrelated to `reader` that attached `frame` to a new
            parent and ended before `reader` started.
        """
        return self.latest_unrelated_run_ended_before(
            changers=[change.run for change in self.changes if frame in change.frames],
            reader=reader,
        )

    @staticmethod
    def latest_unrelated_run_ended_before(
        *, changers: list[NodeRun], reader: NodeRun
    ) -> NodeRun | None:
        """
        :return: The latest of `changers` unrelated to `reader` that ended before it
            started, None if there is none.
        """
        earlier = [
            changer
            for changer in changers
            if changer.end_tick <= reader.start_tick
            and not changer.node.is_ancestor_or_descendant_of(other=reader.node)
        ]
        if not earlier:
            return None
        return max(earlier, key=lambda changer: (changer.end_tick, changer.start_tick))
