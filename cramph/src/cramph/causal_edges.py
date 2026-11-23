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
from cramph.node_runs import NodeRun, StatechartRuns
from cramph.world_access import WorldAccess
from semantic_digital_twin.world_description.degree_of_freedom import DegreeOfFreedom
from semantic_digital_twin.world_description.world_entity import (
    KinematicStructureEntity,
)

# %% what each node does to the world


@dataclass(kw_only=True)
class WorldAccessIndex:
    """
    What each node that ran does to the world, as it declares it.
    """

    runs: StatechartRuns
    """
    The runs of the statechart whose nodes are read.
    """

    @cached_property
    def access_of_node(self) -> dict[StatechartNode, WorldAccess]:
        """
        :return: What each node that ran does to the world.
        """
        return {
            node: node.world_access()
            for node, runs in self.runs.runs_of_node.items()
            if runs
        }

    def __getitem__(self, node: StatechartNode) -> WorldAccess:
        """
        :param node: A node of the statechart.
        :return: What it does to the world, nothing for a node that never ran.
        """
        return self.access_of_node.get(node, WorldAccess())


# %% causal edges between runs


@dataclass(frozen=True, kw_only=True)
class RunEdge(ABC):
    """
    Says that the outcome of :attr:`cause` can decide the outcome of :attr:`effect`.
    """

    cause: NodeRun
    """
    The run whose outcome acts on the other.
    """

    effect: NodeRun
    """
    The run acted on.
    """

    @abstractmethod
    def describe(self) -> str:
        """
        :return: How the cause acts on the effect, as the verb phrase between them.
        """

    @staticmethod
    def names_of(
        *, entities: Iterable[DegreeOfFreedom | KinematicStructureEntity]
    ) -> str:
        """
        :return: The names of `entities`, sorted and joined.
        """
        return ", ".join(sorted(str(entity.name.name) for entity in entities))


@dataclass(frozen=True, kw_only=True)
class ContainmentEdge(RunEdge):
    """
    The effect is a composite node whose outcome is made from that of its child, the
    cause.
    """

    def describe(self) -> str:
        return "is part of"


@dataclass(frozen=True, kw_only=True)
class ControlEdge(RunEdge):
    """
    A transition condition of the effect reads the cause.
    """

    transition_kind: TransitionKind
    """
    The kind of the condition of the effect that reads the cause.
    """

    def describe(self) -> str:
        return f"is read by the {self.transition_kind.name.lower()} condition of"


@dataclass(frozen=True, kw_only=True)
class WorldEdge(RunEdge):
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

    def describe(self) -> str:
        changed = self.names_of(entities=[*self.degrees_of_freedom, *self.frames])
        return f"left ({changed}) where they were read by"


class CausalEdgeSource(ABC):
    """
    Finds one kind of causal edge between the runs of a statechart.
    """

    @abstractmethod
    def edges(self, *, runs: StatechartRuns, access: WorldAccessIndex) -> list[RunEdge]:
        """
        :param runs: The runs of the statechart.
        :param access: What each node that ran does to the world.
        :return: The edges of this kind.
        """


@dataclass(kw_only=True)
class ContainmentEdges(CausalEdgeSource):
    """
    Each run of a child acts on the run of its parent that held it.
    """

    def edges(self, *, runs: StatechartRuns, access: WorldAccessIndex) -> list[RunEdge]:
        edges = []
        for run in runs.all_runs:
            parent = run.node.parent_node
            if parent is None:
                continue
            parent_run = runs.latest_run_started_by(node=parent, tick=run.start_tick)
            if parent_run is not None:
                edges.append(ContainmentEdge(cause=run, effect=parent_run))
        return edges


@dataclass(kw_only=True)
class ControlEdges(CausalEdgeSource):
    """
    Each run of a node is acted on by the runs of the other nodes its written conditions
    read, except the nodes it holds or is held by.
    """

    def edges(self, *, runs: StatechartRuns, access: WorldAccessIndex) -> list[RunEdge]:
        edges = []
        for node, node_runs in runs.runs_of_node.items():
            for condition in node.conditions:
                for read_node in set(condition.node_dependencies):
                    if node.holds_or_is_held_by(other=read_node):
                        continue
                    for run in node_runs:
                        read_run = runs.latest_run_started_by(
                            node=read_node, tick=run.start_tick
                        )
                        if read_run is not None:
                            edges.append(
                                ControlEdge(
                                    cause=read_run,
                                    effect=run,
                                    transition_kind=condition.kind,
                                )
                            )
        return edges


@dataclass(kw_only=True)
class WorldEdges(CausalEdgeSource):
    """
    Each run reading a degree of freedom or frame is acted on by the last run that
    changed it and ended before the reading run started.
    """

    def edges(self, *, runs: StatechartRuns, access: WorldAccessIndex) -> list[RunEdge]:
        movers = ChangesOfTheWorld.from_runs(runs=runs, access=access)
        edges = []
        for reader in runs.all_runs:
            reader_access = access[reader.node]
            left_by: dict[NodeRun, tuple[set, set]] = {}
            for degree_of_freedom in reader_access.read:
                last = movers.last_mover(
                    degree_of_freedom_id=degree_of_freedom.id, reader=reader
                )
                if last is not None:
                    left_by.setdefault(last, (set(), set()))[0].add(degree_of_freedom)
            for frame in reader_access.read_frames:
                last = movers.last_reattacher(frame=frame, reader=reader)
                if last is not None:
                    left_by.setdefault(last, (set(), set()))[1].add(frame)
            edges.extend(
                WorldEdge(
                    cause=mover,
                    effect=reader,
                    degrees_of_freedom=frozenset(degrees_of_freedom),
                    frames=frozenset(frames),
                )
                for mover, (degrees_of_freedom, frames) in left_by.items()
            )
        return edges


@dataclass(kw_only=True)
class ChangesOfTheWorld:
    """
    The ended runs that moved each degree of freedom and attached each frame.
    """

    movers: dict[UUID, list[NodeRun]]
    """
    The ended runs that moved the degree of freedom with each id.
    """

    reattachers: dict[KinematicStructureEntity, list[NodeRun]]
    """
    The ended runs that attached each frame to a new parent.
    """

    @classmethod
    def from_runs(
        cls, *, runs: StatechartRuns, access: WorldAccessIndex
    ) -> ChangesOfTheWorld:
        """
        :return: The changes the ended runs of `runs` made.
        """
        movers, reattachers = {}, {}
        for run in runs.all_runs:
            if run.end_tick is None:
                continue
            for degree_of_freedom in access[run.node].moved:
                movers.setdefault(degree_of_freedom.id, []).append(run)
            for frame in access[run.node].reattached:
                reattachers.setdefault(frame, []).append(run)
        return cls(movers=movers, reattachers=reattachers)

    def last_mover(
        self, *, degree_of_freedom_id: UUID, reader: NodeRun
    ) -> NodeRun | None:
        """
        :return: The last run unrelated to `reader` that moved the degree of freedom
            with `degree_of_freedom_id` and ended before `reader` started.
        """
        return self.last_before(
            changers=self.movers.get(degree_of_freedom_id, []), reader=reader
        )

    def last_reattacher(
        self, *, frame: KinematicStructureEntity, reader: NodeRun
    ) -> NodeRun | None:
        """
        :return: The last run unrelated to `reader` that attached `frame` to a new
            parent and ended before `reader` started.
        """
        return self.last_before(changers=self.reattachers.get(frame, []), reader=reader)

    @staticmethod
    def last_before(*, changers: list[NodeRun], reader: NodeRun) -> NodeRun | None:
        """
        :return: The latest of `changers` unrelated to `reader` that ended before it
            started, None if there is none.
        """
        earlier = [
            changer
            for changer in changers
            if changer.end_tick <= reader.start_tick
            and not changer.node.holds_or_is_held_by(other=reader.node)
        ]
        if not earlier:
            return None
        return max(earlier, key=lambda changer: (changer.end_tick, changer.start_tick))
