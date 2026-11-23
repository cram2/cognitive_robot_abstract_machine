"""
The runs of a statechart's nodes, as its state history recorded them.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import cached_property

from cramph.data_types import LifeCycleValue
from cramph.node import StatechartNode
from cramph.statechart import Statechart

# %% one run of a node


@dataclass(frozen=True, kw_only=True)
class NodeRun:
    """
    One run of a statechart node, from the tick it started on to the tick it ended or
    was reset on.
    """

    node: StatechartNode
    """
    The node that ran.
    """

    number: int
    """
    Which run of :attr:`node` this is, counted from one.
    """

    start_tick: int
    """
    The tick this run started on.
    """

    end_tick: int | None
    """
    The tick this run ended or was reset on, None while it still runs.
    """

    final_state: LifeCycleValue
    """
    The life cycle state this run ended in, or the one it is in while it still runs.
    """

    @property
    def failed(self) -> bool:
        """
        :return: Whether this run ended as FAILED.
        """
        return self.final_state == LifeCycleValue.FAILED

    @property
    def last_tick(self) -> float:
        """
        :return: The tick this run ended on, infinity while it still runs.
        """
        return float("inf") if self.end_tick is None else self.end_tick

    def overlaps(self, *, other: NodeRun) -> bool:
        """
        :param other: Another run.
        :return: Whether both runs were active at a common tick.
        """
        return self.start_tick <= other.last_tick and other.start_tick <= self.last_tick

    def contains(self, *, other: NodeRun) -> bool:
        """
        :param other: Another run.
        :return: Whether `other` started and ended while this run was active.
        """
        return self.start_tick <= other.start_tick and other.last_tick <= self.last_tick

    @property
    def variable_name(self) -> str:
        """
        :return: The name of the outcome variable standing for this run in a causal
            graph, unique within its statechart.
        """
        return f"{self.node.name}[{self.node.index}]#{self.number}.outcome"

    def __str__(self) -> str:
        """
        :return: The type and name of the node, which run this is, its ticks and its
            final state.
        """
        return (
            f"{type(self.node).__name__} '{self.node.name}' "
            f"(run {self.number}, ticks {self.start_tick}-{self.end_tick}, "
            f"{self.final_state.name})"
        )


# %% every run of a statechart


@dataclass(kw_only=True)
class StatechartRuns:
    """
    Every run of every node of a statechart, as its state history recorded them.
    """

    statechart: Statechart
    """
    The statechart whose history is read.
    """

    @cached_property
    def runs_of_node(self) -> dict[StatechartNode, list[NodeRun]]:
        """
        :return: The runs of each node of the statechart, in the order they started.
        """
        return {node: self.read_runs(node=node) for node in self.statechart.nodes}

    @cached_property
    def all_runs(self) -> list[NodeRun]:
        """
        :return: Every run of every node.
        """
        return [run for runs in self.runs_of_node.values() for run in runs]

    def read_runs(self, *, node: StatechartNode) -> list[NodeRun]:
        """
        :param node: A node of the statechart.
        :return: The runs of `node`: each starts when it leaves NOT_STARTED and ends
            when it reaches a final state or is reset.
        """
        runs = []
        start_tick = None
        previous_state = LifeCycleValue.NOT_STARTED
        for item in self.statechart.state_history.items_recording(node=node):
            state = item.life_cycle_state[node]
            if state == LifeCycleValue.NOT_STARTED and start_tick is not None:
                runs.append(
                    NodeRun(
                        node=node,
                        number=len(runs) + 1,
                        start_tick=start_tick,
                        end_tick=item.tick_count,
                        final_state=previous_state,
                    )
                )
                start_tick = None
            elif (
                state != LifeCycleValue.NOT_STARTED
                and start_tick is None
                and not previous_state.is_terminal
            ):
                start_tick = item.tick_count
            if state.is_terminal and start_tick is not None:
                runs.append(
                    NodeRun(
                        node=node,
                        number=len(runs) + 1,
                        start_tick=start_tick,
                        end_tick=item.tick_count,
                        final_state=state,
                    )
                )
                start_tick = None
            previous_state = state
        if start_tick is not None:
            runs.append(
                NodeRun(
                    node=node,
                    number=len(runs) + 1,
                    start_tick=start_tick,
                    end_tick=None,
                    final_state=previous_state,
                )
            )
        return runs

    @property
    def last_tick(self) -> int:
        """
        :return: The tick of the newest snapshot of the statechart.
        """
        return self.statechart.state_history.latest_item.tick_count

    def latest_run_started_by(
        self, *, node: StatechartNode, tick: float
    ) -> NodeRun | None:
        """
        :param node: A node of the statechart.
        :param tick: A tick.
        :return: The last run of `node` that started at or before `tick`, None if none.
        """
        started = [run for run in self.runs_of_node[node] if run.start_tick <= tick]
        return started[-1] if started else None

    def latest_run(self, *, node: StatechartNode) -> NodeRun:
        """
        :param node: A node of the statechart that ran.
        :return: Its most recent run.
        """
        return self.runs_of_node[node][-1]
