"""
The runs of a statechart's nodes, indexed by node, as its state history recorded them.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import cached_property

from cramph.node import StatechartNode
from cramph.state_history import NodeRun
from cramph.statechart import Statechart

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
        return {
            node: self.statechart.state_history.get_runs_of_node(node=node)
            for node in self.statechart.nodes
        }

    @cached_property
    def all_runs(self) -> list[NodeRun]:
        """
        :return: Every run of every node.
        """
        return [run for runs in self.runs_of_node.values() for run in runs]

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
