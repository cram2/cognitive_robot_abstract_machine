"""
The copy of a statechart a server runs, rebuilt from the payloads it reports the
statechart in, see :class:`StatechartPayloadKey`.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum

from typing_extensions import Any, Self

from cramph.context import StatechartContext
from cramph.exceptions import ReportedNodeNotInStatechartError
from cramph.state_history import StateHistoryItem
from cramph.statechart import Statechart
from krrood.adapters.json_serializer import from_json
from semantic_digital_twin.world import World

# %% the keys of a payload


class StatechartPayloadKey(StrEnum):
    """
    The keys of the payloads that report on a statechart a server runs for a goal, and
    of the messages a client answers them with.
    """

    GOAL_ID = "goal_id"
    """
    The goal of the server the payload belongs to.
    """

    STATECHART = "statechart"
    """
    The structure of the statechart, sent once per goal.
    """

    CHANGE_BLOCKS = "change_blocks"
    """
    The structure copies of the change blocks the statechart closed since its structure
    or the change blocks before were sent, oldest first, see
    :meth:`~cramph.statechart_changes.StatechartChangeBlock.create_structure_copy`.
    """

    NODE_STATES = "node_states"
    """
    The states of the nodes at one tick, see :class:`ReportedNodeStates`.
    """


# %% the states of the nodes


@dataclass(kw_only=True)
class ReportedNodeStates:
    """
    The states of the nodes of a statechart at one tick, naming every node by its id, so
    that a copy holding the nodes at other indices takes them over.
    """

    tick_count: int
    """
    The tick the states were taken at.
    """

    node_ids: list[str]
    """
    The id of every reported node, in the order of the values.
    """

    life_cycle_values: list[float]
    """
    The life cycle value of every node of :attr:`node_ids`.
    """

    observation_values: list[float]
    """
    The observation value of every node of :attr:`node_ids`.
    """

    last_observation_values: list[float]
    """
    The observation every node of :attr:`node_ids` took most recently.
    """

    @classmethod
    def from_statechart(cls, *, statechart: Statechart, tick_count: int) -> Self:
        """
        :param statechart: The statechart whose states are reported.
        :param tick_count: The tick the states are taken at.
        :return: The states of every node of `statechart`.
        """
        return cls(
            tick_count=tick_count,
            node_ids=[node._node_id for node in statechart.nodes],
            life_cycle_values=statechart.life_cycle_state.data.tolist(),
            observation_values=statechart.observation_state.data.tolist(),
            last_observation_values=statechart.last_observation_state.data.tolist(),
        )

    def apply_to(self, *, statechart: Statechart) -> None:
        """
        Write the reported states into the nodes of `statechart` with the same ids, and
        record them in its history, which tells its observers. A node of `statechart`
        that is not reported keeps its state.

        :raises ReportedNodeNotInStatechartError: If a reported node is not in
            `statechart`.
        """
        indices = self._indices_in(statechart=statechart)
        for state, values in (
            (statechart.life_cycle_state, self.life_cycle_values),
            (statechart.observation_state, self.observation_values),
            (statechart.last_observation_state, self.last_observation_values),
        ):
            state.data[indices] = values
        statechart.state_history.append(
            next_item=StateHistoryItem(
                tick_count=self.tick_count,
                life_cycle_state=statechart.life_cycle_state,
                observation_state=statechart.observation_state,
            )
        )

    def _indices_in(self, *, statechart: Statechart) -> list[int]:
        """
        :return: The index every node of :attr:`node_ids` has in `statechart`.
        :raises ReportedNodeNotInStatechartError: If a reported node is not in
            `statechart`.
        """
        index_by_node_id = {node._node_id: node.index for node in statechart.nodes}
        missing = [
            node_id for node_id in self.node_ids if node_id not in index_by_node_id
        ]
        if missing:
            raise ReportedNodeNotInStatechartError(node_ids=missing)
        return [index_by_node_id[node_id] for node_id in self.node_ids]


# %% following the payloads


@dataclass(kw_only=True)
class ReportedStatechart:
    """
    A copy of the statechart a server reports on, holding the structure and the states
    of the payloads it took over.
    """

    statechart: Statechart | None = None
    """
    The copy, None until a payload carried the structure of a statechart.
    """

    def take_over(self, *, payload: dict[str, Any]) -> None:
        """
        Take over the structure, the changes to it and the states `payload` reports.

        A payload arriving before any structure is ignored, since its states belong to
        nodes this copy does not know.

        :param payload: A payload of the server.
        """
        if StatechartPayloadKey.STATECHART in payload:
            self.statechart = Statechart.from_json(
                data=payload[StatechartPayloadKey.STATECHART],
                context=StatechartContext(world=World()),
            )
        if self.statechart is None:
            return
        for change_block in payload.get(StatechartPayloadKey.CHANGE_BLOCKS, []):
            self.statechart.apply_change_block_from_json(data=change_block)
        self.statechart.compiler.add_transitions()
        from_json(payload[StatechartPayloadKey.NODE_STATES]).apply_to(
            statechart=self.statechart
        )
