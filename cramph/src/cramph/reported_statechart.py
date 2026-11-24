"""
The copy of a statechart a server runs, rebuilt from the payloads it reports the
statechart in, see :class:`StatechartPayloadKey`.
"""

from __future__ import annotations

import threading
from dataclasses import dataclass, field
from enum import StrEnum

from typing_extensions import Any, Self

from cramph.context import StatechartContext
from cramph.exceptions import ReportedNodeNotInStatechartError
from cramph.state_history import StateHistoryItem
from cramph.statechart import Statechart
from krrood.adapters.json_serializer import from_json
from krrood_synchronization.messages import (
    Message,
    MessageOrigin,
    Sender,
    StreamPosition,
)
from krrood_synchronization.stream_participants import StreamPublisher, StreamReceiver
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
class ReportedNodeStates(Message):
    """
    The states of the nodes of a statechart at one tick, naming every node by its id, so
    that a copy holding the nodes at other indices takes them over. Its position tells a
    receiver whether it is older than a report it already took over.
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
    def from_statechart(
        cls, *, statechart: Statechart, tick_count: int, position: StreamPosition
    ) -> Self:
        """
        :param statechart: The statechart whose states are reported.
        :param tick_count: The tick the states are taken at.
        :param position: Where the report sits in the stream of its publisher.
        :return: The states of every node of `statechart`.
        """
        return cls(
            position=position,
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


@dataclass(kw_only=True)
class NodeStatesPublisher(StreamPublisher):
    """
    Reports the node states of the statecharts a process runs, numbering every report
    by its position in one stream.
    """

    sender: Sender
    """
    The node of the process that sends the reports.
    """

    _origin: MessageOrigin = field(init=False, repr=False)
    """
    The publisher every report comes from.
    """

    def __post_init__(self) -> None:
        """
        Names :attr:`sender` as the publisher of a stream of its own.
        """
        self._origin = MessageOrigin(sender=self.sender)

    @property
    def origin(self) -> MessageOrigin:
        return self._origin

    def report(self, *, statechart: Statechart, tick_count: int) -> ReportedNodeStates:
        """
        :param statechart: The statechart whose states are reported.
        :param tick_count: The tick the states are taken at.
        :return: The states of every node of `statechart`, at the position after the
            report numbered last.
        """
        report = ReportedNodeStates.from_statechart(
            statechart=statechart,
            tick_count=tick_count,
            position=StreamPosition(origin=self.origin),
        )
        self.assign_next_sequence_number(message=report)
        return report


@dataclass(kw_only=True)
class NodeStatesReceiver(StreamReceiver):
    """
    Takes over the reported node states into a copy of the statechart, each report at
    most once and never one older than a report it took over.
    """

    _take_over_lock: threading.Lock = field(
        default_factory=threading.Lock, init=False, repr=False
    )
    """
    Keeps a report taken over on one thread from overwriting a later one taken over on
    another, such as the feedback and the result of a goal.
    """

    def take_over(self, *, report: ReportedNodeStates, statechart: Statechart) -> None:
        """
        Applies `report` to `statechart`, see :meth:`ReportedNodeStates.apply_to`,
        unless this receiver took over `report` or a later report of its publisher.

        :raises ReportedNodeNotInStatechartError: If a reported node is not in
            `statechart`.
        """
        with self._take_over_lock:
            if self.contains(position=report.position):
                return
            report.apply_to(statechart=statechart)
            self.record_applied(position=report.position)


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

    node_states_receiver: NodeStatesReceiver = field(
        default_factory=NodeStatesReceiver, init=False, repr=False
    )
    """
    Takes over the reported node states into :attr:`statechart`.
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
        self.node_states_receiver.take_over(
            report=from_json(payload[StatechartPayloadKey.NODE_STATES]),
            statechart=self.statechart,
        )
