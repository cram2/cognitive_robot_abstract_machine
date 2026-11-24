"""
The recorded sequence of snapshots of a statechart's life cycle and observation state.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field

from cramph.data_types import (
    LifeCycleValue,
    ObservationStateValue,
)
from cramph.node import (
    StatechartNode,
)

from cramph.state import LifeCycleState, ObservationState
from cramph.compiled_tick import LifeCycleChange

# %% recording how the state of a statechart changed


@dataclass(kw_only=True, repr=False, eq=False)
class StateHistoryItem:
    """
    A snapshot of a :class:`~cramph.statechart.Statechart`'s life cycle and observation
    state at one tick.
    """

    tick_count: int
    """
    The number of ticks run when the snapshot was taken.
    """

    life_cycle_state: LifeCycleState
    """
    The life cycle state of every node at that tick.
    """

    observation_state: ObservationState
    """
    The observation state of every node at that tick.
    """

    cancelled_by: StatechartNode | None = field(default=None, kw_only=True)
    """
    The :class:`~cramph.node.CancelStatechart` whose cancellation ended the statechart
    with this snapshot, None for a snapshot that did not end it.
    """

    cancellation_error: Exception | None = field(default=None, kw_only=True)
    """
    The exception :attr:`cancelled_by` ended the statechart with, which says why it was
    cancelled; None for a snapshot that did not end it.
    """

    def __post_init__(self) -> None:
        """
        Snapshots the given states, so later mutation of the live states does not affect
        this item.
        """
        self.life_cycle_state = self.life_cycle_state.snapshot()
        self.observation_state = self.observation_state.snapshot()

    def records(self, *, node: StatechartNode) -> bool:
        """
        :param node: The node to look up.
        :return: Whether `node` had joined the statechart when this snapshot was taken.
        """
        return node.index < len(self.life_cycle_state.data)

    def life_cycle_of(self, *, node: StatechartNode) -> LifeCycleValue:
        """
        :param node: A node of the statechart.
        :return: The life cycle state of `node` in this snapshot, NOT_STARTED if it had
            not joined the statechart yet.
        """
        if not self.records(node=node):
            return LifeCycleValue.NOT_STARTED
        return self.life_cycle_state[node]

    def observation_of(self, *, node: StatechartNode) -> ObservationStateValue:
        """
        :param node: A node of the statechart.
        :return: What `node` observed in this snapshot, UNKNOWN if it had not joined the
            statechart yet.
        """
        if not self.records(node=node):
            return ObservationStateValue.UNKNOWN
        return self.observation_state[node]

    def __eq__(self, other: StateHistoryItem) -> bool:
        """
        :param other: The item to compare against.
        :return: True if `other` has the same life cycle and observation state and was
            ended by the same cancellation, if any.
        .. note:: :attr:`tick_count` is not compared.
        """
        return (
            other.life_cycle_state == self.life_cycle_state
            and other.observation_state == self.observation_state
            and other.cancelled_by is self.cancelled_by
            and other.cancellation_error is self.cancellation_error
        )

    def __repr__(self) -> str:
        """
        :return: The name of every node this snapshot records, mapped to its
            observation state and life cycle state name.
        """
        merged = {
            node.name: (
                f"{self.observation_state[node].name} | "
                f"{self.life_cycle_state[node].name}"
            )
            for node in self.life_cycle_state.keys()
            if self.records(node=node)
        }
        return str(merged)


@dataclass(kw_only=True, frozen=True)
class RunTicks:
    """
    The tick a node's run started and, once it has ended, the tick it ended on.
    """

    start_tick: int
    """
    The tick this run started on.
    """

    end_tick: int | None
    """
    The tick this run ended on, None while it is still running.
    """


@dataclass(kw_only=True, frozen=True)
class NodeRun(RunTicks):
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
        :return: Whether each run started before the other ended. A run that ends on
            the tick the other starts ended before it.
        """
        return self.start_tick < other.last_tick and other.start_tick < self.last_tick

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


@dataclass(kw_only=True)
class StateHistoryObserver(ABC):
    """
    Observes the snapshots a :class:`StateHistory` records.
    """

    @abstractmethod
    def on_state_change(self, *, history: StateHistory) -> None:
        """
        Observe the newest snapshot after it has been appended.

        :param history: The history containing the changed state.
        """


@dataclass(kw_only=True)
class StateHistory:
    """
    The recorded sequence of :class:`StateHistoryItem` snapshots of a
    :class:`~cramph.statechart.Statechart`.
    """

    items: list[StateHistoryItem] = field(default_factory=list)
    """
    The snapshots in the order in which they were recorded, without consecutive
    duplicates.
    """

    observers: list[StateHistoryObserver] = field(
        default_factory=list, init=False, repr=False, compare=False
    )
    """
    The observers subscribed to newly recorded snapshots.
    """

    def add_observer(self, *, observer: StateHistoryObserver) -> None:
        """
        Subscribe an observer once by identity.

        :param observer: The observer to notify when a changed state is recorded.
        """
        if any(registered is observer for registered in self.observers):
            return
        self.observers.append(observer)

    def remove_observer(self, *, observer: StateHistoryObserver) -> None:
        """
        Remove an observer's subscription if it is present.

        :param observer: The observer whose subscription should end.
        """
        self.observers[:] = [
            registered for registered in self.observers if registered is not observer
        ]

    def append(self, *, next_item: StateHistoryItem) -> None:
        """
        Appends `next_item` and notifies the observers, unless it is equal to the last
        recorded item, in which case it is dropped to avoid storing consecutive
        duplicates.

        :param next_item: The snapshot to append.
        """
        latest_item = self.latest_item
        if latest_item is not None and next_item == latest_item:
            return
        self.items.append(next_item)
        for observer in tuple(self.observers):
            observer.on_state_change(history=self)

    @property
    def latest_item(self) -> StateHistoryItem | None:
        """
        :return: The snapshot recorded last, or None while this history is empty.
        """
        if not self.items:
            return None
        return self.items[-1]

    @property
    def item_before_latest(self) -> StateHistoryItem | None:
        """
        :return: The snapshot recorded right before :attr:`latest_item`, or None while
            fewer than two were recorded.
        """
        if len(self.items) < 2:
            return None
        return self.items[-2]

    def nodes_started_in_latest_item(self) -> list[StatechartNode]:
        """
        :return: The nodes the newest snapshot shows running, or past running, that had
            not started in the snapshot before it, in index order.
        """
        return [
            change.node
            for change in self._life_cycle_changes_of_latest_item()
            if change.previous_state == LifeCycleValue.NOT_STARTED
        ]

    def nodes_ended_in_latest_item(self) -> list[StatechartNode]:
        """
        :return: The nodes that reached a terminal life cycle state in the newest
            snapshot, in index order.
        """
        return [
            change.node
            for change in self._life_cycle_changes_of_latest_item()
            if change.new_state.is_terminal
        ]

    def _life_cycle_changes_of_latest_item(self) -> list[LifeCycleChange]:
        """
        :return: Every change of a life cycle state from the snapshot before the newest
            one to the newest one; a node that joined since changed from not started.
        """
        current = self.latest_item
        if current is None:
            return []
        previous = self.item_before_latest
        changes = []
        for node in current.life_cycle_state.keys():
            previous_state = (
                previous.life_cycle_state[node]
                if previous is not None and previous.records(node=node)
                else LifeCycleValue.NOT_STARTED
            )
            current_state = current.life_cycle_state[node]
            if previous_state != current_state:
                changes.append(
                    LifeCycleChange(
                        node=node,
                        previous_state=previous_state,
                        new_state=current_state,
                    )
                )
        return changes

    def get_life_cycle_history_of_node(
        self, *, node: StatechartNode
    ) -> list[LifeCycleValue]:
        """
        :param node: The node to fetch the recorded life cycle state for.
        :return: The recorded life cycle state of `node` at every tick since it
            joined, in order.
        """
        return [
            history_item.life_cycle_state[node]
            for history_item in self.items_recording(node=node)
        ]

    def get_observation_history_of_node(
        self, *, node: StatechartNode
    ) -> list[ObservationStateValue]:
        """
        :param node: The node to fetch the recorded observation state for.
        :return: The recorded observation state of `node` at every tick since it
            joined, in order.
        """
        return [
            history_item.observation_state[node]
            for history_item in self.items_recording(node=node)
        ]

    def item_at(self, *, tick: int) -> StateHistoryItem:
        """
        :param tick: A tick the history covers.
        :return: The state at `tick`: the last snapshot taken at or before it.
        """
        return [item for item in self.items if item.tick_count <= tick][-1]

    def last_observation_of(
        self, *, node: StatechartNode, tick: int
    ) -> ObservationStateValue:
        """
        :param node: A node of the statechart.
        :param tick: A tick the history covers.
        :return: The latest observation other than UNKNOWN `node` took at or before
            `tick`, UNKNOWN if it took none.
        """
        observed = [
            observation
            for item in self.items
            if item.tick_count <= tick
            and (observation := item.observation_of(node=node))
            is not ObservationStateValue.UNKNOWN
        ]
        return observed[-1] if observed else ObservationStateValue.UNKNOWN

    def items_recording(self, *, node: StatechartNode) -> list[StateHistoryItem]:
        """
        :param node: The node to look up.
        :return: The snapshots taken since `node` joined the statechart, in order.
        """
        return [
            history_item
            for history_item in self.items
            if history_item.records(node=node)
        ]

    def get_current_run_ticks_of_node(self, *, node: StatechartNode) -> RunTicks | None:
        """
        :param node: The node to look up.
        :return: The start and, once reached, end tick of `node`'s most recent run
            since its last reset, or None if it has not started since then (or
            this history is still empty).
        """
        history = self.items_recording(node=node)
        if (
            not history
            or history[-1].life_cycle_state[node] == LifeCycleValue.NOT_STARTED
        ):
            return None
        return self.get_runs_of_node(node=node)[-1]

    def get_runs_of_node(self, *, node: StatechartNode) -> list[NodeRun]:
        """
        :param node: The node to look up.
        :return: Every run of `node`, in order: each starts when it leaves NOT_STARTED
            and ends when it reaches a final state or is reset.
        """
        runs = []
        start_tick = None
        previous_state = LifeCycleValue.NOT_STARTED
        for item in self.items_recording(node=node):
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

    def __len__(self) -> int:
        return len(self.items)
