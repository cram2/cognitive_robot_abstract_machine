from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field

from cramph.data_types import (
    LifeCycleValues,
    ObservationStateValues,
)
from cramph.node import (
    StatechartNode,
)

from cramph.state import LifeCycleState, ObservationState
from cramph.compiled_tick import LifeCycleChange

# %% recording how the state of a statechart changed


@dataclass(repr=False, eq=False)
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

    def __post_init__(self) -> None:
        """
        Snapshots the given states, so later mutation of the live states does not affect
        this item.
        """
        self.life_cycle_state = self.life_cycle_state.snapshot()
        self.observation_state = self.observation_state.snapshot()

    def records(self, node: StatechartNode) -> bool:
        """
        :param node: The node to look up.
        :return: Whether `node` had joined the statechart when this snapshot was taken.
        """
        return node.index < len(self.life_cycle_state.data)

    def __eq__(self, other: StateHistoryItem) -> bool:
        """
        :param other: The item to compare against.
        :return: True if `other` has the same life cycle and observation state.
        .. note:: :attr:`tick_count` is not compared.
        """
        return (
            other.life_cycle_state == self.life_cycle_state
            and other.observation_state == self.observation_state
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
            if self.records(node)
        }
        return str(merged)


@dataclass(frozen=True)
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


@dataclass
class StateHistoryObserver(ABC):
    """
    Observes the snapshots a :class:`StateHistory` records.
    """

    @abstractmethod
    def on_state_change(self, history: StateHistory) -> None:
        """
        Observe the newest snapshot after it has been appended.

        :param history: The history containing the changed state.
        """


@dataclass
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

    def add_observer(self, observer: StateHistoryObserver) -> None:
        """
        Subscribe an observer once by identity.

        :param observer: The observer to notify when a changed state is recorded.
        """
        if any(registered is observer for registered in self.observers):
            return
        self.observers.append(observer)

    def remove_observer(self, observer: StateHistoryObserver) -> None:
        """
        Remove an observer's subscription if it is present.

        :param observer: The observer whose subscription should end.
        """
        self.observers[:] = [
            registered for registered in self.observers if registered is not observer
        ]

    def append(self, next_item: StateHistoryItem) -> None:
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
            observer.on_state_change(self)

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
            if change.previous_state == LifeCycleValues.NOT_STARTED
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
                if previous is not None and previous.records(node)
                else LifeCycleValues.NOT_STARTED
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
        self, node: StatechartNode
    ) -> list[LifeCycleValues]:
        """
        :param node: The node to fetch the recorded life cycle state for.
        :return: The recorded life cycle state of `node` at every tick since it
            joined, in order.
        """
        return [
            history_item.life_cycle_state[node]
            for history_item in self.items_recording(node)
        ]

    def get_observation_history_of_node(
        self, node: StatechartNode
    ) -> list[ObservationStateValues]:
        """
        :param node: The node to fetch the recorded observation state for.
        :return: The recorded observation state of `node` at every tick since it
            joined, in order.
        """
        return [
            history_item.observation_state[node]
            for history_item in self.items_recording(node)
        ]

    def items_recording(self, node: StatechartNode) -> list[StateHistoryItem]:
        """
        :param node: The node to look up.
        :return: The snapshots taken since `node` joined the statechart, in order.
        """
        return [
            history_item for history_item in self.items if history_item.records(node)
        ]

    def get_current_run_ticks_of_node(self, node: StatechartNode) -> RunTicks | None:
        """
        :param node: The node to look up.
        :return: The start and, once reached, end tick of `node`'s most recent run
            since its last reset, or None if it has not started since then (or
            this history is still empty).
        """
        history = self.items_recording(node)
        if not history:
            return None
        current_state = history[-1].life_cycle_state[node]
        if current_state == LifeCycleValues.NOT_STARTED:
            return None
        start_index = len(history) - 1
        while (
            start_index > 0
            and history[start_index - 1].life_cycle_state[node]
            != LifeCycleValues.NOT_STARTED
        ):
            start_index -= 1
        if not current_state.is_terminal:
            return RunTicks(start_tick=history[start_index].tick_count, end_tick=None)
        end_index = start_index
        while not history[end_index].life_cycle_state[node].is_terminal:
            end_index += 1
        return RunTicks(
            start_tick=history[start_index].tick_count,
            end_tick=history[end_index].tick_count,
        )

    def __len__(self) -> int:
        return len(self.items)
