"""
Keeping the client's copy of a running statechart in the states Giskard reports.

Giskard ticks the statechart of a goal and reports the state of every node in its
feedback and in its result; the client writes these states into its own copy, so that
whatever observes the copy sees the goal progress.
"""

from __future__ import annotations

from cramph.data_types import StatechartPayloadKey
from dataclasses import dataclass

from typing_extensions import Any, Self

from cramph.state import LastObservationState, LifeCycleState, ObservationState
from cramph.statechart import Statechart
from cramph.state_history import StateHistoryItem
from giskardpy.middleware.ros2.exceptions import StateUpdateAheadOfStatechartError

# %% the states Giskard reports


@dataclass
class StatechartStateUpdate:
    """
    The state of every node of a running goal at one tick, as Giskard reported it.
    """

    statechart: Statechart
    """
    The client's copy of the statechart the states belong to.
    """

    tick_count: int
    """
    The tick the states were taken at.
    """

    life_cycle_state: LifeCycleState
    """
    The life cycle state of every node Giskard holds.
    """

    observation_state: ObservationState
    """
    The observation state of every node Giskard holds.
    """

    last_observation_state: LastObservationState
    """
    The observation every node Giskard holds took most recently.
    """

    @classmethod
    def from_json(cls, payload: dict[str, Any], statechart: Statechart) -> Self:
        """
        :param payload: The feedback or result of a goal, as Giskard sent it.
        :param statechart: The client's copy of the statechart of the goal.
        :return: The states the payload reports.
        """
        return cls(
            statechart=statechart,
            tick_count=payload[StatechartPayloadKey.TICK_COUNT],
            life_cycle_state=LifeCycleState.from_json(
                payload[StatechartPayloadKey.LIFE_CYCLE_STATE],
                statechart=statechart,
            ),
            observation_state=ObservationState.from_json(
                payload[StatechartPayloadKey.OBSERVATION_STATE],
                statechart=statechart,
            ),
            last_observation_state=LastObservationState.from_json(
                payload[StatechartPayloadKey.LAST_OBSERVATION_STATE],
                statechart=statechart,
            ),
        )

    def apply(self) -> None:
        """
        Write the reported states into :attr:`statechart` and record them in its
        history, which tells its observers.

        The client's copy may already hold nodes Giskard has not added yet, such as the
        child the client just chose; those keep their state until Giskard reports them.

        :raises StateUpdateAheadOfStatechartError: If Giskard holds more nodes than
            :attr:`statechart`.
        """
        reported_node_count = len(self.life_cycle_state)
        if reported_node_count > len(self.statechart.nodes):
            raise StateUpdateAheadOfStatechartError(
                node_count=len(self.statechart.nodes),
                reported_node_count=reported_node_count,
            )
        self.statechart.life_cycle_state.data[:reported_node_count] = (
            self.life_cycle_state.data
        )
        self.statechart.observation_state.data[:reported_node_count] = (
            self.observation_state.data
        )
        self.statechart.last_observation_state.data[:reported_node_count] = (
            self.last_observation_state.data
        )
        self.statechart.state_history.append(
            next_item=StateHistoryItem(
                tick_count=self.tick_count,
                life_cycle_state=self.statechart.life_cycle_state,
                observation_state=self.statechart.observation_state,
            )
        )
