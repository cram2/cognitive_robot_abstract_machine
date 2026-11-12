"""
The copy of a statechart a server runs, rebuilt from the payloads it reports the
statechart in, see :class:`StatechartPayloadKey`.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum

from typing_extensions import Any

from cramph.context import StatechartContext
from cramph.state import LastObservationState, LifeCycleState, ObservationState
from cramph.statechart import Statechart
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

    JOINED_NODES = "joined_nodes"
    """
    The structure of the nodes that joined the statechart since the structure of the
    nodes before them was sent, as
    :meth:`~cramph.statechart.Statechart.nodes_from_to_json` writes them.
    """

    LIFE_CYCLE_STATE = "life_cycle_state"
    """
    The life cycle state of every node.
    """

    OBSERVATION_STATE = "observation_state"
    """
    The observation state of every node.
    """

    LAST_OBSERVATION_STATE = "last_observation_state"
    """
    The observation every node took most recently.
    """

    TICK_COUNT = "tick_count"
    """
    The tick the states were taken at.
    """


# %% following the payloads


@dataclass
class ReportedStatechart:
    """
    A copy of the statechart a server reports on, holding the structure and the states
    of the payloads it took over.
    """

    statechart: Statechart | None = None
    """
    The copy, None until a payload carried the structure of a statechart.
    """

    def take_over(self, payload: dict[str, Any]) -> None:
        """
        Take over the structure, the joined nodes and the states `payload` reports.

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
        if StatechartPayloadKey.JOINED_NODES in payload:
            self.statechart.add_nodes_from_json(
                payload[StatechartPayloadKey.JOINED_NODES]
            )
        self.statechart.compiler.add_transitions()
        self.statechart.life_cycle_state.data = LifeCycleState.from_json(
            data=payload[StatechartPayloadKey.LIFE_CYCLE_STATE],
            statechart=self.statechart,
        ).data
        self.statechart.observation_state.data = ObservationState.from_json(
            data=payload[StatechartPayloadKey.OBSERVATION_STATE],
            statechart=self.statechart,
        ).data
        self.statechart.last_observation_state.data = LastObservationState.from_json(
            data=payload[StatechartPayloadKey.LAST_OBSERVATION_STATE],
            statechart=self.statechart,
        ).data
