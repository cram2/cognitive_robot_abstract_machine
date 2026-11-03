"""
The copy of a statechart a server runs, rebuilt from the payloads it reports the
statechart in, see :class:`~cramph.data_types.StatechartPayloadKey`.
"""

from __future__ import annotations

from dataclasses import dataclass

from typing_extensions import Any

from cramph.context import StatechartContext
from cramph.data_types import StatechartPayloadKey
from cramph.state import LastObservationState, LifeCycleState, ObservationState
from cramph.statechart import Statechart
from semantic_digital_twin.world import World

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
                payload[StatechartPayloadKey.STATECHART],
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
            payload[StatechartPayloadKey.LIFE_CYCLE_STATE], statechart=self.statechart
        ).data
        self.statechart.observation_state.data = ObservationState.from_json(
            payload[StatechartPayloadKey.OBSERVATION_STATE], statechart=self.statechart
        ).data
        self.statechart.last_observation_state.data = LastObservationState.from_json(
            payload[StatechartPayloadKey.LAST_OBSERVATION_STATE],
            statechart=self.statechart,
        ).data
