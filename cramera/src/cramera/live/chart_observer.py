"""
Watching the motion statechart an executor is ticking.

A statechart exists only while giskardpy executes it: one is compiled per merged motion
group and thrown away afterwards. What a viewer shows of it, and what a recording keeps
of it, is therefore a snapshot per tick -- taken here, so the live bridge and the
recording use the same observations.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from giskardpy.motion_statechart.data_types import LifeCycleValues
from giskardpy.motion_statechart.motion_statechart import MotionStatechart
from typing_extensions import List, Optional, Tuple

from cramera.live.chart_structure import (
    ChartNodeEntry,
    ChartSnapshot,
    ChartStructure,
    ObservationName,
    structure_of,
)


@dataclass
class ChartObserver:
    """
    One executor's statechart, as snapshots.

    Remembers what it last saw, so a chart is re-serialized only when the executor
    compiled a new one.
    """

    title: str = ""
    """
    What the run calls what the chart is executing, shown above it.
    """

    _chart: Optional[MotionStatechart] = field(default=None, init=False)
    """
    The chart last seen, to recognize a newly compiled one.
    """

    _structure: Optional[ChartStructure] = field(default=None, init=False)
    """
    Its nodes and transitions, which only change when the chart does.
    """

    _sent_node_states: Optional[Tuple[List[str], List[ObservationName]]] = field(
        default=None, init=False
    )
    """
    Where its nodes stood when :meth:`change` last reported them.
    """

    def snapshot(self, chart: Optional[MotionStatechart]) -> Optional[ChartSnapshot]:
        """
        What the chart looks like now, or None when nothing is executing.

        :param chart: The statechart the executor is currently ticking, if any.
        """
        if chart is None:
            return None
        if chart is not self._chart or self._structure is None:
            self._chart = chart
            self._structure = structure_of(chart)
            self._sent_node_states = None
        structure = self._structure
        life_cycle = [
            int(chart.life_cycle_state.data[index])
            for index in structure.node_state_indices
        ]
        observations = [
            float(chart.observation_state.data[index])
            for index in structure.node_state_indices
        ]
        return ChartSnapshot(
            signature=structure.signature,
            title=self.title,
            nodes=[
                ChartNodeEntry(
                    id=node.id,
                    name=node.name,
                    class_name=node.class_name,
                    parent=node.parent,
                    life_cycle=LifeCycleValues(life_cycle[position]).name,
                    observation=_observation_name(observations[position]),
                )
                for position, node in enumerate(structure.nodes)
            ],
            edges=list(structure.edges),
        )

    def change(self, chart: Optional[MotionStatechart]) -> Optional[ChartSnapshot]:
        """
        The snapshot to send a viewer that already holds the last one: None while the
        chart stands where it was reported to stand.

        :param chart: The statechart the executor is currently ticking, if any.
        """
        snapshot = self.snapshot(chart)
        if snapshot is None:
            return None
        node_states = (
            [node.life_cycle for node in snapshot.nodes],
            [node.observation for node in snapshot.nodes],
        )
        if node_states == self._sent_node_states:
            return None
        self._sent_node_states = node_states
        return snapshot


def _observation_name(observation: float) -> ObservationName:
    """
    Trinary observation value → name (0 false, 0.5 unknown, 1 true).

    :param observation: The raw trinary observation value.
    """
    if observation >= 0.75:
        return ObservationName.TRUE
    if observation <= 0.25:
        return ObservationName.FALSE
    return ObservationName.UNKNOWN


__all__ = ["ChartObserver"]
