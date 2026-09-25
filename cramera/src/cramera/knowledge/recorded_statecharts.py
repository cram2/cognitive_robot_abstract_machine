"""
The motion statecharts a live run went through, recorded so a replay can show them.

A statechart only exists while giskardpy executes it — one is compiled per merged
motion group and thrown away afterwards. :class:`RecordedStatecharts` keeps what was
seen: each compiled structure once, the distinct node-state moments the run passed
through, and which moment every recorded tick was in. Written beside the trajectory of
a recording bundle (see :mod:`cramera.live.recording_bundle`) and read back for the
graph panel's statechart tab.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from typing_extensions import (
    Any,
    Dict,
    Iterable,
    List,
    Optional,
    TYPE_CHECKING,
)

from cramera.generated_json import GeneratedJson
from cramera.knowledge.scene_bundle import SceneBundle

if TYPE_CHECKING:
    from cramera.live.bridge import ChartSnapshot

STATECHART_FILE = "statecharts.json"
"""
Name the recorded statecharts are written under inside a scene bundle.
"""

NO_STATECHART = -1
"""
Moment index of a recorded tick during which no statechart was executing.
"""


@dataclass(frozen=True)
class RecordedChartNode:
    """
    One statechart node's structure, which does not change while the chart runs.
    """

    id: str
    """
    The node's id, as the live chart addresses it.
    """

    name: str
    """
    The node's own name.
    """

    class_name: str
    """
    Name of the giskardpy class the node was compiled from.
    """

    parent: Optional[str]
    """
    Id of the node containing this one, or None at the top level.
    """

    def to_payload(self) -> Dict[str, Any]:
        """
        This node in the shape the viewer reads.
        """
        return {
            "id": self.id,
            "name": self.name,
            "class_name": self.class_name,
            "parent": self.parent,
        }

    @classmethod
    def of_payload(cls, payload: Dict[str, Any]) -> RecordedChartNode:
        """
        One node read back from a recorded bundle.

        :param payload: The node as :meth:`to_payload` wrote it.
        """
        return cls(
            id=payload["id"],
            name=payload["name"],
            class_name=payload["class_name"],
            parent=payload["parent"],
        )


@dataclass(frozen=True)
class RecordedChartEdge:
    """
    One transition between two statechart nodes.
    """

    source: str
    """
    Id of the node the transition leaves.
    """

    target: str
    """
    Id of the node the transition enters.
    """

    kind: str
    """
    The transition's kind, as giskardpy names it.
    """

    def to_payload(self) -> Dict[str, str]:
        """
        This edge in the shape the viewer reads.

        Uses ``from``/``to`` rather than :attr:`source`/:attr:`target`, since ``from``
        is a Python keyword and cannot be a dataclass field name.
        """
        return {"from": self.source, "to": self.target, "kind": self.kind}

    @classmethod
    def of_payload(cls, payload: Dict[str, str]) -> RecordedChartEdge:
        """
        One edge read back from a recorded bundle.

        :param payload: The edge as :meth:`to_payload` wrote it.
        """
        return cls(source=payload["from"], target=payload["to"], kind=payload["kind"])


@dataclass(frozen=True)
class RecordedChart:
    """
    One compiled statechart, recorded once however many ticks ran on it.
    """

    signature: str
    """
    The structure's signature, stable while the executor keeps ticking this chart.
    """

    title: str
    """
    Name of the action whose motion group this statechart belongs to.
    """

    nodes: List[RecordedChartNode] = field(default_factory=list)
    """
    The chart's nodes, in the order its state vectors are indexed.
    """

    edges: List[RecordedChartEdge] = field(default_factory=list)
    """
    The transitions between those nodes.
    """

    @classmethod
    def of_snapshot(cls, snapshot: ChartSnapshot) -> RecordedChart:
        """
        The structural part of one live statechart snapshot.

        :param snapshot: The snapshot the bridge published for a tick.
        """
        return cls(
            signature=snapshot.signature,
            title=snapshot.title,
            nodes=[
                RecordedChartNode(
                    id=entry.id,
                    name=entry.name,
                    class_name=entry.class_name,
                    parent=entry.parent,
                )
                for entry in snapshot.nodes
            ],
            edges=[
                RecordedChartEdge(
                    source=edge.source, target=edge.target, kind=edge.kind
                )
                for edge in snapshot.edges
            ],
        )

    def to_payload(self) -> Dict[str, Any]:
        """
        This chart in the shape the viewer reads.
        """
        return {
            "signature": self.signature,
            "title": self.title,
            "nodes": [node.to_payload() for node in self.nodes],
            "edges": [edge.to_payload() for edge in self.edges],
        }

    @classmethod
    def of_payload(cls, payload: Dict[str, Any]) -> RecordedChart:
        """
        One chart read back from a recorded bundle.

        :param payload: The chart as :meth:`to_payload` wrote it.
        """
        return cls(
            signature=payload["signature"],
            title=payload["title"],
            nodes=[RecordedChartNode.of_payload(node) for node in payload["nodes"]],
            edges=[RecordedChartEdge.of_payload(edge) for edge in payload["edges"]],
        )


@dataclass(frozen=True)
class StatechartMoment:
    """
    The node states one or more recorded ticks found a statechart in.
    """

    chart: int
    """
    Index into :attr:`RecordedStatecharts.charts` of the chart this moment belongs to.
    """

    life_cycles: List[str] = field(default_factory=list)
    """
    Each node's ``LifeCycleValues`` name, ordered like the chart's nodes.
    """

    observations: List[str] = field(default_factory=list)
    """
    Each node's trinary observation name, ordered like the chart's nodes.
    """

    @classmethod
    def of_snapshot(cls, chart: int, snapshot: ChartSnapshot) -> StatechartMoment:
        """
        The states one live snapshot reported.

        :param chart: Index of the chart the snapshot belongs to.
        :param snapshot: The snapshot the bridge published for a tick.
        """
        return cls(
            chart=chart,
            life_cycles=[entry.life_cycle for entry in snapshot.nodes],
            observations=[str(entry.observation) for entry in snapshot.nodes],
        )

    def to_payload(self) -> Dict[str, Any]:
        """
        This moment in the shape the viewer reads.
        """
        return {
            "chart": self.chart,
            "lifeCycles": self.life_cycles,
            "observations": self.observations,
        }

    @classmethod
    def of_payload(cls, payload: Dict[str, Any]) -> StatechartMoment:
        """
        One moment read back from a recorded bundle.

        :param payload: The moment as :meth:`to_payload` wrote it.
        """
        return cls(
            chart=payload["chart"],
            life_cycles=list(payload["lifeCycles"]),
            observations=list(payload["observations"]),
        )

    def identity(self) -> Any:
        """
        A hashable key identifying this moment, so a run that keeps returning to the
        same states records them once.
        """
        return (self.chart, tuple(self.life_cycles), tuple(self.observations))


@dataclass(frozen=True)
class RecordedStatecharts:
    """
    Every statechart a recording went through, and where in it each tick was.
    """

    charts: List[RecordedChart] = field(default_factory=list)
    """
    Each compiled structure the run ticked, in the order it was first seen.
    """

    moments: List[StatechartMoment] = field(default_factory=list)
    """
    The distinct node-state moments the run passed through.
    """

    moment_of_frame: List[int] = field(default_factory=list)
    """
    Index into :attr:`moments` per recorded tick, or :data:`NO_STATECHART` for a tick
    that had no statechart executing.
    """

    @classmethod
    def of_snapshots(
        cls, snapshots: Iterable[Optional[ChartSnapshot]]
    ) -> RecordedStatecharts:
        """
        The statecharts a run's ticks were in, structures and moments deduplicated.

        :param snapshots: The statechart snapshot of each recorded tick, in order, None
            for a tick with no statechart executing.
        """
        charts: List[RecordedChart] = []
        chart_of_structure: Dict[Any, int] = {}
        moments: List[StatechartMoment] = []
        moment_of_identity: Dict[Any, int] = {}
        moment_of_frame: List[int] = []
        for snapshot in snapshots:
            if snapshot is None or not snapshot.nodes:
                moment_of_frame.append(NO_STATECHART)
                continue
            structure = RecordedChart.of_snapshot(snapshot)
            key = (structure.signature, structure.title)
            if key not in chart_of_structure:
                chart_of_structure[key] = len(charts)
                charts.append(structure)
            moment = StatechartMoment.of_snapshot(chart_of_structure[key], snapshot)
            if moment.identity() not in moment_of_identity:
                moment_of_identity[moment.identity()] = len(moments)
                moments.append(moment)
            moment_of_frame.append(moment_of_identity[moment.identity()])
        return cls(charts=charts, moments=moments, moment_of_frame=moment_of_frame)

    def is_empty(self) -> bool:
        """
        Whether no tick of this recording had a statechart executing.
        """
        return not self.moments

    def to_payload(self) -> Dict[str, Any]:
        """
        The recorded statecharts in the shape the viewer reads.
        """
        return {
            "charts": [chart.to_payload() for chart in self.charts],
            "moments": [moment.to_payload() for moment in self.moments],
            "frames": list(self.moment_of_frame),
        }

    @classmethod
    def of_payload(cls, payload: Any) -> RecordedStatecharts:
        """
        The statecharts recorded in a bundle, or none from anything unreadable.

        :param payload: The recorded statecharts as :meth:`to_payload` wrote them.
        """
        if not isinstance(payload, dict):
            return cls()
        return cls(
            charts=[RecordedChart.of_payload(chart) for chart in payload["charts"]],
            moments=[
                StatechartMoment.of_payload(moment) for moment in payload["moments"]
            ],
            moment_of_frame=list(payload["frames"]),
        )

    @classmethod
    def of_scene(cls, scene: Optional[str] = None) -> RecordedStatecharts:
        """
        The statecharts recorded beside one scene's trajectory, or none.

        :param scene: Name of the scene to read, or None for the active one.
        """
        directory = SceneBundle.directory_of(scene)
        if not directory or not (directory / STATECHART_FILE).is_file():
            return cls()
        return cls.of_payload(GeneratedJson(directory / STATECHART_FILE).read())
