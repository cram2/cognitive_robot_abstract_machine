"""
What a motion statechart is made of, as the viewer and a recording read it.

Lives apart from the bridge that publishes it: a recording is written by the onboarder
too, and both have to describe a statechart the same way.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field, replace
from enum import StrEnum

from giskardpy.motion_statechart.motion_statechart import MotionStatechart
from typing_extensions import Dict, List, Optional


@dataclass(frozen=True)
class ChartNodeStructure:
    """
    The structural part of one statechart node: what does not change per tick.
    """

    id: str
    name: str
    class_name: str
    parent: Optional[str]


@dataclass(frozen=True)
class ChartEdgeEntry:
    """
    One transition edge between two statechart nodes.
    """

    source: str
    target: str
    kind: str

    def to_payload(self) -> Dict[str, str]:
        """
        This edge as the wire shape the frontend reads.

        Uses ``from``/``to`` rather than :attr:`source`/:attr:`target`, since ``from``
        is a Python keyword and cannot be a dataclass field name.
        """
        return {"from": self.source, "to": self.target, "kind": self.kind}


@dataclass(frozen=True)
class ChartStructure:
    """
    A statechart's cached structure, rebuilt only when the executor compiles a new one.
    """

    nodes: List[ChartNodeStructure] = field(default_factory=list)
    edges: List[ChartEdgeEntry] = field(default_factory=list)
    node_state_indices: List[int] = field(default_factory=list)
    """
    Each node's index into the chart's life-cycle/observation state vectors.
    """

    signature: str = ""
    """
    Content digest identifying the nodes, hierarchy and transitions.
    """


class ObservationName(StrEnum):
    """
    A statechart node's trinary observation value, by name.
    """

    TRUE = "TRUE"
    FALSE = "FALSE"
    UNKNOWN = "UNKNOWN"


@dataclass(frozen=True)
class ChartNodeEntry:
    """
    One statechart node's structure plus its current life cycle and observation.
    """

    id: str
    name: str
    class_name: str
    parent: Optional[str]
    life_cycle: str
    """
    The node's ``LifeCycleValues`` name (e.g. ``RUNNING``).
    """

    observation: ObservationName
    """
    The node's trinary observation name.
    """


@dataclass(frozen=True)
class ChartSnapshot:
    """
    The motion statechart in the shape the viewer renders.
    """

    signature: str = ""
    title: str = ""
    """
    Name of the action whose motion group this statechart belongs to.
    """

    nodes: List[ChartNodeEntry] = field(default_factory=list)
    edges: List[ChartEdgeEntry] = field(default_factory=list)


def structure_of(chart: MotionStatechart) -> ChartStructure:
    """
    Nodes and transition edges of a statechart.

    :param chart: The statechart to serialize.
    """
    nodes: List[ChartNodeStructure] = []
    node_state_indices: List[int] = []
    for node in chart.nodes:
        parent_index = node.parent_node_index
        nodes.append(
            ChartNodeStructure(
                id="chart_node_%d" % node.index,
                name=node.name,
                class_name=type(node).__name__,
                parent=(
                    ("chart_node_%d" % parent_index)
                    if parent_index is not None
                    else None
                ),
            )
        )
        node_state_indices.append(node.index)
    edges = []
    for source, target, transition in chart.rx_graph.edge_index_map().values():
        edges.append(
            ChartEdgeEntry(
                source="chart_node_%d" % chart.rx_graph.get_node_data(source).index,
                target="chart_node_%d" % chart.rx_graph.get_node_data(target).index,
                kind=transition.kind.name,
            )
        )
    structure = ChartStructure(
        nodes=nodes,
        edges=edges,
        node_state_indices=node_state_indices,
    )
    signature = hashlib.sha256(
        json.dumps(asdict(structure), sort_keys=True).encode("utf-8")
    ).hexdigest()
    return replace(structure, signature=signature)
