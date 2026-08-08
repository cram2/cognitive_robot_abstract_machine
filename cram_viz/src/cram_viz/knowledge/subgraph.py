"""
Shared node/edge/detail types and accumulator for building one drill-down/graph-panel
subgraph view.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from typing_extensions import Any, Dict, List, Optional

from cram_viz.knowledge.enums import EdgeKind, NodeGroup


@dataclass
class GraphNode:
    """
    One node of a graph-panel subgraph.
    """

    id: str
    """
    Unique node id within its subgraph.
    """

    label: str
    """
    Display label.
    """

    group: NodeGroup
    """
    Colour group the frontend renders this node with.
    """

    title: str
    """
    Tooltip text (label plus its detail lines, newline-joined).
    """

    status: Optional[str] = None
    """
    Live execution status; only the plan view sets this.
    """

    def to_payload(self) -> Dict[str, Any]:
        """
        The JSON-serializable shape the frontend's graph panel expects.
        """
        payload = {
            "id": self.id,
            "label": self.label,
            "group": self.group,
            "title": self.title,
        }
        if self.status is not None:
            payload["status"] = self.status
        return payload


@dataclass
class DetailEntry:
    """
    The detail-panel entry for one graph node.
    """

    label: str
    """
    Display label, mirrors the node's own label.
    """

    group: NodeGroup
    """
    Colour group, mirrors the node's own group.
    """

    lines: List[str]
    """
    Tooltip/detail-panel lines describing the node.
    """


@dataclass
class GraphEdge:
    """
    One directed edge of a graph-panel subgraph.
    """

    source: str
    """
    Id of the edge's source node.
    """

    target: str
    """
    Id of the edge's target node.
    """

    kind: EdgeKind
    """
    Rendering kind (solid ``prop`` or dashed ``type``).
    """

    label: str
    """
    Edge label shown on hover.
    """

    def to_payload(self) -> Dict[str, str]:
        """
        The JSON-serializable shape the frontend's graph panel expects.

        ``source``/``target`` map to the wire keys ``from``/``to`` — ``from`` is a
        Python keyword and cannot be a dataclass field name.
        """
        return {
            "from": self.source,
            "to": self.target,
            "kind": self.kind,
            "label": self.label,
        }


@dataclass
class LegendEntry:
    """
    One row of a graph panel's colour legend.
    """

    group: NodeGroup
    """
    Node colour group this row explains.
    """

    label: str
    """
    Human-readable name shown next to the group's colour.
    """


@dataclass
class SubgraphAccumulator:
    """
    Accumulates the nodes/edges/details of one drill-down/graph-panel subgraph.
    """

    nodes: List[GraphNode] = field(default_factory=list)
    """
    Every node added to this subgraph so far.
    """

    edges: List[GraphEdge] = field(default_factory=list)
    """
    Every edge added to this subgraph so far.
    """

    details: Dict[str, DetailEntry] = field(default_factory=dict)
    """
    Detail-panel entry per node id.
    """

    def add(
        self,
        node_id: str,
        label: str,
        group: NodeGroup,
        lines: List[str],
        status: Optional[str] = None,
    ) -> None:
        """
        Append one graph node and its detail-panel entry.

        :param node_id: Id of the node to add.
        :param label: Display label of the node, also used as the detail entry's label.
        :param group: Colour group the node and its detail entry belong to.
        :param lines: Detail-panel lines shown under the node's label.
        :param status: Status colouring for the node, if any.
        """
        title = "\n".join([label] + lines)
        self.nodes.append(GraphNode(node_id, label, group, title, status=status))
        self.details[node_id] = DetailEntry(label, group, lines)
