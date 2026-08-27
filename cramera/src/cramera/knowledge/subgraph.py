"""
Shared node/edge/detail types and accumulator for building one drill-down/graph-panel
subgraph view.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import asdict, dataclass, field

from typing_extensions import Any, ClassVar, Dict, List, Optional, TYPE_CHECKING

from cramera.knowledge.enums import ColourGroup, EdgeKind
from cramera.payload import CrameraPayload

if TYPE_CHECKING:
    from cramera.knowledge.knowledge_base import EpisodeKnowledgeBase


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

    group: ColourGroup
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

    group: ColourGroup
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

    group: ColourGroup
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
        group: ColourGroup,
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

    def add_edge(self, source: str, target: str, kind: EdgeKind, label: str) -> None:
        """
        Append one edge between two nodes of this subgraph.

        :param source: Id of the edge's source node.
        :param target: Id of the edge's target node.
        :param kind: Rendering kind of the edge.
        :param label: Edge label shown on hover.
        """
        self.edges.append(GraphEdge(source, target, kind, label))

    def add_edge_to_existing(
        self, source: str, target: str, kind: EdgeKind, label: str
    ) -> None:
        """
        Append one edge, but only if the target is a node of this subgraph.

        Lets a caller wire a node into a cluster that may or may not have been built,
        without leaving an edge pointing at nothing.

        :param source: Id of the edge's source node.
        :param target: Id of the edge's target node; the edge is dropped if this node
            is not in the subgraph.
        :param kind: Rendering kind of the edge.
        :param label: Edge label shown on hover.
        """
        if any(node.id == target for node in self.nodes):
            self.add_edge(source, target, kind, label)


# %% what a graph-panel tab sends to the frontend
@dataclass(kw_only=True)
class GraphPanelPayload(CrameraPayload):
    """
    One tab or drill-down of the graph panel, in the shape the frontend reads.

    Every view sends the same four keys; :meth:`panel_options` adds the ones only that
    view knows about, so no subclass repeats the serialization of nodes and edges.
    """

    nodes: List[GraphNode] = field(default_factory=list)
    """
    Every node in this view.
    """

    edges: List[GraphEdge] = field(default_factory=list)
    """
    Every edge in this view.
    """

    details: Dict[str, DetailEntry] = field(default_factory=dict)
    """
    Detail-panel entry per node id.
    """

    TAB: ClassVar[Optional[str]] = None
    """
    Name of the graph-panel tab this view serves, or None for a view that is only ever
    reached by drilling into a node.
    """

    @classmethod
    def of_tab(cls, knowledge_base: EpisodeKnowledgeBase) -> GraphPanelPayload:
        """
        Build this view as a whole graph-panel tab.

        :param knowledge_base: The recorded episode the view is built from.
        :raises NotImplementedError: For a drill-down-only view, which has no tab of its
            own and is built from the node that was double-clicked instead.
        """
        raise NotImplementedError("%s serves no graph-panel tab" % cls.__name__)

    @abstractmethod
    def panel_options(self) -> Dict[str, Any]:
        """
        The payload keys only this view sends, such as its breadcrumb or legend.
        """

    def to_payload(self) -> Dict[str, Any]:
        """
        The JSON-serializable shape the frontend's graph panel expects.
        """
        return {
            "ok": self.ok,
            "nodes": [node.to_payload() for node in self.nodes],
            "edges": [edge.to_payload() for edge in self.edges],
            "details": {
                node_id: asdict(entry) for node_id, entry in self.details.items()
            },
            **self.panel_options(),
        }
