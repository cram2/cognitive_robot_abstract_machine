from __future__ import annotations

from dataclasses import dataclass, field
from typing_extensions import ClassVar, Optional, List, Any

try:
    from rustworkx_utils import GraphVisualizer
except ImportError:
    GraphVisualizer = None

import rustworkx as rx


@dataclass
class ColorLegend:
    name: str = field(default="Other")
    color: str = field(default="white")


# ---- rustworkx-backed node wrapper to mimic needed anytree.Node API ----
@dataclass
class RWXNode:
    name: str
    weight: str = field(default='')
    data: Optional[Any] = field(default=None)
    _primary_parent_id: Optional[int] = None
    color: ColorLegend = field(default_factory=ColorLegend)
    # Grouping/boxing options
    wrap_subtree: bool = field(default=False)
    wrap_facecolor: Optional[str] = field(default=None)
    wrap_edgecolor: Optional[str] = field(default=None)
    wrap_alpha: float = field(default=0.08)
    # Visual emphasis options
    enclosed: bool = field(default=False)
    id_: int = field(init=False)
    _graph: ClassVar[rx.PyDAG] = rx.PyDAG()
    enclosed_name: ClassVar[str] = 'enclosed'

    def __post_init__(self):
        # store self as node data to keep a 1:1 mapping
        self.id: int = self._graph.add_node(self)

    # Non-primary connect: add edge without changing primary parent pointer
    def add_parent(self, parent: "RWXNode", edge_weight=None):
        # Avoid self-loops
        if parent is self:
            return
        # Do not add duplicate edges between the same two nodes
        if self._graph.has_edge(parent.id, self.id):
            return
        # Avoid creating cycles: PyDAG will raise if creates a cycle
        self._graph.add_edge(parent.id, self.id, edge_weight if edge_weight is not None else self.weight)

    def remove(self):
        self._graph.remove_node(self.id)

    def remove_node(self, node: RWXNode):
        self._graph.remove_node(node.id)

    def remove_child(self, child: RWXNode):
        child.remove_parent(self)

    def remove_parent(self, parent: RWXNode):
        self._graph.remove_edge(parent.id, self.id)

    @property
    def ancestors(self) -> List[RWXNode]:
        node_ids = rx.ancestors(self._graph, self.id)
        return [self._graph[n_id] for n_id in node_ids]

    @property
    def parents(self) -> List[RWXNode]:
        # In this environment rustworkx returns node data objects directly
        return self._graph.predecessors(self.id)

    @property
    def parent(self) -> Optional["RWXNode"]:
        if self._primary_parent_id is None:
            return None
        return self._graph[self._primary_parent_id]

    @parent.setter
    def parent(self, value: Optional["RWXNode"]):
        if value is None:
            # detach current parent
            self._graph.remove_edge(self._primary_parent_id, self.id)
            self._primary_parent_id = None
            return
        # Create edge and set as primary (no need to detach non-primary edges)
        self.add_parent(value)
        self._primary_parent_id = value.id

    @property
    def children(self) -> List["RWXNode"]:
        # In this environment rustworkx returns node data objects directly
        return self._graph.successors(self.id)

    @property
    def descendants(self) -> List["RWXNode"]:
        desc_ids = rx.descendants(self._graph, self.id)
        return [self._graph[nid] for nid in desc_ids]

    @property
    def leaves(self) -> List["RWXNode"]:
        return [n for n in [self] + self.descendants if self._graph.out_degree(n.id) == 0]

    @property
    def root(self) -> "RWXNode":
        n = self
        while n.parent is not None:
            n = n.parent
        return n

    def __str__(self):
        return self.name

    def visualize(self, figsize=(35, 30), node_size=7000, font_size=25, spacing_x: float = 4, spacing_y: float = 4,
                  curve_scale: float = 0.5, layout: str = 'tidy', edge_style: str = 'orthogonal',
                  label_max_chars_per_line: Optional[int] = 13):
        """Render a rooted, top-to-bottom directed graph.
        Delegates to a dedicated visualizer class to keep this method small and reusable.
        """
        if not GraphVisualizer:
            raise RuntimeError("rustworkx_utils is not installed. Please install it with `pip install rustworkx_utils`")
        visualizer = GraphVisualizer(
            node=self,
            figsize=figsize,
            node_size=node_size,
            font_size=font_size,
            spacing_x=spacing_x,
            spacing_y=spacing_y,
            curve_scale=curve_scale,
            layout=layout,
            edge_style=edge_style,
            label_max_chars_per_line=label_max_chars_per_line,
        )
        return visualizer.render()
