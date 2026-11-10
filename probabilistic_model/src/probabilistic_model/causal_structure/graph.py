"""
A causal graph over the variables of a fitted model: which variables cause which, as far
as the data settles it, and the directed acyclic graphs it stands for.
"""

from __future__ import annotations

import itertools
from dataclasses import dataclass, replace
from enum import StrEnum
from functools import cached_property

from krrood.adapters.json_serializer import SubclassJSONSerializer, from_json, to_json
from random_events.variable import Variable
from typing_extensions import Any, Iterable, Self

from probabilistic_model.graph import DirectedAcyclicGraph, Graph
from probabilistic_model.causal_structure.exceptions import (
    CyclicGraph,
    EquivalenceClassTooLarge,
    InconsistentGraph,
    NotAPartAttribute,
    UnknownGraphNode,
)

# %% nodes and edges


class EdgeMark(StrEnum):
    """
    What an edge shows at one of its ends.
    """

    TAIL = "tail"
    """
    The node at this end is not caused by the node at the other.
    """

    ARROW = "arrow"
    """
    The node at this end is caused by the node at the other.
    """


class NodeLevel(StrEnum):
    """
    What a node of a relational model's graph stands for.
    """

    EXAMPLE_ATTRIBUTE = "example_attribute"
    """
    An attribute of the modelled object itself.
    """

    AGGREGATION_STATISTIC = "aggregation_statistic"
    """
    A statistic over the parts of one of its collections, such as how many there are.
    """

    PART_ATTRIBUTE = "part_attribute"
    """
    An attribute shared by every part of one of its collections.
    """


@dataclass(frozen=True)
class GraphNode:
    """
    A node of a causal graph, standing for one variable.
    """

    variable: Variable
    """
    The variable, named as a query's parameters name it.
    """

    level: NodeLevel
    """
    What the variable stands for.
    """

    collection: str | None = None
    """
    The name of the collection whose parts the variable is an attribute of, for a
    :attr:`NodeLevel.PART_ATTRIBUTE` node.
    """

    @property
    def name(self) -> str:
        """
        :return: The name of the node's variable.
        """
        return self.variable.name

    def __lt__(self, other: GraphNode) -> bool:
        """
        Order nodes by name, wherever an order is needed.
        """
        return self.name < other.name

    def variable_of_part(self, index: int) -> Variable:
        """
        :param index: The position of one part in the collection.
        :return: The variable of that part's copy of the attribute, named
            ``Class.collection[index].attribute`` after the node's
            ``Class.collection.attribute``.
        :raises NotAPartAttribute: If the node is not an attribute of a collection's
            parts.
        """
        if self.level is not NodeLevel.PART_ATTRIBUTE:
            raise NotAPartAttribute(self)
        return replace(
            self.variable,
            name=f"{self._part_variable_name_start}{index}{self._part_variable_name_end}",
        )

    def part_index_of(self, variable_name: str) -> int | None:
        """
        :param variable_name: The name of a query's variable.
        :return: The position of the part whose copy of the attribute the variable is,
            the inverse of :meth:`variable_of_part`; ``None`` if it is no such copy.
        """
        if self.level is not NodeLevel.PART_ATTRIBUTE:
            return None
        start, end = self._part_variable_name_start, self._part_variable_name_end
        if not (variable_name.startswith(start) and variable_name.endswith(end)):
            return None
        index = variable_name[len(start) : len(variable_name) - len(end)]
        return int(index) if index.isdigit() else None

    @property
    def _part_variable_name_start(self) -> str:
        """
        :return: What a part's copy of the attribute is named before the part's
            position, ``Class.collection[``.
        """
        return self.name.split(f".{self.collection}.", 1)[0] + f".{self.collection}["

    @property
    def _part_variable_name_end(self) -> str:
        """
        :return: What a part's copy of the attribute is named after the part's
            position, ``].attribute``.
        """
        return "]." + self.name.split(f".{self.collection}.", 1)[1]


@dataclass(frozen=True)
class GraphEdge:
    """
    An edge between two nodes, with what it shows at each end.
    """

    first: GraphNode
    """
    The node at one end.
    """

    second: GraphNode
    """
    The node at the other end.
    """

    mark_at_first: EdgeMark
    """
    What the edge shows at :attr:`first`.
    """

    mark_at_second: EdgeMark
    """
    What the edge shows at :attr:`second`.
    """

    @classmethod
    def directed_edge(cls, parent: GraphNode, child: GraphNode) -> GraphEdge:
        """
        :param parent: The cause.
        :param child: What it causes.
        :return: The edge ``parent → child``.
        """
        return cls(parent, child, EdgeMark.TAIL, EdgeMark.ARROW)

    @classmethod
    def undirected_edge(cls, first: GraphNode, second: GraphNode) -> GraphEdge:
        """
        :param first: The node at one end.
        :param second: The node at the other end.
        :return: The edge ``first — second``, whose direction the data leaves open.
        """
        return cls(first, second, EdgeMark.TAIL, EdgeMark.TAIL)

    @property
    def is_directed(self) -> bool:
        """
        :return: Whether the edge shows a tail at one end and an arrow at the other.
        """
        return {self.mark_at_first, self.mark_at_second} == {
            EdgeMark.TAIL,
            EdgeMark.ARROW,
        }

    @property
    def is_undirected(self) -> bool:
        """
        :return: Whether the edge shows a tail at both ends.
        """
        return self.mark_at_first is self.mark_at_second is EdgeMark.TAIL

    @property
    def parent(self) -> GraphNode:
        """
        :return: The node at the tail of a directed edge.
        """
        return self.first if self.mark_at_first is EdgeMark.TAIL else self.second

    @property
    def child(self) -> GraphNode:
        """
        :return: The node at the arrow of a directed edge.
        """
        return self.second if self.mark_at_first is EdgeMark.TAIL else self.first


# %% the directed acyclic graphs a causal graph stands for


@dataclass(frozen=True)
class ColliderPatternInCausalGraph:
    """
    Two nodes that both cause a third: ``first_parent → child ← second_parent``.
    """

    first_parent: GraphNode
    """
    The parent whose name sorts first.
    """

    child: GraphNode
    """
    The node both parents cause.
    """

    second_parent: GraphNode
    """
    The other parent.
    """


@dataclass(eq=False)
class MemberGraph(DirectedAcyclicGraph[GraphNode]):
    """
    One of the directed acyclic graphs a causal graph stands for, over its nodes.
    """

    def copy(self) -> MemberGraph:
        """
        :return: A graph with the same nodes and edges, changed independently of this
            one.
        """
        return MemberGraph(graph=self.graph.copy(), indices=dict(self.indices))

    def without_edges(
        self, edges: Iterable[tuple[GraphNode, GraphNode]]
    ) -> MemberGraph:
        """
        :param edges: Edges of the graph, each as its parent and its child.
        :return: A copy of the graph without them.
        """
        result = self.copy()
        for parent, child in edges:
            result.graph.remove_edge(self.index_of(parent), self.index_of(child))
        return result


@dataclass(frozen=True)
class NodeReference:
    """
    A graph node as a query's variable refers to it: a part attribute's template node
    together with the position of the part the variable is a copy for.
    """

    node: GraphNode
    """
    The node.
    """

    part_index: int | None = None
    """
    The position of the part, for a part attribute's node; ``None`` otherwise.
    """

    @property
    def variable(self) -> Variable:
        """
        :return: The variable the reference stands for.
        """
        if self.part_index is None:
            return self.node.variable
        return self.node.variable_of_part(self.part_index)


# %% the graph


class CausalGraphJSONKey(StrEnum):
    """
    The keys of a causal graph's JSON dict.
    """

    NODES = "nodes"
    """
    The nodes, in the graph's order.
    """

    EDGES = "edges"
    """
    The edges, in the graph's order.
    """

    MAXIMUM_MEMBER_GRAPHS = "maximum_member_graphs"
    """
    The graph's :attr:`CausalGraph.maximum_member_graphs`.
    """


@dataclass(eq=False)
class CausalGraph(Graph[GraphNode], SubclassJSONSerializer):
    """
    A causal graph: a directed acyclic graph, or the equivalence class of those the data
    cannot tell apart, whose undirected edges any of them may orient either way.
    """

    maximum_member_graphs: int = 1000
    """
    How many directed acyclic graphs the graph may stand for and still have them
    listed.
    """

    @classmethod
    def from_edges(
        cls, nodes: Iterable[GraphNode], edges: Iterable[GraphEdge]
    ) -> CausalGraph:
        """
        :param nodes: The nodes, one per variable.
        :param edges: The edges between them, at most one between two nodes.
        :return: The graph.
        :raises UnknownGraphNode: If an edge joins a node that is not among the nodes.
        """
        graph = cls()
        graph.add_nodes_from(nodes)
        for edge in edges:
            graph.add_edge(edge)
        return graph

    def add_edge(self, edge: GraphEdge) -> None:
        """
        :param edge: An edge between two of the graph's nodes.
        :raises UnknownGraphNode: If the graph lacks one of them.
        """
        for node in (edge.first, edge.second):
            if node not in self.indices:
                raise UnknownGraphNode(node.name, self.node_names)
        self.graph.add_edge(self.index_of(edge.first), self.index_of(edge.second), edge)

    def edges(self) -> list[GraphEdge]:
        """
        :return: The edges, in the order they were added.
        """
        return self.graph.edges()

    def node(self, name: str) -> GraphNode:
        """
        :param name: The name of a node's variable.
        :return: The node.
        :raises UnknownGraphNode: If the graph has no such node.
        """
        for node in self.nodes():
            if node.name == name:
                return node
        raise UnknownGraphNode(name, self.node_names)

    def reference_of(self, variable_name: str) -> NodeReference | None:
        """
        :param variable_name: The name of a query's variable.
        :return: The node the variable stands for, with the part's position for one
            part's copy of a part attribute; ``None`` if the graph has no node for it.
        """
        for node in self.nodes():
            if node.name == variable_name:
                return NodeReference(node)
            part_index = node.part_index_of(variable_name)
            if part_index is not None:
                return NodeReference(node, part_index)
        return None

    @property
    def node_names(self) -> list[str]:
        """
        :return: The names of the nodes, in the graph's order.
        """
        return [node.name for node in self.nodes()]

    def to_json(self, **kwargs) -> dict[str, Any]:
        return {
            **super().to_json(**kwargs),
            CausalGraphJSONKey.NODES: to_json(self.nodes(), **kwargs),
            CausalGraphJSONKey.EDGES: to_json(self.edges(), **kwargs),
            CausalGraphJSONKey.MAXIMUM_MEMBER_GRAPHS: self.maximum_member_graphs,
        }

    @classmethod
    def _from_json(cls, data: dict[str, Any], **kwargs) -> Self:
        graph = cls.from_edges(
            from_json(data[CausalGraphJSONKey.NODES], **kwargs),
            from_json(data[CausalGraphJSONKey.EDGES], **kwargs),
        )
        graph.maximum_member_graphs = data[CausalGraphJSONKey.MAXIMUM_MEMBER_GRAPHS]
        return graph

    def edge_between(self, first: GraphNode, second: GraphNode) -> GraphEdge | None:
        """
        :param first: A node.
        :param second: Another node.
        :return: The edge between them, or ``None`` if they are not adjacent.
        """
        for edge in self.edges():
            if {edge.first, edge.second} == {first, second}:
                return edge
        return None

    def possible_ancestors(self, node: GraphNode) -> set[GraphNode]:
        """
        :param node: A node of the graph.
        :return: The nodes with a directed path to it in at least one member graph.
        """
        return {
            ancestor
            for member in self.member_graphs()
            for ancestor in member.ancestors(node)
        }

    def adjacent(self, first: GraphNode, second: GraphNode) -> bool:
        """
        :param first: A node.
        :param second: Another node.
        :return: Whether an edge joins them.
        """
        return self.edge_between(first, second) is not None

    def unshielded_colliders(self) -> set[ColliderPatternInCausalGraph]:
        """
        :return: Every collider among the directed edges whose two parents are not
            adjacent.
        """
        directed = MemberGraph()
        directed.add_nodes_from(self.nodes())
        directed.add_edges_from(
            (edge.parent, edge.child) for edge in self.edges() if edge.is_directed
        )
        return self.colliders_of(directed)

    def colliders_of(self, directed: MemberGraph) -> set[ColliderPatternInCausalGraph]:
        """
        :param directed: Directed edges over the graph's nodes.
        :return: Every collider among the directed edges whose two parents are not
            :meth:`adjacent` in this graph.
        """
        return {
            ColliderPatternInCausalGraph(first_parent, child, second_parent)
            for child in directed.nodes()
            for first_parent, second_parent in itertools.combinations(
                sorted(directed.predecessors(child)), 2
            )
            if not self.adjacent(first_parent, second_parent)
        }

    def member_graphs(self) -> list[MemberGraph]:
        """
        :return: Every directed acyclic graph the graph stands for: each orientation of
            the undirected edges that is acyclic and has exactly the graph's unshielded
            colliders.
        :raises EquivalenceClassTooLarge: If the graph stands for more than
            :attr:`maximum_member_graphs` directed acyclic graphs.
        :raises CyclicGraph: If the graph has no undirected edges and a directed cycle.
        :raises InconsistentGraph: If no orientation qualifies.
        """
        if not self.has_member_graphs:
            raise InconsistentGraph(
                len([edge for edge in self.edges() if edge.is_undirected])
            )
        return self._member_graphs

    @property
    def has_member_graphs(self) -> bool:
        """
        :return: Whether some orientation of the undirected edges qualifies as a member
            graph, as :meth:`member_graphs` describes them. A graph learned from a
            finite sample may have none.
        :raises EquivalenceClassTooLarge: If the graph stands for more than
            :attr:`maximum_member_graphs` directed acyclic graphs.
        :raises CyclicGraph: If the graph has no undirected edges and a directed cycle.
        """
        return bool(self._member_graphs)

    @cached_property
    def _member_graphs(self) -> list[MemberGraph]:
        """
        :return: The member graphs, as :meth:`member_graphs` describes them, listed
            once; none if no orientation qualifies.
        """
        undirected = self.undirected_edges_in_connected_order()
        directed = MemberGraph()
        directed.add_nodes_from(self.nodes())
        directed.add_edges_from(
            (edge.parent, edge.child) for edge in self.edges() if edge.is_directed
        )
        if not undirected:
            if not directed.is_acyclic:
                raise CyclicGraph(directed.cycle())
            return [directed]
        colliders = self.unshielded_colliders()
        members = []
        partial_orientations = [directed]
        while partial_orientations:
            partial = partial_orientations.pop()
            oriented_count = len(partial.edges()) - len(directed.edges())
            if oriented_count == len(undirected):
                if len(members) == self.maximum_member_graphs:
                    raise EquivalenceClassTooLarge(self.maximum_member_graphs)
                members.append(partial)
                continue
            edge = undirected[oriented_count]
            for parent, child in ((edge.second, edge.first), (edge.first, edge.second)):
                extended = partial.copy()
                extended.add_edge(parent, child)
                if extended.is_acyclic and self.colliders_of(extended) == colliders:
                    partial_orientations.append(extended)
        return members

    def undirected_edges_in_connected_order(self) -> list[GraphEdge]:
        """
        :return: The undirected edges, each sharing a node with an earlier one where
            any of the rest does.
        """
        remaining = [edge for edge in self.edges() if edge.is_undirected]
        ordered, reached = [], set()
        while remaining:
            edge = next(
                (edge for edge in remaining if {edge.first, edge.second} & reached),
                remaining[0],
            )
            remaining.remove(edge)
            ordered.append(edge)
            reached |= {edge.first, edge.second}
        return ordered
