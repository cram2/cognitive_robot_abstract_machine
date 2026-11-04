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
        owner, attribute = self.name.split(f".{self.collection}.", 1)
        return replace(
            self.variable, name=f"{owner}.{self.collection}[{index}].{attribute}"
        )


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
    def directed(cls, parent: GraphNode, child: GraphNode) -> GraphEdge:
        """
        :param parent: The cause.
        :param child: What it causes.
        :return: The edge ``parent → child``.
        """
        return cls(parent, child, EdgeMark.TAIL, EdgeMark.ARROW)

    @classmethod
    def undirected(cls, first: GraphNode, second: GraphNode) -> GraphEdge:
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
class Collider:
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

    MAXIMUM_UNDIRECTED_EDGES = "maximum_undirected_edges"
    """
    The graph's :attr:`CausalGraph.maximum_undirected_edges`.
    """


@dataclass(eq=False)
class CausalGraph(Graph[GraphNode], SubclassJSONSerializer):
    """
    A causal graph: a directed acyclic graph, or the equivalence class of those the data
    cannot tell apart, whose undirected edges any of them may orient either way.
    """

    maximum_undirected_edges: int = 12
    """
    How many undirected edges the graph may have for the directed acyclic graphs it
    stands for to be listed, each undirected edge doubling their candidates.
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
            CausalGraphJSONKey.MAXIMUM_UNDIRECTED_EDGES: self.maximum_undirected_edges,
        }

    @classmethod
    def _from_json(cls, data: dict[str, Any], **kwargs) -> Self:
        graph = cls.from_edges(
            from_json(data[CausalGraphJSONKey.NODES], **kwargs),
            from_json(data[CausalGraphJSONKey.EDGES], **kwargs),
        )
        graph.maximum_undirected_edges = data[
            CausalGraphJSONKey.MAXIMUM_UNDIRECTED_EDGES
        ]
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

    def adjacent(self, first: GraphNode, second: GraphNode) -> bool:
        """
        :param first: A node.
        :param second: Another node.
        :return: Whether an edge joins them.
        """
        return self.edge_between(first, second) is not None

    def unshielded_colliders(self) -> set[Collider]:
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

    def colliders_of(self, directed: MemberGraph) -> set[Collider]:
        """
        :param directed: Directed edges over the graph's nodes.
        :return: Every collider among the directed edges whose two parents are not
            :meth:`adjacent` in this graph.
        """
        return {
            Collider(first_parent, child, second_parent)
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
        :raises EquivalenceClassTooLarge: If the graph has more than
            :attr:`maximum_undirected_edges` undirected edges.
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
        :raises EquivalenceClassTooLarge: If the graph has more than
            :attr:`maximum_undirected_edges` undirected edges.
        :raises CyclicGraph: If the graph has no undirected edges and a directed cycle.
        """
        return bool(self._member_graphs)

    @cached_property
    def _member_graphs(self) -> list[MemberGraph]:
        """
        :return: The member graphs, as :meth:`member_graphs` describes them, listed
            once; none if no orientation qualifies.
        """
        undirected = [edge for edge in self.edges() if edge.is_undirected]
        if len(undirected) > self.maximum_undirected_edges:
            raise EquivalenceClassTooLarge(
                len(undirected), self.maximum_undirected_edges
            )
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
        for reversals in itertools.product((False, True), repeat=len(undirected)):
            member = directed.copy()
            member.add_edges_from(
                (edge.second, edge.first) if reverse else (edge.first, edge.second)
                for edge, reverse in zip(undirected, reversals)
            )
            if member.is_acyclic and (self.colliders_of(member) == colliders):
                members.append(member)
        return members
