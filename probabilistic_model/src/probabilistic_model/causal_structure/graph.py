"""
A causal graph over the variables of a fitted model: which variables cause which, as far
as the data settles it, and the directed acyclic graphs it stands for.
"""

from __future__ import annotations

import itertools
from dataclasses import dataclass, field, replace
from enum import StrEnum
from functools import cached_property

from random_events.variable import Variable
from typing_extensions import Callable, Iterable

from probabilistic_model.directed_acyclic_graph import DirectedAcyclicGraph
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

    first: str
    """
    The name of the node at one end.
    """

    second: str
    """
    The name of the node at the other end.
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
    def directed(cls, parent: str, child: str) -> GraphEdge:
        """
        :param parent: The name of the cause.
        :param child: The name of what it causes.
        :return: The edge ``parent → child``.
        """
        return cls(parent, child, EdgeMark.TAIL, EdgeMark.ARROW)

    @classmethod
    def undirected(cls, first: str, second: str) -> GraphEdge:
        """
        :param first: The name of the node at one end.
        :param second: The name of the node at the other end.
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
    def parent(self) -> str:
        """
        :return: The name of the node at the tail of a directed edge.
        """
        return self.first if self.mark_at_first is EdgeMark.TAIL else self.second

    @property
    def child(self) -> str:
        """
        :return: The name of the node at the arrow of a directed edge.
        """
        return self.second if self.mark_at_first is EdgeMark.TAIL else self.first


# %% the directed acyclic graphs a causal graph stands for


@dataclass
class MemberGraph(DirectedAcyclicGraph[str]):
    """
    One of the directed acyclic graphs a causal graph stands for, over the names of its
    nodes.
    """

    indices: dict[str, int] = field(default_factory=dict, repr=False)
    """
    The index of every node in :attr:`graph`, by name.
    """

    def index_of(self, node: str) -> int:
        return self.indices[node]

    def add_node(self, node: str) -> None:
        if node not in self.indices:
            self.indices[node] = self.graph.add_node(node)

    def copy(self) -> MemberGraph:
        """
        :return: A graph with the same nodes and edges, changed independently of this
            one.
        """
        return MemberGraph(graph=self.graph.copy(), indices=dict(self.indices))

    def without_edges(self, edges: Iterable[tuple[str, str]]) -> MemberGraph:
        """
        :param edges: Edges of the graph, each as its parent and its child.
        :return: A copy of the graph without them.
        """
        result = self.copy()
        for parent, child in edges:
            result.graph.remove_edge(self.index_of(parent), self.index_of(child))
        return result


# %% the graph


@dataclass
class CausalGraph:
    """
    A causal graph: a directed acyclic graph, or the equivalence class of those the data
    cannot tell apart, whose undirected edges any of them may orient either way.
    """

    nodes: list[GraphNode]
    """
    The nodes, one per variable.
    """

    edges: list[GraphEdge]
    """
    The edges, at most one between two nodes.
    """

    maximum_undirected_edges: int = 12
    """
    How many undirected edges the graph may have for the directed acyclic graphs it
    stands for to be listed, each undirected edge doubling their candidates.
    """

    def node(self, name: str) -> GraphNode:
        """
        :param name: The name of a node's variable.
        :return: The node.
        :raises UnknownGraphNode: If the graph has no such node.
        """
        for node in self.nodes:
            if node.name == name:
                return node
        raise UnknownGraphNode(name, [node.name for node in self.nodes])

    @property
    def node_names(self) -> list[str]:
        """
        :return: The names of the nodes, in the graph's order.
        """
        return [node.name for node in self.nodes]

    def edge_between(self, first: str, second: str) -> GraphEdge | None:
        """
        :param first: The name of one node.
        :param second: The name of another.
        :return: The edge between them, or ``None`` if they are not adjacent.
        """
        for edge in self.edges:
            if {edge.first, edge.second} == {first, second}:
                return edge
        return None

    def adjacent(self, first: str, second: str) -> bool:
        """
        :param first: The name of one node.
        :param second: The name of another.
        :return: Whether an edge joins them.
        """
        return self.edge_between(first, second) is not None

    def unshielded_colliders(self) -> set[tuple[str, str, str]]:
        """
        :return: Every ``a → c ← b`` among the directed edges with ``a`` and ``b`` not
            adjacent, as ``(a, c, b)`` with ``a`` sorted before ``b``.
        """
        directed = MemberGraph()
        directed.add_nodes_from(self.node_names)
        directed.add_edges_from(
            (edge.parent, edge.child) for edge in self.edges if edge.is_directed
        )
        return self.colliders_of(directed, self.adjacent)

    @staticmethod
    def colliders_of(
        directed: MemberGraph, adjacent: Callable[[str, str], bool]
    ) -> set[tuple[str, str, str]]:
        """
        :param directed: Directed edges.
        :param adjacent: Whether two nodes are joined by any edge.
        :return: Every unshielded collider among the directed edges, as
            :meth:`unshielded_colliders` gives them.
        """
        return {
            (first, child, second)
            for child in directed.nodes()
            for first, second in itertools.combinations(
                sorted(directed.predecessors(child)), 2
            )
            if not adjacent(first, second)
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
                len([edge for edge in self.edges if edge.is_undirected])
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
        undirected = [edge for edge in self.edges if edge.is_undirected]
        if len(undirected) > self.maximum_undirected_edges:
            raise EquivalenceClassTooLarge(
                len(undirected), self.maximum_undirected_edges
            )
        directed = MemberGraph()
        directed.add_nodes_from(self.node_names)
        directed.add_edges_from(
            (edge.parent, edge.child) for edge in self.edges if edge.is_directed
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
            if member.is_acyclic and (
                self.colliders_of(member, self.adjacent) == colliders
            ):
                members.append(member)
        return members

    def possible_ancestors(self, name: str) -> set[str]:
        """
        :param name: The name of a node.
        :return: The names of the nodes with a directed path to it in at least one
            member graph.
        """
        return set().union(*(member.ancestors(name) for member in self.member_graphs()))
