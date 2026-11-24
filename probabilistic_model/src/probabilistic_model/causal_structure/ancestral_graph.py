"""
An ancestral graph over the variables of a fitted model, which allows hidden variables
that cause two or more of them: a maximal ancestral graph, or the partial ancestral
graph standing for every maximal ancestral graph the data cannot tell apart.
"""

from __future__ import annotations

import itertools
from dataclasses import dataclass
from functools import cached_property

from krrood.patterns.classproperty import classproperty

from probabilistic_model.causal_structure.graph import (
    EdgeMark,
    EdgeMarkedGraph,
    GraphEdge,
    GraphNode,
)
from probabilistic_model.graph import DirectedAcyclicGraph

# %% the graph


@dataclass(eq=False, kw_only=True)
class AncestralGraph(EdgeMarkedGraph):
    """
    A maximal ancestral graph, or a partial ancestral graph standing for several.

    The mark at a node is a tail if the node causes the other end, an arrow if it does
    not, and a circle if the maximal ancestral graphs disagree.

    .. note::
        A directed edge says its parent causes its child, not that no hidden variable
        causes both. Only a visible edge, see :meth:`is_visible`, rules that out.
    """

    @classproperty
    def is_closed_under_union(cls) -> bool:
        """
        :return: False, unlike :meth:`EdgeMarkedGraph.is_closed_under_union`: the edges
            of ancestral graphs over overlapping variables may not form one.
        """
        return False

    # %% possibly directed paths

    def possible_children(self, *, node: GraphNode) -> list[GraphNode]:
        """
        :param node: A node of the graph.
        :return: The nodes joined to it by an edge that shows no arrow at it, the
            nodes a possibly directed path may step to from it.
        """
        return [
            edge.other_end(node=node)
            for edge in self.edges_at(node=node)
            if edge.mark_at(node=node) is not EdgeMark.ARROW
        ]

    def possibly_directed_paths(self, *, start: GraphNode) -> list[list[GraphNode]]:
        """
        :param start: A node of the graph.
        :return: Every possibly directed path from the node, the node alone included:
            every path on which no edge between two of its nodes shows an arrow at the
            earlier one.
        """
        paths, to_extend = [], [[start]]
        while to_extend:
            path = to_extend.pop()
            paths.append(path)
            to_extend.extend(
                path + [child]
                for child in self.possible_children(node=path[-1])
                if child not in path
                and self.points_back_to_none_of(earlier=path[:-1], node=child)
            )
        return paths

    def points_back_to_none_of(
        self, *, earlier: list[GraphNode], node: GraphNode
    ) -> bool:
        """
        :param earlier: Nodes of a path.
        :param node: A node to extend the path by.
        :return: Whether no edge between the node and one of them shows an arrow at
            that one.
        """
        return not any(
            (edge := self.edge_between(previous, node)) is not None
            and edge.mark_at(node=previous) is EdgeMark.ARROW
            for previous in earlier
        )

    @cached_property
    def possible_descendants_of_node(self) -> dict[GraphNode, set[GraphNode]]:
        """
        :return: For every node, the nodes other than itself that a possibly directed
            path from it reaches.
        """
        return {
            node: {path[-1] for path in self.possibly_directed_paths(start=node)}
            - {node}
            for node in self.nodes()
        }

    def possible_descendants(self, *, node: GraphNode) -> set[GraphNode]:
        """
        :param node: A node of the graph.
        :return: The nodes, other than itself, a possibly directed path from it
            reaches.
        """
        return self.possible_descendants_of_node[node]

    def possible_ancestors(self, *, node: GraphNode) -> set[GraphNode]:
        """
        :return: The nodes, other than itself, from which a possibly directed path
            reaches it.
        """
        return {
            ancestor
            for ancestor, descendants in self.possible_descendants_of_node.items()
            if node in descendants
        }

    # %% whether the graph is a valid ancestral graph

    @cached_property
    def directed_part(self) -> DirectedAcyclicGraph[GraphNode]:
        """
        :return: The graph's directed edges, over all of its nodes, each from its
            parent to its child.
        """
        directed = DirectedAcyclicGraph()
        directed.add_nodes_from(self.nodes())
        directed.add_edges_from(
            (edge.parent, edge.child) for edge in self.edges() if edge.is_directed
        )
        return directed

    @cached_property
    def has_selection_bias_edge(self) -> bool:
        """
        :return: Whether an edge says the rows were selected by a variable both its
            ends cause, which the graph's reading does not support.
        """
        return any(edge.implies_selection_bias for edge in self.edges())

    @cached_property
    def is_ancestral(self) -> bool:
        """
        :return: Whether the directed edges form no cycle and no bidirected edge joins
            a node to one of its ancestors.
        """
        if not self.directed_part.is_acyclic:
            return False
        return not any(
            edge.first in self.directed_part.ancestors(edge.second)
            or edge.second in self.directed_part.ancestors(edge.first)
            for edge in self.edges()
            if edge.is_bidirected
        )

    @cached_property
    def representative_maximal_ancestral_graph(self) -> AncestralGraph | None:
        """
        :return: One maximal ancestral graph the graph stands for: each ``o→`` becomes
            ``→``, and the ``o–o`` edges are oriented without unshielded colliders,
            from the earlier node to the later in a maximum cardinality search.
            ``None`` if the graph has a selection bias edge, its ``o–o`` edges do not
            form a chordal graph, or the result is not ancestral.
        """
        if self.has_selection_bias_edge:
            return None
        nondirected = [edge for edge in self.edges() if edge.is_nondirected]
        order = self.maximum_cardinality_order(edges=nondirected)
        if order is None:
            return None
        position = {node: index for index, node in enumerate(order)}
        edges = [
            self.definite_edge_of(edge=edge, position=position) for edge in self.edges()
        ]
        representative = AncestralGraph.from_edges(nodes=self.nodes(), edges=edges)
        if not representative.is_ancestral:
            return None
        return representative

    @staticmethod
    def definite_edge_of(
        *, edge: GraphEdge, position: dict[GraphNode, int]
    ) -> GraphEdge:
        """
        :param edge: An edge of the graph.
        :param position: The position of every end of an ``o–o`` edge in the order it
            is oriented by.
        :return: The edge with its circles replaced as
            :attr:`representative_maximal_ancestral_graph` replaces them.
        """
        if edge.is_partially_directed:
            circle_end = (
                edge.first if edge.mark_at_first is EdgeMark.CIRCLE else edge.second
            )
            return GraphEdge.directed_edge(circle_end, edge.other_end(node=circle_end))
        if edge.is_nondirected:
            earlier, later = sorted((edge.first, edge.second), key=position.__getitem__)
            return GraphEdge.directed_edge(earlier, later)
        return edge

    @staticmethod
    def maximum_cardinality_order(*, edges: list[GraphEdge]) -> list[GraphNode] | None:
        """
        :param edges: Edges read as undirected.
        :return: Their nodes in the order of a maximum cardinality search, which picks
            next the node with the most neighbours already picked, the first by name on
            a tie; ``None`` if the edges do not form a chordal graph.
        """
        neighbours: dict[GraphNode, set[GraphNode]] = {}
        for edge in edges:
            neighbours.setdefault(edge.first, set()).add(edge.second)
            neighbours.setdefault(edge.second, set()).add(edge.first)
        order: list[GraphNode] = []
        while len(order) < len(neighbours):
            picked = max(
                sorted(set(neighbours) - set(order)),
                key=lambda node: len(neighbours[node] & set(order)),
            )
            earlier = neighbours[picked] & set(order)
            if any(
                second not in neighbours[first]
                for first, second in itertools.combinations(sorted(earlier), 2)
            ):
                return None
            order.append(picked)
        return order

    # %% visible edges and the paths an adjustment set must leave alone

    def is_visible(self, *, edge: GraphEdge) -> bool:
        """
        :param edge: A directed edge ``X → W`` of the graph.
        :return: Whether the graph rules out a hidden variable causing both ends: some
            node not adjacent to ``W`` has an edge into ``X``, or a path into ``X``
            whose inner nodes are colliders and parents of ``W``. Circles never count.
        """
        cause, child = edge.parent, edge.child
        visited, to_visit = {cause}, [cause]
        while to_visit:
            node = to_visit.pop()
            for other_edge in self.edges_at(node=node):
                other = other_edge.other_end(node=node)
                if other_edge.mark_at(node=node) is not EdgeMark.ARROW:
                    continue
                if other not in (cause, child) and not self.adjacent(
                    first=other, second=child
                ):
                    return True
                if (
                    other_edge.is_bidirected
                    and other not in visited
                    and self.is_parent_of(parent=other, child=child)
                ):
                    visited.add(other)
                    to_visit.append(other)
        return False

    def is_parent_of(self, *, parent: GraphNode, child: GraphNode) -> bool:
        """
        :param parent: A node.
        :param child: Another node.
        :return: Whether a directed edge leads from the first to the second.
        """
        edge = self.edge_between(parent, child)
        return edge is not None and edge.is_directed and edge.parent == parent

    def possibly_causal_paths(
        self, *, cause: GraphNode, effect: GraphNode
    ) -> list[list[GraphNode]]:
        """
        :param cause: A node.
        :param effect: Another node.
        :return: Every possibly directed path from the cause to the effect.
        """
        return [
            path
            for path in self.possibly_directed_paths(start=cause)
            if path[-1] == effect
        ]

    def forbidden_nodes(
        self, *, possibly_causal_paths: list[list[GraphNode]]
    ) -> set[GraphNode]:
        """
        :param possibly_causal_paths: Every possibly directed path from a cause to an
            effect.
        :return: Every node other than the cause on one of them, and every possible
            descendant of one.
        """
        on_paths = {node for path in possibly_causal_paths for node in path[1:]}
        return on_paths.union(
            *(self.possible_descendants(node=node) for node in on_paths)
        )

    # %% m-separation in a maximal ancestral graph

    def is_m_separated(
        self, *, first: GraphNode, second: GraphNode, given: frozenset[GraphNode]
    ) -> bool:
        """
        :param first: A node of a graph with only directed and bidirected edges.
        :param second: Another node.
        :param given: The nodes conditioned on, neither of the two among them.
        :return: Whether every path between the two has a collider that is neither
            given nor an ancestor of a given node, or a non-collider that is given.
        """
        opens_colliders = set(given).union(
            *(self.directed_part.ancestors(node) for node in given)
        )
        visited: set[tuple[GraphNode, bool]] = set()
        to_visit = [
            (edge.other_end(node=first), self.arrives_by_arrow(edge=edge, at=first))
            for edge in self.edges_at(node=first)
        ]
        while to_visit:
            node, entered_by_arrow = to_visit.pop()
            if node == second:
                return False
            if (node, entered_by_arrow) in visited:
                continue
            visited.add((node, entered_by_arrow))
            for edge in self.edges_at(node=node):
                is_collider = (
                    entered_by_arrow and edge.mark_at(node=node) is EdgeMark.ARROW
                )
                if is_collider and node not in opens_colliders:
                    continue
                if not is_collider and node in given:
                    continue
                to_visit.append(
                    (
                        edge.other_end(node=node),
                        self.arrives_by_arrow(edge=edge, at=node),
                    )
                )
        return True

    @staticmethod
    def arrives_by_arrow(*, edge: GraphEdge, at: GraphNode) -> bool:
        """
        :param edge: An edge.
        :param at: The end a path leaves the edge from.
        :return: Whether the path enters the other end through an arrow.
        """
        return edge.mark_at(node=edge.other_end(node=at)) is EdgeMark.ARROW

    def without_causal_first_edges(
        self, *, cause: GraphNode, effect: GraphNode
    ) -> AncestralGraph:
        """
        :param cause: A node of a graph with only directed and bidirected edges.
        :param effect: Another node.
        :return: The graph without the edges from the cause into a node on a directed
            path from the cause to the effect, which leaves only the paths that are not
            causal to connect the two.
        """
        on_causal_paths = self.directed_part.descendants(cause) & (
            self.directed_part.ancestors(effect) | {effect}
        )
        return AncestralGraph.from_edges(
            nodes=self.nodes(),
            edges=[
                edge
                for edge in self.edges()
                if not (
                    edge.is_directed
                    and edge.parent == cause
                    and edge.child in on_causal_paths
                )
            ],
        )
