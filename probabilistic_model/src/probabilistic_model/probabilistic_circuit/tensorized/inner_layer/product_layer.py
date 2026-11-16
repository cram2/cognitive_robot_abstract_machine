"""
The layer of product units of a layered probabilistic circuit.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from array_api_compat import array_namespace
from random_events.product_algebra import Event, SimpleEvent
from sortedcontainers import SortedSet
from typing_extensions import Self

from probabilistic_model.probabilistic_model import PartialPointType
from probabilistic_model.probabilistic_circuit.tensorized.array_types import (
    BackendArray,
    NodeIndices,
    NodeMask,
    NodeValues,
    NodeVariableValues,
    SampleArray,
    SampleNodeValues,
    VariableIndices,
    VariableMask,
)
from probabilistic_model.probabilistic_circuit.tensorized.forward_sample_assignment import (
    ForwardSampleAssignment,
)
from probabilistic_model.probabilistic_circuit.tensorized.inner_layer.base import (
    InnerLayer,
    Layer,
)
from probabilistic_model.probabilistic_circuit.tensorized.inner_layer.edge_matrix import (
    EdgeMatrix,
    RowReduction,
    SparseEdgeMatrix,
)
from probabilistic_model.probabilistic_circuit.tensorized.inner_layer.inner_layer_edge import (
    InnerLayerEdges,
)
from probabilistic_model.probabilistic_circuit.tensorized.moment_query import (
    MomentQuery,
)
from probabilistic_model.probabilistic_circuit.tensorized.query_cache import (
    QueryCache,
    memoized,
)
from probabilistic_model.probabilistic_circuit.tensorized.structural_query import (
    LayerWithLogProbabilities,
    LogProbabilitiesOfLayers,
    StructuralQuery,
)
from probabilistic_model.probabilistic_circuit.tensorized.utils import remap_indices


@dataclass(eq=False, repr=False)
class ProductLayer(InnerLayer):
    """
    A layer of product units.

    Every node multiplies at most one node of each child layer, so the scope of a node
    is the union of the scopes of its child layers and its likelihood is the sum of
    their log-likelihoods. A node of a child layer may be a factor of several nodes.
    """

    @classmethod
    def product_of(cls, child_layers: list[Layer]) -> Self:
        """
        :param child_layers: The child layers, each contributing its first node.
        :return: A product layer with a single node that multiplies the first node of
            every child layer.
        """
        number_of_child_layers = len(child_layers)
        edges = InnerLayerEdges(
            np.zeros(number_of_child_layers, dtype=np.int64),
            np.arange(number_of_child_layers),
            np.zeros(number_of_child_layers, dtype=np.int64),
        )
        return cls.from_edges(child_layers, edges, 1)

    @classmethod
    def node_wise_product_of(cls, child_layers: list[Layer]) -> Self:
        """
        :param child_layers: The child layers, all with the same number of nodes.
        :return: A product layer whose node ``i`` multiplies node ``i`` of every child
            layer.
        """
        number_of_nodes = child_layers[0].number_of_nodes
        number_of_child_layers = len(child_layers)
        nodes = np.tile(np.arange(number_of_nodes), number_of_child_layers)
        edges = InnerLayerEdges(
            nodes, np.repeat(np.arange(number_of_child_layers), number_of_nodes), nodes
        )
        return cls.from_edges(child_layers, edges, number_of_nodes)

    @classmethod
    def from_edges(
        cls,
        child_layers: list[Layer],
        edges: InnerLayerEdges,
        number_of_nodes: int,
        edge_matrix_type: type[EdgeMatrix] = SparseEdgeMatrix,
    ) -> Self:
        """
        :param child_layers: The child layers.
        :param edges: The edges from the nodes of the new layer into the child layers,
            at most one per node and child layer.
        :param number_of_nodes: The number of nodes of the new layer.
        :param edge_matrix_type: The type of matrix to store the edges in.
        :return: The product layer with these edges.
        """
        offsets = np.cumsum(
            [0] + [child_layer.number_of_nodes for child_layer in child_layers]
        ).astype(np.int64)
        return cls(
            child_layers,
            edge_matrix_type.from_edge_positions(
                np.asarray(edges.nodes, dtype=np.int64),
                offsets[np.asarray(edges.child_layer_indices, dtype=np.int64)]
                + np.asarray(edges.child_nodes, dtype=np.int64),
                (number_of_nodes, int(offsets[-1])),
            ),
        )

    def with_edges(
        self, child_layers: list[Layer], edges: InnerLayerEdges, copies: list[Self]
    ) -> Self:
        return self.from_edges(
            child_layers, edges, len(copies) * self.number_of_nodes, type(self.edges)
        )

    def attach_child_layer(
        self, child_layer: Layer, nodes: NodeIndices, child_nodes: NodeIndices
    ):
        """
        Make some nodes of this layer multiply a node of another layer as well, in
        place.

        :param child_layer: The layer to add as a child layer.
        :param nodes: The nodes of this layer that get a new factor.
        :param child_nodes: The node of ``child_layer`` every one of ``nodes``
            multiplies.
        """
        edges = self.inner_layer_edges
        attached = self.from_edges(
            self.child_layers + [child_layer],
            InnerLayerEdges(
                np.concatenate([edges.nodes, np.asarray(nodes, dtype=np.int64)]),
                np.concatenate(
                    [
                        edges.child_layer_indices,
                        np.full(len(nodes), len(self.child_layers), dtype=np.int64),
                    ]
                ),
                np.concatenate(
                    [edges.child_nodes, np.asarray(child_nodes, dtype=np.int64)]
                ),
            ),
            self.number_of_nodes,
            type(self.edges),
        )
        self.child_layers = attached.child_layers
        self.edges = attached.edges
        self.reset_variables()
        self.__dict__.pop(InnerLayer.inner_layer_edges.attrname, None)

    @property
    def variables(self) -> VariableIndices:
        if self._variables_cache is None:
            self._variables_cache = np.unique(
                np.concatenate(
                    [child_layer.variables for child_layer in self.child_layers]
                )
            )
        return self._variables_cache

    @property
    def number_of_own_parameters(self) -> int:
        # the edges of a product layer are structure, not parameters
        return 0

    def is_decomposable_of_nodes(self) -> NodeMask:
        """
        A product node is decomposable if no variable is in the scope of more than one
        of its factors.
        """
        edges = self.inner_layer_edges
        factors_of_node = np.zeros(
            (len(self.child_layers), self.number_of_nodes), dtype=np.int64
        )
        np.add.at(factors_of_node, (edges.child_layer_indices, edges.nodes), 1)
        scope_of_child_layer = np.array(
            [
                np.isin(self.variables, child_layer.variables)
                for child_layer in self.child_layers
            ],
            dtype=np.int64,
        )
        # how many factors of every node have every variable in their scope
        occurrences = factors_of_node.T @ scope_of_child_layer
        return occurrences.max(axis=1, initial=0) <= 1

    def products_over(self, variables: VariableIndices) -> NodeMask:
        edges = self.inner_layer_edges
        result = np.ones(self.number_of_nodes, dtype=bool)
        for variable in variables:
            child_layer_models_variable = np.array(
                [
                    variable in child_layer.variables
                    for child_layer in self.child_layers
                ],
                dtype=bool,
            )
            models_variable = np.zeros(self.number_of_nodes, dtype=bool)
            models_variable[
                edges.nodes[child_layer_models_variable[edges.child_layer_indices]]
            ] = True
            result &= models_variable
        return result

    # %% queries

    def combine_factors(
        self, child_results: list[BackendArray], reduction: RowReduction
    ) -> BackendArray:
        """
        Combine, per node, the results of the child nodes it multiplies.

        :param child_results: The result per child layer with the child nodes last,
            arrays of one backend.
        :param reduction: How the results of the factors are combined; a node without
            factors gets its identity.
        :return: The combined result with the nodes of this layer last, an array of that
            backend.
        """
        namespace = array_namespace(*child_results)
        values = self.edges.values_of_columns(namespace.concat(child_results, axis=-1))
        return self.edges.reduce_by_row(values, reduction)

    @memoized
    def log_likelihood_of_nodes(
        self, events: SampleArray, cache: QueryCache | None = None
    ) -> SampleNodeValues:
        child_results = [
            child_layer.log_likelihood_of_nodes(events, cache=cache)
            for child_layer in self.child_layers
        ]
        return self.combine_factors(child_results, RowReduction.SUM)

    @memoized
    def cumulative_distribution_of_nodes(
        self, events: SampleArray, cache: QueryCache | None = None
    ) -> SampleNodeValues:
        child_results = [
            child_layer.cumulative_distribution_of_nodes(events, cache=cache)
            for child_layer in self.child_layers
        ]
        return self.combine_factors(child_results, RowReduction.PRODUCT)

    @memoized
    def probability_of_simple_event_of_nodes(
        self,
        event: SimpleEvent,
        variables: SortedSet,
        cache: QueryCache | None = None,
    ) -> NodeValues:
        child_results = [
            child_layer.probability_of_simple_event_of_nodes(
                event, variables, cache=cache
            )
            for child_layer in self.child_layers
        ]
        namespace = array_namespace(*child_results)
        combined = self.combine_factors(
            [namespace.reshape(result, (1, -1)) for result in child_results],
            RowReduction.PRODUCT,
        )
        return namespace.reshape(combined, (-1,))

    @memoized
    def support_of_nodes(
        self, variables: SortedSet, cache: QueryCache | None = None
    ) -> list[Event]:
        child_supports = [
            child_layer.support_of_nodes(variables, cache=cache)
            for child_layer in self.child_layers
        ]

        own_variables = {variables[index] for index in self.variables}
        result: list[Event | None] = [None] * self.number_of_nodes

        for edge in self.iterate_edges():
            support = child_supports[edge.child_layer_index][edge.child_node]
            if result[edge.node] is None:
                result[edge.node] = support.fill_missing_variables_pure(own_variables)
            else:
                result[edge.node] = result[edge.node] & support

        return [Event() if support is None else support for support in result]

    @memoized
    def log_mode_of_nodes(
        self, variables: SortedSet, cache: QueryCache | None = None
    ) -> tuple[list[Event], NodeValues]:
        child_modes = [
            child_layer.log_mode_of_nodes(variables, cache=cache)
            for child_layer in self.child_layers
        ]

        own_variables = {variables[index] for index in self.variables}
        events: list[Event | None] = [None] * self.number_of_nodes
        values = np.zeros(self.number_of_nodes)

        for edge in self.iterate_edges():
            child_event = child_modes[edge.child_layer_index][0][edge.child_node]
            values[edge.node] += child_modes[edge.child_layer_index][1][edge.child_node]
            if events[edge.node] is None:
                events[edge.node] = child_event.fill_missing_variables_pure(
                    own_variables
                )
            else:
                events[edge.node] = events[edge.node].intersection_with(child_event)

        return [Event() if event is None else event for event in events], values

    @memoized
    def moment_of_nodes(
        self,
        query: MomentQuery,
        variables: SortedSet,
        cache: QueryCache | None = None,
    ) -> NodeVariableValues:
        child_results = [
            child_layer.moment_of_nodes(query, variables, cache=cache)
            for child_layer in self.child_layers
        ]
        # the moments of a decomposable product are the moments of the factor that owns
        # the variable, so summing the (zero padded) child moments is the right reduction
        return self.combine_factors(
            [child_result.T for child_result in child_results], RowReduction.SUM
        ).T

    def sample_forward(
        self,
        assignment: ForwardSampleAssignment,
        samples: SampleArray,
        variables: SortedSet,
    ):
        rows_per_node = assignment.rows_of(self)
        for edge in self.iterate_edges():
            rows_of_node = rows_per_node[edge.node]
            if rows_of_node.is_empty:
                continue
            assignment.assign(
                self.child_layers[edge.child_layer_index],
                edge.child_node,
                rows_of_node.rows,
            )

    # %% structural

    def _structural_pass(
        self,
        child_results: list[LayerWithLogProbabilities],
        query: StructuralQuery,
    ) -> LayerWithLogProbabilities:
        """
        Accumulate the log-probabilities of the children of every node.

        :param child_results: The new child layers and the log-probabilities of their
            nodes.
        :param query: The query to record the result in.
        :return: The new layer and the log-probabilities of its nodes.
        """
        result = self.__class__(
            [child_result.layer for child_result in child_results], self.edges
        )

        return query.log_probabilities.record(
            LayerWithLogProbabilities(
                result, result.sum_of_factor_log_probabilities(child_results)
            )
        )

    def sum_of_factor_log_probabilities(
        self, child_results: list[LayerWithLogProbabilities]
    ) -> NodeValues:
        """
        :param child_results: The log-probabilities of the nodes of every child layer.
        :return: The log-probability of every node, the sum over its factors.
        """
        return self.combine_factors(
            [child_result.log_probabilities for child_result in child_results],
            RowReduction.SUM,
        )

    @memoized
    def log_truncated_of_simple_event(
        self,
        event: SimpleEvent,
        query: StructuralQuery,
        cache: QueryCache | None = None,
    ) -> LayerWithLogProbabilities:
        child_results = [
            child_layer.log_truncated_of_simple_event(event, query, cache=cache)
            for child_layer in self.child_layers
        ]
        return self._structural_pass(child_results, query)

    @memoized
    def log_truncated_of_simple_events(
        self,
        events: list[SimpleEvent],
        query: StructuralQuery,
        cache: QueryCache | None = None,
    ) -> LayerWithLogProbabilities:
        number_of_events = len(events)
        edges = self.inner_layer_edges
        blocks = np.repeat(np.arange(number_of_events), len(edges))

        child_results = [
            child_layer.log_truncated_of_simple_events(events, query, cache=cache)
            for child_layer in self.child_layers
        ]

        # every edge is repeated once per event, pointing into that event's block of the
        # child layer
        node_counts = np.array(
            [child_layer.number_of_nodes for child_layer in self.child_layers]
        )
        repeated_edges = InnerLayerEdges(
            np.tile(edges.nodes, number_of_events) + blocks * self.number_of_nodes,
            np.tile(edges.child_layer_indices, number_of_events),
            np.tile(edges.child_nodes, number_of_events)
            + blocks * np.tile(node_counts[edges.child_layer_indices], number_of_events),
        )
        result = self.from_edges(
            [child_result.layer for child_result in child_results],
            repeated_edges,
            number_of_events * self.number_of_nodes,
            type(self.edges),
        )

        return query.log_probabilities.record(
            LayerWithLogProbabilities(
                result, result.sum_of_factor_log_probabilities(child_results)
            )
        )

    @memoized
    def log_conditional_of_point(
        self,
        point: PartialPointType,
        query: StructuralQuery,
        cache: QueryCache | None = None,
    ) -> LayerWithLogProbabilities:
        child_results = [
            child_layer.log_conditional_of_point(point, query, cache=cache)
            for child_layer in self.child_layers
        ]
        return self._structural_pass(child_results, query)

    def required_child_nodes(
        self, alive: NodeMask, log_probabilities: LogProbabilitiesOfLayers
    ) -> list[tuple[Layer, NodeMask]]:
        edges = self.inner_layer_edges
        kept_edges = alive[edges.nodes]
        result = []
        for child_layer_index, child_layer in enumerate(self.child_layers):
            mask = kept_edges & (edges.child_layer_indices == child_layer_index)
            needed = np.zeros(child_layer.number_of_nodes, dtype=bool)
            needed[edges.child_nodes[mask]] = True
            result.append((child_layer, needed))
        return result

    def rebuild(
        self,
        needed: dict[int, NodeMask],
        rebuilt: dict[int, Layer | None],
    ) -> Layer | None:
        alive = needed[id(self)]
        if not alive.any():
            return None

        edges = self.inner_layer_edges
        node_remap, number_of_nodes = remap_indices(alive)
        kept_edges = alive[edges.nodes]

        new_child_layers = []
        new_edges = []

        for child_layer_index, child_layer in enumerate(self.child_layers):
            mask = kept_edges & (edges.child_layer_indices == child_layer_index)
            if not mask.any():
                continue

            pruned_child = rebuilt.get(id(child_layer))
            if pruned_child is None:
                # a factor of the product became impossible, so every node that
                # references it is impossible as well
                return None

            child_remap, _ = remap_indices(needed[id(child_layer)])
            new_edges.append(
                InnerLayerEdges(
                    node_remap[edges.nodes[mask]],
                    np.full(mask.sum(), len(new_child_layers), dtype=np.int64),
                    child_remap[edges.child_nodes[mask]],
                )
            )
            new_child_layers.append(pruned_child)

        if not new_child_layers:
            return None

        return self.from_edges(
            new_child_layers,
            InnerLayerEdges.concatenate(new_edges),
            number_of_nodes,
            type(self.edges),
        )

    @memoized
    def marginal(
        self, kept: VariableMask, cache: QueryCache | None = None
    ) -> Layer | None:
        edges = self.inner_layer_edges
        new_child_layers = []
        new_edges = []

        for child_layer_index, child_layer in enumerate(self.child_layers):
            marginal_child = child_layer.marginal(kept, cache=cache)
            if marginal_child is None:
                continue
            mask = edges.child_layer_indices == child_layer_index
            new_edges.append(
                InnerLayerEdges(
                    edges.nodes[mask],
                    np.full(mask.sum(), len(new_child_layers), dtype=np.int64),
                    edges.child_nodes[mask],
                )
            )
            new_child_layers.append(marginal_child)

        if not new_child_layers:
            return None

        return self.from_edges(
            new_child_layers,
            InnerLayerEdges.concatenate(new_edges),
            self.number_of_nodes,
            type(self.edges),
        )

    @memoized
    def simplify(self, cache: QueryCache | None = None) -> Layer:
        simplified_children = [
            child_layer.simplify(cache=cache) for child_layer in self.child_layers
        ]
        result = self.__class__(simplified_children, self.edges)

        if result.is_identity():
            return simplified_children[0]
        return result

    def is_identity(self) -> bool:
        """
        :return: Whether this layer forwards its single child layer unchanged.
        """
        if len(self.child_layers) != 1:
            return False
        if self.edges.number_of_edges != self.number_of_nodes:
            return False
        if self.child_layers[0].number_of_nodes != self.number_of_nodes:
            return False
        edges = self.inner_layer_edges
        # every node multiplies only the node of the child layer with its own index
        return bool(
            np.array_equal(edges.child_nodes, edges.nodes)
            and len(np.unique(edges.nodes)) == self.number_of_nodes
        )
