from __future__ import annotations

import functools
from dataclasses import (
    dataclass,
)

import numpy as np
from random_events.product_algebra import Event, SimpleEvent
from random_events.variable import Variable
from scipy.sparse import coo_array
from sortedcontainers import SortedSet
from typing_extensions import (
    Any,
    Dict,
    List,
    Optional,
    Self,
    Tuple,
)

from probabilistic_model.exceptions import ShapeMismatchError
from probabilistic_model.probabilistic_circuit.tensorized.array_types import (
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
from probabilistic_model.probabilistic_circuit.tensorized.row_grouped_sparse_array import (
    SparseEntries,
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
    their log-likelihoods.
    """

    edges: coo_array
    """
    The edges as a sparse integer matrix of shape (#child layers, #nodes).

    The value of the entry ``(l, n)`` is the index of the node in the ``l``-th child
    layer that the ``n``-th node of this layer multiplies. A node of a child layer may
    be referenced by several nodes of this layer. A stored ``0`` is an edge to the first
    node of the child layer, not a missing entry.
    """

    @classmethod
    def product_of(cls, child_layers: List[Layer]) -> Self:
        """
        :param child_layers: The child layers, each contributing its first node.
        :return: A product layer with a single node that multiplies the first node of
            every child layer.
        """
        number_of_child_layers = len(child_layers)
        edges = SparseEntries(
            np.zeros(number_of_child_layers, dtype=np.int64),
            np.arange(number_of_child_layers),
            np.zeros(number_of_child_layers, dtype=np.int64),
        ).to_coo_array((number_of_child_layers, 1))
        return cls(child_layers, edges)

    @property
    def number_of_nodes(self) -> int:
        return self.edges.shape[1]

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

    def validate_own(self):
        if self.edges.shape != (len(self.child_layers), self.number_of_nodes):
            raise ShapeMismatchError(
                (len(self.child_layers), self.number_of_nodes), self.edges.shape
            )

    def is_decomposable_of_nodes(self) -> NodeMask:
        """
        A product node is decomposable if no variable is in the scope of more than one
        of its factors.
        """
        factors_of_node = np.zeros(
            (len(self.child_layers), self.number_of_nodes), dtype=np.int64
        )
        np.add.at(factors_of_node, (self.edges.row, self.edges.col), 1)
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

    # %% queries

    @functools.cached_property
    def inner_layer_edges(self) -> InnerLayerEdges:
        return InnerLayerEdges(
            self.edges.col.astype(np.int64),
            self.edges.row.astype(np.int64),
            self.edges.data.astype(np.int64),
        )

    @functools.cached_property
    def edges_per_child_layer(self) -> List[InnerLayerEdges]:
        """
        :return: The edges into every child layer. A decomposable product has at most one
            factor in each child layer, so every node appears at most once in each of
            them; :attr:`InnerLayerEdges.every_node_at_most_once` keeps the reductions
            correct for a circuit that is not decomposable.
        """
        return [
            self.inner_layer_edges.into_child_layer(child_layer_index)
            for child_layer_index in range(len(self.child_layers))
        ]

    def _gather_and_add(
        self, child_results: List[SampleNodeValues], fill: float
    ) -> SampleNodeValues:
        """
        Sum, per node, the results of the child nodes the edges point to.

        :param child_results: The result per child layer with the child nodes last.
        :param fill: The value of a node without any edge.
        :return: The summed result with the nodes of this layer last.
        """
        leading_shape = child_results[0].shape[:-1]
        result = np.zeros(leading_shape + (self.number_of_nodes,))
        touched = np.zeros(self.number_of_nodes, dtype=bool)

        for edges, child_result in zip(self.edges_per_child_layer, child_results):
            if len(edges) == 0:
                continue
            gathered = child_result[..., edges.child_nodes]
            if edges.every_node_at_most_once:
                result[..., edges.nodes] += gathered
            else:
                np.add.at(result, (Ellipsis, edges.nodes), gathered)
            touched[edges.nodes] = True

        if not touched.all():
            result[..., ~touched] = fill
        return result

    @memoized
    def log_likelihood_of_nodes(
        self, events: SampleArray, cache: Optional[QueryCache] = None
    ) -> SampleNodeValues:
        child_results = [
            child_layer.log_likelihood_of_nodes(events, cache=cache)
            for child_layer in self.child_layers
        ]
        return self._gather_and_add(child_results, fill=0.0)

    @memoized
    def cumulative_distribution_of_nodes(
        self, events: SampleArray, cache: Optional[QueryCache] = None
    ) -> SampleNodeValues:
        child_results = [
            child_layer.cumulative_distribution_of_nodes(events, cache=cache)
            for child_layer in self.child_layers
        ]
        return self._gather_and_multiply(child_results)

    def _gather_and_multiply(
        self, child_results: List[SampleNodeValues]
    ) -> SampleNodeValues:
        leading_shape = child_results[0].shape[:-1]
        result = np.ones(leading_shape + (self.number_of_nodes,))
        for edges, child_result in zip(self.edges_per_child_layer, child_results):
            if len(edges) == 0:
                continue
            gathered = child_result[..., edges.child_nodes]
            if edges.every_node_at_most_once:
                result[..., edges.nodes] *= gathered
            else:
                np.multiply.at(result, (Ellipsis, edges.nodes), gathered)
        return result

    @memoized
    def probability_of_simple_event_of_nodes(
        self,
        event: SimpleEvent,
        variables: SortedSet,
        cache: Optional[QueryCache] = None,
    ) -> NodeValues:
        child_results = [
            child_layer.probability_of_simple_event_of_nodes(
                event, variables, cache=cache
            ).reshape(1, -1)
            for child_layer in self.child_layers
        ]
        return self._gather_and_multiply(child_results).reshape(-1)

    @memoized
    def support_of_nodes(
        self, variables: SortedSet, cache: Optional[QueryCache] = None
    ) -> List[Event]:
        child_supports = [
            child_layer.support_of_nodes(variables, cache=cache)
            for child_layer in self.child_layers
        ]

        own_variables = {variables[index] for index in self.variables}
        result: List[Optional[Event]] = [None] * self.number_of_nodes

        for edge in self.iterate_edges():
            support = child_supports[edge.child_layer_index][edge.child_node]
            if result[edge.node] is None:
                result[edge.node] = support.fill_missing_variables_pure(own_variables)
            else:
                result[edge.node] = result[edge.node] & support

        return [Event() if support is None else support for support in result]

    @memoized
    def log_mode_of_nodes(
        self, variables: SortedSet, cache: Optional[QueryCache] = None
    ) -> Tuple[List[Event], NodeValues]:
        child_modes = [
            child_layer.log_mode_of_nodes(variables, cache=cache)
            for child_layer in self.child_layers
        ]

        own_variables = {variables[index] for index in self.variables}
        events: List[Optional[Event]] = [None] * self.number_of_nodes
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
        cache: Optional[QueryCache] = None,
    ) -> NodeVariableValues:
        child_results = [
            child_layer.moment_of_nodes(query, variables, cache=cache)
            for child_layer in self.child_layers
        ]
        # the moments of a decomposable product are the moments of the factor that owns
        # the variable, so summing the (zero padded) child moments is the right reduction
        result = np.zeros((self.number_of_nodes, query.number_of_variables))
        for edges, child_result in zip(self.edges_per_child_layer, child_results):
            if len(edges) == 0:
                continue
            if edges.every_node_at_most_once:
                result[edges.nodes] += child_result[edges.child_nodes]
            else:
                np.add.at(result, edges.nodes, child_result[edges.child_nodes])
        return result

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
        child_results: List[LayerWithLogProbabilities],
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
            [child_result.layer for child_result in child_results], self.edges.copy()
        )

        own_log_probabilities = np.zeros(self.number_of_nodes)
        for edges, child_result in zip(self.edges_per_child_layer, child_results):
            if len(edges) == 0:
                continue
            gathered = child_result.log_probabilities[edges.child_nodes]
            if edges.every_node_at_most_once:
                own_log_probabilities[edges.nodes] += gathered
            else:
                np.add.at(own_log_probabilities, edges.nodes, gathered)

        return query.log_probabilities.record(
            LayerWithLogProbabilities(result, own_log_probabilities)
        )

    @memoized
    def log_truncated_of_simple_event(
        self,
        event: SimpleEvent,
        query: StructuralQuery,
        cache: Optional[QueryCache] = None,
    ) -> LayerWithLogProbabilities:
        child_results = [
            child_layer.log_truncated_of_simple_event(event, query, cache=cache)
            for child_layer in self.child_layers
        ]
        return self._structural_pass(child_results, query)

    @memoized
    def log_truncated_of_simple_events(
        self,
        events: List[SimpleEvent],
        query: StructuralQuery,
        cache: Optional[QueryCache] = None,
    ) -> LayerWithLogProbabilities:
        number_of_events = len(events)
        number_of_nodes = self.number_of_nodes
        number_of_entries = self.edges.nnz
        blocks = np.arange(number_of_events)

        child_results = [
            child_layer.log_truncated_of_simple_events(events, query, cache=cache)
            for child_layer in self.child_layers
        ]

        # every edge is repeated once per event, pointing into that event's block of the
        # child layer
        node_counts = np.array(
            [child_layer.number_of_nodes for child_layer in self.child_layers]
        )
        rows = np.tile(self.edges.row, number_of_events)
        columns = np.tile(self.edges.col, number_of_events) + np.repeat(
            blocks * number_of_nodes, number_of_entries
        )
        data = np.tile(self.edges.data.astype(np.int64), number_of_events) + np.repeat(
            blocks, number_of_entries
        ) * np.tile(node_counts[self.edges.row], number_of_events)

        edges = SparseEntries(data, rows, columns).to_coo_array(
            (len(self.child_layers), number_of_events * number_of_nodes)
        )
        result = self.__class__(
            [child_result.layer for child_result in child_results], edges
        )

        own_log_probabilities = np.zeros(number_of_events * number_of_nodes)
        for child_layer_index, child_result in enumerate(child_results):
            mask = rows == child_layer_index
            np.add.at(
                own_log_probabilities,
                columns[mask],
                child_result.log_probabilities[data[mask]],
            )
        return query.log_probabilities.record(
            LayerWithLogProbabilities(result, own_log_probabilities)
        )

    @memoized
    def log_conditional_of_point(
        self,
        point: Dict[Variable, Any],
        query: StructuralQuery,
        cache: Optional[QueryCache] = None,
    ) -> LayerWithLogProbabilities:
        child_results = [
            child_layer.log_conditional_of_point(point, query, cache=cache)
            for child_layer in self.child_layers
        ]
        return self._structural_pass(child_results, query)

    def required_child_nodes(
        self, alive: NodeMask, log_probabilities: LogProbabilitiesOfLayers
    ) -> List[Tuple[Layer, NodeMask]]:
        kept_edges = alive[self.edges.col]
        result = []
        for child_layer_index, child_layer in enumerate(self.child_layers):
            mask = kept_edges & (self.edges.row == child_layer_index)
            needed = np.zeros(child_layer.number_of_nodes, dtype=bool)
            needed[self.edges.data[mask].astype(np.int64)] = True
            result.append((child_layer, needed))
        return result

    def rebuild(
        self,
        needed: Dict[int, NodeMask],
        rebuilt: Dict[int, Optional[Layer]],
    ) -> Optional[Layer]:
        alive = needed[id(self)]
        if not alive.any():
            return None

        node_remap, number_of_nodes = remap_indices(alive)
        kept_edges = alive[self.edges.col]

        new_child_layers = []
        new_edges = []

        for child_layer_index, child_layer in enumerate(self.child_layers):
            mask = kept_edges & (self.edges.row == child_layer_index)
            if not mask.any():
                continue

            pruned_child = rebuilt.get(id(child_layer))
            if pruned_child is None:
                # a factor of the product became impossible, so every node that
                # references it is impossible as well
                return None

            child_remap, _ = remap_indices(needed[id(child_layer)])
            new_edges.append(
                SparseEntries(
                    child_remap[self.edges.data[mask].astype(np.int64)],
                    np.full(mask.sum(), len(new_child_layers), dtype=np.int64),
                    node_remap[self.edges.col[mask]],
                )
            )
            new_child_layers.append(pruned_child)

        if not new_child_layers:
            return None

        edges = SparseEntries.concatenate(new_edges).to_coo_array(
            (len(new_child_layers), number_of_nodes)
        )
        return self.__class__(new_child_layers, edges)

    @memoized
    def marginal(
        self, kept: VariableMask, cache: Optional[QueryCache] = None
    ) -> Optional[Layer]:
        new_child_layers = []
        new_edges = []

        for child_layer_index, child_layer in enumerate(self.child_layers):
            marginal_child = child_layer.marginal(kept, cache=cache)
            if marginal_child is None:
                continue
            mask = self.edges.row == child_layer_index
            new_edges.append(
                SparseEntries(
                    self.edges.data[mask],
                    np.full(mask.sum(), len(new_child_layers), dtype=np.int64),
                    self.edges.col[mask],
                )
            )
            new_child_layers.append(marginal_child)

        if not new_child_layers:
            return None

        edges = SparseEntries.concatenate(new_edges).to_coo_array(
            (len(new_child_layers), self.number_of_nodes)
        )
        return self.__class__(new_child_layers, edges)

    @memoized
    def simplify(self, cache: Optional[QueryCache] = None) -> Layer:
        simplified_children = [
            child_layer.simplify(cache=cache) for child_layer in self.child_layers
        ]
        result = self.__class__(simplified_children, self.edges.copy())

        if result.is_identity():
            return simplified_children[0]
        return result

    def is_identity(self) -> bool:
        """
        :return: Whether this layer forwards its single child layer unchanged.
        """
        if len(self.child_layers) != 1:
            return False
        if self.edges.nnz != self.number_of_nodes:
            return False
        if self.child_layers[0].number_of_nodes != self.number_of_nodes:
            return False
        # every node multiplies only the node of the child layer with its own index
        return bool(
            np.array_equal(self.edges.data.astype(np.int64), self.edges.col)
            and len(np.unique(self.edges.col)) == self.number_of_nodes
        )

    def __deepcopy__(self, memo=None) -> ProductLayer:
        if memo is None:
            memo = {}
        if id(self) in memo:
            return memo[id(self)]
        child_layers = [
            child_layer.__deepcopy__(memo) for child_layer in self.child_layers
        ]
        result = self.__class__(child_layers, self.edges.copy())
        memo[id(self)] = result
        return result
