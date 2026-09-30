from __future__ import annotations

import functools
from dataclasses import dataclass

import numpy as np
import numpy.typing as npt
from random_events.product_algebra import Event, SimpleEvent
from random_events.variable import Variable
from sortedcontainers import SortedSet
from typing_extensions import (
    Any,
    Dict,
    Iterable,
    List,
    Optional,
    Self,
    Tuple,
)

from probabilistic_model.exceptions import ShapeMismatchError
from probabilistic_model.probabilistic_circuit.tensorized.array_types import (
    EdgeMask,
    EdgeValues,
    NodeMask,
    NodeValues,
    NodeVariableValues,
    SampleArray,
    SampleNodeValues,
    SampleRows,
    VariableIndices,
    VariableMask,
)
from probabilistic_model.probabilistic_circuit.tensorized.exceptions import (
    NumberOfWeightsMismatchError,
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
    RowGroupedSparseArray,
    SparseEntries,
)
from probabilistic_model.probabilistic_circuit.tensorized.structural_query import (
    LayerWithLogProbabilities,
    LogProbabilitiesOfLayers,
    StructuralQuery,
)
from probabilistic_model.probabilistic_circuit.tensorized.utils import (
    embedded_logsumexp,
    remap_indices,
)


@dataclass(eq=False, repr=False)
class SumLayer(InnerLayer):
    """
    A layer of sum units.

    All nodes of a sum layer have the same scope, which is the scope of its child
    layers.

    The weights are always stored sparsely: a sum node usually has few children, so
    the dense weight matrix of a layer with many nodes is mostly empty and can be far
    larger than the circuit itself.
    """

    log_weights: RowGroupedSparseArray
    """
    The logarithmic weights of the edges as a sparse array of shape (#nodes, total
    number of nodes of the child layers).

    The columns are the nodes of the child layers in order: the node ``j`` of the
    ``i``-th child layer is the column ``column_offsets[i] + j``.
    """

    @classmethod
    def mixture_of(
        cls, child_layers: List[Layer], log_weights: Iterable[float]
    ) -> Self:
        """
        :param child_layers: The child layers, each contributing its first node.
        :param log_weights: The logarithmic weight of every child layer.
        :return: A sum layer with a single node that mixes the first node of every
            child layer.
        :raises NumberOfWeightsMismatchError: If there is not one weight per child
            layer.
        """
        weights = np.array(list(log_weights), dtype=float)
        if len(weights) != len(child_layers):
            raise NumberOfWeightsMismatchError(len(weights), len(child_layers))
        offsets = np.cumsum(
            [0] + [child_layer.number_of_nodes for child_layer in child_layers]
        )
        return cls(
            child_layers,
            RowGroupedSparseArray.from_entries(
                SparseEntries(
                    weights, np.zeros(len(child_layers), dtype=np.int64), offsets[:-1]
                ),
                (1, int(offsets[-1])),
            ),
        )

    @property
    def variables(self) -> VariableIndices:
        if self._variables_cache is None:
            self._variables_cache = self.child_layers[0].variables
        return self._variables_cache

    @property
    def number_of_nodes(self) -> int:
        return self.log_weights.shape[0]

    @property
    def number_of_own_parameters(self) -> int:
        return self.log_weights.number_of_stored_entries

    @property
    def column_offsets(self) -> npt.NDArray[np.int64]:
        """
        :return: The first column of every child layer, followed by the number of
            columns.
        """
        return np.concatenate(
            [
                [0],
                np.cumsum(
                    [child_layer.number_of_nodes for child_layer in self.child_layers]
                ),
            ]
        ).astype(np.int64)

    def validate_own(self):
        expected_shape = (self.number_of_nodes, int(self.column_offsets[-1]))
        if self.log_weights.shape != expected_shape:
            raise ShapeMismatchError(expected_shape, self.log_weights.shape)

    # %% edges

    @functools.cached_property
    def inner_layer_edges(self) -> InnerLayerEdges:
        """
        :return: The edges in the order of the stored weights.
        """
        offsets = self.column_offsets
        columns = self.log_weights.columns
        child_layer_indices = np.searchsorted(offsets, columns, side="right") - 1
        return InnerLayerEdges(
            self.log_weights.rows.astype(np.int64),
            child_layer_indices,
            columns - offsets[child_layer_indices],
        )

    def values_of_edges(self, child_results: List[npt.NDArray]) -> npt.NDArray:
        """
        Take the value of the child node of every edge.

        :param child_results: The result per child layer, child nodes last.
        :return: One value per edge, the edges last.
        """
        values = np.concatenate(child_results, axis=-1)
        columns = self.log_weights.columns
        # a sum layer usually points at every node of its child layers exactly once and
        # in order, in which case the gather is an identity copy of an array that has one
        # entry per event per node, and skipping it is worth the comparison
        if len(columns) == values.shape[-1] and np.array_equal(
            columns, np.arange(len(columns))
        ):
            return values
        return values[..., columns]

    # %% weights

    @property
    def log_normalization_constants(self) -> NodeValues:
        """
        :return: ``log(sum(exp(w)))`` over the weights of each node, shape (#nodes,).
        """
        return embedded_logsumexp(
            self.log_weights.group_by_row(self.log_weights.data), axis=-1
        )

    @property
    def normalized_edge_log_weights(self) -> EdgeValues:
        """
        :return: The logarithmic weight of every edge, normalized per node.
        """
        return (
            self.log_weights.data
            - self.log_normalization_constants[self.log_weights.rows]
        )

    @property
    def normalized_edge_weights(self) -> EdgeValues:
        """
        :return: The weight of every edge in linear space, normalized per node.
        """
        shifted = self.normalized_edge_log_weights
        # a node whose weights are all -inf normalizes to nan; it is impossible, and the
        # prune pass removes it, so its weights are simply zero here
        return np.where(np.isfinite(shifted), np.exp(shifted), 0.0)

    def normalize_own(self):
        self.log_weights.data = self.normalized_edge_log_weights

    # %% queries

    def _weighted_forward(
        self, child_results: List[SampleNodeValues]
    ) -> SampleNodeValues:
        """
        Combine the results of the child layers of a linear (non-logarithmic) query
        whose results have the nodes in the last axis.

        :param child_results: The result per child layer, shape (..., #child nodes).
        :return: The result for the nodes of this layer, shape (..., #nodes).
        """
        values = self.values_of_edges(child_results) * self.normalized_edge_weights
        return self.log_weights.group_by_row(values, padding=0.0).sum(axis=-1)

    def _weighted_forward_over_nodes(
        self, child_results: List[NodeVariableValues]
    ) -> NodeVariableValues:
        """
        Combine the results of the child layers of a query whose results have the nodes
        in the first axis, such as the moments.

        :param child_results: The result per child layer, shape (#child nodes, ...).
        :return: The result for the nodes of this layer, shape (#nodes, ...).
        """
        values = self.values_of_edges([result.T for result in child_results])
        values = values * self.normalized_edge_weights
        return self.log_weights.group_by_row(values, padding=0.0).sum(axis=-1).T

    def log_weighted_sum(
        self, child_results: List[SampleNodeValues]
    ) -> SampleNodeValues:
        """
        Reduce the log-results of the child layers with the normalized log-weights.

        :param child_results: The log-results per child layer, with shape (..., #nodes
            of the child layer).
        :return: The log-result of the nodes of this layer with shape (..., #nodes).
        """
        values = self.values_of_edges(child_results) + self.log_weights.data
        grouped = self.log_weights.group_by_row(values)
        return embedded_logsumexp(grouped, axis=-1) - self.log_normalization_constants

    @memoized
    def log_likelihood_of_nodes(
        self, events: SampleArray, cache: Optional[QueryCache] = None
    ) -> SampleNodeValues:
        child_results = [
            child_layer.log_likelihood_of_nodes(events, cache=cache)
            for child_layer in self.child_layers
        ]
        return self.log_weighted_sum(child_results)

    @memoized
    def cumulative_distribution_of_nodes(
        self, events: SampleArray, cache: Optional[QueryCache] = None
    ) -> SampleNodeValues:
        child_results = [
            child_layer.cumulative_distribution_of_nodes(events, cache=cache)
            for child_layer in self.child_layers
        ]
        return self._weighted_forward(child_results)

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
        return self._weighted_forward(child_results).reshape(-1)

    @memoized
    def support_of_nodes(
        self, variables: SortedSet, cache: Optional[QueryCache] = None
    ) -> List[Event]:
        child_supports = [
            child_layer.support_of_nodes(variables, cache=cache)
            for child_layer in self.child_layers
        ]

        result: List[Optional[Event]] = [None] * self.number_of_nodes
        for edge in self.iterate_edges():
            support = child_supports[edge.child_layer_index][edge.child_node]
            if result[edge.node] is None:
                result[edge.node] = support
            else:
                result[edge.node] = result[edge.node] | support

        return [Event() if support is None else support for support in result]

    @memoized
    def log_mode_of_nodes(
        self, variables: SortedSet, cache: Optional[QueryCache] = None
    ) -> Tuple[List[Event], NodeValues]:
        child_modes = [
            child_layer.log_mode_of_nodes(variables, cache=cache)
            for child_layer in self.child_layers
        ]

        best_value = np.full(self.number_of_nodes, -np.inf)
        candidates: List[List[Event]] = [[] for _ in range(self.number_of_nodes)]

        for edge, log_weight in zip(
            self.iterate_edges(), self.normalized_edge_log_weights
        ):
            child_events, child_values = child_modes[edge.child_layer_index]
            value = log_weight + child_values[edge.child_node]
            mode = child_events[edge.child_node]
            if value > best_value[edge.node]:
                best_value[edge.node] = value
                candidates[edge.node] = [mode]
            elif value == best_value[edge.node]:
                candidates[edge.node].append(mode)

        modes = []
        for events in candidates:
            if not events:
                modes.append(Event())
                continue
            mode = events[0]
            for event in events[1:]:
                mode = mode | event
            modes.append(mode)

        return modes, best_value

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
        return self._weighted_forward_over_nodes(child_results)

    def sample_forward(
        self,
        assignment: ForwardSampleAssignment,
        samples: SampleArray,
        variables: SortedSet,
    ):
        # the padding slot gets a weight of zero, so it is never drawn
        weights = self.log_weights.pad(self.normalized_edge_weights, 0.0)
        for node, rows_of_node in enumerate(assignment.rows_of(self)):
            if not rows_of_node.is_empty:
                self.route_rows_of_node(node, rows_of_node.rows, weights, assignment)

    def route_rows_of_node(
        self,
        node: int,
        rows: SampleRows,
        weights: EdgeValues,
        assignment: ForwardSampleAssignment,
    ):
        """
        Split the sample rows of one node among its children, in proportion to the
        weights of its edges.

        :param node: The index of the node.
        :param rows: The sample rows assigned to the node.
        :param weights: The normalized weight of every edge, followed by the zero weight
            of the padding slot.
        :param assignment: The assignment to route the rows into.
        """
        positions = self.log_weights.gather[node]
        probabilities = weights[positions]

        # guard against the accumulated floating point error of the normalization
        total = probabilities.sum()
        if total <= 0:
            return
        counts = np.random.multinomial(len(rows), pvals=probabilities / total)

        # shuffle so that the contiguous chunks handed to the children are an unbiased
        # partition of the rows
        np.random.shuffle(rows)
        chunks = np.split(rows, np.cumsum(counts)[:-1])

        edges = self.inner_layer_edges
        for index in np.flatnonzero(counts):
            position = positions[index]
            assignment.assign(
                self.child_layers[edges.child_layer_indices[position]],
                edges.child_nodes[position],
                chunks[index],
            )

    def is_deterministic_of_nodes(
        self, variables: SortedSet, cache: QueryCache
    ) -> NodeMask:
        """
        A sum node is deterministic if its children have pairwise disjoint supports.
        """
        supports = [
            child_layer.support_of_nodes(variables, cache=cache)
            for child_layer in self.child_layers
        ]

        supports_per_node: List[List[Event]] = [[] for _ in range(self.number_of_nodes)]
        for edge in self.iterate_edges():
            supports_per_node[edge.node].append(
                supports[edge.child_layer_index][edge.child_node]
            )

        return np.array(
            [
                self.are_pairwise_disjoint(node_supports)
                for node_supports in supports_per_node
            ],
            dtype=bool,
        )

    @staticmethod
    def are_pairwise_disjoint(events: List[Event]) -> bool:
        """
        :param events: The events to compare.
        :return: Whether no two of the events intersect.
        """
        return all(
            event.intersection_with(other).is_empty()
            for index, event in enumerate(events)
            for other in events[index + 1 :]
        )

    def __deepcopy__(self, memo=None) -> SumLayer:
        if memo is None:
            memo = {}
        if id(self) in memo:
            return memo[id(self)]
        child_layers = [
            child_layer.__deepcopy__(memo) for child_layer in self.child_layers
        ]
        result = self.__class__(child_layers, self.log_weights.copy())
        memo[id(self)] = result
        return result

    # %% structural

    def _structural_pass(
        self,
        child_results: List[LayerWithLogProbabilities],
        query: StructuralQuery,
    ) -> LayerWithLogProbabilities:
        """
        Update the weights of this layer with the log-probabilities of its children.

        The new weight of an edge is its old weight times the probability of the event
        under the child, and the probability of a node is the sum of its new weights.

        :param child_results: The new child layers and the log-probabilities of their
            nodes.
        :param query: The query to record the result in.
        :return: The new layer and the log-probabilities of its nodes.
        """
        child_log_probabilities = self.values_of_edges(
            [child_result.log_probabilities for child_result in child_results]
        )
        result = self.__class__(
            [child_result.layer for child_result in child_results],
            self.log_weights.with_data(self.log_weights.data + child_log_probabilities),
        )
        # the probability of a node is the sum of its updated weights
        return query.log_probabilities.record(
            LayerWithLogProbabilities(result, result.log_normalization_constants)
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
        child_results = [
            child_layer.log_truncated_of_simple_events(events, query, cache=cache)
            for child_layer in self.child_layers
        ]

        # every child layer grows by the same factor, and the block of event k is the
        # original sparsity pattern shifted into its own rows and into the k-th block
        # of every child layer
        number_of_events = len(events)
        number_of_entries = self.log_weights.number_of_stored_entries
        edges = self.inner_layer_edges
        child_node_counts = np.diff(self.column_offsets)
        blocks = np.repeat(np.arange(number_of_events), number_of_entries)

        rows = np.tile(self.log_weights.rows, number_of_events) + (
            blocks * self.number_of_nodes
        )
        # the child layer i now starts at number_of_events * column_offsets[i], and its
        # block of event k starts k * (its original number of nodes) after that
        first_columns = (
            number_of_events * self.column_offsets[:-1][edges.child_layer_indices]
            + edges.child_nodes
        )
        columns = np.tile(first_columns, number_of_events) + blocks * np.tile(
            child_node_counts[edges.child_layer_indices], number_of_events
        )
        # the weight of an edge times the probability of the event under its child
        data = (
            np.tile(self.log_weights.data, number_of_events)
            + np.concatenate(
                [child_result.log_probabilities for child_result in child_results]
            )[columns]
        )

        result = self.__class__(
            [child_result.layer for child_result in child_results],
            RowGroupedSparseArray.from_entries(
                SparseEntries(data, rows, columns),
                (
                    number_of_events * self.number_of_nodes,
                    number_of_events * int(self.column_offsets[-1]),
                ),
            ),
        )
        return query.log_probabilities.record(
            LayerWithLogProbabilities(result, result.log_normalization_constants)
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

    def live_edges(
        self, alive: NodeMask, log_probabilities: LogProbabilitiesOfLayers
    ) -> EdgeMask:
        """
        Determine the edges that survive a prune.

        :param alive: The live nodes of this layer.
        :param log_probabilities: The log-probabilities of the structural query.
        :return: Which edges survive.
        """
        child_alive = [
            log_probabilities.alive_nodes_of(child_layer)
            for child_layer in self.child_layers
        ]
        return (
            alive[self.log_weights.rows]
            & (self.log_weights.data > -np.inf)
            & self.values_of_edges(child_alive)
        )

    def required_child_nodes(
        self, alive: NodeMask, log_probabilities: LogProbabilitiesOfLayers
    ) -> List[Tuple[Layer, NodeMask]]:
        live = self.live_edges(alive, log_probabilities)
        edges = self.inner_layer_edges
        result = []
        for child_layer_index, child_layer in enumerate(self.child_layers):
            needed = np.zeros(child_layer.number_of_nodes, dtype=bool)
            needed[
                edges.child_nodes[
                    live & (edges.child_layer_indices == child_layer_index)
                ]
            ] = True
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

        edges = self.inner_layer_edges
        kept = alive[self.log_weights.rows] & (self.log_weights.data > -np.inf)

        new_child_layers = []
        new_columns = np.full(self.log_weights.number_of_stored_entries, -1)
        for child_layer_index, child_layer in enumerate(self.child_layers):
            pruned_child = rebuilt.get(id(child_layer))
            if pruned_child is None:
                kept &= edges.child_layer_indices != child_layer_index
                continue
            child_needed = needed[id(child_layer)]
            of_child = edges.child_layer_indices == child_layer_index
            kept[of_child] &= child_needed[edges.child_nodes[of_child]]
            if not (kept & of_child).any():
                continue
            child_remap, number_of_child_nodes = remap_indices(child_needed)
            offset = sum(layer.number_of_nodes for layer in new_child_layers)
            new_columns[of_child] = offset + child_remap[edges.child_nodes[of_child]]
            new_child_layers.append(pruned_child)

        if not new_child_layers:
            return None

        node_remap, number_of_nodes = remap_indices(alive)
        return self.__class__(
            new_child_layers,
            RowGroupedSparseArray.from_entries(
                SparseEntries(
                    self.log_weights.data[kept],
                    node_remap[self.log_weights.rows[kept]],
                    new_columns[kept],
                ),
                (
                    number_of_nodes,
                    sum(layer.number_of_nodes for layer in new_child_layers),
                ),
            ),
        )

    @memoized
    def marginal(
        self, kept: VariableMask, cache: Optional[QueryCache] = None
    ) -> Optional[Layer]:
        edges = self.inner_layer_edges

        new_child_layers = []
        kept_edges = np.zeros(self.log_weights.number_of_stored_entries, dtype=bool)
        new_columns = np.full(self.log_weights.number_of_stored_entries, -1)
        for child_layer_index, child_layer in enumerate(self.child_layers):
            marginal_child = child_layer.marginal(kept, cache=cache)
            if marginal_child is None:
                continue
            of_child = edges.child_layer_indices == child_layer_index
            offset = sum(layer.number_of_nodes for layer in new_child_layers)
            new_columns[of_child] = offset + edges.child_nodes[of_child]
            kept_edges |= of_child
            new_child_layers.append(marginal_child)

        if not new_child_layers:
            return None
        return self.__class__(
            new_child_layers,
            RowGroupedSparseArray.from_entries(
                SparseEntries(
                    self.log_weights.data[kept_edges],
                    self.log_weights.rows[kept_edges],
                    new_columns[kept_edges],
                ),
                (
                    self.number_of_nodes,
                    sum(layer.number_of_nodes for layer in new_child_layers),
                ),
            ),
        )

    @memoized
    def simplify(self, cache: Optional[QueryCache] = None) -> Layer:
        simplified_children = [
            child_layer.simplify(cache=cache) for child_layer in self.child_layers
        ]
        result = self.__class__(simplified_children, self.log_weights.copy())

        if result.is_identity():
            return simplified_children[0]
        return result

    def is_identity(self) -> bool:
        """
        :return: Whether this layer passes its single child layer through unchanged, so
            that it can be removed without changing the distribution.
        """
        if len(self.child_layers) != 1:
            return False
        log_weights = self.log_weights
        if log_weights.shape[0] != log_weights.shape[1]:
            return False
        if log_weights.number_of_stored_entries != self.number_of_nodes:
            return False
        # every node points at the node with its own index, and at nothing else
        return bool(
            np.array_equal(log_weights.rows, log_weights.columns)
            and len(np.unique(log_weights.rows)) == self.number_of_nodes
        )
