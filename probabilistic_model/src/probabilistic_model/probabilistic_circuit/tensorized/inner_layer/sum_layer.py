"""
The layer of sum units of a layered probabilistic circuit.
"""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass

import numpy as np
import numpy.typing as npt
from random_events.product_algebra import Event, SimpleEvent
from sortedcontainers import SortedSet
from typing_extensions import Iterable, Self

from probabilistic_model.exceptions import ShapeMismatchError
from probabilistic_model.probabilistic_model import PartialPointType
from probabilistic_model.probabilistic_circuit.tensorized.array_types import (
    BackendArray,
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
from probabilistic_model.exceptions import NumberOfWeightsMismatchError
from probabilistic_model.probabilistic_circuit.tensorized.forward_sample_assignment import (
    ForwardSampleAssignment,
)
from probabilistic_model.probabilistic_circuit.tensorized.has_parameter_arrays import (
    HasParameterArrays,
)
from probabilistic_model.probabilistic_circuit.tensorized.inner_layer.base import (
    InnerLayer,
    Layer,
)
from probabilistic_model.probabilistic_circuit.tensorized.inner_layer.edge_matrix import (
    EdgeMatrix,
    RowReduction,
    SparseEdgeMatrix,
    SparseEntries,
)
from probabilistic_model.probabilistic_circuit.tensorized.inner_layer.inner_layer_edge import (
    InnerLayerEdges,
)
from probabilistic_model.probabilistic_circuit.tensorized.moment_query import (
    MomentQuery,
)
from probabilistic_model.probabilistic_circuit.tensorized.node_component_supports import (
    NodeComponentSupports,
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
from probabilistic_model.probabilistic_circuit.tensorized.utils import (
    remap_indices,
)


@dataclass(eq=False, repr=False)
class SumLayer(HasParameterArrays, InnerLayer):
    """
    A layer of sum units.

    All nodes of a sum layer have the same scope, which is the scope of its child
    layers.
    """

    log_weights: BackendArray
    """
    The logarithmic weight of every edge, laid out as :attr:`edges` stores values.

    It is the only array an array backend may replace.
    """

    @classmethod
    def from_entries(
        cls,
        child_layers: list[Layer],
        log_weights: SparseEntries,
        number_of_nodes: int,
        edge_matrix_type: type[EdgeMatrix] = SparseEdgeMatrix,
    ) -> Self:
        """
        :param child_layers: The child layers.
        :param log_weights: The logarithmic weight of every edge, at the node and the
            column of its child node.
        :param number_of_nodes: The number of nodes of the new layer.
        :param edge_matrix_type: The type of matrix to store the edges in.
        :return: The sum layer with these edges and weights.
        """
        number_of_child_nodes = sum(layer.number_of_nodes for layer in child_layers)
        edges = edge_matrix_type.from_edge_positions(
            log_weights.rows,
            log_weights.columns,
            (number_of_nodes, number_of_child_nodes),
        )
        return cls(child_layers, edges, edges.stored_values(log_weights, -np.inf))

    @classmethod
    def mixture_of(
        cls, child_layers: list[Layer], log_weights: Iterable[float]
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
        return cls.from_entries(
            child_layers,
            SparseEntries(
                weights, np.zeros(len(child_layers), dtype=np.int64), offsets[:-1]
            ),
            1,
        )

    @classmethod
    def from_edges(
        cls,
        child_layers: list[Layer],
        edges: InnerLayerEdges,
        log_weights: EdgeValues,
        number_of_nodes: int,
        edge_matrix_type: type[EdgeMatrix] = SparseEdgeMatrix,
    ) -> Self:
        """
        :param child_layers: The child layers.
        :param edges: The edges from the nodes of the new layer into the child layers.
        :param log_weights: The logarithmic weight of every edge.
        :param number_of_nodes: The number of nodes of the new layer.
        :param edge_matrix_type: The type of matrix to store the edges in.
        :return: The sum layer with these edges and weights.
        """
        offsets = np.concatenate(
            [
                [0],
                np.cumsum(
                    [child_layer.number_of_nodes for child_layer in child_layers]
                ),
            ]
        ).astype(np.int64)
        return cls.from_entries(
            child_layers,
            SparseEntries(
                log_weights,
                edges.nodes,
                offsets[edges.child_layer_indices] + edges.child_nodes,
            ),
            number_of_nodes,
            edge_matrix_type,
        )

    @classmethod
    def mixture_of_pieces(
        cls, pieces: list[LayerWithLogProbabilities]
    ) -> LayerWithLogProbabilities:
        """
        Mix the pieces a structural query splits every node of a layer into.

        Node ``i`` of the result mixes node ``i`` of every piece, weighted by the
        probability of that piece. Pieces of the same type are joined into one child
        layer, so the number of layers does not grow with the number of pieces.

        :param pieces: The pieces, each with as many nodes as the layer that was split,
            and the log-probabilities of their nodes.
        :return: The mixture and the log-probabilities of its nodes.
        """
        number_of_nodes = pieces[0].layer.number_of_nodes
        pieces_by_type: dict[type[Layer], list[LayerWithLogProbabilities]] = {}
        for piece in pieces:
            pieces_by_type.setdefault(type(piece.layer), []).append(piece)

        child_layers = []
        log_probabilities_per_child_layer = []
        for layer_type, typed_pieces in pieces_by_type.items():
            child_layers.append(
                layer_type.concatenate([piece.layer for piece in typed_pieces])
            )
            log_probabilities_per_child_layer.extend(
                piece.log_probabilities for piece in typed_pieces
            )

        # the pieces are the columns in order, and node i of every piece sits in row i
        number_of_pieces = len(pieces)
        log_weights = SparseEntries(
            np.concatenate(log_probabilities_per_child_layer),
            np.tile(np.arange(number_of_nodes), number_of_pieces),
            np.arange(number_of_pieces * number_of_nodes),
        )

        node_log_probabilities = np.logaddexp.reduce(
            [piece.log_probabilities for piece in pieces], axis=0
        )
        return LayerWithLogProbabilities(
            cls.from_entries(child_layers, log_weights, number_of_nodes),
            node_log_probabilities,
        )

    @property
    def variables(self) -> VariableIndices:
        if self._variables_cache is None:
            self._variables_cache = self.child_layers[0].variables
        return self._variables_cache

    @property
    def number_of_own_parameters(self) -> int:
        return self.edges.number_of_edges

    @property
    def parameters(self) -> list[BackendArray]:
        return [self.log_weights]

    def copy_with_parameters(self, parameters: list[BackendArray]) -> Self:
        (log_weights,) = parameters
        return dataclasses.replace(self, log_weights=log_weights)

    def validate_own(self):
        super().validate_own()
        if tuple(self.log_weights.shape) != self.edges.stored_shape:
            raise ShapeMismatchError(
                self.edges.stored_shape, tuple(self.log_weights.shape)
            )

    # %% edges

    def nodes_with_components(self) -> NodeMask:
        result = np.zeros(self.number_of_nodes, dtype=bool)
        result[self.edges.rows] = True
        return result

    def component_supports(
        self, variables: SortedSet, cache: QueryCache | None = None
    ) -> list[NodeComponentSupports]:
        child_supports = [
            child_layer.support_of_nodes(variables, cache=cache)
            for child_layer in self.child_layers
        ]
        result = [
            NodeComponentSupports(node, []) for node in range(self.number_of_nodes)
        ]
        for edge in self.iterate_edges():
            result[edge.node].supports.append(
                child_supports[edge.child_layer_index][edge.child_node]
            )
        return result

    def mixture_components(self) -> list[tuple[Layer, int]]:
        if self.number_of_nodes != 1:
            return []
        edges = self.inner_layer_edges
        return [
            (self.child_layers[child_layer_index], int(child_node))
            for child_layer_index, child_node in zip(
                edges.child_layer_indices, edges.child_nodes
            )
        ]

    def values_of_edges(self, child_results: list[npt.NDArray]) -> npt.NDArray:
        """
        Take the value of the child node of every edge.

        :param child_results: The result per child layer, child nodes last.
        :return: One value per edge, the edges last.
        """
        return np.concatenate(child_results, axis=-1)[..., self.edges.columns]

    def has_equal_edges(self, other: Self) -> bool:
        return super().has_equal_edges(other) and np.array_equal(
            self.edge_log_weights, other.edge_log_weights
        )

    def with_edges(
        self, child_layers: list[Layer], edges: InnerLayerEdges, copies: list[Self]
    ) -> Self:
        return self.from_edges(
            child_layers,
            edges,
            np.concatenate([copy.edge_log_weights for copy in copies]),
            len(copies) * self.number_of_nodes,
            type(self.edges),
        )

    def store_edges_as(self, edge_matrix_type: type[EdgeMatrix]):
        log_weights = SparseEntries(
            self.edge_log_weights, self.edges.rows, self.edges.columns
        )
        super().store_edges_as(edge_matrix_type)
        self.log_weights = self.edges.stored_values(log_weights, -np.inf)

    # %% weights

    @property
    def edge_log_weights(self) -> EdgeValues:
        """
        :return: The logarithmic weight of every edge, in the order of :attr:`edges`.
        """
        return self.edges.edge_values(self.log_weights)

    @property
    def log_normalization_constants(self) -> NodeValues:
        """
        :return: ``log(sum(exp(w)))`` over the weights of each node, shape (#nodes,).
        """
        return self.edges.reduce_by_row(self.log_weights, RowReduction.LOG_SUM_EXP)

    @property
    def normalized_log_weights(self) -> BackendArray:
        """
        :return: The logarithmic weights normalized per node, laid out as
            :attr:`log_weights`.
        """
        return self.log_weights - self.edges.values_of_rows(
            self.log_normalization_constants
        )

    @property
    def normalized_edge_log_weights(self) -> EdgeValues:
        """
        :return: The logarithmic weight of every edge, normalized per node.
        """
        return self.edges.edge_values(self.normalized_log_weights)

    @property
    def normalized_weights(self) -> BackendArray:
        """
        :return: The weights in linear space normalized per node, laid out as
            :attr:`log_weights`.
        """
        shifted = self.normalized_log_weights
        # a node whose weights are all -inf normalizes to nan; it is impossible, and the
        # prune pass removes it, so its weights are simply zero here
        return self.namespace.where(
            self.namespace.isfinite(shifted), self.namespace.exp(shifted), 0.0
        )

    def normalize_own(self):
        self.log_weights = self.normalized_log_weights

    # %% queries

    def _weighted_forward(
        self, child_results: list[SampleNodeValues]
    ) -> SampleNodeValues:
        """
        Combine the results of the child layers of a linear (non-logarithmic) query
        whose results have the nodes in the last axis.

        :param child_results: The result per child layer, shape (..., #child nodes).
        :return: The result for the nodes of this layer, shape (..., #nodes).
        """
        values = self.edges.values_of_columns(
            self.namespace.concat(child_results, axis=-1)
        )
        return self.edges.reduce_by_row(
            values * self.normalized_weights, RowReduction.SUM
        )

    def _weighted_forward_over_nodes(
        self, child_results: list[NodeVariableValues]
    ) -> NodeVariableValues:
        """
        Combine the results of the child layers of a query whose results have the nodes
        in the first axis, such as the moments.

        :param child_results: The result per child layer, shape (#child nodes, ...).
        :return: The result for the nodes of this layer, shape (#nodes, ...).
        """
        return self._weighted_forward([result.T for result in child_results]).T

    def log_weighted_sum(
        self, child_results: list[SampleNodeValues]
    ) -> SampleNodeValues:
        """
        Reduce the log-results of the child layers with the normalized log-weights.

        :param child_results: The log-results per child layer, with shape (..., #nodes
            of the child layer).
        :return: The log-result of the nodes of this layer with shape (..., #nodes).
        """
        values = self.edges.values_of_columns(
            self.namespace.concat(child_results, axis=-1)
        )
        return (
            self.edges.reduce_by_row(
                values + self.log_weights, RowReduction.LOG_SUM_EXP
            )
            - self.log_normalization_constants
        )

    @memoized
    def log_likelihood_of_nodes(
        self, events: SampleArray, cache: QueryCache | None = None
    ) -> SampleNodeValues:
        child_results = [
            child_layer.log_likelihood_of_nodes(events, cache=cache)
            for child_layer in self.child_layers
        ]
        return self.log_weighted_sum(child_results)

    @memoized
    def cumulative_distribution_of_nodes(
        self, events: SampleArray, cache: QueryCache | None = None
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
        cache: QueryCache | None = None,
    ) -> NodeValues:
        child_results = [
            self.namespace.reshape(
                child_layer.probability_of_simple_event_of_nodes(
                    event, variables, cache=cache
                ),
                (1, -1),
            )
            for child_layer in self.child_layers
        ]
        return self.namespace.reshape(self._weighted_forward(child_results), (-1,))

    @memoized
    def support_of_nodes(
        self, variables: SortedSet, cache: QueryCache | None = None
    ) -> list[Event]:
        child_supports = [
            child_layer.support_of_nodes(variables, cache=cache)
            for child_layer in self.child_layers
        ]

        result: list[Event | None] = [None] * self.number_of_nodes
        for edge in self.iterate_edges():
            support = child_supports[edge.child_layer_index][edge.child_node]
            if result[edge.node] is None:
                result[edge.node] = support
            else:
                result[edge.node] = result[edge.node] | support

        return [Event() if support is None else support for support in result]

    @memoized
    def log_mode_of_nodes(
        self, variables: SortedSet, cache: QueryCache | None = None
    ) -> tuple[list[Event], NodeValues]:
        child_modes = [
            child_layer.log_mode_of_nodes(variables, cache=cache)
            for child_layer in self.child_layers
        ]

        best_value = np.full(self.number_of_nodes, -np.inf)
        candidates: list[list[Event]] = [[] for _ in range(self.number_of_nodes)]

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
        cache: QueryCache | None = None,
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
        weights = self.edges.pad(
            np.asarray(self.edges.edge_values(self.normalized_weights)), 0.0
        )
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
        positions = self.edges.gather[node]
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

        supports_per_node: list[list[Event]] = [[] for _ in range(self.number_of_nodes)]
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
    def are_pairwise_disjoint(events: list[Event]) -> bool:
        """
        :param events: The events to compare.
        :return: Whether no two of the events intersect.
        """
        return all(
            event.intersection_with(other).is_empty()
            for index, event in enumerate(events)
            for other in events[index + 1 :]
        )

    # %% structural

    def _structural_pass(
        self,
        child_results: list[LayerWithLogProbabilities],
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
        log_weights = SparseEntries(
            self.edge_log_weights + child_log_probabilities,
            self.edges.rows,
            self.edges.columns,
        )
        result = self.__class__(
            [child_result.layer for child_result in child_results],
            self.edges,
            self.edges.stored_values(log_weights, -np.inf),
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
        child_results = [
            child_layer.log_truncated_of_simple_events(events, query, cache=cache)
            for child_layer in self.child_layers
        ]

        # every child layer grows by the same factor, and the block of event k is the
        # original sparsity pattern shifted into its own rows and into the k-th block
        # of every child layer
        number_of_events = len(events)
        number_of_entries = self.edges.number_of_edges
        edges = self.inner_layer_edges
        child_node_counts = np.diff(self.column_offsets)
        blocks = np.repeat(np.arange(number_of_events), number_of_entries)

        rows = np.tile(self.edges.rows, number_of_events) + (
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
            np.tile(self.edge_log_weights, number_of_events)
            + np.concatenate(
                [child_result.log_probabilities for child_result in child_results]
            )[columns]
        )

        result = self.from_entries(
            [child_result.layer for child_result in child_results],
            SparseEntries(data, rows, columns),
            number_of_events * self.number_of_nodes,
            type(self.edges),
        )
        return query.log_probabilities.record(
            LayerWithLogProbabilities(result, result.log_normalization_constants)
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
            alive[self.edges.rows]
            & (self.edge_log_weights > -np.inf)
            & self.values_of_edges(child_alive)
        )

    def required_child_nodes(
        self, alive: NodeMask, log_probabilities: LogProbabilitiesOfLayers
    ) -> list[tuple[Layer, NodeMask]]:
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
        needed: dict[int, NodeMask],
        rebuilt: dict[int, Layer | None],
    ) -> Layer | None:
        alive = needed[id(self)]
        if not alive.any():
            return None

        edges = self.inner_layer_edges
        edge_log_weights = self.edge_log_weights
        kept = alive[self.edges.rows] & (edge_log_weights > -np.inf)

        new_child_layers = []
        new_columns = np.full(self.edges.number_of_edges, -1)
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
        return self.from_entries(
            new_child_layers,
            SparseEntries(
                edge_log_weights[kept],
                node_remap[self.edges.rows[kept]],
                new_columns[kept],
            ),
            number_of_nodes,
            type(self.edges),
        )

    @memoized
    def marginal(
        self, kept: VariableMask, cache: QueryCache | None = None
    ) -> Layer | None:
        edges = self.inner_layer_edges

        new_child_layers = []
        kept_edges = np.zeros(self.edges.number_of_edges, dtype=bool)
        new_columns = np.full(self.edges.number_of_edges, -1)
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
        return self.from_entries(
            new_child_layers,
            SparseEntries(
                self.edge_log_weights[kept_edges],
                self.edges.rows[kept_edges],
                new_columns[kept_edges],
            ),
            self.number_of_nodes,
            type(self.edges),
        )

    @memoized
    def simplify(self, cache: QueryCache | None = None) -> Layer:
        simplified_children = [
            child_layer.simplify(cache=cache) for child_layer in self.child_layers
        ]
        result = self.__class__(
            simplified_children,
            self.edges,
            self.namespace.asarray(self.log_weights, copy=True),
        )

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
        if self.edges.shape[0] != self.edges.shape[1]:
            return False
        if self.edges.number_of_edges != self.number_of_nodes:
            return False
        # every node points at the node with its own index, and at nothing else
        return bool(
            np.array_equal(self.edges.rows, self.edges.columns)
            and len(np.unique(self.edges.rows)) == self.number_of_nodes
        )
