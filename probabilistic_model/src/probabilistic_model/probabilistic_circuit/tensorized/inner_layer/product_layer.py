from __future__ import annotations

from dataclasses import (
    dataclass,
)

import numpy as np
import numpy.typing as npt
import tqdm
from random_events.product_algebra import Event, SimpleEvent
from random_events.variable import Variable
from sortedcontainers import SortedSet
from typing_extensions import (
    Any,
    Dict,
    Iterator,
    List,
    Optional,
    Tuple,
)

from probabilistic_model.exceptions import ShapeMismatchError
from probabilistic_model.probabilistic_circuit.tensorized.utils import (
    SparseArray,
    remap_indices,
)
from probabilistic_model.probabilistic_circuit.rx.probabilistic_circuit import (
    ProbabilisticCircuit as RustworkxProbabilisticCircuit,
    ProductUnit,
    Unit,
)
from probabilistic_model.probabilistic_circuit.tensorized.inner_layer.base import (
    Edge,
    ForwardSampleAssignment,
    InnerLayer,
    Layer,
    LayerConverter,
    LayerQuery,
    QueryCache,
    memoized,
)


@dataclass(eq=False, repr=False)
class ProductLayer(InnerLayer[ProductUnit]):
    """
    A layer of decomposable product units.

    Every node multiplies at most one node of each child layer, so the scope of a node
    is the union of the scopes of its child layers and its likelihood is the sum of
    their log-likelihoods.
    """

    edges: SparseArray
    """
    The edges as a sparse integer matrix of shape (#child layers, #nodes).

    The value of the entry ``(l, n)`` is the index of the node in the ``l``-th child
    layer that the ``n``-th node of this layer multiplies. A node of a child layer may
    be referenced by several nodes of this layer.
    """

    @property
    def number_of_nodes(self) -> int:
        return self.edges.shape[1]

    @property
    def number_of_components(self) -> int:
        return (
            sum(child_layer.number_of_components for child_layer in self.child_layers)
            + self.edges.number_of_stored_entries
        )

    @property
    def variables(self) -> npt.NDArray:
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

    def is_decomposable_own(self) -> bool:
        seen = set()
        for child_layer in self.child_layers:
            variables = set(child_layer.variables.tolist())
            if seen & variables:
                return False
            seen |= variables
        return True

    # %% queries

    def edges_of_child_layer(
        self, child_layer_index: int
    ) -> Tuple[npt.NDArray, npt.NDArray, bool]:
        """
        The edges into one child layer.

        :param child_layer_index: The index of the child layer.
        :return: The nodes of this layer, the nodes of the child layer they point to,
            and whether every node appears at most once. A decomposable product has at
            most one factor in each child layer, so the fast path is the normal one; the
            check keeps the reduction correct for a circuit that is not decomposable.
        """
        mask = self.edges.rows == child_layer_index
        nodes = self.edges.columns[mask]
        child_nodes = self.edges.data[mask].astype(np.int64)
        unique = len(np.unique(nodes)) == len(nodes)
        return nodes, child_nodes, unique

    def _gather_and_add(
        self, child_results: List[npt.NDArray], fill: float
    ) -> npt.NDArray:
        """
        Sum, per node, the results of the child nodes the edges point to.

        :param child_results: The result per child layer with the child nodes last.
        :param fill: The value of a node without any edge.
        :return: The summed result with the nodes of this layer last.
        """
        leading_shape = child_results[0].shape[:-1]
        result = np.zeros(leading_shape + (self.number_of_nodes,))
        touched = np.zeros(self.number_of_nodes, dtype=bool)

        for child_layer_index, child_result in enumerate(child_results):
            nodes, child_nodes, unique = self.edges_of_child_layer(child_layer_index)
            if len(nodes) == 0:
                continue
            gathered = child_result[..., child_nodes]
            if unique:
                result[..., nodes] += gathered
            else:
                np.add.at(result, (Ellipsis, nodes), gathered)
            touched[nodes] = True

        if not touched.all():
            result[..., ~touched] = fill
        return result

    @memoized(LayerQuery.LOG_LIKELIHOOD)
    def log_likelihood_of_nodes(
        self, x: npt.NDArray, cache: Optional[QueryCache] = None
    ) -> npt.NDArray:
        child_results = [
            child_layer.log_likelihood_of_nodes(x, cache=cache)
            for child_layer in self.child_layers
        ]
        return self._gather_and_add(child_results, fill=0.0)

    @memoized(LayerQuery.CUMULATIVE_DISTRIBUTION)
    def cumulative_distribution_of_nodes(
        self, x: npt.NDArray, cache: Optional[QueryCache] = None
    ) -> npt.NDArray:
        child_results = [
            child_layer.cumulative_distribution_of_nodes(x, cache=cache)
            for child_layer in self.child_layers
        ]
        return self._gather_and_multiply(child_results)

    def _gather_and_multiply(self, child_results: List[npt.NDArray]) -> npt.NDArray:
        leading_shape = child_results[0].shape[:-1]
        result = np.ones(leading_shape + (self.number_of_nodes,))
        for child_layer_index, child_result in enumerate(child_results):
            nodes, child_nodes, unique = self.edges_of_child_layer(child_layer_index)
            if len(nodes) == 0:
                continue
            gathered = child_result[..., child_nodes]
            if unique:
                result[..., nodes] *= gathered
            else:
                np.multiply.at(result, (Ellipsis, nodes), gathered)
        return result

    @memoized(LayerQuery.PROBABILITY_OF_SIMPLE_EVENT)
    def probability_of_simple_event_of_nodes(
        self,
        event: SimpleEvent,
        variables: SortedSet,
        cache: Optional[QueryCache] = None,
    ) -> npt.NDArray:
        child_results = [
            child_layer.probability_of_simple_event_of_nodes(
                event, variables, cache=cache
            ).reshape(1, -1)
            for child_layer in self.child_layers
        ]
        return self._gather_and_multiply(child_results).reshape(-1)

    @memoized(LayerQuery.SUPPORT)
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
            support = child_supports[edge.child_layer_index][
                edge.child_node
            ].__deepcopy__()
            if result[edge.node] is None:
                support.fill_missing_variables(own_variables)
                result[edge.node] = support
            else:
                result[edge.node] = result[edge.node] & support

        return [Event() if support is None else support for support in result]

    @memoized(LayerQuery.LOG_MODE)
    def log_mode_of_nodes(
        self, variables: SortedSet, cache: Optional[QueryCache] = None
    ) -> Tuple[List[Event], npt.NDArray]:
        child_modes = [
            child_layer.log_mode_of_nodes(variables, cache=cache)
            for child_layer in self.child_layers
        ]

        own_variables = {variables[index] for index in self.variables}
        events: List[Optional[Event]] = [None] * self.number_of_nodes
        values = np.zeros(self.number_of_nodes)

        for edge in self.iterate_edges():
            child_event = child_modes[edge.child_layer_index][0][
                edge.child_node
            ].__deepcopy__()
            values[edge.node] += child_modes[edge.child_layer_index][1][edge.child_node]
            if events[edge.node] is None:
                child_event.fill_missing_variables(own_variables)
                events[edge.node] = child_event
            else:
                events[edge.node] = events[edge.node].intersection_with(child_event)

        return [Event() if event is None else event for event in events], values

    def iterate_edges(self) -> Iterator[Edge]:
        for (child_layer_index, node), child_node in zip(
            self.edges.indices, self.edges.data
        ):
            yield Edge(int(node), int(child_layer_index), int(child_node))

    @memoized(LayerQuery.MOMENT)
    def moment_of_nodes(
        self,
        order: npt.NDArray,
        center: npt.NDArray,
        requested: npt.NDArray,
        variables: SortedSet,
        cache: Optional[QueryCache] = None,
    ) -> npt.NDArray:
        child_results = [
            child_layer.moment_of_nodes(
                order, center, requested, variables, cache=cache
            )
            for child_layer in self.child_layers
        ]
        # the moments of a decomposable product are the moments of the factor that owns
        # the variable, so summing the (zero padded) child moments is the right reduction
        result = np.zeros((self.number_of_nodes, len(order)))
        for child_layer_index, child_result in enumerate(child_results):
            nodes, child_nodes, unique = self.edges_of_child_layer(child_layer_index)
            if len(nodes) == 0:
                continue
            if unique:
                result[nodes] += child_result[child_nodes]
            else:
                np.add.at(result, nodes, child_result[child_nodes])
        return result

    def sample_forward(
        self,
        assignment: ForwardSampleAssignment,
        samples: npt.NDArray,
        variables: SortedSet,
    ):
        own_assignment = assignment.rows_of(self)

        rows_per_node = [
            np.concatenate(rows) if rows else None for rows in own_assignment
        ]

        for edge in self.iterate_edges():
            rows = rows_per_node[edge.node]
            if rows is None:
                continue
            child_layer = self.child_layers[edge.child_layer_index]
            assignment.assign(child_layer, edge.child_node, rows)

    # %% structural

    def _structural_pass(
        self,
        child_results: List[Tuple[Layer, npt.NDArray]],
        log_probabilities: Dict[int, npt.NDArray],
    ) -> Tuple[Layer, npt.NDArray]:
        """
        Accumulate the log-probabilities of the children of every node.

        :param child_results: The new child layer and its node log-probabilities.
        :param log_probabilities: The map to record the result in.
        :return: The new layer and the log-probabilities of its nodes.
        """
        result = self.__class__(
            [child_layer for child_layer, _ in child_results], self.edges.copy()
        )

        own_log_probabilities = np.zeros(self.number_of_nodes)
        for child_layer_index, (_, child_log_probabilities) in enumerate(child_results):
            nodes, child_nodes, unique = self.edges_of_child_layer(child_layer_index)
            if len(nodes) == 0:
                continue
            if unique:
                own_log_probabilities[nodes] += child_log_probabilities[child_nodes]
            else:
                np.add.at(
                    own_log_probabilities, nodes, child_log_probabilities[child_nodes]
                )

        log_probabilities[id(result)] = own_log_probabilities
        return result, own_log_probabilities

    def log_truncated_of_simple_event(
        self,
        event: SimpleEvent,
        variables: SortedSet,
        singleton_allowed: bool,
        cache: Optional[QueryCache] = None,
        log_probabilities: Optional[Dict[int, npt.NDArray]] = None,
    ) -> Tuple[Layer, npt.NDArray]:
        if cache is None:
            cache = QueryCache()
        if cache.has(LayerQuery.TRUNCATED, self):
            return cache.get(LayerQuery.TRUNCATED, self)

        child_results = [
            child_layer.log_truncated_of_simple_event(
                event,
                variables,
                singleton_allowed,
                cache=cache,
                log_probabilities=log_probabilities,
            )
            for child_layer in self.child_layers
        ]
        return cache.set(
            LayerQuery.TRUNCATED,
            self,
            self._structural_pass(child_results, log_probabilities),
        )

    def log_truncated_of_simple_events(
        self,
        events: List[SimpleEvent],
        variables: SortedSet,
        singleton_allowed: bool,
        cache: Optional[QueryCache] = None,
        log_probabilities: Optional[Dict[int, npt.NDArray]] = None,
    ) -> Optional[Tuple[Layer, npt.NDArray]]:
        if cache is None:
            cache = QueryCache()
        if cache.has(LayerQuery.BATCHED_TRUNCATED, self):
            return cache.get(LayerQuery.BATCHED_TRUNCATED, self)

        number_of_events = len(events)
        number_of_nodes = self.number_of_nodes
        number_of_entries = self.edges.number_of_stored_entries
        blocks = np.arange(number_of_events)

        new_child_layers = []
        child_log_probabilities = []
        for child_layer in self.child_layers:
            truncated_child = child_layer.log_truncated_of_simple_events(
                events,
                variables,
                singleton_allowed,
                cache=cache,
                log_probabilities=log_probabilities,
            )
            if truncated_child is None:
                return None
            new_child_layer, child_log_probability = truncated_child
            new_child_layers.append(new_child_layer)
            child_log_probabilities.append(child_log_probability)

        # every edge is repeated once per event, pointing into that event's block of the
        # child layer
        node_counts = np.array(
            [child_layer.number_of_nodes for child_layer in self.child_layers]
        )
        rows = np.tile(self.edges.rows, number_of_events)
        columns = np.tile(self.edges.columns, number_of_events) + np.repeat(
            blocks * number_of_nodes, number_of_entries
        )
        data = np.tile(self.edges.data.astype(np.int64), number_of_events) + np.repeat(
            blocks, number_of_entries
        ) * np.tile(node_counts[self.edges.rows], number_of_events)

        edges = SparseArray.from_coordinates(
            rows,
            columns,
            data,
            (len(self.child_layers), number_of_events * number_of_nodes),
        )
        result = self.__class__(new_child_layers, edges)

        own_log_probabilities = np.zeros(number_of_events * number_of_nodes)
        for child_layer_index, child_log_probability in enumerate(
            child_log_probabilities
        ):
            mask = rows == child_layer_index
            np.add.at(
                own_log_probabilities,
                columns[mask],
                child_log_probability[data[mask]],
            )
        log_probabilities[id(result)] = own_log_probabilities

        return cache.set(
            LayerQuery.BATCHED_TRUNCATED, self, (result, own_log_probabilities)
        )

    def log_conditional_of_point(
        self,
        point: Dict[Variable, Any],
        variables: SortedSet,
        cache: Optional[QueryCache] = None,
        log_probabilities: Optional[Dict[int, npt.NDArray]] = None,
    ) -> Tuple[Layer, npt.NDArray]:
        if cache is None:
            cache = QueryCache()
        if cache.has(LayerQuery.CONDITIONAL, self):
            return cache.get(LayerQuery.CONDITIONAL, self)

        child_results = [
            child_layer.log_conditional_of_point(
                point,
                variables,
                cache=cache,
                log_probabilities=log_probabilities,
            )
            for child_layer in self.child_layers
        ]
        return cache.set(
            LayerQuery.CONDITIONAL,
            self,
            self._structural_pass(child_results, log_probabilities),
        )

    def required_child_nodes(
        self, alive: npt.NDArray, log_probabilities: Dict[int, npt.NDArray]
    ) -> List[Tuple[Layer, npt.NDArray]]:
        kept_edges = alive[self.edges.columns]
        result = []
        for child_layer_index, child_layer in enumerate(self.child_layers):
            mask = kept_edges & (self.edges.rows == child_layer_index)
            needed = np.zeros(child_layer.number_of_nodes, dtype=bool)
            needed[self.edges.data[mask].astype(np.int64)] = True
            result.append((child_layer, needed))
        return result

    def rebuild(
        self,
        needed: Dict[int, npt.NDArray],
        rebuilt: Dict[int, Optional[Layer]],
    ) -> Optional[Layer]:
        alive = needed[id(self)]
        if not alive.any():
            return None

        node_remap, number_of_nodes = remap_indices(alive)
        kept_edges = alive[self.edges.columns]

        new_child_layers = []
        new_rows = []
        new_columns = []
        new_data = []

        for child_layer_index, child_layer in enumerate(self.child_layers):
            mask = kept_edges & (self.edges.rows == child_layer_index)
            if not mask.any():
                continue

            pruned_child = rebuilt.get(id(child_layer))
            if pruned_child is None:
                # a factor of the product became impossible, so every node that
                # references it is impossible as well
                return None

            child_remap, _ = remap_indices(needed[id(child_layer)])
            new_rows.append(np.full(mask.sum(), len(new_child_layers), dtype=np.int64))
            new_columns.append(node_remap[self.edges.columns[mask]])
            new_data.append(child_remap[self.edges.data[mask].astype(np.int64)])
            new_child_layers.append(pruned_child)

        if not new_child_layers:
            return None

        edges = SparseArray.from_coordinates(
            np.concatenate(new_rows),
            np.concatenate(new_columns),
            np.concatenate(new_data),
            (len(new_child_layers), number_of_nodes),
        )
        return self.__class__(new_child_layers, edges)

    def marginal(
        self, kept: npt.NDArray, cache: Optional[QueryCache] = None
    ) -> Optional[Layer]:
        if cache is None:
            cache = QueryCache()
        if cache.has(LayerQuery.MARGINAL, self):
            return cache.get(LayerQuery.MARGINAL, self)

        new_child_layers = []
        new_rows = []
        new_columns = []
        new_data = []

        for child_layer_index, child_layer in enumerate(self.child_layers):
            marginal_child = child_layer.marginal(kept, cache)
            if marginal_child is None:
                continue
            mask = self.edges.rows == child_layer_index
            new_rows.append(np.full(mask.sum(), len(new_child_layers), dtype=np.int64))
            new_columns.append(self.edges.columns[mask])
            new_data.append(self.edges.data[mask])
            new_child_layers.append(marginal_child)

        if not new_child_layers:
            return cache.set(LayerQuery.MARGINAL, self, None)

        edges = SparseArray.from_coordinates(
            np.concatenate(new_rows),
            np.concatenate(new_columns),
            np.concatenate(new_data),
            (len(new_child_layers), self.number_of_nodes),
        )
        result = self.__class__(new_child_layers, edges)
        return cache.set(LayerQuery.MARGINAL, self, result)

    def simplify(self, cache: Optional[QueryCache] = None) -> Layer:
        if cache is None:
            cache = QueryCache()
        if cache.has(LayerQuery.SIMPLIFY, self):
            return cache.get(LayerQuery.SIMPLIFY, self)

        simplified_children = [
            child_layer.simplify(cache) for child_layer in self.child_layers
        ]
        result = self.__class__(simplified_children, self.edges.copy())

        if result.is_identity():
            result = simplified_children[0]

        return cache.set(LayerQuery.SIMPLIFY, self, result)

    def is_identity(self) -> bool:
        """
        :return: Whether this layer forwards its single child layer unchanged.
        """
        if len(self.child_layers) != 1:
            return False
        if self.edges.number_of_stored_entries != self.number_of_nodes:
            return False
        if self.child_layers[0].number_of_nodes != self.number_of_nodes:
            return False
        sorted_edges = self.edges.sort_indices()
        expected = np.arange(self.number_of_nodes)
        return bool(
            np.array_equal(sorted_edges.columns, expected)
            and np.array_equal(sorted_edges.data.astype(np.int64), expected)
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

    # %% conversion

    @classmethod
    def create_layer_from_nodes_with_same_type_and_scope(
        cls,
        nodes: List[ProductUnit],
        child_layers: List[LayerConverter],
        progress_bar: bool = False,
    ) -> LayerConverter:
        hash_remap = {hash(node): index for index, node in enumerate(nodes)}

        # only the candidates that at least one of these nodes points to become child
        # layers, so that the edge matrix has no empty rows
        used_child_layers: List[LayerConverter] = []
        row_of_child_layer: Dict[int, int] = {}

        rows, columns, values = [], [], []

        iterator = (
            tqdm.tqdm(nodes, desc="Assembling product layer") if progress_bar else nodes
        )
        for node_index, node in enumerate(iterator):
            subcircuit_hashes = {hash(subcircuit) for subcircuit in node.subcircuits}
            for child_layer_index, child_layer in enumerate(child_layers):
                for subcircuit_hash in subcircuit_hashes:
                    if subcircuit_hash not in child_layer.hash_remap:
                        continue
                    if child_layer_index not in row_of_child_layer:
                        row_of_child_layer[child_layer_index] = len(used_child_layers)
                        used_child_layers.append(child_layer)
                    rows.append(row_of_child_layer[child_layer_index])
                    columns.append(node_index)
                    values.append(child_layer.hash_remap[subcircuit_hash])

        edges = SparseArray.from_coordinates(
            np.array(rows, dtype=np.int64),
            np.array(columns, dtype=np.int64),
            np.array(values, dtype=np.int64),
            (len(used_child_layers), len(nodes)),
        )
        layer = cls([cl.layer for cl in used_child_layers], edges)
        return LayerConverter(layer, nodes, hash_remap)

    def to_rustworkx(
        self,
        variables: SortedSet,
        result: RustworkxProbabilisticCircuit,
        cache: Optional[QueryCache] = None,
        progress_bar: Optional[tqdm.tqdm] = None,
    ) -> List[Unit]:
        if cache is None:
            cache = QueryCache()
        if cache.has(LayerQuery.TO_RUSTWORKX, self):
            return cache.get(LayerQuery.TO_RUSTWORKX, self)

        if progress_bar:
            progress_bar.set_postfix_str(
                f"Parsing product layer of {[variables[i] for i in self.variables]}"
            )

        units = [
            ProductUnit(probabilistic_circuit=result)
            for _ in range(self.number_of_nodes)
        ]
        child_units = [
            child_layer.to_rustworkx(variables, result, cache, progress_bar)
            for child_layer in self.child_layers
        ]

        for edge in self.iterate_edges():
            units[edge.node].add_subcircuit(
                child_units[edge.child_layer_index][edge.child_node]
            )
            if progress_bar:
                progress_bar.update()

        return cache.set(LayerQuery.TO_RUSTWORKX, self, units)
