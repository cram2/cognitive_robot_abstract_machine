from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import numpy.typing as npt
import tqdm
from random_events.product_algebra import (
    SimpleEvent,
)
from random_events.variable import Variable
from sortedcontainers import SortedSet
from typing_extensions import (
    Any,
    Dict,
    Iterator,
    List,
    Optional,
    Self,
    Tuple,
)

from probabilistic_model.probabilistic_circuit.tensorized.utils import (
    SparseArray,
    embedded_logsumexp,
    remap_indices,
)
from probabilistic_model.probabilistic_circuit.rx.probabilistic_circuit import (
    ProbabilisticCircuit as RustworkxProbabilisticCircuit,
    SumUnit,
    Unit,
)
from probabilistic_model.probabilistic_circuit.tensorized.inner_layer.base import (
    ForwardSampleAssignment,
    Layer,
    LayerConverter,
)
from probabilistic_model.probabilistic_circuit.tensorized.inner_layer.sum_layer import (
    SumLayer,
)


@dataclass(eq=False, repr=False)
class SparseSumLayer(SumLayer):
    """
    A sum layer whose weights are stored sparsely.

    This is the layer that a circuit of the ``rx`` package is converted into: sum units
    there usually have few children, so the dense weight matrix would be mostly empty.
    """

    _edge_gather: Optional[npt.NDArray] = field(default=None, init=False, repr=False)
    """
    Cached index matrix of :attr:`edge_gather`.
    """

    _edges_are_contiguous: bool = field(default=False, init=False, repr=False)
    """
    Whether the edges are stored node by node with the same number of edges per node,
    filled together with :attr:`_edge_gather`.
    """

    _edge_targets: Optional[Tuple[npt.NDArray, npt.NDArray]] = field(
        default=None, init=False, repr=False
    )
    """
    Cached targets of :attr:`edge_targets`.
    """

    @property
    def number_of_nodes(self) -> int:
        return self.log_weights[0].shape[0]

    @property
    def number_of_own_parameters(self) -> int:
        return sum(
            log_weights.number_of_stored_entries for log_weights in self.log_weights
        )

    @property
    def number_of_components(self) -> int:
        return sum(
            child_layer.number_of_components for child_layer in self.child_layers
        ) + sum(
            log_weights.number_of_stored_entries for log_weights in self.log_weights
        )

    @property
    def concatenated_rows(self) -> npt.NDArray:
        """
        :return: The node of every edge, with the child layers concatenated in order.
        """
        return np.concatenate([log_weights.rows for log_weights in self.log_weights])

    @property
    def concatenated_edge_log_weights(self) -> npt.NDArray:
        """
        :return: The weight of every edge, with the child layers concatenated in order.
        """
        return np.concatenate([log_weights.data for log_weights in self.log_weights])

    @property
    def edge_gather(self) -> npt.NDArray:
        """
        The positions of the edges of every node, as a rectangular index matrix of shape
        (#nodes, largest number of edges of a node).

        Rows of nodes with fewer edges are padded with the position one past the last
        edge, which the queries fill with ``-inf``. Gathering with this matrix turns the
        per-node reduction over a ragged set of edges into one reduction over the last
        axis of a rectangular array, which is what keeps the likelihood of a whole batch
        of events a handful of numpy calls instead of a loop over nodes or events.
        """
        if self._edge_gather is None:
            rows = self.concatenated_rows
            number_of_edges = len(rows)
            counts = np.bincount(rows, minlength=self.number_of_nodes)
            width = max(int(counts.max()) if len(counts) else 0, 1)

            gather = np.full(
                (self.number_of_nodes, width), number_of_edges, dtype=np.int64
            )
            # the position of every edge inside the row of its node
            order = np.argsort(rows, kind="stable")
            sorted_rows = rows[order]
            offsets = np.arange(number_of_edges) - np.repeat(
                np.concatenate([[0], np.cumsum(counts)[:-1]]), counts
            )
            gather[sorted_rows, offsets] = order
            self._edge_gather = gather

            # when every node has the same number of edges and the edges are already
            # stored node by node, grouping them is a reshape rather than a gather
            self._edges_are_contiguous = bool(
                number_of_edges == self.number_of_nodes * width
                and np.array_equal(
                    gather, np.arange(number_of_edges).reshape(-1, width)
                )
            )
        return self._edge_gather

    @property
    def edges_are_contiguous(self) -> bool:
        """
        :return: Whether :meth:`group_edges_by_node` can reshape instead of gather.
        """
        self.edge_gather  # fills the flag along with the gather matrix
        return self._edges_are_contiguous

    def group_edges_by_node(
        self, values: npt.NDArray, padding: float = -np.inf
    ) -> npt.NDArray:
        """
        Rearrange per-edge values into one row per node.

        :param values: Per-edge values with the edges in the last axis.
        :param padding: The value for nodes with fewer edges than the widest one.
        :return: The values with shape ``(..., #nodes, edges per node)``.
        """
        if self.edges_are_contiguous:
            return values.reshape(values.shape[:-1] + (self.number_of_nodes, -1))
        return self.pad_edges(values, padding)[..., self.edge_gather]

    def pad_edges(self, values: npt.NDArray, padding: float) -> npt.NDArray:
        """
        Append the slot that :attr:`edge_gather` pads with.

        :param values: Per-edge values with the edges in the last axis.
        :param padding: The value of the padding slot. ``-inf`` is neutral for a
            logarithmic reduction, ``0`` for a linear one.
        :return: The values with one extra entry in the last axis.
        """
        return np.concatenate(
            [values, np.full(values.shape[:-1] + (1,), padding)], axis=-1
        )

    @property
    def log_normalization_constants(self) -> npt.NDArray:
        gathered = self.group_edges_by_node(self.concatenated_edge_log_weights)
        return embedded_logsumexp(gathered, axis=-1)

    @property
    def normalized_edge_weights(self) -> npt.NDArray:
        """
        :return: The weight of every edge in linear space, normalized per node.
        """
        normalization = self.log_normalization_constants
        rows = self.concatenated_rows
        shifted = self.concatenated_edge_log_weights - normalization[rows]
        # a node whose weights are all -inf normalizes to nan; it is impossible, and the
        # prune pass removes it, so its weights are simply zero here
        return np.where(np.isfinite(shifted), np.exp(shifted), 0.0)

    @property
    def edge_targets(self) -> Tuple[npt.NDArray, npt.NDArray]:
        """
        :return: The index of the child layer and the index of the node in it that every
            edge points to, in the order of the concatenated edges.
        """
        if self._edge_targets is None:
            self._edge_targets = (
                np.concatenate(
                    [
                        np.full(log_weights.number_of_stored_entries, index, np.int64)
                        for index, log_weights in enumerate(self.log_weights)
                    ]
                ),
                np.concatenate(
                    [log_weights.columns for log_weights in self.log_weights]
                ),
            )
        return self._edge_targets

    def _weighted_forward(self, child_results: List[npt.NDArray]) -> npt.NDArray:
        # the dense weight block of a layer with many nodes is mostly empty and can be
        # far larger than the circuit itself, so the linear queries reduce over the
        # stored edges instead of building it
        values = np.concatenate(
            [
                child_result[..., log_weights.columns]
                for log_weights, child_result in zip(self.log_weights, child_results)
            ],
            axis=-1,
        )
        values = values * self.normalized_edge_weights
        return self.group_edges_by_node(values, padding=0.0).sum(axis=-1)

    def _weighted_forward_over_nodes(
        self, child_results: List[npt.NDArray]
    ) -> npt.NDArray:
        values = np.concatenate(
            [
                child_result[log_weights.columns]
                for log_weights, child_result in zip(self.log_weights, child_results)
            ],
            axis=0,
        )
        values = values * self.normalized_edge_weights[:, None]
        padded = np.concatenate([values, np.zeros((1, values.shape[1]))], axis=0)
        return padded[self.edge_gather].sum(axis=1)

    def sample_forward(
        self,
        assignment: ForwardSampleAssignment,
        samples: npt.NDArray,
        variables: SortedSet,
    ):
        own_assignment = assignment.rows_of(self)
        gather = self.edge_gather
        # the padding slot gets a weight of zero, so it is never drawn
        weights = np.append(self.normalized_edge_weights, 0.0)
        child_layer_of_edge, child_node_of_edge = self.edge_targets

        for node, rows_of_node in enumerate(own_assignment):
            if not rows_of_node:
                continue
            rows = np.concatenate(rows_of_node)

            positions = gather[node]
            probabilities = weights[positions]

            # guard against the accumulated floating point error of the normalization
            total = probabilities.sum()
            if total <= 0:
                continue
            counts = np.random.multinomial(len(rows), pvals=probabilities / total)

            # shuffle so that the contiguous chunks handed to the children are an
            # unbiased partition of the rows
            np.random.shuffle(rows)

            offset = 0
            for count, position in zip(counts, positions):
                if not count:
                    continue
                child_layer = self.child_layers[child_layer_of_edge[position]]
                assignment.assign(
                    child_layer,
                    child_node_of_edge[position],
                    rows[offset : offset + count],
                )
                offset += count

    @property
    def normalized_weights(self) -> List[SparseArray]:
        normalization = self.log_normalization_constants
        result = []
        for log_weights in self.log_weights:
            normalized = log_weights.copy()
            with np.errstate(invalid="ignore"):
                normalized.data = np.exp(
                    normalized.data - normalization[normalized.rows]
                )
            normalized.data = np.nan_to_num(normalized.data, nan=0.0)
            result.append(normalized)
        return result

    def normalized_weights_per_child_layer(self) -> List[npt.NDArray]:
        return [weights.to_dense(0.0) for weights in self.normalized_weights]

    def normalized_log_weights_per_child_layer(self) -> List[npt.NDArray]:
        normalization = self.log_normalization_constants
        result = []
        for log_weights in self.log_weights:
            normalized = log_weights.copy()
            normalized.data = normalized.data - normalization[normalized.rows]
            result.append(normalized.to_dense(-np.inf))
        return result

    def edges(self) -> Iterator[Tuple[int, int, int]]:
        for child_layer_index, log_weights in enumerate(self.log_weights):
            for node, child_node in log_weights.indices:
                yield int(node), child_layer_index, int(child_node)

    def weighted_child_values(
        self, log_weights: SparseArray, child_result: npt.NDArray
    ) -> npt.NDArray:
        """
        Take the value of the child node of every edge and add the weight of that edge.

        :param log_weights: The weights of the edges into one child layer.
        :param child_result: The result of that child layer, child nodes last.
        :return: One value per edge, the edges last.
        """
        columns = log_weights.columns
        # a sum layer usually points at every node of its child layer exactly once and in
        # order, in which case the gather is an identity copy of an array that has one
        # entry per event per node, and skipping it is worth the comparison
        if len(columns) == child_result.shape[-1] and np.array_equal(
            columns, np.arange(len(columns))
        ):
            return child_result + log_weights.data
        return child_result[..., columns] + log_weights.data

    def log_weighted_sum(self, child_results: List[npt.NDArray]) -> npt.NDArray:
        values = np.concatenate(
            [
                self.weighted_child_values(log_weights, child_result)
                for log_weights, child_result in zip(self.log_weights, child_results)
            ],
            axis=-1,
        )
        gathered = self.group_edges_by_node(values)
        return embedded_logsumexp(gathered, axis=-1) - self.log_normalization_constants

    def normalize_own(self):
        normalization = self.log_normalization_constants
        for log_weights in self.log_weights:
            log_weights.data = log_weights.data - normalization[log_weights.rows]

    def __deepcopy__(self, memo=None) -> SparseSumLayer:
        if memo is None:
            memo = {}
        if id(self) in memo:
            return memo[id(self)]
        child_layers = [
            child_layer.__deepcopy__(memo) for child_layer in self.child_layers
        ]
        result = self.__class__(
            child_layers, [log_weights.copy() for log_weights in self.log_weights]
        )
        memo[id(self)] = result
        return result

    def to_json(self, **kwargs) -> Dict[str, Any]:
        result = super().to_json(**kwargs)
        result["log_weights"] = [
            log_weights.to_json() for log_weights in self.log_weights
        ]
        return result

    @classmethod
    def _from_json(cls, data: Dict[str, Any], **kwargs) -> Self:
        child_layers = [
            Layer.from_json(child_layer, **kwargs)
            for child_layer in data["child_layers"]
        ]
        log_weights = [
            SparseArray.from_json(log_weights) for log_weights in data["log_weights"]
        ]
        return cls(child_layers, log_weights)

    # %% structural

    def _structural_pass(
        self,
        child_results: List[Tuple[Layer, npt.NDArray]],
        log_probabilities: Dict[int, npt.NDArray],
    ) -> Tuple[Layer, npt.NDArray]:
        """
        Update the weights of this layer with the log-probabilities of its children.

        This is the layered equivalent of ``SumUnit.log_forward_conditioning``: the new
        weight of an edge is its old weight times the probability of the event under the
        child, and the probability of a node is the sum of its new weights.

        :param child_results: The new child layer and its node log-probabilities.
        :param log_probabilities: The map to record the result in.
        :return: The new layer and the log-probabilities of its nodes.
        """
        new_log_weights = []
        for log_weights, (_, child_log_probabilities) in zip(
            self.log_weights, child_results
        ):
            updated = log_weights.copy()
            updated.data = updated.data + child_log_probabilities[updated.columns]
            new_log_weights.append(updated)

        result = self.__class__(
            [child_layer for child_layer, _ in child_results], new_log_weights
        )
        # the probability of a node is the sum of its updated weights
        own_log_probabilities = result.log_normalization_constants
        log_probabilities[id(result)] = own_log_probabilities
        return result, own_log_probabilities

    def log_truncated_of_simple_event(
        self,
        event: SimpleEvent,
        variables: SortedSet,
        singleton_allowed: bool,
        cache: Optional[Dict] = None,
        log_probabilities: Optional[Dict[int, npt.NDArray]] = None,
    ) -> Tuple[Layer, npt.NDArray]:
        if cache is None:
            cache = {}
        key = ("truncated", id(self))
        if key in cache:
            return cache[key]

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
        result = self._structural_pass(child_results, log_probabilities)
        cache[key] = result
        return result

    def log_truncated_of_simple_events(
        self,
        events: List[SimpleEvent],
        variables: SortedSet,
        singleton_allowed: bool,
        cache: Optional[Dict] = None,
        log_probabilities: Optional[Dict[int, npt.NDArray]] = None,
    ) -> Optional[Tuple[Layer, npt.NDArray]]:
        if cache is None:
            cache = {}
        key = ("batched truncated", id(self))
        if key in cache:
            return cache[key]

        number_of_events = len(events)
        number_of_nodes = self.number_of_nodes

        new_child_layers = []
        new_log_weights = []
        for log_weights, child_layer in self.log_weighted_child_layers:
            truncated_child = child_layer.log_truncated_of_simple_events(
                events,
                variables,
                singleton_allowed,
                cache=cache,
                log_probabilities=log_probabilities,
            )
            if truncated_child is None:
                return None
            new_child_layer, child_log_probabilities = truncated_child
            new_child_layers.append(new_child_layer)

            # the block of event k is the original sparsity pattern shifted into its own
            # rows and columns
            number_of_entries = log_weights.number_of_stored_entries
            blocks = np.arange(number_of_events)
            rows = np.tile(log_weights.rows, number_of_events) + np.repeat(
                blocks * number_of_nodes, number_of_entries
            )
            columns = np.tile(log_weights.columns, number_of_events) + np.repeat(
                blocks * child_layer.number_of_nodes, number_of_entries
            )
            # the weight of an edge times the probability of the event under its child
            data = (
                np.tile(log_weights.data, number_of_events)
                + child_log_probabilities[columns]
            )

            new_log_weights.append(
                SparseArray.from_coordinates(
                    rows,
                    columns,
                    data,
                    (
                        number_of_events * number_of_nodes,
                        number_of_events * child_layer.number_of_nodes,
                    ),
                )
            )

        result = self.__class__(new_child_layers, new_log_weights)
        own_log_probabilities = result.log_normalization_constants
        log_probabilities[id(result)] = own_log_probabilities

        cache[key] = (result, own_log_probabilities)
        return cache[key]

    def log_conditional_of_point(
        self,
        point: Dict[Variable, Any],
        variables: SortedSet,
        cache: Optional[Dict] = None,
        log_probabilities: Optional[Dict[int, npt.NDArray]] = None,
    ) -> Tuple[Layer, npt.NDArray]:
        if cache is None:
            cache = {}
        key = ("conditional", id(self))
        if key in cache:
            return cache[key]

        child_results = [
            child_layer.log_conditional_of_point(
                point,
                variables,
                cache=cache,
                log_probabilities=log_probabilities,
            )
            for child_layer in self.child_layers
        ]
        result = self._structural_pass(child_results, log_probabilities)
        cache[key] = result
        return result

    def live_entries(
        self,
        alive: npt.NDArray,
        log_weights: SparseArray,
        child_layer: Layer,
        log_probabilities: Dict[int, npt.NDArray],
    ) -> npt.NDArray:
        """
        Determine the edges into one child layer that survive a prune.

        :param alive: The live nodes of this layer.
        :param log_weights: The weights of the edges into the child layer.
        :param child_layer: The child layer.
        :param log_probabilities: The per-layer log-probabilities of the structural
            pass.
        :return: A boolean mask over the stored weight entries.
        """
        mask = alive[log_weights.rows] & (log_weights.data > -np.inf)
        child_log_probabilities = log_probabilities.get(id(child_layer))
        if child_log_probabilities is not None:
            mask = mask & (child_log_probabilities[log_weights.columns] > -np.inf)
        return mask

    def required_child_nodes(
        self, alive: npt.NDArray, log_probabilities: Dict[int, npt.NDArray]
    ) -> List[Tuple[Layer, npt.NDArray]]:
        result = []
        for log_weights, child_layer in self.log_weighted_child_layers:
            mask = self.live_entries(alive, log_weights, child_layer, log_probabilities)
            needed = np.zeros(child_layer.number_of_nodes, dtype=bool)
            needed[log_weights.columns[mask]] = True
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

        new_child_layers = []
        new_log_weights = []
        for log_weights, child_layer in self.log_weighted_child_layers:
            pruned_child = rebuilt.get(id(child_layer))
            if pruned_child is None:
                continue
            child_needed = needed[id(child_layer)]
            mask = (
                alive[log_weights.rows]
                & (log_weights.data > -np.inf)
                & child_needed[log_weights.columns]
            )
            if not mask.any():
                continue
            child_remap, number_of_child_nodes = remap_indices(child_needed)
            new_child_layers.append(pruned_child)
            new_log_weights.append(
                SparseArray.from_coordinates(
                    node_remap[log_weights.rows[mask]],
                    child_remap[log_weights.columns[mask]],
                    log_weights.data[mask],
                    (number_of_nodes, number_of_child_nodes),
                )
            )

        if not new_child_layers:
            return None

        return self.__class__(new_child_layers, new_log_weights)

    def marginal(
        self, kept: npt.NDArray, cache: Optional[Dict] = None
    ) -> Optional[Layer]:
        if cache is None:
            cache = {}
        key = ("marginal", id(self))
        if key in cache:
            return cache[key]

        new_child_layers = []
        new_log_weights = []
        for log_weights, child_layer in self.log_weighted_child_layers:
            marginal_child = child_layer.marginal(kept, cache)
            if marginal_child is None:
                continue
            new_child_layers.append(marginal_child)
            new_log_weights.append(log_weights.copy())

        result = (
            None
            if not new_child_layers
            else self.__class__(new_child_layers, new_log_weights)
        )
        cache[key] = result
        return result

    def simplify(self, cache: Optional[Dict] = None) -> Layer:
        if cache is None:
            cache = {}
        key = ("simplify", id(self))
        if key in cache:
            return cache[key]

        # placed before the recursion so that a cycle-free DAG with shared layers
        # resolves to the same object for every parent
        simplified_children = [
            child_layer.simplify(cache) for child_layer in self.child_layers
        ]
        result = self.__class__(
            simplified_children,
            [log_weights.copy() for log_weights in self.log_weights],
        )

        if result.is_identity():
            result = simplified_children[0]

        cache[key] = result
        return result

    def is_identity(self) -> bool:
        """
        :return: Whether this layer passes its single child layer through unchanged, so
            that it can be removed without changing the distribution.
        """
        if len(self.log_weights) != 1:
            return False
        log_weights = self.log_weights[0]
        if log_weights.shape[0] != log_weights.shape[1]:
            return False
        if log_weights.number_of_stored_entries != self.number_of_nodes:
            return False
        sorted_weights = log_weights.sort_indices()
        expected = np.arange(self.number_of_nodes)
        return bool(
            np.array_equal(sorted_weights.rows, expected)
            and np.array_equal(sorted_weights.columns, expected)
        )

    # %% conversion

    @classmethod
    def create_layer_from_nodes_with_same_type_and_scope(
        cls,
        nodes: List[SumUnit],
        child_layers: List[LayerConverter],
        progress_bar: bool = False,
    ) -> LayerConverter:
        hash_remap = {hash(node): index for index, node in enumerate(nodes)}
        variables = np.array(
            [
                nodes[0].probabilistic_circuit.variables.index(variable)
                for variable in nodes[0].variables
            ]
        )

        # only the child layers with the same scope can be children of these sum units
        filtered_child_layers = [
            child_layer
            for child_layer in child_layers
            if np.array_equal(child_layer.layer.variables, variables)
        ]

        used_child_layers = []
        log_weights = []
        for child_layer in filtered_child_layers:
            rows, columns, values = [], [], []
            for index, node in enumerate(
                tqdm.tqdm(nodes, desc="Assembling sum layer") if progress_bar else nodes
            ):
                for log_weight, subcircuit in node.log_weighted_subcircuits:
                    if hash(subcircuit) in child_layer.hash_remap:
                        rows.append(index)
                        columns.append(child_layer.hash_remap[hash(subcircuit)])
                        values.append(log_weight)

            # a candidate that none of these nodes points to is not a child layer
            if not rows:
                continue

            used_child_layers.append(child_layer.layer)
            log_weights.append(
                SparseArray.from_coordinates(
                    np.array(rows, dtype=np.int64),
                    np.array(columns, dtype=np.int64),
                    np.array(values, dtype=float),
                    (len(nodes), child_layer.layer.number_of_nodes),
                )
            )

        layer = cls(used_child_layers, log_weights)
        return LayerConverter(layer, nodes, hash_remap)

    def to_rustworkx(
        self,
        variables: SortedSet,
        result: RustworkxProbabilisticCircuit,
        cache: Optional[Dict] = None,
        progress_bar: Optional[tqdm.tqdm] = None,
    ) -> List[Unit]:
        if cache is None:
            cache = {}
        if id(self) in cache:
            return cache[id(self)]

        if progress_bar:
            progress_bar.set_postfix_str(
                f"Parsing sum layer of {[variables[i] for i in self.variables]}"
            )

        units = [
            SumUnit(probabilistic_circuit=result) for _ in range(self.number_of_nodes)
        ]
        child_units = [
            child_layer.to_rustworkx(variables, result, cache, progress_bar)
            for child_layer in self.child_layers
        ]

        for log_weights, child_layer_units in zip(self.log_weights, child_units):
            for (node, child_node), log_weight in zip(
                log_weights.indices, log_weights.data
            ):
                units[node].add_subcircuit(
                    child_layer_units[child_node], float(log_weight)
                )
                if progress_bar:
                    progress_bar.update()

        for unit in units:
            unit.normalize()

        cache[id(self)] = units
        return units
