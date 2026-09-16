from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import (
    dataclass,
)

import numpy as np
import numpy.typing as npt
from random_events.product_algebra import Event, SimpleEvent
from sortedcontainers import SortedSet
from typing_extensions import (
    Dict,
    Iterator,
    List,
    Optional,
    Tuple,
)

from probabilistic_model.exceptions import ShapeMismatchError
from probabilistic_model.probabilistic_circuit.tensorized.utils import (
    SparseArray,
)
from probabilistic_model.probabilistic_circuit.rx.probabilistic_circuit import (
    SumUnit,
)
from probabilistic_model.probabilistic_circuit.tensorized.inner_layer.base import (
    InnerLayer,
    Layer,
    memoized,
)


@dataclass(eq=False, repr=False)
class SumLayer(InnerLayer[SumUnit], ABC):
    """
    Abstract base class for layers of sum units.

    The weights of all sum units of a layer are grouped per child layer: the ``i``-th
    entry of :attr:`log_weights` holds, for every node of this layer, the logarithmic
    weights of the edges into the ``i``-th child layer. All nodes of a sum layer have the
    same scope, which is the scope of the child layers.
    """

    log_weights: List[SparseArray]

    @property
    def variables(self) -> npt.NDArray:
        if self._variables_cache is None:
            self._variables_cache = self.child_layers[0].variables
        return self._variables_cache

    @property
    def log_weighted_child_layers(self) -> Iterator[Tuple[SparseArray, Layer]]:
        """
        :return: The log-weights and the child layers, zipped together.
        """
        return zip(self.log_weights, self.child_layers)

    @property
    @abstractmethod
    def log_normalization_constants(self) -> npt.NDArray:
        """
        :return: ``log(sum(exp(w)))`` over the weights of each node, shape (#nodes,).
        """
        raise NotImplementedError

    @property
    @abstractmethod
    def normalized_weights(self) -> List[SparseArray]:
        """
        :return: The weights of each node in linear space, normalized to sum to one.
        """
        raise NotImplementedError

    def validate_own(self):
        for log_weights in self.log_weights:
            if log_weights.shape[0] != self.number_of_nodes:
                raise ShapeMismatchError(self.number_of_nodes, log_weights.shape[0])

        for log_weights, child_layer in self.log_weighted_child_layers:
            if log_weights.shape[1] != child_layer.number_of_nodes:
                raise ShapeMismatchError(
                    child_layer.number_of_nodes, log_weights.shape[1]
                )

    # %% queries

    def _weighted_forward(self, child_results: List[npt.NDArray]) -> npt.NDArray:
        """
        Combine the results of the child layers of a linear (non-logarithmic) query
        whose results have the nodes in the last axis.

        :param child_results: The result per child layer, shape (..., #child nodes).
        :return: The result for the nodes of this layer, shape (..., #nodes).
        """
        weights = self.normalized_weights_per_child_layer()
        result = None
        for weight, child_result in zip(weights, child_results):
            contribution = child_result @ weight.T
            result = contribution if result is None else result + contribution
        return result

    def _weighted_forward_over_nodes(
        self, child_results: List[npt.NDArray]
    ) -> npt.NDArray:
        """
        Combine the results of the child layers of a query whose results have the nodes
        in the first axis, such as the moments.

        :param child_results: The result per child layer, shape (#child nodes, ...).
        :return: The result for the nodes of this layer, shape (#nodes, ...).
        """
        weights = self.normalized_weights_per_child_layer()
        result = None
        for weight, child_result in zip(weights, child_results):
            contribution = weight @ child_result
            result = contribution if result is None else result + contribution
        return result

    @abstractmethod
    def normalized_weights_per_child_layer(self) -> List[npt.NDArray]:
        """
        :return: The dense, normalized weight block per child layer with shape
            (#nodes, #nodes of the child layer).
        """
        raise NotImplementedError

    @memoized("log_likelihood")
    def log_likelihood_of_nodes(
        self, x: npt.NDArray, cache: Optional[Dict] = None
    ) -> npt.NDArray:
        child_results = [
            child_layer.log_likelihood_of_nodes(x, cache=cache)
            for child_layer in self.child_layers
        ]
        return self.log_weighted_sum(child_results)

    @abstractmethod
    def log_weighted_sum(self, child_results: List[npt.NDArray]) -> npt.NDArray:
        """
        Reduce the log-results of the child layers with the normalized log-weights.

        :param child_results: The log-results per child layer, with shape (..., #nodes
            of the child layer).
        :return: The log-result of the nodes of this layer with shape (..., #nodes).
        """
        raise NotImplementedError

    @memoized("cumulative_distribution")
    def cumulative_distribution_of_nodes(
        self, x: npt.NDArray, cache: Optional[Dict] = None
    ) -> npt.NDArray:
        child_results = [
            child_layer.cumulative_distribution_of_nodes(x, cache=cache)
            for child_layer in self.child_layers
        ]
        return self._weighted_forward(child_results)

    @memoized("probability_of_simple_event")
    def probability_of_simple_event_of_nodes(
        self,
        event: SimpleEvent,
        variables: SortedSet,
        cache: Optional[Dict] = None,
    ) -> npt.NDArray:
        child_results = [
            child_layer.probability_of_simple_event_of_nodes(
                event, variables, cache=cache
            ).reshape(1, -1)
            for child_layer in self.child_layers
        ]
        return self._weighted_forward(child_results).reshape(-1)

    @memoized("support")
    def support_of_nodes(
        self, variables: SortedSet, cache: Optional[Dict] = None
    ) -> List[Event]:
        child_supports = [
            child_layer.support_of_nodes(variables, cache=cache)
            for child_layer in self.child_layers
        ]

        result: List[Optional[Event]] = [None] * self.number_of_nodes
        for node, child_layer_index, child_node in self.edges():
            support = child_supports[child_layer_index][child_node]
            if result[node] is None:
                result[node] = support.__deepcopy__()
            else:
                result[node] = result[node] | support.__deepcopy__()

        return [Event() if support is None else support for support in result]

    @memoized("log_mode")
    def log_mode_of_nodes(
        self, variables: SortedSet, cache: Optional[Dict] = None
    ) -> Tuple[List[Event], npt.NDArray]:
        child_modes = [
            child_layer.log_mode_of_nodes(variables, cache=cache)
            for child_layer in self.child_layers
        ]

        log_weights = self.normalized_log_weights_per_child_layer()

        best_value = np.full(self.number_of_nodes, -np.inf)
        candidates: List[List[Event]] = [[] for _ in range(self.number_of_nodes)]

        for node, child_layer_index, child_node in self.edges():
            log_weight = log_weights[child_layer_index][node, child_node]
            value = log_weight + child_modes[child_layer_index][1][child_node]
            mode = child_modes[child_layer_index][0][child_node]
            if value > best_value[node]:
                best_value[node] = value
                candidates[node] = [mode]
            elif value == best_value[node]:
                candidates[node].append(mode)

        modes = []
        for events in candidates:
            if not events:
                modes.append(Event())
                continue
            mode = events[0].__deepcopy__()
            for event in events[1:]:
                mode |= event.__deepcopy__()
            modes.append(mode)

        return modes, best_value

    @memoized("moment")
    def moment_of_nodes(
        self,
        order: npt.NDArray,
        center: npt.NDArray,
        requested: npt.NDArray,
        variables: SortedSet,
        cache: Optional[Dict] = None,
    ) -> npt.NDArray:
        child_results = [
            child_layer.moment_of_nodes(
                order, center, requested, variables, cache=cache
            )
            for child_layer in self.child_layers
        ]
        return self._weighted_forward_over_nodes(child_results)

    # %% structural

    @abstractmethod
    def edges(self) -> Iterator[Tuple[int, int, int]]:
        """
        :return: Yields ``(node, child layer index, child node)`` for every edge of this
            layer.
        """
        raise NotImplementedError

    @abstractmethod
    def normalized_log_weights_per_child_layer(self) -> List[npt.NDArray]:
        """
        :return: The dense, normalized log-weight block per child layer.
        """
        raise NotImplementedError

    def is_deterministic_own(self, variables: SortedSet, cache: Dict) -> bool:
        """
        Check whether every node of this sum layer has children with pairwise disjoint
        supports.

        :param variables: The variables of the circuit.
        :param cache: The shared cache of the supports computed so far.
        :return: Whether all nodes are deterministic.
        """
        supports = [
            child_layer.support_of_nodes(variables, cache=cache)
            for child_layer in self.child_layers
        ]

        supports_per_node: List[List[Event]] = [[] for _ in range(self.number_of_nodes)]
        for node, child_layer_index, child_node in self.edges():
            supports_per_node[node].append(supports[child_layer_index][child_node])

        for node_supports in supports_per_node:
            for index, support in enumerate(node_supports):
                for other in node_supports[index + 1 :]:
                    if not support.intersection_with(other).is_empty():
                        return False
        return True
