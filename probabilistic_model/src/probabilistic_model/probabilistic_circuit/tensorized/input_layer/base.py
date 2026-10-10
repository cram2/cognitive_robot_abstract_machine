from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass

import numpy as np
from random_events.interval import Interval
from random_events.product_algebra import Event, SimpleEvent, VariableMap
from random_events.sigma_algebra import AbstractCompositeSet
from random_events.variable import Variable
from sortedcontainers import SortedSet
from typing_extensions import (
    Any,
    Dict,
    List,
    Optional,
    Self,
    Tuple,
    Type,
)

from probabilistic_model.distributions.distributions import UnivariateDistribution
from probabilistic_model.probabilistic_model import PartialPointType
from probabilistic_model.probabilistic_circuit.tensorized.array_types import (
    NodeIndices,
    NodeMask,
    NodeValues,
    NodeVariableValues,
    SampleArray,
    SampleColumn,
    SampleNodeValues,
    VariableIndices,
    VariableMask,
)
from probabilistic_model.probabilistic_circuit.tensorized.forward_sample_assignment import (
    ForwardSampleAssignment,
)
from probabilistic_model.probabilistic_circuit.tensorized.inner_layer.base import Layer
from probabilistic_model.probabilistic_circuit.tensorized.moment_query import (
    MomentQuery,
)
from probabilistic_model.probabilistic_circuit.tensorized.stacked_copies import (
    StackedLayer,
)
from probabilistic_model.probabilistic_circuit.tensorized.query_cache import (
    QueryCache,
    memoized,
)
from probabilistic_model.probabilistic_circuit.tensorized.structural_query import (
    LayerWithLogProbabilities,
    StructuralQuery,
)


@dataclass(eq=False, repr=False)
class InputLayer(Layer, ABC):
    """
    Abstract base class for the input layers of a layered circuit.

    An input layer holds univariate distributions of one single type over one single
    variable, so that the likelihood of all of its nodes is evaluated without any
    branching.
    """

    variable: int
    """
    The index of the variable of this layer.
    """

    @property
    def child_layers(self) -> List[Layer]:
        """
        :return: An empty list. An input layer is a leaf of the layer graph.
        """
        return []

    @property
    def variables(self) -> VariableIndices:
        return np.array([self.variable], dtype=np.int64)

    @memoized
    def remap_variables(
        self, remap: VariableIndices, cache: Optional[QueryCache] = None
    ):
        self.variable = int(remap[self.variable])

    def column_of(self, events: SampleArray) -> SampleColumn:
        """
        Select the column of the variable of this layer from an event array.

        :param events: The events with shape (#events, #variables of the circuit).
        :return: The column of this layer's variable.
        """
        return events[:, self.variable]

    # %% per node view

    @abstractmethod
    def node_distribution(
        self, index: int, variable: Variable
    ) -> UnivariateDistribution:
        """
        Materialize one node of this layer as a univariate distribution.

        :param index: The index of the node.
        :param variable: The variable of this layer.
        :return: The distribution of that node.
        """
        raise NotImplementedError

    @classmethod
    @abstractmethod
    def from_distributions(
        cls, variable_index: int, distributions: List[UnivariateDistribution]
    ) -> Self:
        """
        Create a layer from a list of distributions of the type of this layer.

        :param variable_index: The index of the variable of the distributions.
        :param distributions: The distributions.
        :return: The layer.
        """
        raise NotImplementedError

    @abstractmethod
    def select_nodes(self, mask: NodeMask) -> Self:
        """
        Create a layer that only holds the nodes selected by a mask.

        :param mask: A boolean mask over the nodes of this layer.
        :return: The reduced layer.
        """
        raise NotImplementedError

    @classmethod
    def stacked(
        cls, copies: List[Self], stacked_child_layers: List[StackedLayer]
    ) -> StackedLayer:
        return StackedLayer.of_input_layer_copies(copies)

    @classmethod
    @abstractmethod
    def concatenate(cls, layers: List[Self]) -> Self:
        """
        Join layers of this type over the same variable into one layer.

        The nodes keep the order of the layers, so the nodes of ``layers[k]`` occupy one
        contiguous block. Only layers that were truncated from the same layer are
        concatenated, which is why the shared parameters may be taken from the first one.

        :param layers: The layers to join.
        :return: The joined layer.
        """
        raise NotImplementedError

    def node_distributions(self, variable: Variable) -> List[UnivariateDistribution]:
        """
        :param variable: The variable of this layer.
        :return: Every node of this layer as a univariate distribution.
        """
        return [
            self.node_distribution(index, variable)
            for index in range(self.number_of_nodes)
        ]

    # %% queries

    @memoized
    def support_of_nodes(
        self, variables: SortedSet, cache: Optional[QueryCache] = None
    ) -> List[Event]:
        variable = variables[self.variable]
        return [
            distribution.support for distribution in self.node_distributions(variable)
        ]

    @memoized
    def log_mode_of_nodes(
        self, variables: SortedSet, cache: Optional[QueryCache] = None
    ) -> Tuple[List[Event], NodeValues]:
        variable = variables[self.variable]
        modes = [
            distribution.log_mode()
            for distribution in self.node_distributions(variable)
        ]
        return [mode for mode, _ in modes], np.array(
            [value for _, value in modes], dtype=float
        )

    @memoized
    def moment_of_nodes(
        self,
        query: MomentQuery,
        variables: SortedSet,
        cache: Optional[QueryCache] = None,
    ) -> NodeVariableValues:
        result = np.zeros((self.number_of_nodes, query.number_of_variables))
        if not query.requested[self.variable]:
            return result
        result[:, self.variable] = self.moment_of_nodes_own(
            int(query.order[self.variable]),
            float(query.center[self.variable]),
            variables[self.variable],
        )
        return result

    def moment_of_nodes_own(
        self, order: int, center: float, variable: Variable
    ) -> NodeValues:
        """
        Calculate the moment of the variable of this layer for every node.

        The fallback evaluates the nodes one by one through their distributions. Layers
        whose moment has a closed form that numpy can evaluate for all nodes at once
        override this.

        :param order: The order of the moment.
        :param center: The center of the moment.
        :param variable: The variable of this layer.
        :return: The moments with shape (#nodes,).
        """
        order_map = VariableMap({variable: order})
        center_map = VariableMap({variable: center})
        return np.array(
            [
                distribution.moment(order_map, center_map)[variable]
                for distribution in self.node_distributions(variable)
            ],
            dtype=float,
        )

    def sample_forward(
        self,
        assignment: ForwardSampleAssignment,
        samples: SampleArray,
        variables: SortedSet,
    ):
        rows, nodes = [], []
        for node, rows_of_node in enumerate(assignment.rows_of(self)):
            if rows_of_node.is_empty:
                continue
            rows.append(rows_of_node.rows)
            nodes.append(np.full(len(rows_of_node.rows), node))
        if not rows:
            return
        samples[np.concatenate(rows), self.variable] = self.sample_of_nodes(
            np.concatenate(nodes), variables
        )

    def sample_of_nodes(self, nodes: NodeIndices, variables: SortedSet) -> SampleColumn:
        """
        Draw one sample from the node of every entry of ``nodes``.

        The fallback draws from the nodes one by one through :meth:`sample_of_node`.
        Layers whose distributions numpy or scipy can sample for all nodes at once
        override this.

        :param nodes: The node to draw each sample from.
        :param variables: The variables of the circuit.
        :return: The samples with shape (#entries of ``nodes``,).
        """
        result = np.empty(len(nodes))
        for node in np.unique(nodes):
            of_node = nodes == node
            result[of_node] = self.sample_of_node(
                int(node), int(of_node.sum()), variables
            )
        return result

    def sample_of_node(
        self, node: int, amount: int, variables: SortedSet
    ) -> SampleColumn:
        """
        Draw samples from a single node of this layer.

        :param node: The index of the node.
        :param amount: The number of samples.
        :param variables: The variables of the circuit.
        :return: The samples with shape (amount,).
        """
        distribution = self.node_distribution(node, variables[self.variable])
        return distribution.sample(amount)[:, 0]

    # %% structural

    @abstractmethod
    def log_truncated_of_assignment(
        self, assignment: AbstractCompositeSet, singleton_allowed: bool
    ) -> LayerWithLogProbabilities:
        """
        Truncate every node of this layer to the assignment of its variable at once.

        :param assignment: The assignment of the variable of this layer.
        :param singleton_allowed: Whether singletons are allowed.
        :return: The truncated layer, with as many nodes as this one, and the
            log-probabilities of its nodes.
        """
        raise NotImplementedError

    @abstractmethod
    def type_of_truncated_layer(
        self, assignment: AbstractCompositeSet, singleton_allowed: bool
    ) -> Type[Layer]:
        """
        :param assignment: The assignment of the variable of this layer.
        :param singleton_allowed: Whether singletons are allowed.
        :return: The type of the layer :meth:`log_truncated_of_assignment` returns for
            that assignment.
        """
        raise NotImplementedError

    @memoized
    def log_truncated_of_simple_event(
        self,
        event: SimpleEvent,
        query: StructuralQuery,
        cache: Optional[QueryCache] = None,
    ) -> LayerWithLogProbabilities:
        return query.log_probabilities.record(
            self.log_truncated_of_assignment(
                event[query.variables[self.variable]], query.singleton_allowed
            )
        )

    def can_truncate_in_one_batch(
        self, events: List[SimpleEvent], query: StructuralQuery
    ) -> bool:
        """
        The truncations to the events are joined with :meth:`concatenate`, which needs
        them all to be input layers of one type.
        """
        variable = query.variables[self.variable]
        types = {
            self.type_of_truncated_layer(event[variable], query.singleton_allowed)
            for event in events
        }
        return len(types) == 1 and issubclass(types.pop(), InputLayer)

    @memoized
    def log_truncated_of_simple_events(
        self,
        events: List[SimpleEvent],
        query: StructuralQuery,
        cache: Optional[QueryCache] = None,
    ) -> LayerWithLogProbabilities:
        variable = query.variables[self.variable]
        truncated = [
            self.log_truncated_of_assignment(event[variable], query.singleton_allowed)
            for event in events
        ]
        layers = [piece.layer for piece in truncated]
        return query.log_probabilities.record(
            LayerWithLogProbabilities(
                type(layers[0]).concatenate(layers),
                np.concatenate([piece.log_probabilities for piece in truncated]),
            )
        )

    @abstractmethod
    def log_conditional_of_value(self, value: Any) -> LayerWithLogProbabilities:
        """
        Condition every node of this layer on a value of its variable at once.

        :param value: The value.
        :return: The conditioned layer, with as many nodes as this one, and the
            log-likelihoods of the value under its nodes.
        """
        raise NotImplementedError

    @memoized
    def log_conditional_of_point(
        self,
        point: PartialPointType,
        query: StructuralQuery,
        cache: Optional[QueryCache] = None,
    ) -> LayerWithLogProbabilities:
        variable = query.variables[self.variable]
        if variable not in point:
            result = LayerWithLogProbabilities(
                self.__deepcopy__(), np.zeros(self.number_of_nodes)
            )
        else:
            result = self.log_conditional_of_value(point[variable])
        return query.log_probabilities.record(result)

    def rebuild(
        self,
        needed: Dict[int, NodeMask],
        rebuilt: Dict[int, Optional[Layer]],
    ) -> Optional[Layer]:
        alive = needed[id(self)]
        if not alive.any():
            return None
        return self.select_nodes(alive)

    def marginal(
        self, kept: VariableMask, cache: Optional[QueryCache] = None
    ) -> Optional[Layer]:
        if not kept[self.variable]:
            return None
        return self.__deepcopy__()


@dataclass(eq=False, repr=False)
class AbstractContinuousLayer(InputLayer, ABC):
    """
    Abstract base class for the input layers of continuous univariate distributions.
    """

    @memoized
    def log_likelihood_of_nodes(
        self, events: SampleArray, cache: Optional[QueryCache] = None
    ) -> SampleNodeValues:
        return self.log_likelihood_of_nodes_from_column(self.column_of(events))

    @abstractmethod
    def log_likelihood_of_nodes_from_column(
        self, values: SampleColumn
    ) -> SampleNodeValues:
        """
        Calculate the log-likelihood of every node for a column of values of the
        variable of this layer.

        :param values: The values with shape (#events,).
        :return: The log-likelihoods with shape (#events, #nodes).
        """
        raise NotImplementedError

    @memoized
    def probability_of_simple_event_of_nodes(
        self,
        event: SimpleEvent,
        variables: SortedSet,
        cache: Optional[QueryCache] = None,
    ) -> NodeValues:
        interval: Interval = event[variables[self.variable]]
        result = np.zeros(self.number_of_nodes)
        for simple_interval in interval.simple_sets:
            values = self.cumulative_distribution_of_nodes_from_column(
                np.array([simple_interval.lower, simple_interval.upper], dtype=float)
            )
            result += values[1] - values[0]
        return result

    @memoized
    def cumulative_distribution_of_nodes(
        self, events: SampleArray, cache: Optional[QueryCache] = None
    ) -> SampleNodeValues:
        return self.cumulative_distribution_of_nodes_from_column(self.column_of(events))

    @abstractmethod
    def cumulative_distribution_of_nodes_from_column(
        self, values: SampleColumn
    ) -> SampleNodeValues:
        """
        Calculate the cumulative distribution function of every node for a column of
        values of the variable of this layer.

        :param values: The values with shape (#events,).
        :return: The values with shape (#events, #nodes).
        """
        raise NotImplementedError
