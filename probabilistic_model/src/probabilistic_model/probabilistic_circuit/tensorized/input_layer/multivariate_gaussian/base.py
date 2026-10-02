from __future__ import annotations

import itertools
from abc import ABC, abstractmethod
from dataclasses import dataclass

import numpy as np
from random_events.product_algebra import Event, SimpleEvent
from random_events.variable import Variable
from sortedcontainers import SortedSet
from typing_extensions import Any, Dict, List, Optional, Self, Type

from probabilistic_model.distributions.multivariate_gaussian import (
    MultivariateGaussianDistribution,
)
from probabilistic_model.exceptions import ShapeMismatchError
from probabilistic_model.probabilistic_model import PartialPointType
from probabilistic_model.probabilistic_circuit.tensorized.array_types import (
    NodeIndices,
    NodeMask,
    NodeScopeValues,
    SampleArray,
    SampleScopeValues,
    VariableIndices,
    VariableMask,
    VariableValues,
)
from probabilistic_model.probabilistic_circuit.tensorized.forward_sample_assignment import (
    ForwardSampleAssignment,
)
from probabilistic_model.probabilistic_circuit.tensorized.inner_layer.base import Layer
from probabilistic_model.probabilistic_circuit.tensorized.inner_layer.product_layer import (
    ProductLayer,
)
from probabilistic_model.probabilistic_circuit.tensorized.inner_layer.sum_layer import (
    SumLayer,
)
from probabilistic_model.probabilistic_circuit.tensorized.input_layer.dirac_delta_layer import (
    DiracDeltaLayer,
)
from probabilistic_model.probabilistic_circuit.tensorized.input_layer.multivariate_gaussian.covariance_array import (
    CovarianceArray,
)
from probabilistic_model.probabilistic_circuit.tensorized.input_layer.multivariate_gaussian.hyperrectangle_array import (
    HyperrectangleArray,
)
from probabilistic_model.probabilistic_circuit.tensorized.input_layer.multivariate_gaussian.multivariate_gaussian_array import (
    MultivariateGaussianArray,
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
class AbstractMultivariateGaussianLayer(Layer, ABC):
    """
    Abstract base class for the input layers of Gaussians over several continuous
    variables at once.

    Every node is a Gaussian over the same variables, and its mean and covariance are
    laid out in the order of those variables in the circuit.
    """

    scope: VariableIndices
    """
    The sorted indices of the variables of every node.
    """

    mean: NodeScopeValues
    """
    The mean of every node.
    """

    covariance: CovarianceArray
    """
    The covariance matrix of every node.
    """

    @property
    def child_layers(self) -> List[Layer]:
        """
        :return: An empty list. An input layer is a leaf of the layer graph.
        """
        return []

    @property
    def variables(self) -> VariableIndices:
        return self.scope

    @property
    def number_of_nodes(self) -> int:
        return len(self.mean)

    @property
    def number_of_scope_variables(self) -> int:
        """
        :return: How many variables every node is over.
        """
        return len(self.scope)

    @property
    def number_of_own_parameters(self) -> int:
        dimension = self.number_of_scope_variables
        return self.number_of_nodes * (dimension + dimension * (dimension + 1) // 2)

    def validate_own(self):
        dimension = self.number_of_scope_variables
        if self.mean.shape != (self.number_of_nodes, dimension):
            raise ShapeMismatchError(self.mean.shape, (self.number_of_nodes, dimension))
        expected = (self.number_of_nodes, dimension * (dimension + 1) // 2)
        if self.covariance.lower_triangles.shape != expected:
            raise ShapeMismatchError(self.covariance.lower_triangles.shape, expected)

    def values_of_scope(self, events: SampleArray) -> SampleScopeValues:
        """
        :param events: The events with shape (#events, #variables of the circuit).
        :return: The columns of the variables of this layer.
        """
        return events[:, self.scope]

    def scope_variables(self, variables: SortedSet) -> List[Variable]:
        """
        :param variables: The variables of the circuit.
        :return: The variables of this layer.
        """
        return [variables[index] for index in self.scope]

    @staticmethod
    def scope_of(
        variables: SortedSet, distribution: MultivariateGaussianDistribution
    ) -> VariableIndices:
        """
        :param variables: The variables of the circuit.
        :param distribution: A Gaussian over some of them.
        :return: The sorted indices of the variables of the Gaussian.
        """
        return np.sort(
            [variables.index(variable) for variable in distribution.variables]
        ).astype(np.int64)

    # %% the untruncated Gaussians

    @property
    def untruncated_gaussians(self) -> MultivariateGaussianArray:
        """
        :return: The Gaussian of every node, without any hyperrectangle that confines
            it.
        """
        return MultivariateGaussianArray(self.mean, self.covariance)

    def hyperrectangles_of(
        self, event: SimpleEvent, variables: SortedSet
    ) -> List[HyperrectangleArray]:
        """
        :param event: A simple event.
        :param variables: The variables of the circuit.
        :return: The hyperrectangles the event makes of the variables of this layer,
            one per combination of their simple intervals.
        """
        return [
            HyperrectangleArray.of_simple_intervals(intervals)
            for intervals in itertools.product(
                *(
                    event[variable].simple_sets
                    for variable in self.scope_variables(variables)
                )
            )
        ]

    def untruncated_distribution(
        self, index: int, variables: SortedSet
    ) -> MultivariateGaussianDistribution:
        """
        :param index: The index of a node.
        :param variables: The variables of the circuit.
        :return: The untruncated Gaussian of that node.
        """
        return MultivariateGaussianDistribution(
            variables=tuple(self.scope_variables(variables)),
            mean=self.mean[index].copy(),
            covariance=self.covariance.covariance_at(index),
        )

    # %% per node view

    @abstractmethod
    def node_distribution(self, index: int, variables: SortedSet) -> Any:
        """
        :param index: The index of a node.
        :param variables: The variables of the circuit.
        :return: The distribution of that node.
        """
        raise NotImplementedError

    def node_distributions(self, variables: SortedSet) -> List[Any]:
        """
        :param variables: The variables of the circuit.
        :return: The distribution of every node.
        """
        return [
            self.node_distribution(index, variables)
            for index in range(self.number_of_nodes)
        ]

    @abstractmethod
    def select_nodes(self, mask: NodeMask) -> Self:
        """
        :param mask: A boolean mask over the nodes of this layer.
        :return: A layer that only holds the selected nodes.
        """
        raise NotImplementedError

    @classmethod
    @abstractmethod
    def concatenate(cls, layers: List[Self]) -> Self:
        """
        Join layers of this type over the same variables into one layer, the nodes of
        ``layers[k]`` as one contiguous block.

        :param layers: The layers to join.
        :return: The joined layer.
        """
        raise NotImplementedError

    # %% queries

    @memoized
    def support_of_nodes(
        self, variables: SortedSet, cache: Optional[QueryCache] = None
    ) -> List[Event]:
        return [
            distribution.support for distribution in self.node_distributions(variables)
        ]

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
        rows = np.concatenate(rows)
        samples[rows[:, None], self.scope[None, :]] = self.samples_of_nodes(
            np.concatenate(nodes)
        )

    @abstractmethod
    def samples_of_nodes(self, nodes: NodeIndices) -> SampleScopeValues:
        """
        :param nodes: The node to draw each sample from.
        :return: One sample per entry of ``nodes``.
        """
        raise NotImplementedError

    # %% structural

    @abstractmethod
    def type_of_layer_truncated_to_hyperrectangle(
        self, hyperrectangle: HyperrectangleArray
    ) -> Type[Layer]:
        """
        :param hyperrectangle: A single hyperrectangle.
        :return: The type of the layer :meth:`log_truncated_of_hyperrectangle` returns
            for it.
        """
        raise NotImplementedError

    @abstractmethod
    def log_truncated_of_hyperrectangle(
        self, hyperrectangle: HyperrectangleArray
    ) -> LayerWithLogProbabilities:
        """
        Truncate every node to the same hyperrectangle.

        :param hyperrectangle: A single hyperrectangle.
        :return: The truncated layer, with as many nodes as this one, and the log-
            probability of the hyperrectangle under every node.
        """
        raise NotImplementedError

    def log_truncated_of_hyperrectangles(
        self, hyperrectangles: List[HyperrectangleArray]
    ) -> LayerWithLogProbabilities:
        """
        Truncate every node to the union of disjoint hyperrectangles.

        :param hyperrectangles: The hyperrectangles.
        :return: The truncated layer and the log-probabilities of its nodes. A node
            truncated to several hyperrectangles becomes a mixture of its truncations.
        """
        if not hyperrectangles:
            return LayerWithLogProbabilities(
                self.__deepcopy__(), np.full(self.number_of_nodes, -np.inf)
            )
        pieces = [
            self.log_truncated_of_hyperrectangle(hyperrectangle)
            for hyperrectangle in hyperrectangles
        ]
        if len(pieces) == 1:
            return pieces[0]
        return SumLayer.mixture_of_pieces(pieces)

    def type_of_truncated_layer(
        self, event: SimpleEvent, variables: SortedSet
    ) -> Type[Layer]:
        """
        :param event: A simple event.
        :param variables: The variables of the circuit.
        :return: The type of the layer truncating to the event creates.
        """
        hyperrectangles = self.hyperrectangles_of(event, variables)
        if len(hyperrectangles) > 1:
            return SumLayer
        if not hyperrectangles:
            return self.__class__
        return self.type_of_layer_truncated_to_hyperrectangle(hyperrectangles[0])

    @memoized
    def log_truncated_of_simple_event(
        self,
        event: SimpleEvent,
        query: StructuralQuery,
        cache: Optional[QueryCache] = None,
    ) -> LayerWithLogProbabilities:
        # a Gaussian gives every single point probability zero, so a singleton makes a
        # node impossible whether singletons are allowed or not
        return query.log_probabilities.record(
            self.log_truncated_of_hyperrectangles(
                self.hyperrectangles_of(event, query.variables)
            )
        )

    def can_truncate_in_one_batch(
        self, events: List[SimpleEvent], query: StructuralQuery
    ) -> bool:
        # the truncations are joined with concatenate, which needs them all to be
        # layers of one Gaussian type
        types = {
            self.type_of_truncated_layer(event, query.variables) for event in events
        }
        return len(types) == 1 and issubclass(
            types.pop(), AbstractMultivariateGaussianLayer
        )

    @memoized
    def log_truncated_of_simple_events(
        self,
        events: List[SimpleEvent],
        query: StructuralQuery,
        cache: Optional[QueryCache] = None,
    ) -> LayerWithLogProbabilities:
        truncated = [
            self.log_truncated_of_hyperrectangles(
                self.hyperrectangles_of(event, query.variables)
            )
            for event in events
        ]
        layers = [piece.layer for piece in truncated]
        return query.log_probabilities.record(
            LayerWithLogProbabilities(
                type(layers[0]).concatenate(layers),
                np.concatenate([piece.log_probabilities for piece in truncated]),
            )
        )

    @memoized
    def log_conditional_of_point(
        self,
        point: PartialPointType,
        query: StructuralQuery,
        cache: Optional[QueryCache] = None,
    ) -> LayerWithLogProbabilities:
        # the fixed variables become Dirac deltas next to the Gaussians of the others,
        # conditioned on them
        variables = self.scope_variables(query.variables)
        fixed = np.array(
            [
                position
                for position, variable in enumerate(variables)
                if variable in point
            ],
            dtype=np.int64,
        )
        if len(fixed) == 0:
            return query.log_probabilities.record(
                LayerWithLogProbabilities(
                    self.__deepcopy__(), np.zeros(self.number_of_nodes)
                )
            )

        values = np.array([float(point[variables[position]]) for position in fixed])
        free = np.setdiff1d(np.arange(self.number_of_scope_variables), fixed)
        conditioned = self.log_conditional_of_values(fixed, free, values)

        point_masses = [
            DiracDeltaLayer(
                int(self.scope[position]),
                np.full(self.number_of_nodes, value),
                np.ones(self.number_of_nodes),
            )
            for position, value in zip(fixed, values)
        ]
        factors = point_masses if len(free) == 0 else [conditioned.layer] + point_masses
        return query.log_probabilities.record(
            LayerWithLogProbabilities(
                ProductLayer.node_wise_product_of(factors),
                conditioned.log_probabilities,
            )
        )

    @abstractmethod
    def log_conditional_of_values(
        self, fixed: NodeIndices, free: NodeIndices, values: np.ndarray
    ) -> LayerWithLogProbabilities:
        """
        :param fixed: The positions in the scope of the variables held at a value.
        :param free: The positions in the scope of the other variables, possibly none.
        :param values: What the fixed variables are held at.
        :return: The layer over the free variables conditioned on the values, which is
            meaningless without free variables, and the log-likelihood of the values
            under every node.
        """
        raise NotImplementedError

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
        positions = np.flatnonzero(kept[self.scope])
        if len(positions) == 0:
            return None
        if len(positions) == self.number_of_scope_variables:
            return self.__deepcopy__()
        return self.marginal_over(positions)

    @abstractmethod
    def marginal_over(self, positions: NodeIndices) -> Layer:
        """
        :param positions: Some, but not all, positions in the scope, ascending.
        :return: The marginal of every node over those variables.
        """
        raise NotImplementedError

    @memoized
    def remap_variables(
        self, remap: VariableIndices, cache: Optional[QueryCache] = None
    ):
        remapped = remap[self.scope]
        order = np.argsort(remapped)
        self.scope = remapped[order]
        self.reorder_scope(order)

    def reorder_scope(self, order: NodeIndices):
        """
        Lay the parameters out in a new order of the variables, in place.

        :param order: The old position of every variable, in the new order.
        """
        self.mean = self.mean[:, order]
        self.covariance = self.covariance.marginal(order)

    def apply_translation_own(self, translation: VariableValues):
        self.mean = self.mean + translation[self.scope]

    def apply_scaling_own(self, scaling: VariableValues):
        factors = scaling[self.scope]
        self.mean = self.mean * factors
        self.covariance = self.covariance.scaled(factors)
