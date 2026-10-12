from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from random_events.interval import singleton
from random_events.product_algebra import Event, SimpleEvent
from scipy.stats import multivariate_normal
from sortedcontainers import SortedSet
from typing_extensions import List, Optional, Self, Tuple, Type

from probabilistic_model.distributions.multivariate_gaussian import (
    MultivariateGaussianDistribution,
)
from probabilistic_model.probabilistic_circuit.tensorized.array_types import (
    NodeIndices,
    NodeMask,
    NodeValues,
    NodeVariableValues,
    SampleArray,
    SampleNodeValues,
    SampleScopeValues,
    ScopeValues,
)
from probabilistic_model.probabilistic_circuit.tensorized.inner_layer.base import Layer
from probabilistic_model.probabilistic_circuit.tensorized.input_layer.gaussian_layer import (
    GaussianLayer,
)
from probabilistic_model.probabilistic_circuit.tensorized.input_layer.multivariate_gaussian.base import (
    AbstractMultivariateGaussianLayer,
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
from probabilistic_model.probabilistic_circuit.tensorized.input_layer.multivariate_gaussian.truncated_multivariate_gaussian_layer import (
    TruncatedMultivariateGaussianLayer,
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
)


@dataclass(eq=False, repr=False)
class MultivariateGaussianLayer(AbstractMultivariateGaussianLayer):
    """
    A layer of Gaussians over several continuous variables at once.
    """

    def node_distribution(
        self, index: int, variables: SortedSet
    ) -> MultivariateGaussianDistribution:
        return self.untruncated_distribution(index, variables)

    @classmethod
    def from_distributions(
        cls,
        variables: SortedSet,
        distributions: List[MultivariateGaussianDistribution],
    ) -> Self:
        """
        :param variables: The variables of the circuit.
        :param distributions: Gaussians over the same variables, in any order.
        :return: The layer with one node per distribution.
        """
        scope = cls.scope_of(variables, distributions[0])
        gaussians = MultivariateGaussianArray.from_distributions(
            distributions, [variables[index] for index in scope]
        )
        return cls(scope=scope, mean=gaussians.mean, covariance=gaussians.covariance)

    def select_nodes(self, mask: NodeMask) -> Self:
        return self.__class__(
            scope=self.scope.copy(),
            mean=self.mean[mask],
            covariance=self.covariance.select(mask),
        )

    @classmethod
    def concatenate(cls, layers: List[Self]) -> Self:
        return cls(
            scope=layers[0].scope.copy(),
            mean=np.concatenate([layer.mean for layer in layers]),
            covariance=CovarianceArray.concatenate(
                [layer.covariance for layer in layers]
            ),
        )

    # %% queries

    @memoized
    def log_likelihood_of_nodes(
        self, events: SampleArray, cache: Optional[QueryCache] = None
    ) -> SampleNodeValues:
        return self.untruncated_gaussians.log_density(self.values_of_scope(events))

    @memoized
    def cumulative_distribution_of_nodes(
        self, events: SampleArray, cache: Optional[QueryCache] = None
    ) -> SampleNodeValues:
        values = self.values_of_scope(events)
        return np.stack(
            [
                np.atleast_1d(
                    multivariate_normal(
                        self.mean[node], self.covariance.matrices[node]
                    ).cdf(values)
                )
                for node in range(self.number_of_nodes)
            ],
            axis=1,
        )

    @memoized
    def probability_of_simple_event_of_nodes(
        self,
        event: SimpleEvent,
        variables: SortedSet,
        cache: Optional[QueryCache] = None,
    ) -> NodeValues:
        return sum(
            (
                self.untruncated_gaussians.probability_of_hyperrectangles(
                    hyperrectangle.broadcast_to(self.number_of_nodes)
                )
                for hyperrectangle in self.hyperrectangles_of(event, variables)
            ),
            np.zeros(self.number_of_nodes),
        )

    @memoized
    def log_mode_of_nodes(
        self, variables: SortedSet, cache: Optional[QueryCache] = None
    ) -> Tuple[List[Event], NodeValues]:
        # a Gaussian is most dense at its mean
        scope_variables = self.scope_variables(variables)
        modes = [
            SimpleEvent.from_data(
                {
                    variable: singleton(float(value))
                    for variable, value in zip(scope_variables, mean)
                }
            ).as_composite_set()
            for mean in self.mean
        ]
        _, log_determinant = np.linalg.slogdet(self.covariance.matrices)
        return modes, -0.5 * (
            self.number_of_scope_variables * math.log(2 * math.pi) + log_determinant
        )

    @memoized
    def moment_of_nodes(
        self,
        query: MomentQuery,
        variables: SortedSet,
        cache: Optional[QueryCache] = None,
    ) -> NodeVariableValues:
        # the moment of one variable is the moment of its univariate marginal
        result = np.zeros((self.number_of_nodes, query.number_of_variables))
        for position, index in enumerate(self.scope):
            if not query.requested[index]:
                continue
            marginal = GaussianLayer(
                int(index),
                self.mean[:, position],
                np.sqrt(self.covariance.variances[:, position]),
            )
            result[:, index] = marginal.moment_of_nodes_own(
                int(query.order[index]), float(query.center[index]), variables[index]
            )
        return result

    def samples_of_nodes(self, nodes: NodeIndices) -> SampleScopeValues:
        standard = np.random.standard_normal(
            (len(nodes), self.number_of_scope_variables)
        )
        cholesky = np.linalg.cholesky(self.covariance.matrices)
        return self.mean[nodes] + np.einsum("rij,rj->ri", cholesky[nodes], standard)

    # %% structural

    def type_of_layer_truncated_to_hyperrectangle(
        self, hyperrectangle: HyperrectangleArray
    ) -> Type[Layer]:
        if hyperrectangle.is_whole_space:
            return MultivariateGaussianLayer
        return TruncatedMultivariateGaussianLayer

    def log_truncated_of_hyperrectangle(
        self, hyperrectangle: HyperrectangleArray
    ) -> LayerWithLogProbabilities:
        # the whole space leaves every node a Gaussian, any other hyperrectangle
        # confines it
        if (
            self.type_of_layer_truncated_to_hyperrectangle(hyperrectangle)
            is MultivariateGaussianLayer
        ):
            return LayerWithLogProbabilities(
                self.__deepcopy__(), np.zeros(self.number_of_nodes)
            )
        hyperrectangles = hyperrectangle.broadcast_to(self.number_of_nodes)
        log_probabilities = (
            self.untruncated_gaussians.log_probability_of_hyperrectangles(
                hyperrectangles
            )
        )
        return LayerWithLogProbabilities(
            TruncatedMultivariateGaussianLayer(
                scope=self.scope.copy(),
                mean=self.mean.copy(),
                covariance=self.covariance.copy(),
                hyperrectangles=hyperrectangles,
                log_normalizing_constant=log_probabilities.copy(),
            ),
            log_probabilities,
        )

    def log_conditional_of_values(
        self, fixed: NodeIndices, free: NodeIndices, values: ScopeValues
    ) -> LayerWithLogProbabilities:
        gaussians = self.untruncated_gaussians
        log_likelihood = gaussians.marginal(fixed).log_density(values[None, :])[0]
        if len(free) == 0:
            return LayerWithLogProbabilities(self.__deepcopy__(), log_likelihood)
        return LayerWithLogProbabilities(
            self.over(free, gaussians.conditional(fixed, free, values)),
            log_likelihood,
        )

    def marginal_over(self, positions: NodeIndices) -> Layer:
        return self.over(positions, self.untruncated_gaussians.marginal(positions))

    def over(
        self, positions: NodeIndices, gaussians: MultivariateGaussianArray
    ) -> Self:
        """
        :param positions: Positions in the scope of this layer, ascending.
        :param gaussians: One Gaussian per node over the variables at those positions.
        :return: The layer of those Gaussians.
        """
        return self.__class__(
            scope=self.scope[positions],
            mean=gaussians.mean,
            covariance=gaussians.covariance,
        )

    def __deepcopy__(self, memo=None) -> MultivariateGaussianLayer:
        if memo is None:
            memo = {}
        if id(self) in memo:
            return memo[id(self)]
        result = self.__class__(
            scope=self.scope.copy(),
            mean=self.mean.copy(),
            covariance=self.covariance.copy(),
        )
        memo[id(self)] = result
        return result
