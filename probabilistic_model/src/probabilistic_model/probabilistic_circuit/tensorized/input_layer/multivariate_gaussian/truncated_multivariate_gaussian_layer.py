from __future__ import annotations

import dataclasses

import numpy as np
from random_events.product_algebra import Event, SimpleEvent
from scipy.stats import multivariate_normal, truncnorm
from sortedcontainers import SortedSet
from typing_extensions import List, Optional, Self, Tuple, Type

from probabilistic_model.distributions.truncated_multivariate_gaussian import (
    MomentIntegration,
    TruncatedMultivariateGaussianDistribution,
)
from probabilistic_model.exceptions import ShapeMismatchError
from probabilistic_model.probabilistic_circuit.tensorized.array_types import (
    NodeIndices,
    NodeSelection,
    NodeMask,
    NodeValues,
    NodeVariableValues,
    SampleArray,
    SampleNodeValues,
    SampleScopeValues,
    ScopeValues,
    VariableValues,
)
from probabilistic_model.exceptions import NoClosedFormError
from probabilistic_model.probabilistic_circuit.tensorized.inner_layer.base import Layer
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


@dataclasses.dataclass(eq=False, repr=False)
class TruncatedMultivariateGaussianLayer(AbstractMultivariateGaussianLayer):
    """
    A layer of Gaussians over several continuous variables, each confined to a
    hyperrectangle.

    This is the layer that truncating a :class:`MultivariateGaussianLayer` to a
    hyperrectangle produces.
    """

    hyperrectangles: HyperrectangleArray
    """
    The hyperrectangle that every node is confined to.
    """

    log_normalizing_constant: NodeValues
    """
    The log-probability of the hyperrectangle of every node under its untruncated
    Gaussian.
    """

    burn_in_period_length: int = 100
    """
    How many times each chain of the sampler draws every variable before its last state
    becomes a sample.
    """

    moment_integration: MomentIntegration = dataclasses.field(
        default_factory=MomentIntegration
    )
    """
    How the moments of every node are integrated numerically.
    """

    @property
    def number_of_own_parameters(self) -> int:
        return (
            super().number_of_own_parameters
            + 2 * self.number_of_nodes * self.number_of_scope_variables
        )

    def validate_own(self):
        super().validate_own()
        expected = (self.number_of_nodes, self.number_of_scope_variables, 2)
        if self.hyperrectangles.interval.shape != expected:
            raise ShapeMismatchError(self.hyperrectangles.interval.shape, expected)
        if self.hyperrectangles.bounds.shape != expected:
            raise ShapeMismatchError(self.hyperrectangles.bounds.shape, expected)
        if self.log_normalizing_constant.shape != (self.number_of_nodes,):
            raise ShapeMismatchError(
                self.log_normalizing_constant.shape, (self.number_of_nodes,)
            )

    def node_distribution(
        self, index: int, variables: SortedSet
    ) -> TruncatedMultivariateGaussianDistribution:
        return TruncatedMultivariateGaussianDistribution(
            untruncated=self.untruncated_distribution(index, variables),
            box=self.hyperrectangles.simple_event_of(
                index, self.scope_variables(variables)
            ),
            burn_in_period_length=self.burn_in_period_length,
            moment_integration=self.moment_integration,
        )

    @classmethod
    def from_distributions(
        cls,
        variables: SortedSet,
        distributions: List[TruncatedMultivariateGaussianDistribution],
    ) -> Self:
        """
        :param variables: The variables of the circuit.
        :param distributions: Truncated Gaussians over the same variables, in any order.
        :return: The layer with one node per distribution.
        """
        scope = cls.scope_of(variables, distributions[0].untruncated)
        scope_variables = [variables[index] for index in scope]
        untruncated = MultivariateGaussianArray.from_distributions(
            [distribution.untruncated for distribution in distributions],
            scope_variables,
        )
        hyperrectangles = HyperrectangleArray.stack(
            [
                HyperrectangleArray.of_simple_intervals(
                    [distribution.interval_of(variable) for variable in scope_variables]
                )
                for distribution in distributions
            ]
        )
        return cls(
            scope=scope,
            mean=untruncated.mean,
            covariance=untruncated.covariance,
            hyperrectangles=hyperrectangles,
            log_normalizing_constant=np.log(
                [distribution.normalizing_constant for distribution in distributions]
            ),
            burn_in_period_length=distributions[0].burn_in_period_length,
            moment_integration=distributions[0].moment_integration,
        )

    def with_nodes(self, indices: NodeSelection) -> Self:
        """
        :param indices: A mask or index array over the nodes.
        :return: A layer of only those nodes.
        """
        return dataclasses.replace(
            self,
            scope=self.scope.copy(),
            mean=self.mean[indices],
            covariance=self.covariance.select(indices),
            hyperrectangles=self.hyperrectangles.select(indices),
            log_normalizing_constant=self.log_normalizing_constant[indices],
        )

    def select_nodes(self, mask: NodeMask) -> Self:
        return self.with_nodes(mask)

    @classmethod
    def concatenate(cls, layers: List[Self]) -> Self:
        return dataclasses.replace(
            layers[0],
            scope=layers[0].scope.copy(),
            mean=np.concatenate([layer.mean for layer in layers]),
            covariance=CovarianceArray.concatenate(
                [layer.covariance for layer in layers]
            ),
            hyperrectangles=HyperrectangleArray.concatenate(
                [layer.hyperrectangles for layer in layers]
            ),
            log_normalizing_constant=np.concatenate(
                [layer.log_normalizing_constant for layer in layers]
            ),
        )

    def has_equal_parameters(self, other: Self) -> bool:
        return (
            super().has_equal_parameters(other)
            and np.array_equal(
                self.hyperrectangles.interval, other.hyperrectangles.interval
            )
            and np.array_equal(
                self.hyperrectangles.bounds, other.hyperrectangles.bounds
            )
            and np.array_equal(
                self.log_normalizing_constant, other.log_normalizing_constant
            )
        )

    # %% queries

    @memoized
    def log_likelihood_of_nodes(
        self, events: SampleArray, cache: Optional[QueryCache] = None
    ) -> SampleNodeValues:
        values = self.values_of_scope(events)
        return np.where(
            self.hyperrectangles.contains(values),
            self.untruncated_gaussians.log_density(values)
            - self.log_normalizing_constant,
            -np.inf,
        )

    @memoized
    def cumulative_distribution_of_nodes(
        self, events: SampleArray, cache: Optional[QueryCache] = None
    ) -> SampleNodeValues:
        # the probability of the part of the hyperrectangle below the point
        values = self.values_of_scope(events)
        lower, upper = self.hyperrectangles.lower, self.hyperrectangles.upper
        result = np.zeros((len(values), self.number_of_nodes))
        for node in range(self.number_of_nodes):
            below_the_hyperrectangle = np.any(values < lower[node], axis=1)
            result[:, node] = np.where(
                below_the_hyperrectangle,
                0.0,
                np.atleast_1d(
                    multivariate_normal(
                        self.mean[node], self.covariance.matrices[node]
                    ).cdf(np.minimum(values, upper[node]), lower_limit=lower[node])
                ),
            )
        return np.clip(result / np.exp(self.log_normalizing_constant), 0.0, 1.0)

    @memoized
    def probability_of_simple_event_of_nodes(
        self,
        event: SimpleEvent,
        variables: SortedSet,
        cache: Optional[QueryCache] = None,
    ) -> NodeValues:
        untruncated = sum(
            (
                self.untruncated_gaussians.probability_of_hyperrectangles(
                    self.hyperrectangles.intersection_with(hyperrectangle)
                )
                for hyperrectangle in self.hyperrectangles_of(event, variables)
            ),
            np.zeros(self.number_of_nodes),
        )
        return untruncated / np.exp(self.log_normalizing_constant)

    @memoized
    def log_mode_of_nodes(
        self, variables: SortedSet, cache: Optional[QueryCache] = None
    ) -> Tuple[List[Event], NodeValues]:
        modes = [
            distribution.log_mode()
            for distribution in self.node_distributions(variables)
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
        requested = self.scope[query.requested[self.scope]]
        if len(requested) == 0:
            return result
        order = {variables[index]: int(query.order[index]) for index in requested}
        center = {variables[index]: float(query.center[index]) for index in requested}
        for node, distribution in enumerate(self.node_distributions(variables)):
            moments = distribution.moment(order, center)
            for index in requested:
                result[node, index] = moments[variables[index]]
        return result

    def samples_of_nodes(self, nodes: NodeIndices) -> SampleScopeValues:
        # Gibbs sampling, all chains at once: every sweep draws each variable from its
        # Gaussian given the others, confined to its interval (scipy's truncnorm). Every
        # chain starts at the mean of its node moved into the hyperrectangle. The
        # samples follow the distribution only approximately, closer the more sweeps
        # each chain makes.
        lower = self.hyperrectangles.lower[nodes]
        upper = self.hyperrectangles.upper[nodes]
        mean = self.mean[nodes]
        precision = np.linalg.inv(self.covariance.matrices)[nodes]
        precision_of_itself = np.diagonal(precision, axis1=1, axis2=2)
        conditional_deviation = 1 / np.sqrt(precision_of_itself)

        samples = np.clip(mean, lower, upper)
        for _ in range(self.burn_in_period_length):
            for position in range(self.number_of_scope_variables):
                difference = samples - mean
                conditional_mean = (
                    mean[:, position]
                    - (
                        np.einsum("rj,rj->r", difference, precision[:, position])
                        - precision_of_itself[:, position] * difference[:, position]
                    )
                    / precision_of_itself[:, position]
                )
                deviation = conditional_deviation[:, position]
                samples[:, position] = truncnorm.rvs(
                    a=(lower[:, position] - conditional_mean) / deviation,
                    b=(upper[:, position] - conditional_mean) / deviation,
                    loc=conditional_mean,
                    scale=deviation,
                    size=len(nodes),
                )
        return samples

    # %% structural

    def type_of_layer_truncated_to_hyperrectangle(
        self, hyperrectangle: HyperrectangleArray
    ) -> Type[Layer]:
        return TruncatedMultivariateGaussianLayer

    def log_truncated_of_hyperrectangle(
        self, hyperrectangle: HyperrectangleArray
    ) -> LayerWithLogProbabilities:
        # every node keeps its Gaussian, confined to the intersection of the two
        # hyperrectangles
        intersection = self.hyperrectangles.intersection_with(hyperrectangle)
        log_normalizing_constant = (
            self.untruncated_gaussians.log_probability_of_hyperrectangles(intersection)
        )
        alive = np.isfinite(log_normalizing_constant)

        # impossible nodes keep their parameters and are dropped by the prune pass
        truncated = dataclasses.replace(
            self,
            scope=self.scope.copy(),
            mean=self.mean.copy(),
            covariance=self.covariance.copy(),
            hyperrectangles=HyperrectangleArray(
                np.where(
                    alive[:, None, None],
                    intersection.interval,
                    self.hyperrectangles.interval,
                ),
                np.where(
                    alive[:, None, None],
                    intersection.bounds,
                    self.hyperrectangles.bounds,
                ),
            ),
            log_normalizing_constant=np.where(
                alive, log_normalizing_constant, self.log_normalizing_constant
            ),
        )
        return LayerWithLogProbabilities(
            truncated, log_normalizing_constant - self.log_normalizing_constant
        )

    def log_conditional_of_values(
        self, fixed: NodeIndices, free: NodeIndices, values: ScopeValues
    ) -> LayerWithLogProbabilities:
        # the Gaussian conditional of every node, confined to the slice its
        # hyperrectangle makes at the values
        inside = self.hyperrectangles.over(fixed).contains(values[None, :])[0]
        gaussians = self.untruncated_gaussians
        log_density = gaussians.marginal(fixed).log_density(values[None, :])[0]
        if len(free) == 0:
            return LayerWithLogProbabilities(
                self.__deepcopy__(),
                np.where(inside, log_density - self.log_normalizing_constant, -np.inf),
            )

        conditionals = gaussians.conditional(fixed, free, values)
        slices = self.hyperrectangles.over(free)
        log_probability_of_slice = conditionals.log_probability_of_hyperrectangles(
            slices
        )
        alive = inside & np.isfinite(log_probability_of_slice)
        conditioned = dataclasses.replace(
            self,
            scope=self.scope[free],
            mean=conditionals.mean,
            covariance=conditionals.covariance,
            hyperrectangles=slices,
            log_normalizing_constant=np.where(alive, log_probability_of_slice, 0.0),
        )
        return LayerWithLogProbabilities(
            conditioned,
            np.where(
                alive,
                log_density + log_probability_of_slice - self.log_normalizing_constant,
                -np.inf,
            ),
        )

    def marginal_over(self, positions: NodeIndices) -> Layer:
        # the marginal of a Gaussian confined to a hyperrectangle is not a Gaussian
        # confined to a hyperrectangle
        raise NoClosedFormError(type(self), type(self).marginal)

    def reorder_scope(self, order: NodeIndices):
        super().reorder_scope(order)
        self.hyperrectangles = self.hyperrectangles.over(order)

    def apply_translation_own(self, translation: VariableValues):
        super().apply_translation_own(translation)
        self.hyperrectangles = self.hyperrectangles.translated(translation[self.scope])

    def apply_scaling_own(self, scaling: VariableValues):
        super().apply_scaling_own(scaling)
        self.hyperrectangles = self.hyperrectangles.scaled(scaling[self.scope])
        # the hyperrectangle scales with the Gaussian, so its probability stays the same

    def __deepcopy__(self, memo=None) -> TruncatedMultivariateGaussianLayer:
        if memo is None:
            memo = {}
        if id(self) in memo:
            return memo[id(self)]
        # selecting by an index array copies every parameter array
        result = self.with_nodes(np.arange(self.number_of_nodes))
        memo[id(self)] = result
        return result
