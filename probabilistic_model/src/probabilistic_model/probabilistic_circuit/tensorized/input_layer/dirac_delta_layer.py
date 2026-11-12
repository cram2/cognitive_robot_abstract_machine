from __future__ import annotations

import copy
from dataclasses import dataclass

import numpy as np
from random_events.interval import Bound, Interval
from random_events.product_algebra import SimpleEvent
from random_events.variable import Variable
from sortedcontainers import SortedSet
from typing_extensions import Any, Self

from probabilistic_model.distributions.distributions import DiracDeltaDistribution
from probabilistic_model.exceptions import ShapeMismatchError
from probabilistic_model.probabilistic_circuit.tensorized.array_types import (
    BackendArray,
    NodeIndices,
    NodeMask,
    NodeValues,
    SampleColumn,
    SampleNodeValues,
    VariableValues,
)
from probabilistic_model.probabilistic_circuit.tensorized.inner_layer.base import Layer
from probabilistic_model.probabilistic_circuit.tensorized.input_layer.base import (
    AbstractContinuousLayer,
)
from probabilistic_model.probabilistic_circuit.tensorized.query_cache import (
    QueryCache,
    memoized,
)
from probabilistic_model.probabilistic_circuit.tensorized.structural_query import (
    LayerWithLogProbabilities,
)


@dataclass(eq=False, repr=False)
class DiracDeltaLayer(AbstractContinuousLayer):
    """
    A layer of Dirac delta distributions over one continuous variable.
    """

    location: NodeValues
    """
    The location of every node.
    """

    density_cap: NodeValues
    """
    The value that replaces the infinite density of every node.
    """

    tolerance: float = 1e-6
    """
    The tolerance with which a value is considered equal to the location.
    """

    @property
    def number_of_nodes(self) -> int:
        return len(self.location)

    @property
    def number_of_own_parameters(self) -> int:
        return 2 * self.number_of_nodes

    @property
    def parameters(self) -> list[BackendArray]:
        return [self.location, self.density_cap]

    def validate_own(self):
        if self.location.shape != self.density_cap.shape:
            raise ShapeMismatchError(self.location.shape, self.density_cap.shape)

    def node_distribution(
        self, index: int, variable: Variable
    ) -> DiracDeltaDistribution:
        return DiracDeltaDistribution(
            variable=variable,
            location=float(self.location[index]),
            density_cap=float(self.density_cap[index]),
            tolerance=self.tolerance,
        )

    @classmethod
    def from_distributions(
        cls, variable_index: int, distributions: list[DiracDeltaDistribution]
    ) -> Self:
        return cls(
            variable_index,
            np.array([distribution.location for distribution in distributions]),
            np.array([distribution.density_cap for distribution in distributions]),
            distributions[0].tolerance,
        )

    def select_nodes(self, mask: NodeMask) -> Self:
        return self.__class__(
            self.variable, self.location[mask], self.density_cap[mask], self.tolerance
        )

    @classmethod
    def concatenate(cls, layers: list[Self]) -> Self:
        return cls(
            layers[0].variable,
            np.concatenate([layer.location for layer in layers]),
            np.concatenate([layer.density_cap for layer in layers]),
            layers[0].tolerance,
        )

    def has_equal_parameters(self, other: Self) -> bool:
        return np.array_equal(self.location, other.location) and np.array_equal(
            self.density_cap, other.density_cap
        )

    def log_likelihood_of_nodes_from_column(
        self, values: SampleColumn
    ) -> SampleNodeValues:
        hit = (
            self.namespace.abs(self.float_column(values=values) - self.location)
            < self.tolerance
        )
        with np.errstate(divide="ignore"):
            return self.namespace.where(
                hit, self.namespace.log(self.density_cap), -np.inf
            )

    def cumulative_distribution_of_nodes_from_column(
        self, values: SampleColumn
    ) -> SampleNodeValues:
        reached = self.float_column(values=values) >= self.location - self.tolerance
        return self.namespace.astype(reached, self.namespace.float64)

    def moment_of_nodes_own(
        self, order: int, center: float, variable: Variable
    ) -> NodeValues:
        """
        Like :meth:`~probabilistic_model.probabilistic_circuit.tensorized.input_layer.ba
        se.UnivariateInputLayer.moment_of_nodes_own`, in closed form: all the mass of a
        node lies at its location.
        """
        if order == 0:
            return self.namespace.ones_like(self.location)
        if order == 1:
            return self.location - center
        return self.namespace.zeros_like(self.location)

    def sample_of_nodes(self, nodes: NodeIndices, variables: SortedSet) -> SampleColumn:
        return self.location[nodes].astype(float)

    def type_of_truncated_layer(
        self, assignment: Interval, singleton_allowed: bool
    ) -> type[Layer]:
        return self.__class__

    def contains(self, assignment: Interval) -> NodeMask:
        """
        :param assignment: The assignment of the variable of this layer.
        :return: Whether the assignment reaches the mass of every node, up to
            :attr:`tolerance`.
        """
        lower_of_window = self.location - self.tolerance
        upper_of_window = self.location + self.tolerance
        inside = self.namespace.zeros(self.number_of_nodes, dtype=self.namespace.bool)
        for simple_interval in assignment.simple_sets:
            left = (
                simple_interval.lower <= upper_of_window
                if simple_interval.left == Bound.CLOSED
                else simple_interval.lower < upper_of_window
            )
            right = (
                lower_of_window <= simple_interval.upper
                if simple_interval.right == Bound.CLOSED
                else lower_of_window < simple_interval.upper
            )
            inside = inside | (left & right)
        return inside

    def log_truncated_of_assignment(
        self, assignment: Interval, singleton_allowed: bool
    ) -> LayerWithLogProbabilities:
        """
        Truncating a Dirac delta either keeps it unchanged or makes it impossible, so
        the whole layer is truncated by testing which locations the assignment contains.

        See :meth:`~probabilistic_model.probabilistic_circuit.tensorized.input_layer.base.UnivariateInputLayer.log_truncated_of_assignment`
        for the parameters and the result.
        """
        return LayerWithLogProbabilities(
            copy.deepcopy(self), np.where(self.contains(assignment), 0.0, -np.inf)
        )

    @memoized
    def probability_of_simple_event_of_nodes(
        self,
        event: SimpleEvent,
        variables: SortedSet,
        cache: QueryCache | None = None,
    ) -> NodeValues:
        """
        Like :meth:`~probabilistic_model.probabilistic_circuit.tensorized.input_layer.base.AbstractContinuousLayer.probability_of_simple_event_of_nodes`, but a
        node has probability one if the event reaches its location up to
        :attr:`tolerance` and zero otherwise.
        """
        return self.namespace.astype(
            self.contains(event[variables[self.variable]]), self.namespace.float64
        )

    def log_conditional_of_value(self, value: float) -> LayerWithLogProbabilities:
        log_likelihood = self.log_likelihood_of_nodes_from_column(np.array([value]))[0]
        conditioned = self.__class__(
            self.variable,
            np.full(self.number_of_nodes, float(value)),
            self.density_cap.copy(),
            self.tolerance,
        )
        return LayerWithLogProbabilities(conditioned, log_likelihood)

    def apply_translation_own(self, translation: VariableValues):
        self.location = self.location + translation[self.variable]

    def apply_scaling_own(self, scaling: VariableValues):
        self.location = self.location * scaling[self.variable]
