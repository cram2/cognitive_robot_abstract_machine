from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from random_events.interval import Bound, Interval
from random_events.product_algebra import SimpleEvent
from random_events.variable import Variable
from sortedcontainers import SortedSet
from typing_extensions import List, Optional, Self, Type

from probabilistic_model.distributions.distributions import DiracDeltaDistribution
from probabilistic_model.exceptions import ShapeMismatchError
from probabilistic_model.probabilistic_circuit.tensorized.array_types import (
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
        cls, variable_index: int, distributions: List[DiracDeltaDistribution]
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
    def concatenate(cls, layers: List[Self]) -> Self:
        return cls(
            layers[0].variable,
            np.concatenate([layer.location for layer in layers]),
            np.concatenate([layer.density_cap for layer in layers]),
            layers[0].tolerance,
        )

    def log_likelihood_of_nodes_from_column(
        self, values: SampleColumn
    ) -> SampleNodeValues:
        column = np.asarray(values, dtype=float).reshape(-1, 1)
        hit = np.abs(column - self.location) < self.tolerance
        with np.errstate(divide="ignore"):
            return np.where(hit, np.log(self.density_cap), -np.inf)

    def cumulative_distribution_of_nodes_from_column(
        self, values: SampleColumn
    ) -> SampleNodeValues:
        column = np.asarray(values, dtype=float).reshape(-1, 1)
        return (column >= self.location - self.tolerance).astype(float)

    def moment_of_nodes_own(
        self, order: int, center: float, variable: Variable
    ) -> NodeValues:
        if order == 0:
            return np.ones(self.number_of_nodes)
        if order == 1:
            return self.location - center
        return np.zeros(self.number_of_nodes)

    def sample_of_nodes(self, nodes: NodeIndices, variables: SortedSet) -> SampleColumn:
        return self.location[nodes].astype(float)

    def type_of_truncated_layer(
        self, assignment: Interval, singleton_allowed: bool
    ) -> Type[Layer]:
        return self.__class__

    def contains(self, assignment: Interval) -> NodeMask:
        """
        :param assignment: The assignment of the variable of this layer.
        :return: Whether the assignment contains the location of every node.
        """
        inside = np.zeros(self.number_of_nodes, dtype=bool)
        for interval in assignment.simple_sets:
            left = (
                interval.lower <= self.location
                if interval.left == Bound.CLOSED
                else interval.lower < self.location
            )
            right = (
                self.location <= interval.upper
                if interval.right == Bound.CLOSED
                else self.location < interval.upper
            )
            inside |= left & right
        return inside

    def log_truncated_of_assignment(
        self, assignment: Interval, singleton_allowed: bool
    ) -> LayerWithLogProbabilities:
        """
        Truncating a Dirac delta either keeps it unchanged or makes it impossible, so
        the whole layer is truncated by testing which locations the assignment contains.
        """
        return LayerWithLogProbabilities(
            self.__deepcopy__(), np.where(self.contains(assignment), 0.0, -np.inf)
        )

    @memoized
    def probability_of_simple_event_of_nodes(
        self,
        event: SimpleEvent,
        variables: SortedSet,
        cache: Optional[QueryCache] = None,
    ) -> NodeValues:
        # all of the mass sits on one point, so whether a bound is open or closed
        # decides whether it is counted
        return self.contains(event[variables[self.variable]]).astype(float)

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

    def __deepcopy__(self, memo=None) -> DiracDeltaLayer:
        if memo is None:
            memo = {}
        if id(self) in memo:
            return memo[id(self)]
        result = self.__class__(
            self.variable,
            self.location.copy(),
            self.density_cap.copy(),
            self.tolerance,
        )
        memo[id(self)] = result
        return result
