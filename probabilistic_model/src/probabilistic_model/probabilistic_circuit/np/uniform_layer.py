from __future__ import annotations

import numpy as np
import numpy.typing as npt
from random_events.interval import Interval
from random_events.variable import Variable
from sortedcontainers import SortedSet
from typing_extensions import Dict, List, Optional, Self, Tuple, Type

from probabilistic_model.distributions.uniform import UniformDistribution
from probabilistic_model.probabilistic_circuit.np.inner_layer import memoized
from probabilistic_model.probabilistic_circuit.np.input_layer import (
    ContinuousLayerWithFiniteSupport,
)


class UniformLayer(ContinuousLayerWithFiniteSupport):
    """
    A layer of uniform distributions over one continuous variable.
    """

    @classmethod
    def rustworkx_classes(cls) -> Tuple[Type, ...]:
        return (UniformDistribution,)

    @property
    def number_of_own_parameters(self) -> int:
        return 2 * self.number_of_nodes

    def log_probability_density_function_value(self) -> npt.NDArray:
        """
        :return: The log-density of every node.
        """
        with np.errstate(divide="ignore"):
            return -np.log(self.upper - self.lower)

    @memoized("log_likelihood")
    def log_likelihood_of_nodes(
        self, x: npt.NDArray, cache: Optional[Dict] = None
    ) -> npt.NDArray:
        return np.where(
            self.included_condition(self.column_of(x)),
            self.log_probability_density_function_value(),
            -np.inf,
        )

    def cumulative_distribution_of_nodes_from_column(
        self, x: npt.NDArray
    ) -> npt.NDArray:
        column = np.asarray(x, dtype=float).reshape(-1, 1)
        result = (column - self.lower) / (self.upper - self.lower)
        return np.clip(result, 0.0, 1.0)

    def moment_of_nodes_own(
        self, order: int, center: float, variable: Variable
    ) -> npt.NDArray:
        density = np.exp(self.log_probability_density_function_value())

        def evaluate_integral_at(x: npt.NDArray) -> npt.NDArray:
            return density * (x - center) ** (order + 1) / (order + 1)

        return evaluate_integral_at(self.upper) - evaluate_integral_at(self.lower)

    def node_distribution(self, index: int, variable: Variable) -> UniformDistribution:
        return UniformDistribution(
            variable=variable, interval=self.simple_interval_of(index)
        )

    @classmethod
    def from_distributions(
        cls, variable_index: int, distributions: List[UniformDistribution]
    ) -> Self:
        interval = np.array(
            [
                [distribution.interval.lower, distribution.interval.upper]
                for distribution in distributions
            ],
            dtype=float,
        )
        bounds = np.array(
            [
                [int(distribution.interval.left), int(distribution.interval.right)]
                for distribution in distributions
            ],
            dtype=np.int64,
        )
        return cls(variable_index, interval, bounds)

    def sample_of_node(
        self, node: int, amount: int, variables: SortedSet
    ) -> npt.NDArray:
        return np.random.uniform(self.lower[node], self.upper[node], amount)

    def log_truncated_of_assignment(
        self, assignment: Interval, singleton_allowed: bool
    ) -> Optional[Tuple["UniformLayer", npt.NDArray]]:
        """
        Truncate all nodes to one simple interval at once.

        A uniform truncated to an interval is the uniform over the intersection of the
        two, so the whole layer is truncated by intersecting its bounds with the interval
        and reading the probabilities off its own cumulative distribution. This mirrors
        :meth:`UniformDistribution.log_conditional_from_simple_interval_if_not_singleton`
        exactly, node by node.

        A composite assignment splits a node into one piece per simple interval and a
        singleton turns it into a Dirac delta; neither keeps the layer a uniform layer, so
        both fall back to the generic path.
        """
        if len(assignment.simple_sets) != 1:
            return None

        interval = assignment.simple_sets[0]
        if singleton_allowed and interval.is_singleton():
            return None

        lower, upper = float(interval.lower), float(interval.upper)
        left_bound, right_bound = int(interval.left), int(interval.right)

        cumulative = self.cumulative_distribution_of_nodes_from_column(
            np.array([lower, upper])
        )
        probability = cumulative[1] - cumulative[0]
        alive = probability > 0

        # the bounds of the intersection: the tighter side wins, and where the two
        # bounds coincide the interval is open if either of them is open. Bound.OPEN is
        # the larger value, so that is a maximum.
        own_left, own_right = self.bounds[:, 0], self.bounds[:, 1]
        new_left = np.where(
            self.lower > lower,
            own_left,
            np.where(self.lower < lower, left_bound, np.maximum(own_left, left_bound)),
        )
        new_right = np.where(
            self.upper < upper,
            own_right,
            np.where(
                self.upper > upper, right_bound, np.maximum(own_right, right_bound)
            ),
        )

        # impossible nodes keep their parameters and are dropped by the prune pass
        interval_of_nodes = np.where(
            alive[:, None],
            np.stack([np.maximum(self.lower, lower), np.minimum(self.upper, upper)], 1),
            self.interval,
        )
        bounds_of_nodes = np.where(
            alive[:, None], np.stack([new_left, new_right], axis=1), self.bounds
        )
        log_probabilities = np.where(
            alive, np.log(np.where(alive, probability, 1.0)), -np.inf
        )

        return (
            self.__class__(self.variable, interval_of_nodes, bounds_of_nodes),
            log_probabilities,
        )
