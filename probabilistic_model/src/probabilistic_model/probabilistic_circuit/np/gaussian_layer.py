from __future__ import annotations

import math

import numpy as np
import numpy.typing as npt
from random_events.variable import Variable
from scipy.stats import norm
from sortedcontainers import SortedSet
from typing_extensions import Any, Dict, List, Optional, Self, Tuple, Type

from probabilistic_model.distributions.gaussian import (
    GaussianDistribution,
    TruncatedGaussianDistribution,
)
from probabilistic_model.exceptions import ShapeMismatchError
from probabilistic_model.probabilistic_circuit.np.inner_layer import memoized
from probabilistic_model.probabilistic_circuit.np.input_layer import (
    ContinuousLayer,
    ContinuousLayerWithFiniteSupport,
)


class GaussianLayer(ContinuousLayer):
    """
    A layer of Gaussian distributions over one continuous variable.
    """

    location: npt.NDArray
    """
    The mean of every node.
    """

    scale: npt.NDArray
    """
    The standard deviation of every node.
    """

    def __init__(self, variable: int, location: npt.NDArray, scale: npt.NDArray):
        super().__init__(variable)
        self.location = np.asarray(location, dtype=float).reshape(-1)
        self.scale = np.asarray(scale, dtype=float).reshape(-1)

    @property
    def number_of_nodes(self) -> int:
        return len(self.location)

    @property
    def number_of_own_parameters(self) -> int:
        return 2 * self.number_of_nodes

    @classmethod
    def rustworkx_classes(cls) -> Tuple[Type, ...]:
        return (GaussianDistribution,)

    def validate_own(self):
        if self.location.shape != self.scale.shape:
            raise ShapeMismatchError(self.location.shape, self.scale.shape)

    @memoized("log_likelihood")
    def log_likelihood_of_nodes(
        self, x: npt.NDArray, cache: Optional[Dict] = None
    ) -> npt.NDArray:
        column = self.column_of(x).astype(float).reshape(-1, 1)
        return norm.logpdf(column, loc=self.location, scale=self.scale)

    def cumulative_distribution_of_nodes_from_column(
        self, x: npt.NDArray
    ) -> npt.NDArray:
        column = np.asarray(x, dtype=float).reshape(-1, 1)
        return norm.cdf(column, loc=self.location, scale=self.scale)

    def raw_moments(self, order: int) -> List[npt.NDArray]:
        r"""
        The raw moments of order ``0`` to ``order`` of every node.

        .. math::

            E(X^n) = \sum_{j=0}^{\lfloor \frac{n}{2}\rfloor}
            \binom{n}{2j}\dfrac{\mu^{n-2j}\sigma^{2j}(2j)!}{j!2^j}

        :param order: The highest order to calculate.
        :return: One array of shape (#nodes,) per order.
        """
        result = []
        for current_order in range(order + 1):
            raw_moment = np.zeros(self.number_of_nodes)
            for j in range(math.floor(current_order / 2) + 1):
                raw_moment += (
                    math.comb(current_order, 2 * j)
                    * self.location ** (current_order - 2 * j)
                    * self.scale ** (2 * j)
                    * math.factorial(2 * j)
                    / (math.factorial(j) * (2**j))
                )
            result.append(raw_moment)
        return result

    def moment_of_nodes_own(
        self, order: int, center: float, variable: Variable
    ) -> npt.NDArray:
        raw_moments = self.raw_moments(order)
        result = np.zeros(self.number_of_nodes)
        for current_order in range(order + 1):
            result += (
                math.comb(order, current_order)
                * raw_moments[current_order]
                * (-center) ** (order - current_order)
            )
        return result

    def node_distribution(self, index: int, variable: Variable) -> GaussianDistribution:
        return GaussianDistribution(
            variable=variable,
            location=float(self.location[index]),
            scale=float(self.scale[index]),
        )

    @classmethod
    def from_distributions(
        cls, variable_index: int, distributions: List[GaussianDistribution]
    ) -> Self:
        return cls(
            variable_index,
            np.array([distribution.location for distribution in distributions]),
            np.array([distribution.scale for distribution in distributions]),
        )

    def select_nodes(self, mask: npt.NDArray) -> Self:
        return self.__class__(self.variable, self.location[mask], self.scale[mask])

    def sample_of_node(
        self, node: int, amount: int, variables: SortedSet
    ) -> npt.NDArray:
        return norm.rvs(loc=self.location[node], scale=self.scale[node], size=amount)

    def apply_translation_own(self, translation: npt.NDArray):
        self.location = self.location + translation[self.variable]

    def apply_scaling_own(self, scaling: npt.NDArray):
        self.location = self.location * scaling[self.variable]
        self.scale = self.scale * scaling[self.variable]

    def __deepcopy__(self, memo=None) -> GaussianLayer:
        if memo is None:
            memo = {}
        if id(self) in memo:
            return memo[id(self)]
        result = self.__class__(self.variable, self.location.copy(), self.scale.copy())
        memo[id(self)] = result
        return result

    def to_json(self, **kwargs) -> Dict[str, Any]:
        result = super().to_json(**kwargs)
        result["location"] = self.location.tolist()
        result["scale"] = self.scale.tolist()
        return result

    @classmethod
    def _from_json(cls, data: Dict[str, Any], **kwargs) -> Self:
        return cls(
            data["variable"], np.array(data["location"]), np.array(data["scale"])
        )


class TruncatedGaussianLayer(ContinuousLayerWithFiniteSupport):
    """
    A layer of truncated Gaussian distributions over one continuous variable.

    This is the layer that truncating a :class:`GaussianLayer` to a bounded interval
    produces.
    """

    location: npt.NDArray
    """
    The mean of the untruncated Gaussian of every node.
    """

    scale: npt.NDArray
    """
    The standard deviation of the untruncated Gaussian of every node.
    """

    def __init__(
        self,
        variable: int,
        interval: npt.NDArray,
        location: npt.NDArray,
        scale: npt.NDArray,
        bounds: Optional[npt.NDArray] = None,
    ):
        super().__init__(variable, interval, bounds)
        self.location = np.asarray(location, dtype=float).reshape(-1)
        self.scale = np.asarray(scale, dtype=float).reshape(-1)

    @property
    def number_of_own_parameters(self) -> int:
        return 4 * self.number_of_nodes

    @classmethod
    def rustworkx_classes(cls) -> Tuple[Type, ...]:
        return (TruncatedGaussianDistribution,)

    @property
    def cumulative_distribution_to_lower(self) -> npt.NDArray:
        """
        :return: The untruncated cumulative distribution at the lower bound of every
            node.
        """
        return norm.cdf(self.lower, loc=self.location, scale=self.scale)

    @property
    def normalizing_constant(self) -> npt.NDArray:
        """
        :return: The probability of the support of every node under its untruncated
            Gaussian.
        """
        return (
            norm.cdf(self.upper, loc=self.location, scale=self.scale)
            - self.cumulative_distribution_to_lower
        )

    @memoized("log_likelihood")
    def log_likelihood_of_nodes(
        self, x: npt.NDArray, cache: Optional[Dict] = None
    ) -> npt.NDArray:
        column = self.column_of(x).astype(float).reshape(-1, 1)
        with np.errstate(divide="ignore"):
            density = norm.logpdf(
                column, loc=self.location, scale=self.scale
            ) - np.log(self.normalizing_constant)
        return np.where(self.included_condition(self.column_of(x)), density, -np.inf)

    def cumulative_distribution_of_nodes_from_column(
        self, x: npt.NDArray
    ) -> npt.NDArray:
        column = np.asarray(x, dtype=float).reshape(-1, 1)
        left_included = np.where(
            self.left_closed, self.lower <= column, self.lower < column
        )
        untruncated = norm.cdf(column, loc=self.location, scale=self.scale)
        result = (
            untruncated - self.cumulative_distribution_to_lower
        ) / self.normalizing_constant
        return np.minimum(1.0, np.where(left_included, result, 0.0))

    def node_distribution(
        self, index: int, variable: Variable
    ) -> TruncatedGaussianDistribution:
        return TruncatedGaussianDistribution(
            variable=variable,
            interval=self.simple_interval_of(index),
            location=float(self.location[index]),
            scale=float(self.scale[index]),
        )

    @classmethod
    def from_distributions(
        cls, variable_index: int, distributions: List[TruncatedGaussianDistribution]
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
        return cls(
            variable_index,
            interval,
            np.array([distribution.location for distribution in distributions]),
            np.array([distribution.scale for distribution in distributions]),
            bounds,
        )

    def select_nodes(self, mask: npt.NDArray) -> Self:
        return self.__class__(
            self.variable,
            self.interval[mask],
            self.location[mask],
            self.scale[mask],
            self.bounds[mask],
        )

    @classmethod
    def concatenate(cls, layers: List[Self]) -> Self:
        return cls(
            layers[0].variable,
            np.concatenate([layer.interval for layer in layers]),
            np.concatenate([layer.location for layer in layers]),
            np.concatenate([layer.scale for layer in layers]),
            np.concatenate([layer.bounds for layer in layers]),
        )

    def apply_translation_own(self, translation: npt.NDArray):
        super().apply_translation_own(translation)
        self.location = self.location + translation[self.variable]

    def apply_scaling_own(self, scaling: npt.NDArray):
        super().apply_scaling_own(scaling)
        self.location = self.location * scaling[self.variable]
        self.scale = self.scale * scaling[self.variable]

    def __deepcopy__(self, memo=None) -> TruncatedGaussianLayer:
        if memo is None:
            memo = {}
        if id(self) in memo:
            return memo[id(self)]
        result = self.__class__(
            self.variable,
            self.interval.copy(),
            self.location.copy(),
            self.scale.copy(),
            self.bounds.copy(),
        )
        memo[id(self)] = result
        return result

    def to_json(self, **kwargs) -> Dict[str, Any]:
        result = super().to_json(**kwargs)
        result["location"] = self.location.tolist()
        result["scale"] = self.scale.tolist()
        return result

    @classmethod
    def _from_json(cls, data: Dict[str, Any], **kwargs) -> Self:
        return cls(
            data["variable"],
            np.array(data["interval"]),
            np.array(data["location"]),
            np.array(data["scale"]),
            np.array(data["bounds"]),
        )
