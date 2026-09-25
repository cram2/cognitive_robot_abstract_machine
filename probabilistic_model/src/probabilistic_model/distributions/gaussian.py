from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional

import numpy as np
import numpy.typing as npt
from numpy import nextafter
from scipy.stats import norm, truncnorm
from typing_extensions import Self, Tuple, TYPE_CHECKING

from probabilistic_model.distributions.distributions import (
    ContinuousDistribution,
    ContinuousDistributionWithFiniteSupport,
    DiracDeltaDistribution,
)
from probabilistic_model.probabilistic_model import OrderType, CenterType, MomentType
from probabilistic_model.utils import simple_interval_as_array
from random_events.interval import Interval, reals, singleton, SimpleInterval, Bound
from random_events.product_algebra import VariableMap
from random_events.sigma_algebra import AbstractCompositeSet
from random_events.variable import Variable

if TYPE_CHECKING:
    from scipy.stats.distributions import rv_frozen


@dataclass
class GaussianDistribution(ContinuousDistribution):
    """
    Class for Gaussian distributions.
    """

    location: float
    """
    The mean of the Gaussian distribution.
    """

    scale: float
    """
    The standard deviation of the Gaussian distribution.
    """

    @property
    def univariate_support(self) -> Interval:
        return reals()

    def log_likelihood(self, x: npt.NDArray) -> npt.NDArray:
        return norm.logpdf(x[:, 0], loc=self.location, scale=self.scale)

    def cumulative_distribution_function(self, x: npt.NDArray) -> npt.NDArray:
        return norm.cdf(x[:, 0], loc=self.location, scale=self.scale)

    def univariate_log_mode(self) -> Tuple[AbstractCompositeSet, float]:
        return (
            singleton(self.location),
            self.log_likelihood(np.array([[self.location]]))[0],
        )

    def sample(self, amount: int) -> npt.NDArray:
        return norm.rvs(loc=self.location, scale=self.scale, size=(amount, 1))

    def probability_point(self, value):
        return norm.ppf(value, loc=self.location, scale=self.scale)

    def raw_moment(self, order: int) -> float:
        r"""
        Helper method to calculate the raw moment of a Gaussian distribution.

        The raw moment is given by:

        .. math::

            E(X^n) = \sum_{j=0}^{\lfloor \frac{n}{2}\rfloor}\binom{n}{2j}\dfrac{\mu^{n-2j}\sigma^{2j}(2j)!}{j!2^j}.


        """
        raw_moment = 0  # Initialize the raw moment
        for j in range(math.floor(order / 2) + 1):
            mu_term = self.location ** (order - 2 * j)
            sigma_term = self.scale ** (2 * j)

            raw_moment += (
                math.comb(order, 2 * j)
                * mu_term
                * sigma_term
                * math.factorial(2 * j)
                / (math.factorial(j) * (2**j))
            )

        return raw_moment

    def moment(self, order: OrderType, center: CenterType) -> MomentType:
        r"""
        Calculate the moment of the distribution using Alessandro's (made up) Equation:

        .. math::

            E(X-center)^i = \sum_{i=0}^{order} \binom{order}{i} E[X^i] * (- center)^{(order-i)}
        """
        order = order[self.variable]
        center = center[self.variable]

        # get the raw moments from 0 to i
        raw_moments = [self.raw_moment(i) for i in range(order + 1)]

        moment = 0

        # Compute the desired moment:
        for order_ in range(order + 1):
            moment += (
                math.comb(order, order_)
                * raw_moments[order_]
                * (-center) ** (order - order_)
            )

        return VariableMap({self.variable: moment})

    def log_conditional_from_simple_interval_if_not_singleton(
        self, interval: SimpleInterval
    ) -> Tuple[Optional[ContinuousDistribution], float]:
        cdf_values = self.cumulative_distribution_function(
            simple_interval_as_array(interval).reshape(-1, 1)
        )
        probability: float = cdf_values[1] - cdf_values[0]
        if probability <= 0.0:
            return None, -np.inf

        if interval.as_composite_set() == reals():
            return GaussianDistribution(
                variable=self.variable, location=self.location, scale=self.scale
            ), np.log(probability)

        return TruncatedGaussianDistribution(
            variable=self.variable,
            interval=interval,
            location=self.location,
            scale=self.scale,
        ), np.log(probability)

    @property
    def representation(self):
        return f"N({self.variable.name} | {self.location}, {self.scale})"

    def __repr__(self):
        return f"N({self.variable.name})"

    def __copy__(self):
        return self.__class__(
            variable=self.variable, location=self.location, scale=self.scale
        )

    def __deepcopy__(self, memo=None):
        if memo is None:
            memo = {}
        id_self = id(self)
        if id_self in memo:
            return memo[id_self]

        variable = self.variable.__class__(
            name=self.variable.name, domain=self.variable.domain
        )
        result = self.__class__(
            variable=variable, location=self.location, scale=self.scale
        )
        memo[id_self] = result
        return result

    @property
    def abbreviated_symbol(self) -> str:
        return "N"

    def apply_translation(self, translation: VariableMap[Variable, float]):
        self.location += translation[self.variable]

    def apply_scaling(self, scaling: VariableMap[Variable, float]):
        self.location *= scaling[self.variable]
        self.scale *= scaling[self.variable]


@dataclass
class TruncatedGaussianDistribution(
    ContinuousDistributionWithFiniteSupport, GaussianDistribution
):
    """
    Class for Truncated Gaussian distributions.

    The computations are delegated to :data:`scipy.stats.truncnorm`.
    """

    @property
    def standardized_bounds(self) -> Tuple[float, float]:
        """
        :return: The bounds of the interval in standard deviations from the location,
            as :data:`scipy.stats.truncnorm` expects them.
        """
        return (
            (self.lower - self.location) / self.scale,
            (self.upper - self.location) / self.scale,
        )

    @property
    def truncated_normal(self) -> rv_frozen:
        """
        :return: This distribution as a frozen :data:`scipy.stats.truncnorm` distribution.
        """
        lower, upper = self.standardized_bounds
        return truncnorm(lower, upper, loc=self.location, scale=self.scale)

    def log_likelihood_without_bounds_check(self, x: npt.NDArray) -> npt.NDArray:
        return self.truncated_normal.logpdf(x[:, 0])

    def cumulative_distribution_function(self, x: npt.NDArray) -> npt.NDArray:
        return self.truncated_normal.cdf(x[:, 0])

    def univariate_log_mode(self) -> Tuple[Interval, float]:
        if self.interval.contains(self.location):
            value = self.location
        elif self.location < self.lower:
            value = self.lower
            if self.interval.left == Bound.OPEN:
                value = nextafter(value, np.inf)
        else:
            value = self.upper
            if self.interval.right == Bound.OPEN:
                value = nextafter(value, -np.inf)
        return (
            singleton(value),
            self.log_likelihood_without_bounds_check(np.array([[value]]))[0],
        )

    def sample(self, amount: int) -> npt.NDArray:
        return self.truncated_normal.rvs(size=(amount, 1))

    def moment(self, order: OrderType, center: CenterType) -> MomentType:
        """
        Calculate the moment about the center as the raw moment of the distribution
        shifted by the center. This avoids expanding it into raw moments of the
        unshifted distribution, whose differences cancel far in the tails.

        :param order: The order of the moment for the variable of this distribution.
        :param center: The center of the moment for the variable of this distribution.
        :return: The moment for the variable of this distribution.
        """
        lower, upper = self.standardized_bounds
        moment = truncnorm.moment(
            order[self.variable],
            lower,
            upper,
            loc=self.location - center[self.variable],
            scale=self.scale,
        )
        return VariableMap({self.variable: moment})

    def __eq__(self, other):
        return super().__eq__(other) and self.interval == other.interval

    @property
    def representation(self):
        return (
            f"N({self.variable.name} | {self.location}, {self.scale}, {self.interval})"
        )

    def __copy__(self):
        return self.__class__(
            variable=self.variable,
            interval=self.interval,
            location=self.location,
            scale=self.scale,
        )

    def __deepcopy__(self, memo=None):
        if memo is None:
            memo = {}
        id_self = id(self)
        if id_self in memo:
            return memo[id_self]

        variable = self.variable.__class__(self.variable.name)
        interval = self.interval.__deepcopy__()
        result = self.__class__(
            variable=variable,
            interval=interval,
            location=self.location,
            scale=self.scale,
        )
        memo[id_self] = result
        return result

    def apply_translation(self, translation: VariableMap[Variable, float]):
        super().apply_translation(translation)
        GaussianDistribution.apply_translation(self, translation)

    def apply_scaling(self, scale: VariableMap[Variable, float]):
        super().apply_scaling(scale)
        GaussianDistribution.apply_scaling(self, scale)
