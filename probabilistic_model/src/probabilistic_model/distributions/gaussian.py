from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional

import numpy as np
import numpy.typing as npt
from numpy import nextafter
from scipy.stats import gamma, norm
from typing_extensions import Self, Tuple

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

    def log_conditional_from_simple_interval(
        self, interval: SimpleInterval, singleton_allowed: bool = False
    ) -> Tuple[Optional[ContinuousDistribution], float]:
        if singleton_allowed and interval.is_singleton():
            log_likelihood = self.log_likelihood(np.array([[interval.lower]]))[0]
            if log_likelihood == -np.inf:
                return None, -np.inf

            return (
                DiracDeltaDistribution(
                    variable=self.variable,
                    location=interval.lower,
                    density_cap=1.0,
                ),
                log_likelihood,
            )

        cdf_values = self.cumulative_distribution_function(
            simple_interval_as_array(interval).reshape(-1, 1)
        )
        probability: float = cdf_values[1] - cdf_values[0]
        if probability <= 0.0:
            return None, -np.inf

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
    """

    @property
    def normalizing_constant(self) -> float:
        r"""
        Helper method to calculate

        .. math::

            Z = \mathbf{\Phi}\left( \frac{\text{self.interval.upper} - \mu}{\sigma} \right)
            - \mathbf{\Phi}\left( \frac{\text{self.interval.lower} - \mu}{\sigma} \right)
        """
        return (
            GaussianDistribution.cumulative_distribution_function(
                self, np.array([[self.upper]])
            )
            - GaussianDistribution.cumulative_distribution_function(
                self, np.array([[self.lower]])
            )
        )[0]

    @property
    def cumulative_density_function_to_lower(self):
        return GaussianDistribution.cumulative_distribution_function(
            self, np.array([[self.lower]])
        )[0]

    def log_likelihood_without_bounds_check(self, x: npt.NDArray) -> npt.NDArray:
        return GaussianDistribution.log_likelihood(self, x) - np.log(
            self.normalizing_constant
        )

    def cumulative_distribution_function(self, x: npt.NDArray) -> npt.NDArray:
        result = np.zeros(len(x))
        non_zero_condition = self.left_included_condition(x)
        x_non_zero = x[non_zero_condition].reshape(-1, 1)
        cumulative_density_function_non_zero = (
            GaussianDistribution.cumulative_distribution_function(self, x_non_zero)
        )
        result[non_zero_condition[:, 0]] = (
            cumulative_density_function_non_zero
            - self.cumulative_density_function_to_lower
        ) / self.normalizing_constant
        result = np.minimum(1, result)
        return result

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

    def rejection_sample(self, amount: int) -> npt.NDArray:
        """
        Rejection sample from the distribution.

        .. note::
            Be aware that this may be inefficient.
            The acceptance probability is self.normalizing_constant.
        """
        samples = super().sample(amount)
        log_likelihoods = self.log_likelihood(samples)
        samples = samples[log_likelihoods > -np.inf]
        rejected_samples = amount - len(samples)
        if rejected_samples > 0:
            samples = np.concatenate((samples, self.rejection_sample(rejected_samples)))
        return samples

    def moment(self, order: OrderType, center: CenterType) -> MomentType:
        r"""
        Helper method to calculate the moment of a Truncated Gaussian distribution.

        .. note::
            This method follows the equation (2.8) in :cite:p:`ogasawara2022moments`.

        .. math::

            \mathbb{E} \left[ \left( X-center \right)^{order} \right]\mathbb{1}_{\left[ lower , upper \right]}(x)
            = \sigma^{order} \frac{1}{\Phi(upper)-\Phi(lower)} \sum_{k=0}^{order} \binom{order}{k} I_k (-center)^{(order-k)}.

        where:

        .. math::

            I_k = \frac{2^{\frac{k}{2}}}{\sqrt{\pi}}\Gamma \left( \frac{k+1}{2} \right) \left[ sgn \left(upper\right)
            \mathbb{1}\left \{ k=2 \nu \right \} + \mathbb{1} \left\{k = 2\nu -1 \right\} \frac{1}{2}
            F_{\Gamma} \left( \frac{upper^2}{2},\frac{k+1}{2} \right) - sgn \left(lower\right) \mathbb{1}\left \{ k=2 \nu \right \}
            + \mathbb{1} \left\{k = 2\nu -1 \right\} \frac{1}{2} F_{\Gamma} \left( \frac{lower^2}{2},\frac{k+1}{2} \right) \right]

        :return: The moment of the distribution.
        """

        order = order[self.variable]
        center = center[self.variable]

        lower_bound = self.transform_to_standard_normal(
            self.lower
        )  # normalize the lower bound
        upper_bound = self.transform_to_standard_normal(
            self.upper
        )  # normalize the upper bound
        normalized_center = self.transform_to_standard_normal(
            center
        )  # normalize the center
        truncated_moment = 0

        for value in range(order + 1):

            multiplying_constant = (
                math.comb(order, value)
                * 2 ** (value / 2)
                * math.gamma((value + 1) / 2)
                / math.sqrt(math.pi)
            )

            if value % 2 == 0:
                bound_selection_lower = np.sign(lower_bound)
                bound_selection_upper = np.sign(upper_bound)
            else:
                bound_selection_lower = 1
                bound_selection_upper = 1

            gamma_term_lower = (
                -0.5
                * gamma.cdf(lower_bound**2 / 2, (value + 1) / 2)
                * bound_selection_lower
            )
            gamma_term_upper = (
                0.5
                * gamma.cdf(upper_bound**2 / 2, (value + 1) / 2)
                * bound_selection_upper
            )

            truncated_moment += (
                multiplying_constant
                * (gamma_term_lower + gamma_term_upper)
                * (-normalized_center) ** (order - value)
            )

        truncated_moment *= (self.scale**order) / self.normalizing_constant

        return VariableMap({self.variable: truncated_moment})

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

    def transform_to_standard_normal(self, number: float) -> float:
        """
        Transform the number to the standard normal distribution.
        :param number: The number to transform
        :return: The transformed bound
        """
        return (number - self.location) / self.scale

    def robert_rejection_sample(self, amount: int) -> npt.NDArray:
        """
        Use robert rejection sampling to sample from the truncated Gaussian distribution.

        :param amount: The amount of samples to generate
        :return: The samples
        """
        # handle the case where the distribution is not the standard normal
        new_interval = SimpleInterval.from_data(
            self.transform_to_standard_normal(self.interval.lower),
            self.transform_to_standard_normal(self.interval.upper),
            self.interval.left,
            self.interval.right,
        )
        standard_distribution = self.__class__(
            variable=self.variable, interval=new_interval, location=0, scale=1
        )

        # enforce an upper bound if it is infinite
        if standard_distribution.interval.upper == np.inf:
            standard_distribution.interval.upper = (
                standard_distribution.interval.lower + 10
            )

        # enforce a lower bound if it is infinite
        if standard_distribution.interval.lower == -np.inf:
            standard_distribution.interval.lower = (
                standard_distribution.interval.upper - 10
            )

        # sample from double truncated standard normal instead
        samples = standard_distribution.robert_rejection_sample_from_standard_normal_with_double_truncation(
            amount
        )

        # transform samples to this distributions mean and scale
        samples *= self.scale
        samples += self.location

        return samples

    def robert_rejection_sample_from_standard_normal_with_double_truncation(
        self, amount: int
    ) -> np.ndarray:
        """
        Use robert rejection sampling to sample from the truncated standard normal distribution.
        Resamples as long as the amount of samples is not reached.

        :param amount: The amount of samples to generate
        :return: The samples
        """
        assert self.scale == 1 and self.location == 0
        # sample from uniform distribution over this distribution's interval
        accepted_samples = np.array([])
        while len(accepted_samples) < amount:
            accepted_samples = np.append(
                accepted_samples,
                self.robert_rejection_sample_from_standard_normal_with_double_truncation_helper(
                    amount - len(accepted_samples)
                ),
            )
        return accepted_samples

    def robert_rejection_sample_from_standard_normal_with_double_truncation_helper(
        self, amount: int
    ) -> np.ndarray:
        """
        Use robert rejection sampling to sample from the truncated standard normal distribution.

        :param amount: The maximum number of samples to generate. The actual number of samples can be lower due to
            rejection sampling.
        :return: The samples
        """
        uniform_samples = np.random.uniform(self.lower, self.upper, amount)

        # if the mean in the interval
        if self.interval.contains(0):
            limiting_function = np.exp((uniform_samples**2) / -2)

        # if the mean is below the interval
        elif self.upper <= 0:
            limiting_function = np.exp(
                (self.interval.upper**2 - uniform_samples**2) / 2
            )

        # if the mean is above the interval
        elif self.lower >= 0:
            limiting_function = np.exp(
                (self.interval.lower**2 - uniform_samples**2) / 2
            )
        else:
            raise ValueError("This should never happen")

        # generate standard uniform samples as acceptance probabilities
        acceptance_probabilities = np.random.uniform(0, 1, amount)

        # accept samples that are below the limiting function
        accepted_samples = uniform_samples[
            acceptance_probabilities <= limiting_function
        ]
        return accepted_samples

    def sample(self, amount: int) -> npt.NDArray:
        if self.upper == np.inf and self.lower == -np.inf:
            return super().sample(amount)
        return self.robert_rejection_sample(amount).reshape(-1, 1)

    def apply_translation(self, translation: VariableMap[Variable, float]):
        super().apply_translation(translation)
        GaussianDistribution.apply_translation(self, translation)

    def apply_scaling(self, scale: VariableMap[Variable, float]):
        super().apply_scaling(scale)
        GaussianDistribution.apply_scaling(self, scale)
