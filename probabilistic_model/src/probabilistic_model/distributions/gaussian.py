from __future__ import annotations
from numpy import nextafter

from scipy.stats import gamma, norm

from .distributions import *
from ..utils import simple_interval_as_array


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

    def __init__(self, variable: Continuous, location: float, scale: float):
        super().__init__()
        self.variable = variable
        self.location = location
        self.scale = scale

    @property
    def univariate_support(self) -> Interval:
        return reals()

    def log_likelihood(self, x: np.array) -> np.array:
        return norm.logpdf(x[:, 0], loc=self.location, scale=self.scale)

    def cdf(self, x: np.array) -> np.array:
        return norm.cdf(x[:, 0], loc=self.location, scale=self.scale)

    def univariate_log_mode(self) -> Tuple[AbstractCompositeSet, float]:
        return singleton(self.location), self.log_likelihood(np.array([[self.location]]))[0]

    def sample(self, amount: int) -> np.array:
        return norm.rvs(loc=self.location, scale=self.scale, size=(amount, 1))

    def ppf(self, value):
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

            raw_moment += (math.comb(order, 2 * j) * mu_term * sigma_term * math.factorial(2 * j) / (
                    math.factorial(j) * (2 ** j)))

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
            moment += math.comb(order, order_) * raw_moments[order_] * (-center) ** (order - order_)

        return VariableMap({self.variable: moment})

    def log_conditional_from_simple_interval(self, interval: SimpleInterval) -> Tuple[Optional[TruncatedGaussianDistribution], float]:
        cdf_values = self.cdf(simple_interval_as_array(interval).reshape(-1, 1))
        probability = cdf_values[1] - cdf_values[0]
        if probability <= 0.0:
            return None, -np.inf
        return TruncatedGaussianDistribution(self.variable, interval, self.location, self.scale), np.log(probability)

    def __eq__(self, other: Self):
        return super().__eq__(other) and self.location == other.location and self.scale == other.scale

    @property
    def representation(self):
        return f"N({self.variable.name} | {self.location}, {self.scale})"

    def __repr__(self):
        return f"N({self.variable.name})"

    def __copy__(self):
        return self.__class__(self.variable, self.location, self.scale)

    def __deepcopy__(self, memo=None):
        if memo is None:
            memo = {}
        id_self = id(self)
        if id_self in memo:
            return memo[id_self]

        variable = self.variable.__class__(self.variable.name)
        result = self.__class__(variable, self.location, self.scale)
        memo[id_self] = result
        return result

    def to_json(self) -> Dict[str, Any]:
        return {**super().to_json(), "location": self.location, "scale": self.scale}

    @classmethod
    def _from_json(cls, data: Dict[str, Any]) -> Self:
        variable = Continuous.from_json(data["variable"])
        return cls(variable, data["location"], data["scale"])

    @property
    def abbreviated_symbol(self) -> str:
        return "N"

    def translate(self, translation: VariableMap[Variable, float]):
        self.location += translation[self.variable]

    def scale(self, scaling: VariableMap[Variable, float]):
        self.location *= scaling[self.variable]
        self.scale *= scaling[self.variable]

class TruncatedGaussianDistribution(ContinuousDistributionWithFiniteSupport, GaussianDistribution):
    """
    Class for Truncated Gaussian distributions.
    """

    def __init__(self, variable: Continuous, interval: SimpleInterval, location: float, scale: float):
        GaussianDistribution.__init__(self, variable, location, scale)
        self.interval = interval

    @property
    def normalizing_constant(self) -> float:
        r"""
            Helper method to calculate

            .. math::

            Z = {\mathbf{\Phi}\left ( \frac{self.interval.upper-\mu}{\sigma} \right )-\mathbf{\Phi}
            \left( \frac{self.interval.lower-\mu}{\sigma} \right )}

        """
        return (GaussianDistribution.cdf(self, np.array([[self.upper]])) - GaussianDistribution.cdf(self, np.array(
            [[self.lower]])))[0]

    @property
    def cdf_of_lower(self):
        return GaussianDistribution.cdf(self, np.array([[self.lower]]))[0]

    def log_likelihood_without_bounds_check(self, x: np.array) -> np.array:
        return GaussianDistribution.log_likelihood(self, x) - np.log(self.normalizing_constant)

    def cdf(self, x: np.array) -> np.array:
        result = np.zeros(len(x))
        non_zero_condition = self.left_included_condition(x)
        x_non_zero = x[non_zero_condition].reshape(-1, 1)
        cdf_non_zero = GaussianDistribution.cdf(self, x_non_zero)
        result[non_zero_condition[:, 0]] = (cdf_non_zero - self.cdf_of_lower) / self.normalizing_constant
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
        return singleton(value), self.log_likelihood_without_bounds_check(np.array([[value]]))[0]

    def rejection_sample(self, amount: int) -> np.array:
        """
        .. note::
            This uses rejection sampling and hence is inefficient.
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

                    \mathbb{E} \left[ \left( X-center \right)^{order} \right]\mathds{1}_{\left[ lower , upper \right]}(x)
                    = \sigma^{order} \frac{1}{\Phi(upper)-\Phi(lower)} \sum_{k=0}^{order} \binom{order}{k} I_k (-center)^{(order-k)}.

                    where:

                    .. math::

                        I_k = \frac{2^{\frac{k}{2}}}{\sqrt{\pi}}\Gamma \left( \frac{k+1}{2} \right) \left[ sgn \left(upper\right)
                         \mathds{1}\left \{ k=2 \nu \right \} + \mathds{1} \left\{k = 2\nu -1 \right\} \frac{1}{2}
                          F_{\Gamma} \left( \frac{upper^2}{2},\frac{k+1}{2} \right) - sgn \left(lower\right) \mathds{1}\left \{ k=2 \nu \right \}
                         + \mathds{1} \left\{k = 2\nu -1 \right\} \frac{1}{2} F_{\Gamma} \left( \frac{lower^2}{2},\frac{k+1}{2} \right) \right]

                :return: The moment of the distribution.

                """

        order = order[self.variable]
        center = center[self.variable]

        lower_bound = self.transform_to_standard_normal(self.lower)  # normalize the lower bound
        upper_bound = self.transform_to_standard_normal(self.upper)  # normalize the upper bound
        normalized_center = self.transform_to_standard_normal(center)  # normalize the center
        truncated_moment = 0

        for k in range(order + 1):

            multiplying_constant = math.comb(order, k) * 2 ** (k / 2) * math.gamma((k + 1) / 2) / math.sqrt(math.pi)

            if k % 2 == 0:
                bound_selection_lower = np.sign(lower_bound)
                bound_selection_upper = np.sign(upper_bound)
            else:
                bound_selection_lower = 1
                bound_selection_upper = 1

            gamma_term_lower = -0.5 * gamma.cdf(lower_bound ** 2 / 2, (k + 1) / 2) * bound_selection_lower
            gamma_term_upper = 0.5 * gamma.cdf(upper_bound ** 2 / 2, (k + 1) / 2) * bound_selection_upper

            truncated_moment += (
                    multiplying_constant * (gamma_term_lower + gamma_term_upper) * (-normalized_center) ** (order - k))

        truncated_moment *= (self.scale ** order) / self.normalizing_constant

        return VariableMap({self.variable: truncated_moment})

    def __eq__(self, other):
        return super().__eq__(other) and self.interval == other.interval

    @property
    def representation(self):
        return f"N({self.variable.name} | {self.location}, {self.scale}, {self.interval})"

    def __copy__(self):
        return self.__class__(self.variable, self.interval, self.location, self.scale)

    def __deepcopy__(self, memo=None):
        if memo is None:
            memo = {}
        id_self = id(self)
        if id_self in memo:
            return memo[id_self]
        import copy
        variable = self.variable.__class__(self.variable.name)
        interval = self.interval.__deepcopy__()
        result = self.__class__(variable, interval, self.location, self.scale)
        memo[id_self] = result
        return result

    def to_json(self) -> Dict[str, Any]:
        return {**super().to_json(), "interval": self.interval.to_json()}

    @classmethod
    def _from_json(cls, data: Dict[str, Any]) -> Self:
        variable = Continuous.from_json(data["variable"])
        interval = SimpleInterval.from_json(data["interval"])
        return cls(variable, interval, data["location"], data["scale"])

    def transform_to_standard_normal(self, number: float) -> float:
        """
        Transform the number to the standard normal distribution.
        :param number: The number to transform
        :return: The transformed bound
        """
        return (number - self.location) / self.scale

    def robert_rejection_sample(self, amount: int) -> np.ndarray:
        """
        Use robert rejection sampling to sample from the truncated Gaussian distribution.

        :param amount: The amount of samples to generate
        :return: The samples
        """
        # handle the case where the distribution is not the standard normal
        new_interval = SimpleInterval(self.transform_to_standard_normal(self.interval.lower),
                                      self.transform_to_standard_normal(self.interval.upper), self.interval.left,
                                      self.interval.right)
        standard_distribution = self.__class__(self.variable, new_interval, 0, 1)

        # enforce an upper bound if it is infinite
        if standard_distribution.interval.upper == np.inf:
            standard_distribution.interval.upper = standard_distribution.interval.lower + 10

        # enforce a lower bound if it is infinite
        if standard_distribution.interval.lower == -np.inf:
            standard_distribution.interval.lower = standard_distribution.interval.upper - 10

        # sample from double truncated standard normal instead
        samples = standard_distribution.robert_rejection_sample_from_standard_normal_with_double_truncation(amount)

        # transform samples to this distributions mean and scale
        samples *= self.scale
        samples += self.location

        return samples

    def robert_rejection_sample_from_standard_normal_with_double_truncation(self, amount: int) -> np.ndarray:
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
            accepted_samples = np.append(accepted_samples,
                self.robert_rejection_sample_from_standard_normal_with_double_truncation_helper(
                    amount - len(accepted_samples)))
        return accepted_samples

    def robert_rejection_sample_from_standard_normal_with_double_truncation_helper(self, amount: int) -> np.ndarray:
        """
        Use robert rejection sampling to sample from the truncated standard normal distribution.

        :param amount: The maximum number of samples to generate. The actual number of samples can be lower due to
            rejection sampling.
        :return: The samples
        """
        uniform_samples = np.random.uniform(self.lower, self.upper, amount)

        # if the mean in the interval
        if self.interval.contains(0):
            limiting_function = np.exp((uniform_samples ** 2) / -2)

        # if the mean is below the interval
        elif self.upper <= 0:
            limiting_function = np.exp((self.interval.upper ** 2 - uniform_samples ** 2) / 2)

        # if the mean is above the interval
        elif self.lower >= 0:
            limiting_function = np.exp((self.interval.lower ** 2 - uniform_samples ** 2) / 2)
        else:
            raise ValueError("This should never happen")

        # generate standard uniform samples as acceptance probabilities
        acceptance_probabilities = np.random.uniform(0, 1, amount)

        # accept samples that are below the limiting function
        accepted_samples = uniform_samples[acceptance_probabilities <= limiting_function]
        return accepted_samples

    def sample(self, amount: int) -> np.array:
        if self.upper == np.inf and self.lower == -np.inf:
            return super().sample(amount)
        return self.robert_rejection_sample(amount).reshape(-1, 1)

    def translate(self, translation: VariableMap[Variable, float]):
        super().translate(translation)
        GaussianDistribution.translate(self, translation)

    def scale(self, scale: VariableMap[Variable, float]):
        super().scale(scale)
        GaussianDistribution.scale(self, scale)
