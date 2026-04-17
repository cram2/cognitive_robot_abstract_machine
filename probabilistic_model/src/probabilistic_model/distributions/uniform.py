from probabilistic_model.distributions.distributions import *
from probabilistic_model.constants import (
    PADDING_FACTOR_FOR_X_AXIS_IN_PLOT,
    EXPECTATION_TRACE_NAME,
    MODE_TRACE_NAME,
    MODE_TRACE_COLOR,
    PDF_TRACE_NAME,
    CDF_TRACE_NAME,
    CDF_TRACE_COLOR,
    PDF_TRACE_COLOR,
)


@dataclass(eq=False)
class UniformDistribution(ContinuousDistributionWithFiniteSupport):
    """
    Class for uniform distributions over the half-open interval [lower, upper).
    """

    def log_likelihood_without_bounds_check(self, x: npt.NDArray) -> npt.NDArray:
        return np.full((len(x),), self.log_probability_density_function_value())

    def cumulative_distribution_function(self, x: npt.NDArray) -> npt.NDArray:
        result = (x - self.lower) / (self.upper - self.lower)
        result = np.minimum(1, np.maximum(0, result))
        return result[:, 0]

    def univariate_log_mode(self) -> Tuple[AbstractCompositeSet, float]:
        return (
            self.interval.as_composite_set(),
            self.log_probability_density_function_value(),
        )

    def log_conditional_from_simple_interval(
        self, interval: SimpleInterval, singleton_allowed: bool = False
    ) -> Tuple[Optional[ContinuousDistribution], float]:
        probability = self.probability_of_simple_event(
            SimpleEvent.from_data({self.variable: interval})
        )
        if singleton_allowed and interval.is_singleton():
            log_likelihood = self.log_likelihood(np.array([[interval.lower]]))[0]
            if log_likelihood == -np.inf:
                return None, -np.inf

            return (
                DiracDeltaDistribution(
                    variable=self.variable, location=interval.lower, density_cap=1.0
                ),
                log_likelihood,
            )

        if probability == 0.0:
            return None, -np.inf

        # construct new interval
        new_interval = self.interval.intersection_with(interval)

        if new_interval.is_empty():
            return None, -np.inf

        return self.__class__(variable=self.variable, interval=new_interval), np.log(
            probability
        )

    def sample(self, amount: int) -> npt.NDArray:
        return np.random.uniform(self.lower, self.upper, (amount, 1))

    def probability_density_function_value(self) -> float:
        """
        Calculate the density of the uniform distribution.
        """
        return np.exp(self.log_probability_density_function_value())

    def log_probability_density_function_value(self) -> float:
        """
        Calculate the log-density of the uniform distribution.
        """
        return -np.log(self.upper - self.lower)

    def moment(self, order: OrderType, center: CenterType) -> MomentType:

        order = order[self.variable]
        center = center[self.variable]

        def evaluate_integral_at(x) -> float:
            r"""
            Helper method to calculate

            .. math::

               \int_{-\infty}^{\infty} (x - center)^{order} pdf(x) dx = \frac{p(x-center)^(1+order)}{1+order}

            """
            return (
                self.probability_density_function_value() * (x - center) ** (order + 1)
            ) / (order + 1)

        result = evaluate_integral_at(self.upper) - evaluate_integral_at(self.lower)

        return VariableMap({self.variable: result})

    @property
    def representation(self):
        return f"U({self.variable.name} | {self.interval})"

    @property
    def abbreviated_symbol(self) -> str:
        return "U"

    def __repr__(self):
        return f"U({self.variable.name})"

    def __copy__(self):
        return self.__class__(variable=self.variable, interval=self.interval)

    def __deepcopy__(self, memo=None):
        if memo is None:
            memo = {}
        id_self = id(self)
        if id_self in memo:
            return memo[id_self]

        variable = Continuous(self.variable.name)
        interval = self.interval.__deepcopy__()
        result = self.__class__(variable=variable, interval=interval)
        memo[id_self] = result
        return result

    def x_axis_points_for_plotly(self) -> List[Union[None, float]]:
        interval_size = self.upper - self.lower
        x = [
            self.lower - interval_size * PADDING_FACTOR_FOR_X_AXIS_IN_PLOT,
            self.lower,
            None,
            self.lower,
            self.upper,
            None,
            self.upper,
            self.upper + interval_size * PADDING_FACTOR_FOR_X_AXIS_IN_PLOT,
        ]
        return x

    def probability_density_function_trace(self) -> go.Scatter:
        """
        Create a Plotly trace for the probability density function (PDF) of the uniform distribution.
        """
        probability_density_values = [
            0,
            0,
            None,
            self.probability_density_function_value(),
            self.probability_density_function_value(),
            None,
            0,
            0,
        ]
        probability_density_trace = go.Scatter(
            x=self.x_axis_points_for_plotly(),
            y=probability_density_values,
            mode="lines",
            name=PDF_TRACE_NAME,
            line=dict(color=PDF_TRACE_COLOR),
        )
        return probability_density_trace

    def cumulative_density_function_trace(self) -> go.Scatter:
        """
        Create a Plotly trace for the cumulative distribution function (CDF) of the uniform distribution.
        """
        x = self.x_axis_points_for_plotly()
        cumulative_density_values = [
            (
                value
                if value is None
                else self.cumulative_distribution_function(np.array([[value]]))[0]
            )
            for value in x
        ]
        cumulative_density_trace = go.Scatter(
            x=x,
            y=cumulative_density_values,
            mode="lines",
            name=CDF_TRACE_NAME,
            line=dict(color=CDF_TRACE_COLOR),
        )
        return cumulative_density_trace

    def plot(self, **kwargs) -> List:
        probability_density_trace = self.probability_density_function_trace()
        cumulative_density_trace = self.cumulative_density_function_trace()

        height = (
            self.probability_density_function_value()
            * SCALING_FACTOR_FOR_EXPECTATION_IN_PLOT
        )

        mode_trace = self.univariate_mode_traces(self.mode()[0], height)
        expectation_trace = self.univariate_expectation_trace(height)
        return [
            probability_density_trace,
            cumulative_density_trace,
            expectation_trace,
        ] + mode_trace

    def __hash__(self):
        return hash((self.variable.name, hash(self.interval)))
