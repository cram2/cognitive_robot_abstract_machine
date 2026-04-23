from __future__ import annotations

import os
from abc import abstractmethod
from dataclasses import dataclass, field
from typing import Optional, Set as PythonSet

import numpy as np
import numpy.typing as npt
import plotly.graph_objects as go
from probabilistic_model.exceptions import UndefinedOperationError
from random_events.interval import Interval, SimpleInterval, Bound, singleton, closed
from random_events.product_algebra import Event, SimpleEvent, VariableMap
from typing_extensions import Union, Iterable, Any, Self, Dict, List, Tuple

from probabilistic_model.constants import SCALING_FACTOR_FOR_EXPECTATION_IN_PLOT
from probabilistic_model.probabilistic_model import (
    ProbabilisticModel,
    OrderType,
    MomentType,
    CenterType,
)
from probabilistic_model.utils import (
    MissingDict,
    interval_as_array,
    simple_interval_as_array,
)
from random_events.set import SetElement, Set
from random_events.sigma_algebra import AbstractCompositeSet
from random_events.variable import Variable, Continuous, Symbolic, Integer


@dataclass
class UnivariateDistribution(ProbabilisticModel):
    """
    Abstract Base class for Univariate distributions.
    """

    variable: Variable

    @property
    def variables(self) -> Tuple[Variable, ...]:
        return (self.variable,)

    @property
    def support(self) -> Event:
        return SimpleEvent.from_data(
            {self.variable: self.univariate_support}
        ).as_composite_set()

    @property
    @abstractmethod
    def univariate_support(self) -> AbstractCompositeSet:
        """
        :return: The univariate support of the distribution. This is not an Event.
        """
        raise NotImplementedError

    def log_mode(self) -> Tuple[Event, float]:
        mode, log_likelihood = self.univariate_log_mode()
        return (
            SimpleEvent.from_data({self.variable: mode}).as_composite_set(),
            log_likelihood,
        )

    @abstractmethod
    def univariate_log_mode(self) -> Tuple[AbstractCompositeSet, float]:
        """
        :return: The univariate mode of the distribution and its log-likelihood. The mode is not an Event.
        """
        raise NotImplementedError

    def marginal(self, variables: Iterable[Variable]) -> Optional[Self]:
        if self.variable in variables:
            return self
        else:
            return None

    def composite_set_from_event(self, event: Event) -> AbstractCompositeSet:
        """
        Extract the composite set from the event that is relevant for this distribution.
        :param event: The event
        :return: The composite set
        """
        return event.marginal(set(self.variables)).simple_sets[0][self.variable]

    @property
    def abbreviated_symbol(self) -> str:
        return "P"


@dataclass(eq=False)
class ContinuousDistribution(UnivariateDistribution):
    """
    Abstract base class for continuous distributions.
    """

    variable: Continuous = field(kw_only=True)

    @property
    @abstractmethod
    def univariate_support(self) -> Interval:
        raise NotImplementedError

    def cumulative_distribution_function(self, x: npt.NDArray) -> npt.NDArray:
        """
        Calculate the cumulative distribution function at x.
        :param x: The data
        :return: The cumulative distribution function at x
        """
        raise NotImplementedError

    def probability_of_simple_event(self, event: SimpleEvent) -> float:
        interval: Interval = event[self.variable]
        points = interval_as_array(interval)
        upper_bound_cdf = self.cumulative_distribution_function(points[:, (1,)])
        lower_bound_cdf = self.cumulative_distribution_function(points[:, (0,)])
        return (upper_bound_cdf - lower_bound_cdf).sum()

    def log_truncated(
        self, event: Event, singleton_allowed: bool = False
    ) -> Tuple[Optional[ContinuousDistribution], float]:
        if event.is_empty():
            return None, -np.inf

        interval = self.composite_set_from_event(event)

        if len(interval.simple_sets) == 1:
            return self.log_conditional_from_simple_interval(
                interval.simple_sets[0], singleton_allowed
            )
        else:
            raise UndefinedOperationError(self)

    def log_conditional(
        self, point: Dict[Variable, Any]
    ) -> Tuple[Optional[Union[ProbabilisticModel, Self]], float]:
        value = point[self.variable]
        log_pdf_value = self.log_likelihood(np.array([[value]]))[0]

        if log_pdf_value == -np.inf:
            return None, -np.inf

        return (
            DiracDeltaDistribution(
                variable=self.variable,
                location=value,
                density_cap=np.exp(log_pdf_value),
            ),
            log_pdf_value,
        )

    def log_conditional_from_simple_interval(
        self, interval: SimpleInterval, singleton_allowed: bool = False
    ) -> Tuple[Optional[ContinuousDistribution], float]:
        """
        Calculate the truncated distribution given a simple interval.

        :param interval: The simple interval
        :param singleton_allowed: Whether the simple interval is allowed to be a singleton.
        :return: The truncated distribution and the log-probability of the interval.
        """
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
        else:
            return self.log_conditional_from_simple_interval_if_not_singleton(interval)

    @abstractmethod
    def log_conditional_from_simple_interval_if_not_singleton(
        self, interval: SimpleInterval
    ) -> Tuple[Optional[ContinuousDistribution], float]:
        """
        Calculate the truncated distribution given a simple interval that is not a singleton.

        :param interval: The simple interval
        :return: The truncated distribution and the log-probability of the interval.
        """


@dataclass(eq=False)
class ContinuousDistributionWithFiniteSupport(ContinuousDistribution):
    """
    Abstract base class for continuous distributions with finite support.
    """

    interval: SimpleInterval
    """
    The interval of the distribution.
    """

    @property
    def lower(self) -> float:
        return self.interval.lower

    @property
    def upper(self) -> float:
        return self.interval.upper

    @property
    def univariate_support(self) -> Interval:
        return self.interval.as_composite_set()

    def left_included_condition(self, x: npt.NDArray) -> npt.NDArray:
        """
        Check if x is included in the left bound of the interval.
        :param x: The data
        :return: A boolean array
        """
        return (
            self.interval.lower <= x
            if self.interval.left == Bound.CLOSED
            else self.interval.lower < x
        )

    def right_included_condition(self, x: npt.NDArray) -> npt.NDArray:
        """
        Check if x is included in the right bound of the interval.
        :param x: The data
        :return: A boolean array
        """
        return (
            x < self.interval.upper
            if self.interval.right == Bound.OPEN
            else x <= self.interval.upper
        )

    def included_condition(self, x: npt.NDArray) -> npt.NDArray:
        """
        Check if x is included in interval.
        :param x: The data
        :return: A boolean array
        """
        return self.left_included_condition(x) & self.right_included_condition(x)

    def log_likelihood(self, x: npt.NDArray) -> npt.NDArray:
        result = np.full(x.shape[:-1], -np.inf)
        include_condition = self.included_condition(x)
        filtered_x = x[include_condition].reshape(-1, 1)
        result[include_condition[:, 0]] = self.log_likelihood_without_bounds_check(
            filtered_x
        )
        return result

    @abstractmethod
    def log_likelihood_without_bounds_check(self, x: npt.NDArray) -> npt.NDArray:
        """
        Evaluate the logarithmic likelihood function at `x` without checking the inclusion into the interval.
        :param x: x where p(x) > 0
        :return: log(p(x))
        """
        raise NotImplementedError

    def apply_translation(self, translation: Dict[Variable, float]):
        new_interval = SimpleInterval.from_data(
            self.interval.lower + translation[self.variable],
            self.interval.upper + translation[self.variable],
            self.interval.left,
            self.interval.right,
        )
        self.interval = new_interval

    def apply_scaling(self, scaling: Dict[Variable, float]):
        new_interval = SimpleInterval.from_data(
            self.interval.lower * scaling[self.variable],
            self.interval.upper * scaling[self.variable],
            self.interval.left,
            self.interval.right,
        )
        self.interval = new_interval


@dataclass(eq=False)
class DiscreteDistribution(UnivariateDistribution):
    """
    Abstract base class for univariate discrete distributions.
    """

    variable: Union[Symbolic, Integer] = field(kw_only=True)

    probabilities: MissingDict = field(
        kw_only=True, default_factory=lambda: MissingDict(float)
    )
    """
    A dict that maps from integers (hash(symbol) for symbols) to probabilities.
    """

    def log_likelihood(self, events: npt.NDArray) -> npt.NDArray:
        events = events[:, 0]

        events = np.array([hash(e) for e in events])
        result = np.full(len(events), -np.inf)
        for x, p in self.probabilities.items():
            result[events == hash(x)] = np.log(p)

        # for indices that are still -inf, it might be because the category was not in the domain
        # but with laplace smoothing we should have covered the whole domain.
        # However, if the event itself is not in the domain (e.g. 'bird' when domain is ['cat', 'dog']),
        # hash(e) won't be in self.probabilities.
        # We should assign the same probability as other unobserved categories if possible,
        # or at least the laplace smoothed probability for a generic unobserved category.

        if np.any(result == -np.inf):
            if isinstance(self.variable, (Symbolic, Integer)):
                domain_size = len(self.variable.domain.all_elements)
            else:
                domain_size = len(self.probabilities)

            # This length should be from the data used to fit, but we don't store it.
            # We can approximate it or use a very small probability.
            # Since we used (count + 1) / (len(data) + domain_size),
            # a missing value should get 1 / (len(data) + domain_size).
            # We don't have len(data) here, so we'll use a small value.
            result[result == -np.inf] = np.log(1e-10)

        return result

    def fit(self, data: npt.NDArray) -> Self:
        """
        Fit the distribution to the data.

        The probabilities are set using Laplace smoothing with a pseudocount of 1.
        The data contains the indices of the domain elements (if symbolic) or the values (if integer).

        :param data: The data.
        :return: The fitted distribution
        """
        unique, counts = np.unique(data, return_counts=True)
        probabilities = MissingDict(float)

        # get the domain size
        if isinstance(self.variable, Symbolic):
            domain_size = len(self.variable.domain.all_elements)
        elif isinstance(self.variable, Integer):
            domain_size = len(self.variable.domain.all_elements)
        else:
            domain_size = len(unique)

        # apply laplace smoothing
        for value, count in zip(unique, counts):
            probabilities[hash(value)] = (count + 1) / (len(data) + domain_size)

        # fill missing values with laplace smoothing
        if isinstance(self.variable, (Symbolic, Integer)):
            for element in self.variable.domain.all_elements:
                if hash(element) not in probabilities:
                    probabilities[hash(element)] = 1 / (len(data) + domain_size)

        self.probabilities = probabilities
        return self

    @abstractmethod
    def probabilities_for_plotting(self) -> Dict[Union[int, str], float]:
        """
        :return: The probabilities as dict that can be plotted.
        """
        raise NotImplementedError

    def plot(self, **kwargs) -> List[go.Bar]:
        """
        Plot the distribution.
        """
        probabilities = self.probabilities_for_plotting()
        max_likelihood = max(probabilities.values())
        non_mode_trace = {x: p for x, p in probabilities.items() if p != max_likelihood}
        traces = [
            go.Bar(
                x=list(non_mode_trace.keys()),
                y=list(non_mode_trace.values()),
                name="Probability",
            )
        ]

        mode = [key for key, value in probabilities.items() if value == max_likelihood]
        traces.append(go.Bar(x=mode, y=[max_likelihood] * len(mode), name="Mode"))
        return traces

    def normalize(self):
        """
        Normalize the distribution.
        """
        total = sum(self.probabilities.values())
        for key in self.probabilities:
            self.probabilities[key] /= total

    def log_truncated(
        self, event: Event, singleton_allowed: bool = False
    ) -> Tuple[Optional[Self], float]:
        # construct event
        condition = self.composite_set_from_event(event)
        return self.log_conditional_of_composite_set(condition)

    def log_conditional(
        self, point: Dict[Variable, Any]
    ) -> Tuple[Optional[Self], float]:
        return self.log_truncated(
            SimpleEvent.from_data(
                {self.variable: point[self.variable]}
            ).as_composite_set()
        )

    def log_conditional_of_composite_set(
        self, event: AbstractCompositeSet
    ) -> Tuple[Optional[Self], float]:
        # calculate new probabilities
        new_probabilities = MissingDict(float)
        for x, p_x in self.probabilities.items():
            if x in event:
                new_probabilities[x] = p_x

        # if the event is impossible, return None and 0
        probability = sum(new_probabilities.values())

        if probability == 0:
            return None, -np.inf

        result = self.__class__(variable=self.variable, probabilities=new_probabilities)
        result.normalize()
        return result, np.log(probability)

    def __copy__(self) -> Self:
        return self.__class__(variable=self.variable, probabilities=self.probabilities)

    def __deepcopy__(self, memo=None) -> Self:
        if memo is None:
            memo = {}
        id_self = id(self)
        if id_self in memo:
            return memo[id_self]
        import copy

        variable = self.variable.__class__(
            name=self.variable.name, domain=self.variable.domain
        )
        probabilities = copy.deepcopy(self.probabilities, memo)
        result = self.__class__(variable=variable, probabilities=probabilities)
        memo[id_self] = result
        return result

    def __repr__(self):
        return f"P({self.variable.name})"

    def sample(self, amount: int) -> npt.NDArray:
        sample_space = np.array(list(self.probabilities.keys()))
        sample_probabilities = np.array(
            [value for value in self.probabilities.values()]
        )
        return np.random.choice(
            sample_space, size=(amount, 1), replace=True, p=sample_probabilities
        )


@dataclass(eq=False)
class SymbolicDistribution(DiscreteDistribution):
    """
    Class for symbolic (categorical) distributions.
    """

    variable: Symbolic = field(kw_only=True)

    def univariate_log_mode(self) -> Tuple[Set, float]:
        max_likelihood = max(self.probabilities.values())

        mode_hashes = {
            key for key, value in self.probabilities.items() if value == max_likelihood
        }
        domain_hash_map = self.variable.domain.hash_map

        mode_symbols = {domain_hash_map[hash_value] for hash_value in mode_hashes}
        mode = self.variable.make_value(mode_symbols)
        return mode, np.log(max_likelihood)

    def log_conditional_of_composite_set(
        self, event: AbstractCompositeSet
    ) -> Tuple[Optional[Self], float]:
        new_probabilities = MissingDict(float)
        for x in event:
            hash_x = hash(x)
            if self.probabilities[hash_x] > 0:
                new_probabilities[hash_x] = self.probabilities[hash_x]

        probability = sum(new_probabilities.values())

        if probability == 0:
            return None, -np.inf

        result = self.__class__(variable=self.variable, probabilities=new_probabilities)
        result.normalize()
        return result, np.log(probability)

    def probabilities_for_plotting(self) -> Dict[Union[int, str], float]:
        return {
            str(element): self.probabilities[hash(element)]
            for element in self.variable.domain.simple_sets
        }

    @property
    def univariate_support(self) -> Set:
        hash_map = self.variable.domain.hash_map
        return self.variable.make_value(
            [hash_map[key] for key, value in self.probabilities.items() if value > 0]
        )

    def probability_of_simple_event(self, event: SimpleEvent) -> float:
        return sum(
            self.probabilities[hash(key)] for key in event[self.variable].simple_sets
        )

    @property
    def representation(self):
        return f"Nominal({self.variable.name})"

    def fit(self, data: npt.NDArray) -> Self:
        unique, counts = np.unique(data, return_counts=True)
        probabilities = MissingDict(float)
        for value, count in zip(unique, counts):
            set_element = [
                element
                for element in self.variable.domain.simple_sets
                if element == value
            ][0]
            probabilities[hash(set_element)] = count / len(data)
        self.probabilities = probabilities
        return self

    def fit_from_indices(self, data: npt.NDArray) -> Self:
        """
        Fit the distribution to the data, where the data contains the indices of the domain elements.
        :param data: The data.
        :return: The fitted distribution
        """
        unique, counts = np.unique(data, return_counts=True)
        probabilities = MissingDict(float)

        domain_size = len(self.variable.domain.all_elements)

        for value, count in zip(unique, counts):
            set_element = self.variable.domain.simple_sets[value]
            probabilities[hash(set_element)] = (count + 1) / (len(data) + domain_size)

        for element in self.variable.domain.all_elements:
            if hash(element) not in probabilities:
                probabilities[hash(element)] = 1 / (len(data) + domain_size)

        self.probabilities = probabilities
        return self


@dataclass
class IntegerDistribution(ContinuousDistribution, DiscreteDistribution):
    """
    Abstract base class for integer distributions. Integer distributions also implement the methods of continuous
    distributions.
    """

    variable: Integer = field(kw_only=True)

    def log_truncated(
        self, event: Event, singleton_allowed: bool = False
    ) -> Tuple[Optional[DiscreteDistribution], float]:
        return DiscreteDistribution.log_truncated(self, event, singleton_allowed)

    def univariate_log_mode(self) -> Tuple[AbstractCompositeSet, float]:
        max_likelihood = max(self.probabilities.values())
        mode = Interval()
        for key, value in self.probabilities.items():
            if value == max_likelihood:
                mode |= singleton(key)
        return mode, np.log(max_likelihood)

    def probabilities_for_plotting(self) -> Dict[Union[int, str], float]:
        return {x: p_x for x, p_x in self.probabilities.items() if p_x > 0}

    def log_conditional_from_simple_interval_if_not_singleton(
        self, interval: SimpleInterval
    ) -> Tuple[Optional[ContinuousDistribution], float]:
        raise UndefinedOperationError(self)

    @property
    def univariate_support(self) -> Interval:
        result = Interval()
        for key, value in self.probabilities.items():
            if value > 0:
                result |= singleton(key)
        return result

    def cumulative_distribution_function(self, x: npt.NDArray) -> npt.NDArray:
        result = np.zeros((len(x),))
        maximum_value = max(x)
        for value, p in self.probabilities.items():
            if value > maximum_value:
                break
            else:
                result[x[:, 0] >= value] += p

        return result

    def probability_of_simple_event(self, event: SimpleEvent) -> float:
        interval: Interval = event[self.variable]
        result = 0

        for x, p_x in self.probabilities.items():
            if x in interval:
                result += p_x

        return result

    @property
    def representation(self):
        return f"Ordinal({self.variable.name})"

    def moment(self, order: OrderType, center: CenterType) -> MomentType:
        order = order[self.variable]
        center = center[self.variable]
        result = sum(
            [p_x * (x - center) ** order for x, p_x in self.probabilities.items()]
        )
        return VariableMap({self.variable: result})

    def plot(self, **kwargs) -> List[go.Bar]:
        height = (
            max(self.probabilities.values()) * SCALING_FACTOR_FOR_EXPECTATION_IN_PLOT
        )
        return super().plot() + [self.univariate_expectation_trace(height)]

    def apply_translation(self, translation: Dict[Variable, int]):
        new_probabilities = MissingDict(float)
        for key, value in self.probabilities.items():
            new_probabilities[key + translation[self.variable]] = value
        self.probabilities = new_probabilities

    def apply_scaling(self, scaling: Dict[Variable, int]):
        new_probabilities = MissingDict(float)
        for key, value in self.probabilities.items():
            new_probabilities[key * scaling[self.variable]] = value
        self.probabilities = new_probabilities


@dataclass(eq=False)
class BernoulliDistribution(IntegerDistribution):
    """
    Bernoulli distribution over the integer domain {0, 1}.
    :param

    """

    variable: Integer

    def __init__(
        self,
        variable: Integer,
        p: Optional[float] = None,
        probabilities: Optional[MissingDict[Union[int, SetElement], float]] = None,
    ):
        if probabilities is None:
            if p is None:
                p = 0.5
            if not (0.0 <= p <= 1.0):
                raise ValueError("Parameter p must be in [0, 1].")
            probs = MissingDict(float)
            probs[0] = 1.0 - p
            probs[1] = p
            probabilities = probs

        super().__init__(variable=variable, probabilities=probabilities)

    @property
    def p(self) -> float:
        """Return probability of 1."""
        return float(self.probabilities[1])

    @property
    def representation(self) -> str:
        return f"B({self.variable.name} | p={self.p})"

    @property
    def abbreviated_symbol(self) -> str:
        return "B"

    def __repr__(self):
        return f"B({self.variable.name})"


@dataclass
class DiracDeltaDistribution(ContinuousDistribution):
    """
    Class for Dirac delta distributions.
    The Dirac measure is used whenever evidence is given as a singleton instance.

    https://en.wikipedia.org/wiki/Dirac_delta_function
    """

    variable: Continuous = field(kw_only=True)

    location: float
    """
    The location of the Dirac delta distribution.
    """

    density_cap: float = field(default=np.inf)
    """
    The density cap of the Dirac delta distribution.
    This value will be used to replace infinity in likelihood.
    """

    tolerance: float = field(default=1e-6, compare=False)
    """
    The tolerance of deviations of the `location` of the Dirac delta distribution.
    This is used during calculations to take precision problems into account.
    """

    def log_likelihood(self, events: npt.NDArray) -> npt.NDArray:
        result = np.full(len(events), -np.inf)
        # Check if the event is within the tolerance of the location
        within_tolerance = np.abs(events[:, 0] - self.location) < self.tolerance
        # If it is, set the log likelihood to the log of the density cap
        result[within_tolerance] = np.log(self.density_cap)
        return result

    def cumulative_distribution_function(self, x: npt.NDArray) -> npt.NDArray:
        result = np.zeros((len(x),))
        result[x[:, 0] >= self.location - self.tolerance] = 1.0
        return result

    @property
    def abbreviated_symbol(self) -> str:
        return "δ"

    @property
    def univariate_support(self) -> Interval:
        return singleton(self.location)

    def log_conditional_from_simple_interval_if_not_singleton(
        self, interval: SimpleInterval
    ) -> Tuple[Optional[ContinuousDistribution], float]:
        if interval.contains(self.location):
            return self, 0.0
        else:
            return None, -np.inf

    def probability_of_simple_event(self, event: SimpleEvent) -> float:
        interval: Interval = event[self.variable]

        return (
            0.0
            if (
                closed(self.location - self.tolerance, self.location + self.tolerance)
                & interval
            ).is_empty()
            else 1.0
        )

    def univariate_log_mode(self) -> Tuple[AbstractCompositeSet, float]:
        return self.univariate_support, np.log(self.density_cap)

    def sample(self, amount: int) -> npt.NDArray:
        return np.full((amount, 1), self.location)

    def moment(self, order: OrderType, center: CenterType) -> MomentType:
        order = order[self.variable]
        center = center[self.variable]

        if order == 0:
            moment = 1.0
        elif order == 1:
            moment = self.location - center
        else:
            moment = 0.0

        return VariableMap({self.variable: moment})

    @property
    def representation(self):
        return f"δ({self.location}, {self.density_cap})"

    def __repr__(self):
        return f"δ({self.variable.name})"

    def __copy__(self):
        return self.__class__(
            variable=self.variable, location=self.location, density_cap=self.density_cap
        )

    def __deepcopy__(self, memo=None):
        if memo is None:
            memo = {}
        id_self = id(self)
        if id_self in memo:
            return memo[id_self]

        variable = Continuous(self.variable.name)
        result = self.__class__(
            variable=variable, location=self.location, density_cap=self.density_cap
        )
        memo[id_self] = result
        return result

    def plot(self, **kwargs) -> List:
        lower_border = self.location - 1
        upper_border = self.location + 1
        pdf_trace = go.Scatter(
            x=[lower_border, self.location, self.location, self.location, upper_border],
            y=[0, 0, self.density_cap, 0, 0],
            mode="lines",
            name="PDF",
        )
        cdf_trace = go.Scatter(
            x=[lower_border, self.location, self.location, upper_border],
            y=[0, 0, 1, 1],
            mode="lines",
            name="CDF",
        )
        expectation_trace = go.Scatter(
            x=[self.location, self.location],
            y=[0, self.density_cap * SCALING_FACTOR_FOR_EXPECTATION_IN_PLOT],
            mode="lines+markers",
            name="Expectation",
        )
        mode_trace = go.Scatter(
            x=[self.location, self.location],
            y=[0, self.density_cap * SCALING_FACTOR_FOR_EXPECTATION_IN_PLOT],
            mode="lines+markers",
            name="Mode",
        )
        return [pdf_trace, cdf_trace, expectation_trace, mode_trace]

    def apply_translation(self, translation: VariableMap[Variable, float]):
        self.location += translation[self.variable]

    def apply_scaling(self, scaling: VariableMap[Variable, float]):
        self.location += scaling[self.variable]
