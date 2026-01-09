from __future__ import annotations

import os

import numpy as np
import plotly.graph_objects as go
from random_events.interval import *
from random_events.product_algebra import Event, SimpleEvent, VariableMap
from random_events.variable import *
from typing_extensions import Union, Iterable, Any, Self, Dict, List, Tuple

from ..constants import SCALING_FACTOR_FOR_EXPECTATION_IN_PLOT
from ..interfaces.drawio.drawio import DrawIOInterface
from ..probabilistic_model import ProbabilisticModel, OrderType, MomentType, CenterType
from ..utils import MissingDict, interval_as_array


class UnivariateDistribution(ProbabilisticModel, SubclassJSONSerializer, DrawIOInterface):
    """
    Abstract Base class for Univariate distributions.
    """

    variable: Variable

    @property
    def variables(self) -> Tuple[Variable, ...]:
        return (self.variable,)

    @property
    def support(self) -> Event:
        return SimpleEvent({self.variable: self.univariate_support}).as_composite_set()

    @property
    @abstractmethod
    def univariate_support(self) -> AbstractCompositeSet:
        """
        :return: The univariate support of the distribution. This is not an Event.
        """
        raise NotImplementedError

    def log_mode(self) -> Tuple[Event, float]:
        mode, log_likelihood = self.univariate_log_mode()
        return SimpleEvent({self.variable: mode}).as_composite_set(), log_likelihood

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

    def __eq__(self, other: Self):
        return self.variables == other.variables and isinstance(other, self.__class__)

    def to_json(self) -> Dict[str, Any]:
        return {**super().to_json(), "variable": self.variable.to_json()}

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

    @property
    def drawio_style(self) -> Dict[str, Any]:
        return {"style": self.drawio_label, "width": 30, "height": 30, "label": self.abbreviated_symbol}


class ContinuousDistribution(UnivariateDistribution):
    """
    Abstract base class for continuous distributions.
    """

    variable: Continuous

    @property
    @abstractmethod
    def univariate_support(self) -> Interval:
        raise NotImplementedError

    def cdf(self, x: np.array) -> np.array:
        """
        Calculate the cumulative distribution function at x.
        :param x: The data
        :return: The cumulative distribution function at x
        """
        raise NotImplementedError

    def probability_of_simple_event(self, event: SimpleEvent) -> float:
        interval: Interval = event[self.variable]
        points = interval_as_array(interval)
        upper_bound_cdf = self.cdf(points[:, (1,)])
        lower_bound_cdf = self.cdf(points[:, (0,)])
        return (upper_bound_cdf - lower_bound_cdf).sum()

    def log_truncated(self, event: Event) -> Tuple[Optional[Self], float]:
        if event.is_empty():
            return None, -np.inf

        interval = self.composite_set_from_event(event)

        if len(interval.simple_sets) == 1:
            return self.log_conditional_from_simple_interval(interval.simple_sets[0])

        else:
            return self.log_conditional_from_interval(interval)

    def log_conditional(self, point: Dict[Variable, Any]) -> Tuple[Optional[Union[ProbabilisticModel, Self]], float]:
        value = point[self.variable]
        log_pdf_value = self.log_likelihood(np.array([[value]]))[0]

        if log_pdf_value == -np.inf:
            return None, -np.inf

        return DiracDeltaDistribution(self.variable, value, np.exp(log_pdf_value)), log_pdf_value

    def log_conditional_from_simple_interval(self, interval: SimpleInterval) -> Tuple[Self, float]:
        """
        Calculate the truncated distribution given a simple interval.

        :param interval: The simple interval
        :return: The truncated distribution and the log-probability of the interval.
        """
        raise NotImplementedError

    def log_conditional_from_interval(self, interval) -> Tuple[Self, float]:
        """
        Calculate the truncated distribution given an interval with p(interval) > 0.
        :param interval: The simple interval
        :return: The truncated distribution and the log-probability of the interval.
        """
        raise NotImplementedError


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

    def left_included_condition(self, x: np.array) -> np.array:
        """
        Check if x is included in the left bound of the interval.
        :param x: The data
        :return: A boolean array
        """
        return self.interval.lower <= x if self.interval.left == Bound.CLOSED else self.interval.lower < x

    def right_included_condition(self, x: np.array) -> np.array:
        """
         Check if x is included in the right bound of the interval.
         :param x: The data
         :return: A boolean array
         """
        return x < self.interval.upper if self.interval.right == Bound.OPEN else x <= self.interval.upper

    def included_condition(self, x: np.array) -> np.array:
        """
         Check if x is included in interval.
         :param x: The data
         :return: A boolean array
         """
        return self.left_included_condition(x) & self.right_included_condition(x)

    def log_likelihood(self, x: np.array) -> np.array:
        result = np.full(x.shape[:-1], -np.inf)
        include_condition = self.included_condition(x)
        filtered_x = x[include_condition].reshape(-1, 1)
        result[include_condition[:, 0]] = self.log_likelihood_without_bounds_check(filtered_x)
        return result

    @abstractmethod
    def log_likelihood_without_bounds_check(self, x: np.array) -> np.array:
        """
        Evaluate the logarithmic likelihood function at `x` without checking the inclusion into the interval.
        :param x: x where p(x) > 0
        :return: log(p(x))
        """
        raise NotImplementedError

    def translate(self, translation: Dict[Variable, float]):
        new_interval = SimpleInterval(self.interval.lower + translation[self.variable],
                                      self.interval.upper + translation[self.variable],
                                      self.interval.left, self.interval.right)
        self.interval = new_interval

    def scale(self, scaling: Dict[Variable, float]):
        new_interval = SimpleInterval(self.interval.lower * scaling[self.variable],
                                      self.interval.upper * scaling[self.variable],
                                      self.interval.left, self.interval.right)
        self.interval = new_interval


class DiscreteDistribution(UnivariateDistribution):
    """
    Abstract base class for univariate discrete distributions.
    """

    variable: Union[Symbolic, Integer]

    probabilities: MissingDict[int, float] = MissingDict(float)
    """
    A dict that maps from integers (hash(symbol) for symbols) to probabilities.
    """

    def __init__(self, variable: Union[Symbolic, Integer],
                 probabilities: Optional[MissingDict[Union[int, SetElement], float]]):
        super().__init__()
        self.variable = variable

        if probabilities is not None:
            self.probabilities = probabilities

    def __eq__(self, other):
        return (isinstance(other,
                           DiscreteDistribution) and self.probabilities == other.probabilities and super().__eq__(
            other))

    def __hash__(self):
        return hash((self.variable, tuple(self.probabilities.items())))

    def log_likelihood(self, events: np.array) -> np.array:
        events = events[:, 0]

        events = np.array([hash(e) for e in events])
        result = np.full(len(events), -np.inf)
        for x, p in self.probabilities.items():
            result[events == hash(x)] = np.log(p)
        return result

    def fit(self, data: np.array) -> Self:
        """
        Fit the distribution to the data.

        The probabilities are set equal to the frequencies in the data.
        The data contains the indices of the domain elements (if symbolic) or the values (if integer).

        :param data: The data.
        :return: The fitted distribution
        """
        unique, counts = np.unique(data, return_counts=True)
        probabilities = MissingDict(float)
        for value, count in zip(unique, counts):
            probabilities[hash(value)] = count / len(data)
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
        traces = [go.Bar(x=list(non_mode_trace.keys()), y=list(non_mode_trace.values()), name="Probability")]

        mode = [key for key, value in probabilities.items() if value == max_likelihood]
        traces.append(go.Bar(x=mode, y=[max_likelihood] * len(mode), name="Mode"))
        return traces

    def to_json(self) -> Dict[str, Any]:
        return {**super().to_json(), "probabilities": list(self.probabilities.items())}

    @classmethod
    def _from_json(cls, data: Dict[str, Any]) -> Self:
        variable = Variable.from_json(data["variable"])
        probabilities = MissingDict(float)
        for key, value in data["probabilities"]:
            probabilities[key] = value
        return cls(variable, probabilities)

    def normalize(self):
        """
        Normalize the distribution.
        """
        total = sum(self.probabilities.values())
        for key in self.probabilities:
            self.probabilities[key] /= total

    def log_truncated(self, event: Event) -> Tuple[Optional[Self], float]:
        # construct event
        condition = self.composite_set_from_event(event)
        return self.log_conditional_of_composite_set(condition)

    def log_conditional(self, point: Dict[Variable, Any]) -> Tuple[Optional[Self], float]:
        return self.log_truncated(SimpleEvent({self.variable: point[self.variable]}).as_composite_set())

    def log_conditional_of_composite_set(self, event: AbstractCompositeSet) -> Tuple[Optional[Self], float]:
        # calculate new probabilities
        new_probabilities = MissingDict(float)
        for x, p_x in self.probabilities.items():
            if x in event:
                new_probabilities[x] = p_x

        # if the event is impossible, return None and 0
        probability = sum(new_probabilities.values())

        if probability == 0:
            return None, -np.inf

        result = self.__class__(self.variable, new_probabilities)
        result.normalize()
        return result, np.log(probability)

    def __copy__(self) -> Self:
        return self.__class__(self.variable, self.probabilities)

    def __deepcopy__(self, memo=None) -> Self:
        if memo is None:
            memo = {}
        id_self = id(self)
        if id_self in memo:
            return memo[id_self]
        import copy
        variable = self.variable.__class__(self.variable.name, self.variable.domain)
        probabilities = copy.deepcopy(self.probabilities, memo)
        result = self.__class__(variable, probabilities)
        memo[id_self] = result
        return result

    def __repr__(self):
        return f"P({self.variable.name})"

    def sample(self, amount: int) -> np.array:
        sample_space = np.array(list(self.probabilities.keys()))
        sample_probabilities = np.array([value for value in self.probabilities.values()])
        return np.random.choice(sample_space, size=(amount, 1), replace=True, p=sample_probabilities)


class SymbolicDistribution(DiscreteDistribution):
    """
    Class for symbolic (categorical) distributions.
    """

    variable: Symbolic

    def univariate_log_mode(self) -> Tuple[Set, float]:
        max_likelihood = max(self.probabilities.values())

        mode_hashes = {key for key, value in self.probabilities.items() if value == max_likelihood}
        domain_hash_map = self.variable.domain.hash_map

        mode_symbols = {domain_hash_map[hash_value] for hash_value in mode_hashes}
        mode = self.variable.make_value(mode_symbols)
        return mode, np.log(max_likelihood)

    def log_conditional_of_composite_set(self, event: AbstractCompositeSet) -> Tuple[Optional[Self], float]:
        new_probabilities = MissingDict(float)
        for x in event:
            hash_x = hash(x)
            if self.probabilities[hash_x] > 0:
                new_probabilities[hash_x] = self.probabilities[hash_x]

        probability = sum(new_probabilities.values())

        if probability == 0:
            return None, -np.inf

        result = self.__class__(self.variable, new_probabilities)
        result.normalize()
        return result, np.log(probability)

    def probabilities_for_plotting(self) -> Dict[Union[int, str], float]:
        return {str(element): self.probabilities[hash(element)] for element in self.variable.domain.simple_sets}

    @property
    def univariate_support(self) -> Set:
        hash_map = self.variable.domain.hash_map
        return self.variable.make_value([hash_map[key] for key, value in self.probabilities.items() if value > 0])

    def probability_of_simple_event(self, event: SimpleEvent) -> float:
        return sum(self.probabilities[hash(key)] for key in event[self.variable].simple_sets)

    @property
    def representation(self):
        return f"Nominal({self.variable.name})"

    @property
    def drawio_label(self):
        return "rounded=1;whiteSpace=wrap;html=1;labelPosition=center;verticalLabelPosition=top;align=center;verticalAlign=bottom;"

    @property
    def image(self):
        return os.path.join(os.path.dirname(__file__), "../../../", "resources", "icons", "defaultIcon.png")

    def fit(self, data: np.array) -> Self:
        unique, counts = np.unique(data, return_counts=True)
        probabilities = MissingDict(float)
        for value, count in zip(unique, counts):
            set_element = [element for element in self.variable.domain.simple_sets if element == value][0]
            probabilities[hash(set_element)] = count / len(data)
        self.probabilities = probabilities
        return self

    def fit_from_indices(self, data: np.array) -> Self:
        unique, counts = np.unique(data, return_counts=True)
        probabilities = MissingDict(float)
        for value, count in zip(unique, counts):
            set_element = self.variable.domain.simple_sets[value]
            probabilities[hash(set_element)] = count / len(data)
        self.probabilities = probabilities
        return self

    def to_json(self) -> Dict[str, Any]:
        hashes = list(self.variable.domain.hash_map.keys())
        probabilities = {hashes.index(h): p for h, p in self.probabilities.items()}
        result = super().to_json()
        result["probabilities"] = list(probabilities.items())
        return result

    @classmethod
    def _from_json(cls, data: Dict[str, Any]) -> Self:
        variable = Variable.from_json(data["variable"])
        probabilities = MissingDict(float)
        for key, value in data["probabilities"]:
            probabilities[hash(variable.domain.simple_sets[key])] = value
        return cls(variable, probabilities)


class IntegerDistribution(ContinuousDistribution, DiscreteDistribution):
    """
    Abstract base class for integer distributions. Integer distributions also implement the methods of continuous
    distributions.
    """

    variable: Integer

    def __init__(self, variable: Integer, probabilities: Optional[MissingDict[Union[int, SetElement], float]]):
        DiscreteDistribution.__init__(self, variable, probabilities)

    def log_truncated(self, event: Event) -> Tuple[Optional[Self], float]:
        return DiscreteDistribution.log_truncated(self, event)

    def univariate_log_mode(self) -> Tuple[AbstractCompositeSet, float]:
        max_likelihood = max(self.probabilities.values())
        mode = Interval()
        for key, value in self.probabilities.items():
            if value == max_likelihood:
                mode |= singleton(key)
        return mode, np.log(max_likelihood)

    def probabilities_for_plotting(self) -> Dict[Union[int, str], float]:
        return {x: p_x for x, p_x in self.probabilities.items() if p_x > 0}

    @property
    def univariate_support(self) -> Interval:
        result = Interval()
        for key, value in self.probabilities.items():
            if value > 0:
                result |= singleton(key)
        return result

    def cdf(self, x: np.array) -> np.array:
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
        result = sum([p_x * (x - center) ** order for x, p_x in self.probabilities.items()])
        return VariableMap({self.variable: result})

    def plot(self, **kwargs) -> List[go.Bar]:
        height = max(self.probabilities.values()) * SCALING_FACTOR_FOR_EXPECTATION_IN_PLOT
        return super().plot() + [self.univariate_expectation_trace(height)]

    def translate(self, translation: Dict[Variable, int]):
        new_probabilities = MissingDict(float)
        for key, value in self.probabilities.items():
            new_probabilities[key + translation[self.variable]] = value
        self.probabilities = new_probabilities

    def scale(self, scaling: Dict[Variable, int]):
        new_probabilities = MissingDict(float)
        for key, value in self.probabilities.items():
            new_probabilities[key * scaling[self.variable]] = value
        self.probabilities = new_probabilities


class DiracDeltaDistribution(ContinuousDistribution):
    """
    Class for Dirac delta distributions.
    The Dirac measure is used whenever evidence is given as a singleton instance.

    https://en.wikipedia.org/wiki/Dirac_delta_function
    """

    variable: Continuous

    location: float
    """
    The location of the Dirac delta distribution.
    """

    density_cap: float
    """
    The density cap of the Dirac delta distribution.
    This value will be used to replace infinity in likelihood.
    """

    tolerance: float = 1e-6
    """
    The tolerance of deviations of the `location` of the Dirac delta distribution.
    This is used during calculations to take precision problems into account.
    """

    def __init__(self, variable: Continuous, location: float, density_cap: float = np.inf, tolerance: float = 1e-6):
        super().__init__()
        self.variable = variable
        self.location = location
        self.density_cap = density_cap
        self.tolerance = tolerance

    def log_likelihood(self, events: np.array) -> np.array:
        result = np.full(len(events), -np.inf)
        # Check if the event is within the tolerance of the location
        within_tolerance = np.abs(events[:, 0] - self.location) < self.tolerance
        # If it is, set the log likelihood to the log of the density cap
        result[within_tolerance] = np.log(self.density_cap)
        return result

    def cdf(self, x: np.array) -> np.array:
        result = np.zeros((len(x),))
        result[x[:, 0] >= self.location - self.tolerance] = 1.
        return result

    @property
    def abbreviated_symbol(self) -> str:
        return "δ"

    @property
    def univariate_support(self) -> Interval:
        return singleton(self.location)

    def log_conditional_from_simple_interval(self, interval: SimpleInterval) -> Tuple[Self, float]:
        if interval.contains(self.location):
            return self, 0.
        else:
            return None, -np.inf

    def probability_of_simple_event(self, event: SimpleEvent) -> float:
        interval: Interval = event[self.variable]

        return 0. if (closed(self.location - self.tolerance, self.location + self.tolerance) & interval).is_empty() \
            else 1.

    def univariate_log_mode(self) -> Tuple[AbstractCompositeSet, float]:
        return self.univariate_support, np.log(self.density_cap)

    def sample(self, amount: int) -> np.array:
        return np.full((amount, 1), self.location)

    def moment(self, order: OrderType, center: CenterType) -> MomentType:
        order = order[self.variable]
        center = center[self.variable]

        if order == 0:
            moment = 1.
        elif order == 1:
            moment = self.location - center
        else:
            moment = 0.

        return VariableMap({self.variable: moment})

    def __eq__(self, other):
        return (isinstance(other, self.__class__) and super().__eq__(
            other) and self.location == other.location and self.density_cap == other.density_cap)

    def __hash__(self):
        return hash((self.variable, self.location, self.density_cap))

    @property
    def representation(self):
        return f"δ({self.location}, {self.density_cap})"

    def __repr__(self):
        return f"δ({self.variable.name})"

    def __copy__(self):
        return self.__class__(self.variable, self.location, self.density_cap)

    def __deepcopy__(self, memo=None):
        if memo is None:
            memo = {}
        id_self = id(self)
        if id_self in memo:
            return memo[id_self]

        variable = Continuous(self.variable.name)
        result = self.__class__(variable, self.location, self.density_cap)
        memo[id_self] = result
        return result

    def to_json(self) -> Dict[str, Any]:
        return {**super().to_json(), "location": self.location, "density_cap": self.density_cap}

    @classmethod
    def _from_json(cls, data: Dict[str, Any]) -> Self:
        variable = Continuous.from_json(data["variable"])
        location = data["location"]
        density_cap = data["density_cap"]
        return cls(variable, location, density_cap)

    def plot(self, **kwargs) -> List:
        lower_border = self.location - 1
        upper_border = self.location + 1
        pdf_trace = go.Scatter(x=[lower_border, self.location, self.location, self.location, upper_border],
                               y=[0, 0, self.density_cap, 0, 0], mode="lines", name="PDF")
        cdf_trace = go.Scatter(x=[lower_border, self.location, self.location, upper_border], y=[0, 0, 1, 1],
                               mode="lines", name="CDF")
        expectation_trace = go.Scatter(x=[self.location, self.location],
                                       y=[0, self.density_cap * SCALING_FACTOR_FOR_EXPECTATION_IN_PLOT],
                                       mode="lines+markers", name="Expectation")
        mode_trace = go.Scatter(x=[self.location, self.location],
                                y=[0, self.density_cap * SCALING_FACTOR_FOR_EXPECTATION_IN_PLOT], mode="lines+markers",
                                name="Mode")
        return [pdf_trace, cdf_trace, expectation_trace, mode_trace]

    def translate(self, translation: VariableMap[Variable, float]):
        self.location += translation[self.variable]

    def scale(self, scaling: VariableMap[Variable, float]):
        self.location += scaling[self.variable]
