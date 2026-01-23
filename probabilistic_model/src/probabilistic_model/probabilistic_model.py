from __future__ import annotations

import abc
import math

import tqdm
from random_events.interval import closed, SimpleInterval
from random_events.product_algebra import *
from random_events.set import *
from random_events.variable import *

from .constants import *
from .error import IntractableError, UndefinedOperationError
from .utils import neighbouring_points

# Type definitions
FullEvidenceType = np.array  # [Union[float, int, SetElement]]

# # Type hinting for Python 3.7 to 3.9
if TYPE_CHECKING:
    OrderType = VariableMap[Union[Integer, Continuous], int]
    CenterType = VariableMap[Union[Integer, Continuous], float]
    MomentType = VariableMap[Union[Integer, Continuous], float]
else:
    OrderType = VariableMap
    CenterType = VariableMap
    MomentType = VariableMap


class ProbabilisticModel(abc.ABC):
    """
    Abstract base class for probabilistic models.

    The definition of events follows the definition of events in the random_events package.
    The definition of functions is motivated by the background knowledge provided in the probabilistic circuits.

    This class can be used as an interface to any kind of probabilistic model, tractable or not.

    """

    @property
    def representation(self) -> str:
        """
        The symbol used to represent this distribution.
        """
        return self.__class__.__name__

    @property
    @abstractmethod
    def variables(self) -> Tuple[Variable, ...]:
        """
        :return: The variables of the model.
        """
        raise NotImplementedError

    @property
    @abstractmethod
    def support(self) -> Event:
        """
        :return: The support of the model.
        """
        raise NotImplementedError

    def likelihood(self, events: np.array) -> np.array:
        """
        Calculate the likelihood of an array of events.

        The likelihood is a full evidence query, i.e., an assignment to all variables in the model.
        The order of elements in the event has to correspond to the order of variables in the model.

        The event belongs to the class of full evidence queries.

        ..
        Note:: You can read more about this query class in Definition 1 in :cite:p:`choi2020probabilistic`
            or watch the `video tutorial <https://youtu.be/2RAG5-L9R70?si=TAfIX2LmOWM-Fd2B&t=785>`_.
            :cite:p:`youtube2020probabilistic`

        :param events: The array of full evidence events.
        The shape of the array has to be (n, len(self.variables)).
        :return: The likelihood of the events as an array with shape (n,).
        """
        return np.exp(self.log_likelihood(events))

    @abstractmethod
    def log_likelihood(self, events: np.array) -> np.array:
        """
        Calculate the log-likelihood of an event.

        Check the documentation of `likelihood` for more information.

        :param events: The full evidence event with shape (#events, #variables)
        :return: The log-likelihood of the event with shape (#events).
        """
        raise NotImplementedError

    def cdf(self, events: np.array) -> np.array:
        """
        Calculate the cumulative distribution function of an event-array.

        The event belongs to the class of full evidence queries.

        ..Note:: The cdf only exists if all variables are continuous or integers.

        :param events: The array of full evidence events.
                       The shape of the array has to be (n, len(self.variables)).
        :return: The cumulative distribution function of the event as an array of shape (n,).
        """
        raise NotImplementedError

    def probability(self, event: Event) -> float:
        """
        Calculate the probability of an event.
        The event is richly described by the random_events package.

        :param event: The event.
        :return: The probability of the event.
        """
        event.fill_missing_variables(set(self.variables))
        return sum(self.probability_of_simple_event(simple_set) for simple_set in event.simple_sets)

    @abstractmethod
    def probability_of_simple_event(self, event: SimpleEvent) -> float:
        """
        Calculate the probability of a simple event.

        The event belongs to the class of marginal queries.

        .. Note:: You can read more about queries of this class in Definition 11 in :cite:p:`choi2020probabilistic`
            or watch the `video tutorial <https://youtu.be/2RAG5-L9R70?si=8aEGIqmoDTiUR2u6&t=1089>`_.
            :cite:p:`youtube2020probabilistic`

        :param event: The event.
        :return: The probability of the event.
        """
        raise NotImplementedError

    def mode(self) -> Tuple[Event, float]:
        """
        Calculate the mode of the model.
        The mode is the **set** of most likely events.

        The calculation belongs to the map query class.

        .. Note:: You can read more about queries of this class in Definition 26 in :cite:p:`choi2020probabilistic`
            or watch the `video tutorial <https://youtu.be/2RAG5-L9R70?si=FjREKNtAV0owm27A&t=1962>`_.
            :cite:p:`youtube2020probabilistic`

        :return: The mode and its likelihood.
        """
        mode, log_likelihood = self.log_mode()
        return mode, np.exp(log_likelihood)

    @abstractmethod
    def log_mode(self) -> Tuple[Event, float]:
        """
        Calculate the mode of the model.

        Check the documentation of `mode` for more information.

        :return: The mode and its log-likelihood.
        """
        raise NotImplementedError

    def marginal(self, variables: Iterable[Variable]) -> Optional[Self]:
        """
        Calculate the marginal distribution of a set of variables.

        :param variables: The variables to calculate the marginal distribution on.
        :return: The marginal distribution over the variables.
        """
        raise NotImplementedError

    def truncated(self, event: Event) -> Tuple[Optional[Union[ProbabilisticModel, Self]], float]:
        """
        Calculate the truncated distribution P(*| event) and the probability of the event.

        If the event is impossible, the truncated distribution is None and the probability is 0.

        :param event: The event to condition on.
        :return: The truncated distribution and the probability of the event.
        """
        event.fill_missing_variables(set(self.variables))
        conditional, log_probability = self.log_truncated(event)
        return conditional, np.exp(log_probability)

    @abstractmethod
    def log_truncated(self, event: Event) -> Tuple[Optional[Union[ProbabilisticModel, Self]], float]:
        """
        Calculate the truncated distribution P(*| event) and the probability of the event.

        Check the documentation of `truncated` for more information.

        :param event: The event to condition on.
        :return: The truncated distribution and the log-probability of the event.
        """
        raise NotImplementedError

    def conditional(self, point: Dict[Variable, Any]) -> Tuple[Optional[Self], float]:
        """
        Calculate the truncated distribution P(*| point) and the probability of the event.

        :param point: A partial point to calculate the truncated distribution on.
        :return: The truncated distribution and the log-probability of the point.
        """
        conditional, log_probability = self.log_conditional(point)
        return conditional, np.exp(log_probability)

    @abstractmethod
    def log_conditional(self, point: Dict[Variable, Any]) -> Tuple[Optional[Self], float]:
        """
        Calculate the truncated distribution P(*| point) and the probability of the event.
        Check the documentation of `conditional` for more information.

        :param point: A partial point to calculate the truncated distribution on.
        :return: The truncated distribution and the log-probability of the point.
        """
        raise NotImplementedError

    @abstractmethod
    def sample(self, amount: int) -> np.array:
        """
        Sample from the model.

        :param amount: The number of samples to draw.
        :return: The samples.
        """
        raise NotImplementedError

    def moment(self, order: OrderType, center: CenterType) -> MomentType:
        """
        Calculate the (centralized) moment of the distribution.

        .. math::

            \int_{-\infty}^{\infty} (x - center)^{order} pdf(x) dx

        .. Note:: You can read more about queries of this class in Definition 22 in :cite:p:`choi2020probabilistic`_.
            :cite:p:`youtube2020probabilistic`


        :param order: The orders of the moment as a variable map for every continuous and integer variable.
        :param center: The center of the moment as a variable map for every continuous and integer variable.
        :return: The moments of the variables in `order`.
        """
        raise NotImplementedError

    def expectation(self, variables: Optional[Iterable[Variable]] = None) -> MomentType:
        """
        Calculate the expectation of the numeric variables in `variables`.

        :param variables: The variable to calculate the expectation of.
        :return: The expectation of the variable.
        """

        if variables is None:
            variables = [variable for variable in self.variables if isinstance(variable, (Continuous, Integer))]

        order = VariableMap({variable: 1 for variable in variables})
        center = VariableMap({variable: 0 for variable in variables})
        return self.moment(order, center)

    def variance(self, variables: Optional[Iterable[Variable]] = None) -> MomentType:
        """
        Calculate the variance of the numeric variables in `variables`.

        :param variables: The variable to calculate the variance of.
        :return: The variance of the variable.
        """

        if variables is None:
            variables = [variable for variable in self.variables if isinstance(variable, (Continuous, Integer))]

        order = VariableMap({variable: 2 for variable in variables})
        center = self.expectation(variables)
        return self.moment(order, center)

    def universal_simple_event(self) -> SimpleEvent:
        """
        :return: A simple event that contains every possible value.
        """
        return SimpleEvent({variable: variable.domain for variable in self.variables})

    def translate(self, translation: Dict[Variable, float]):
        """
        Translate the model in-place.
        Translation is done by adding the translation to the variable location influencing values.
        The translation can be viewed as what happens
        when you shift the numeric variables of the model by a constant vector.

        :param translation: The variable value pairs to translate the model by.
        """
        raise NotImplementedError

    def scale(self, scaling: Dict[Variable, float]):
        """
        Scale the model in-place.
        Scaling is done by multiplying the variable location influencing values.
        The scaling can be viewed as what happens
        when you multiply the numeric variables of the model by a constant vector.

        :param scaling: The variable value pairs to scale the model by.
        """
        ...

    def __copy__(self):
        raise NotImplementedError

    def plotly_layout(self) -> Dict[str, Any]:
        """
        Create a layout for the plotly plot.

        :return: The layout.
        """
        if len(self.variables) == 1:
            return self.plotly_layout_1d()
        elif len(self.variables) == 2:
            return self.plotly_layout_2d()
        elif len(self.variables) == 3:
            return self.plotly_layout_3d()
        else:
            raise NotImplementedError("Plotting is only supported for models with up to three variables.")

    def plotly_layout_1d(self) -> Dict[str, Any]:
        """
        :return: The layout argument for plotly figures as dict
        """
        return {"title": f"{self.representation}", "xaxis": {"title": self.variables[0].name}}

    def plotly_layout_2d(self) -> Dict[str, Any]:
        """
        :return: The layout argument for plotly figures as dict
        """
        return {"title": f"{self.representation}", "xaxis": {"title": self.variables[0].name},
                "yaxis": {"title": self.variables[1].name}}

    def plotly_layout_3d(self) -> Dict[str, Any]:
        """
        :return: The layout argument for plotly figures as dict
        """
        return {"title": f"{self.representation}",
                "scene": {"xaxis": {"title": self.variables[0].name}, "yaxis": {"title": self.variables[1].name},
                          "zaxis": {"title": self.variables[2].name}}}

    def plot(self, number_of_samples: int = 1000, surface=False, mode=False) -> List:
        """
        Generate traces that can be plotted with plotly.

        :param number_of_samples: The number of samples to draw.
        :param surface: If True, plot the model as a surface plot.
        :param mode: If True, plot the mode of the model.
        :return: The traces.
        """
        if len(self.variables) == 1:
            if self.variables[0].is_numeric:
                return self.plot_1d_numeric(number_of_samples, mode)
            else:
                return self.plot_1d_symbolic()
        elif len(self.variables) == 2:
            if surface:
                return self.plot_2d_surface(number_of_samples)
            else:
                return self.plot_2d(number_of_samples, mode)
        elif len(self.variables) == 3:
            return self.plot_3d(number_of_samples, mode)
        else:
            raise NotImplementedError("Plotting is only supported for models with up to three variables.")

    def plot_1d_symbolic(self) -> List:
        variable: Symbolic = self.variables[0]

        # calculate probabilities of every element in the domain
        probabilities = {str(element): self.probability_of_simple_event(SimpleEvent({variable: element})) for element in
                         variable.domain}

        maximum = max(probabilities.values())

        # highlight the mode
        color = [MODE_TRACE_COLOR if probability == maximum else PDF_TRACE_COLOR for probability in
                 probabilities.values()]

        return [go.Bar(x=list(probabilities.keys()), y=list(probabilities.values()), name=PDF_TRACE_NAME,
                       marker=dict(color=color))]

    def plot_1d_numeric(self, number_of_samples: int, mode=False) -> List:
        """
        Plot a one-dimensional model using samples.

        :param number_of_samples: The number of samples to draw.
        :param mode: If True, plot the mode of the model.
        :return: The traces.
        """

        # sample for the plot
        samples = self.sample(number_of_samples)[:, 0]

        # prepare pdf trace
        supporting_interval: Interval = self.support.simple_sets[0][self.variables[0]]

        # add border points to samples
        for simple_interval in supporting_interval.simple_sets:
            simple_interval: SimpleInterval
            lower, upper = simple_interval.lower, simple_interval.upper
            if lower > -np.inf:
                samples = np.concatenate((samples, neighbouring_points(lower)))
            if upper < np.inf:
                samples = np.concatenate((samples, neighbouring_points(upper)))

        samples = np.sort(samples)
        lowest = samples[0]
        highest = samples[-1]
        size = highest - lowest
        samples = np.concatenate((np.array([lowest - size * 0.05]), samples, np.array([highest + size * 0.05])))

        # add cdf trace if implemented
        try:
            cdf = self.cdf(samples.reshape(-1, 1))
            cdf_trace = [go.Scatter(x=samples, y=cdf, mode="lines", legendgroup="CDF", name=CDF_TRACE_NAME,
                                    line=dict(color=CDF_TRACE_COLOR))]
        except UndefinedOperationError:
            cdf_trace = []

        pdf = self.likelihood(samples.reshape(-1, 1))
        pdf_trace = go.Scatter(x=samples, y=pdf, mode="lines", legendgroup="PDF", name=PDF_TRACE_NAME,
                               line=dict(color=PDF_TRACE_COLOR))

        # plot the mode if possible
        if mode:
            try:
                mode, maximum_likelihood = self.mode()
            except IntractableError:
                mode, maximum_likelihood = None, max(pdf)
        else:
            mode, maximum_likelihood = None, max(pdf)

        height = maximum_likelihood * SCALING_FACTOR_FOR_EXPECTATION_IN_PLOT
        mode_traces = self.univariate_mode_traces(mode, height)

        return ([pdf_trace,
                 self.univariate_expectation_trace(height)] + mode_traces + self.univariate_complement_of_support_trace(
            min(samples), max(samples)) + cdf_trace)

    def univariate_expectation_trace(self, height: float) -> go.Scatter:
        """
        Create a trace for the expectation of the model in 1d.
        :param height: The height of the trace.
        :return: The trace.
        """
        mean = self.expectation(self.variables)[self.variables[0]]
        mean_trace = go.Scatter(x=[mean, mean], y=[0, height], mode="lines+markers", name=EXPECTATION_TRACE_NAME,
                                marker=dict(color=EXPECTATION_TRACE_COLOR), line=dict(color=EXPECTATION_TRACE_COLOR))
        return mean_trace

    def univariate_mode_traces(self, mode: Optional[Event], height: float):
        if mode is None:
            return []

        interval = mode.simple_sets[0][self.variables[0]]
        x_values = []
        y_values = []
        for simple_interval in interval.simple_sets:
            simple_interval: SimpleInterval
            x_values += (
                [simple_interval.lower, simple_interval.lower, simple_interval.upper, simple_interval.upper, None])
            y_values += ([0, height, height, 0, None])
        return [go.Scatter(x=x_values, y=y_values, mode="lines+markers", name=MODE_TRACE_NAME, fill="toself",
                           line=dict(color=MODE_TRACE_COLOR))]

    def univariate_complement_of_support_trace(self, min_of_samples: float, max_of_samples: float) -> List:
        """
        Create a trace for the complement of the support of the model in 1d.
        :param min_of_samples: The minimum value of the samples.
        :param max_of_samples: The maximum value of the samples.
        :return: A list of traces for the support of the model.
        """
        supporting_interval: Interval = self.support.simple_sets[0][self.variables[0]]
        complement_of_support = supporting_interval.complement()
        limiting_interval = closed(min_of_samples - min_of_samples * PADDING_FACTOR_FOR_X_AXIS_IN_PLOT,
                                   max_of_samples + max_of_samples * PADDING_FACTOR_FOR_X_AXIS_IN_PLOT)
        limited_complement_of_support = complement_of_support & limiting_interval
        traces = SimpleEvent({self.variables[0]: limited_complement_of_support}).plot()
        for trace in traces:
            trace.update(name=PDF_TRACE_NAME, marker=dict(color=PDF_TRACE_COLOR))
        return traces

    def plot_2d(self, number_of_samples: int, mode=False) -> List:
        """
        Plot a two-dimensional model.

        :param number_of_samples: The number of samples to draw.
        :param mode: If True, plot the mode of the model.
        :return: The traces.
        """
        samples = self.sample(number_of_samples)
        likelihood = self.likelihood(samples)
        expectation = self.expectation(self.variables)
        likelihood_trace = go.Scatter(x=samples[:, 0], y=samples[:, 1], mode="markers", marker=dict(color=likelihood),
                                      name=SAMPLES_TRACE_NAME)
        expectation_trace = go.Scatter(x=[expectation[self.variables[0]]], y=[expectation[self.variables[1]]],
                                       mode="markers", marker=dict(color=EXPECTATION_TRACE_COLOR),
                                       name=EXPECTATION_TRACE_NAME)

        if mode:
            mode_traces = self.multivariate_mode_traces()
        else:
            mode_traces = []

        return [likelihood_trace, expectation_trace] + mode_traces

    def plot_2d_surface(self, number_of_samples: int) -> List:
        """
        Plot a two-dimensional model as a surface plot.

        :param number_of_samples: The number of samples to draw.
        :return: The traces.
        """
        samples = self.sample(math.ceil(math.sqrt(number_of_samples)))
        support = self.support
        likelihood = self.likelihood(samples)
        max_likelihood = max(likelihood)

        support_trace = self.bounding_box_trace_of_simple_event(support.bounding_box(), samples, 0.)
        support_trace.showscale = False
        support_trace.cmin = 0
        support_trace.cmax = max_likelihood

        expectation_trace = self.expectation_trace_2d_surface(max_likelihood * SCALING_FACTOR_FOR_EXPECTATION_IN_PLOT)

        traces = [support_trace, expectation_trace]

        first = True
        for simple_set in tqdm.tqdm(support.simple_sets):
            for i1, i2 in itertools.product(*simple_set.values()):
                simple_event = SimpleEvent({self.variables[0]: i1, self.variables[1]: i2})
                trace = self.plot_2d_surface_of_simple_event(simple_event, samples)
                if not first:
                    trace.showscale = False
                    first = False
                trace.cmin = 0
                trace.cmax = max_likelihood
                traces.append(trace)

        return traces

    def expectation_trace_2d_surface(self, height: float) -> go.Scatter3d:
        expectation = self.expectation(self.variables)
        x = expectation[self.variables[0]]
        y = expectation[self.variables[1]]
        return go.Scatter3d(x=[x, x], y=[y, y], z=[0, height], mode="lines+markers", name=EXPECTATION_TRACE_NAME, )

    def bounding_box_trace_of_simple_event(self, simple_event: SimpleEvent, samples: np.array,
                                           fill_value=0.) -> go.Surface:
        """
        Create a bounding box trace for a simple event.
        :param simple_event: The simple event.
        :param samples: The samples to read from if bounds are infinite.
        :param fill_value: The height of the box.

        :return: The trace.
        """
        x_variable = self.variables[0]
        y_variable = self.variables[1]
        min_x = simple_event[x_variable].simple_sets[0].lower
        max_x = simple_event[x_variable].simple_sets[-1].upper
        min_y = simple_event[y_variable].simple_sets[0].lower
        max_y = simple_event[y_variable].simple_sets[-1].upper

        min_x = min_x if min_x > -np.inf else min(samples[:, 0])
        min_x = np.nextafter(min_x, -np.inf)
        max_x = max_x if max_x < np.inf else max(samples[:, 0])
        max_x = np.nextafter(max_x, np.inf)

        min_y = min_y if min_y > -np.inf else min(samples[:, 1])
        min_y = np.nextafter(min_y, -np.inf)
        max_y = max_y if max_y < np.inf else max(samples[:, 1])
        max_y = np.nextafter(max_y, np.inf)

        return go.Surface(x=[min_x, max_x], y=[min_y, max_y], z=[[fill_value, fill_value], [fill_value, fill_value]],
                          showscale=False)

    def plot_2d_surface_of_simple_event(self, simple_event: SimpleEvent, samples: np.array):
        # filter samples by this event
        samples_of_this_event = [s for s in samples if simple_event.contains(s)]

        if len(samples_of_this_event) == 0:
            return go.Surface()

        samples_of_this_event = np.stack(samples_of_this_event, axis=0)

        x_variable = self.variables[0]
        y_variable = self.variables[1]

        x_support: SimpleInterval = simple_event[x_variable].simple_sets[0]
        y_support: SimpleInterval = simple_event[y_variable].simple_sets[0]

        # create border points
        min_x = x_support.lower if x_support.lower > -np.inf else min(samples_of_this_event[:, 0])
        min_x_next_after = np.nextafter(min_x, -np.inf)
        max_x = x_support.upper if x_support.upper < np.inf else max(samples_of_this_event[:, 0])
        max_x_next_after = np.nextafter(max_x, np.inf)
        min_y = y_support.lower if y_support.lower > -np.inf else min(samples_of_this_event[:, 1])
        min_y_next_after = np.nextafter(min_y, -np.inf)
        max_y = y_support.upper if y_support.upper < np.inf else max(samples_of_this_event[:, 1])
        max_y_next_after = np.nextafter(max_y, np.inf)

        # create x axis
        x = samples_of_this_event[:, 0]
        x = np.append(x, [min_x, max_x, min_x_next_after, max_x_next_after])
        x.sort()

        # create y axis
        y = samples_of_this_event[:, 1]
        y = np.append(y, [min_y, max_y, min_y_next_after, max_y_next_after])
        y.sort()

        meshgrid = np.array(np.meshgrid(x, y)).T.reshape(-1, 2)
        likelihood = self.likelihood(meshgrid).reshape(len(x), len(y))
        trace = go.Surface(z=likelihood, x=x, y=y)
        return trace

    def plot_3d(self, number_of_samples: int, mode=False) -> List:
        """
        Plot a three-dimensional model using samples.

        :param number_of_samples: The number of samples to draw.
        :param mode: If True, plot the mode of the model.
        :return: The traces.s
        """
        samples = self.sample(number_of_samples)
        likelihood = self.likelihood(samples)
        expectation = self.expectation(self.variables)
        likelihood_trace = go.Scatter3d(x=samples[:, 0], y=samples[:, 1], z=samples[:, 2], mode="markers",
                                        marker=dict(color=likelihood), name=SAMPLES_TRACE_NAME)
        expectation_trace = go.Scatter3d(x=[expectation[self.variables[0]]], y=[expectation[self.variables[1]]],
                                         z=[expectation[self.variables[2]]], mode="markers",
                                         name=EXPECTATION_TRACE_NAME, marker=dict(color=EXPECTATION_TRACE_COLOR))

        if mode:
            mode_traces = self.multivariate_mode_traces()
        else:
            mode_traces = []

        return [likelihood_trace, expectation_trace] + mode_traces

    def multivariate_mode_traces(self):
        """
        :return: traces for the mode of a multivariate model.
        """
        try:
            mode, _ = self.mode()
            mode_traces = mode.plot(color=MODE_TRACE_COLOR)
            for trace in mode_traces:
                trace.update(name=MODE_TRACE_NAME, mode="lines+markers")
        except IntractableError:
            mode_traces = []
        return mode_traces
