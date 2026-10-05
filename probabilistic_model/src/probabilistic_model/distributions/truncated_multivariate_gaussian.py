from __future__ import annotations

import copy
import dataclasses
import functools
import math

import numpy as np
import numpy.typing as npt
from random_events.interval import Bound, SimpleInterval, singleton
from random_events.product_algebra import Event, SimpleEvent, VariableMap
from random_events.variable import Variable
from scipy.optimize import lsq_linear
from scipy.special import ndtr
from scipy.stats import multivariate_normal, norm, truncnorm
from typing_extensions import (
    Any,
    Dict,
    Iterable,
    Optional,
    Self,
    Tuple,
    TYPE_CHECKING,
)

from probabilistic_model.exceptions import (
    EventIsNotABoxError,
    InvalidMomentOrderError,
    NoClosedFormError,
)
from probabilistic_model.probabilistic_model import (
    CenterType,
    MomentType,
    OrderType,
    ProbabilisticModel,
)

if TYPE_CHECKING:
    from probabilistic_model.distributions.multivariate_gaussian import (
        MultivariateGaussianDistribution,
    )


# %% how moments are integrated numerically


@dataclasses.dataclass
class QuadraturePoints:
    """
    Where an integrand is evaluated and what each evaluation is weighted with.
    """

    values: npt.NDArray
    """
    The points to evaluate the integrand at.
    """

    weights: npt.NDArray
    """
    The weight of the integrand at every point.
    """


@dataclasses.dataclass(frozen=True)
class MomentIntegration:
    """
    How the moments of a Gaussian confined to a box are integrated numerically: by
    Gauss-Legendre quadrature over panels of equal width.
    """

    deviations_integrated_over: float = 10.0
    """
    How many standard deviations around its mean an unbounded variable is integrated
    over.
    """

    panels: int = 16
    """
    How many panels the interval of a variable is split into.
    """

    nodes_per_panel: int = 16
    """
    How many Gauss-Legendre nodes every panel is integrated with.
    """

    def points_between(self, lower: float, upper: float) -> QuadraturePoints:
        """
        :param lower: The lower end of the interval to integrate over.
        :param upper: The upper end of the interval to integrate over.
        :return: The points and weights that integrate a function over the interval.
        """
        edges = np.linspace(lower, upper, self.panels + 1)
        half_widths = np.diff(edges)[:, None] / 2
        nodes, node_weights = np.polynomial.legendre.leggauss(self.nodes_per_panel)
        return QuadraturePoints(
            values=(edges[:-1, None] + half_widths + half_widths * nodes).ravel(),
            weights=(half_widths * node_weights).ravel(),
        )


# %% a Gaussian that has been confined to a box


@dataclasses.dataclass
class TruncatedMultivariateGaussianDistribution(ProbabilisticModel):
    """
    A Gaussian confined to a box, which is what is left of one once part of the space is
    ruled out.

    It is not itself a Gaussian: cutting a correlated Gaussian along an axis leaves a
    shape no Gaussian has. What it keeps is the shape of the original inside the box,
    scaled up so that the box is certain.

    A box is one simple interval per variable, which is as far as one such shape
    reaches: ruling out a hole in the middle of the space leaves several of them, which
    is a circuit rather than a distribution.
    """

    untruncated: MultivariateGaussianDistribution
    """
    The Gaussian before anything was ruled out.
    """

    box: SimpleEvent
    """
    Everything still considered possible, one simple interval per variable.
    """

    burn_in_period_length: int = 100
    """
    How many times each chain draws every variable before its last state becomes a
    sample.
    """

    moment_integration: MomentIntegration = dataclasses.field(
        default_factory=MomentIntegration
    )
    """
    How the moments are integrated numerically.
    """

    @property
    def variables(self) -> Tuple[Variable, ...]:
        return self.untruncated.variables

    @property
    def support(self) -> Event:
        return self.box.as_composite_set()

    @functools.cached_property
    def normalizing_constant(self) -> float:
        """
        :return: How probable the box was before it was the only thing left, which is
            what every density here is scaled up by. It is a numerical integral, so it
            is computed once.
        """
        return self.untruncated.probability_of_simple_event(self.box)

    def interval_of(self, variable: Variable) -> SimpleInterval:
        """
        :param variable: The variable to read.
        :return: The one simple interval the box leaves it.
        """
        return self.box[variable].simple_sets[0]

    def box_contains(self, point: npt.NDArray) -> bool:
        """
        :param point: A point, laid out by the variables.
        :return: Whether the box allows the point.
        """
        return self.box.contains(
            tuple(
                point[self.untruncated.index_of(variable)]
                for variable in self.box.variables
            )
        )

    def log_likelihood(self, events: npt.NDArray) -> npt.NDArray:
        return np.where(
            [self.box_contains(event) for event in events],
            np.atleast_1d(self.untruncated.scipy_distribution.logpdf(events))
            - math.log(self.normalizing_constant),
            -np.inf,
        )

    def cumulative_distribution_function(self, events: npt.NDArray) -> npt.NDArray:
        """
        :param events: The points to evaluate at, one row per point.
        :return: For each point, how probable it is that every variable is at most its
            value there, through :mod:`scipy.stats.multivariate_normal`'s ``cdf``
            confined to the box.
        """
        return np.array([self._cumulative_probability_at(event) for event in events])

    def _cumulative_probability_at(self, event: npt.NDArray) -> float:
        """
        :param event: One point.
        :return: How probable it is that every variable is at most its value there.
        """
        intervals = [self.interval_of(variable) for variable in self.variables]
        lower = np.array([interval.lower for interval in intervals])
        upper = np.array([interval.upper for interval in intervals])
        if np.any(event < lower):
            return 0.0
        probability = self.untruncated.scipy_distribution.cdf(
            np.minimum(event, upper), lower_limit=lower
        )
        return max(float(probability), 0.0) / self.normalizing_constant

    def probability_of_simple_event(self, event: SimpleEvent) -> float:
        surviving = event.intersection_with(self.box)
        if surviving.is_empty():
            return 0.0
        scipy_distribution = self.untruncated.scipy_distribution
        probability = sum(
            max(float(scipy_distribution.cdf(upper, lower_limit=lower)), 0.0)
            for lower, upper in self.untruncated._bounds_of_boxes(surviving)
        )
        return probability / self.normalizing_constant

    # %% the most likely point the box still allows

    def log_mode(self) -> Tuple[Event, float]:
        """
        The density falls away from the untruncated mean in every direction and the box
        is convex, so there is exactly one most likely point: the mean itself while the
        box still contains it, and otherwise the point of the box nearest to it in the
        distribution's own metric.

        :return: That point and its log-density.
        """
        most_likely = self._most_likely_point(self.untruncated.precision)
        mode = SimpleEvent.from_data(
            {
                variable: singleton(value)
                for variable, value in zip(self.variables, most_likely)
            }
        ).as_composite_set()
        return mode, float(self.log_likelihood(most_likely.reshape(1, -1))[0])

    def _most_likely_point(self, precision: npt.NDArray) -> npt.NDArray:
        """
        The point of the box nearest to the mean in the distribution's own metric
        minimises :math:`\\lVert W (x - \\mu) \\rVert^2` over the box, where
        :math:`W^T W` is the inverse of the covariance. That is a bounded linear least
        squares problem, which :func:`scipy.optimize.lsq_linear` solves exactly.

        :param precision: The untruncated distribution's precision matrix.
        :return: Where the density is greatest within the box, moved just inside an
            interval that excludes its own end.
        """
        mean = self.untruncated.mean
        if self.box_contains(mean):
            return mean
        intervals = [self.interval_of(variable) for variable in self.variables]
        whitening = np.linalg.cholesky(precision).T
        found = lsq_linear(
            whitening,
            whitening @ mean,
            bounds=(
                [interval.lower for interval in intervals],
                [interval.upper for interval in intervals],
            ),
            method="bvls",
        ).x
        return np.array(
            [
                interval.nearest_contained_value(float(value))
                for value, interval in zip(found, intervals)
            ]
        )

    # %% ruling out more, and fixing a variable

    def log_truncated(
        self, event: Event, singleton_allowed: bool = False
    ) -> Tuple[Optional[ProbabilisticModel], float]:
        """
        Rule out more of the space.

        :param event: What is still to be considered possible.
        :param singleton_allowed: Ignored, since every single point has probability
            zero.
        :return: The further confined distribution and the log-probability of the event
            under this one, or nothing at all if nothing is left.
        :raises EventIsNotABoxError: If the event is not one simple interval per
            variable.
        """
        event.fill_missing_variables(set(self.variables))
        if not event.is_box():
            raise EventIsNotABoxError(model=self, event=event)
        [box] = event.simple_sets
        surviving = box.intersection_with(self.box)
        probability = self.probability_of_simple_event(surviving)
        if probability == 0.0:
            return None, -np.inf
        return (
            dataclasses.replace(self, box=surviving),
            math.log(probability),
        )

    def log_conditional(
        self, point: Dict[Variable, Any]
    ) -> Tuple[Optional[ProbabilisticModel], float]:
        """
        Fix some of the variables at the values given and answer with the distribution
        over the rest, which is the Gaussian conditional confined to the slice the box
        makes at those values.

        :param point: What each fixed variable is known to be.
        :return: That distribution and the log-likelihood of the values given, or
            nothing at all if the box rules them out.
        :raises VariableNotInDistributionError: If a fixed variable is not one of this
            distribution's.
        :raises ProbabilisticCircuitRequiredError: If every variable is fixed, which
            leaves a product of Dirac impulses.
        """
        conditional, log_density = self.untruncated.log_conditional(point)
        free = [variable for variable in self.variables if variable not in point]
        slice_of_the_box = SimpleEvent.from_data(
            {variable: self.box[variable] for variable in free}
        ).as_composite_set()
        confined, log_probability = conditional.log_truncated(slice_of_the_box)

        inside = all(
            self.box[variable].contains(value) for variable, value in point.items()
        )
        log_likelihood = (
            log_density + log_probability - math.log(self.normalizing_constant)
            if inside
            else -np.inf
        )
        if log_likelihood == -np.inf:
            return None, -np.inf
        return (
            dataclasses.replace(
                self, untruncated=confined.untruncated, box=confined.box
            ),
            log_likelihood,
        )

    # %% marginals and moments

    def marginal(self, variables: Iterable[Variable]) -> Optional[Self]:
        """
        :param variables: The variables to keep.
        :return: A copy of this distribution if all of its variables are kept, or
            nothing if none of them is.
        :raises NoClosedFormError: If only some of its variables are kept. Integrating
            a variable out of a correlated Gaussian confined to a box does not leave a
            Gaussian confined to a box.
        """
        kept = set(variables)
        number_of_kept_variables = sum(variable in kept for variable in self.variables)
        if number_of_kept_variables == 0:
            return None
        if number_of_kept_variables < len(self.variables):
            raise NoClosedFormError(type(self), type(self).marginal)
        return self.__copy__()

    def moment(self, order: OrderType, center: CenterType) -> MomentType:
        """
        Every moment asked for here is of one variable on its own, and is answered by
        integrating the density of that variable numerically.

        :param order: The order of the moment of each variable to answer for.
        :param center: What to take each of those moments about.
        :return: The moment of each variable asked for.
        :raises InvalidMomentOrderError: If an order is not a whole number or is
            negative.
        """
        moments = VariableMap()
        for variable in order:
            if order[variable] < 0 or order[variable] != int(order[variable]):
                raise InvalidMomentOrderError(order[variable])
            requested_order = int(order[variable])
            moments_about_zero = self.moments_about_zero_of(
                self.untruncated.index_of(variable), requested_order
            )
            # the binomial expansion of (x - center) ** order
            moments[variable] = float(
                sum(
                    math.comb(requested_order, power)
                    * (-center[variable]) ** (requested_order - power)
                    * moments_about_zero[power]
                    for power in range(requested_order + 1)
                )
            )
        return moments

    def moments_about_zero_of(self, index: int, highest_order: int) -> npt.NDArray:
        """
        The density of one variable on its own is its Gaussian density times the
        probability that the other variables stay inside the box given its value
        (Cartinhour, 1990). It is integrated over the interval of the variable as
        :attr:`moment_integration` says.

        :param index: The index of a variable.
        :param highest_order: The highest order to answer.
        :return: The moments of that variable about zero, the entry at ``k`` being the
            moment of order ``k``.
        """
        interval = self.interval_of(self.variables[index])
        mean = self.untruncated.mean[index]
        deviation = math.sqrt(self.untruncated.covariance.between(index, index))
        reach = self.moment_integration.deviations_integrated_over * deviation
        points = self.moment_integration.points_between(
            max(interval.lower, min(mean - reach, interval.upper - reach)),
            min(interval.upper, max(mean + reach, interval.lower + reach)),
        )
        density = (
            points.weights
            * norm.pdf(points.values, loc=mean, scale=deviation)
            * self.probability_of_the_other_variables_given(index, points.values)
        )
        return np.array(
            [
                np.sum(density * points.values**power)
                for power in range(highest_order + 1)
            ]
        ) / np.sum(density)

    def probability_of_the_other_variables_given(
        self, index: int, values: npt.NDArray
    ) -> npt.NDArray:
        """
        :param index: The index of a variable.
        :param values: Values of that variable.
        :return: For each value, how probable it is under the untruncated Gaussian that
            every other variable is inside its interval given that the variable takes
            the value.
        """
        others = [other for other in range(len(self.variables)) if other != index]
        if not others:
            return np.ones(len(values))
        intervals = [self.interval_of(self.variables[other]) for other in others]
        lower = np.array([interval.lower for interval in intervals])
        upper = np.array([interval.upper for interval in intervals])

        covariance = self.untruncated.covariance.matrix
        gain = covariance[others, index] / covariance[index, index]
        conditional_means = self.untruncated.mean[others] + np.outer(
            values - self.untruncated.mean[index], gain
        )
        conditional_covariance = covariance[np.ix_(others, others)] - np.outer(
            gain, covariance[index, others]
        )
        if len(others) == 1:
            deviation = math.sqrt(conditional_covariance[0, 0])
            return ndtr((upper - conditional_means[:, 0]) / deviation) - ndtr(
                (lower - conditional_means[:, 0]) / deviation
            )
        probability = multivariate_normal(
            mean=np.zeros(len(others)), cov=conditional_covariance
        ).cdf(upper - conditional_means, lower_limit=lower - conditional_means)
        return np.clip(np.atleast_1d(probability), 0.0, 1.0)

    # %% sampling

    def sample(self, amount: int) -> npt.NDArray:
        """
        Draw by Gibbs sampling: every sweep draws each variable from
        :mod:`scipy.stats.truncnorm`, the Gaussian of that variable given the current
        values of the others, confined to the variable's interval. Each sample ends a
        chain of its own that starts at the mode, so the samples are independent of each
        other.

        .. note::
            The samples follow this distribution only approximately, closer the more
            sweeps each chain makes. Variables that do not co-vary do not depend on each
            other, so for them the first sweep is already exact.

        :param amount: How many samples to draw.
        :return: That many samples, all of them inside the box.
        """
        intervals = [self.interval_of(variable) for variable in self.variables]
        lower = np.array([interval.lower for interval in intervals])
        upper = np.array([interval.upper for interval in intervals])
        mean = self.untruncated.mean
        precision = self.untruncated.precision
        conditional_deviation = 1 / np.sqrt(np.diag(precision))

        samples = np.tile(self._most_likely_point(precision), (amount, 1))
        for _ in range(self.burn_in_period_length):
            for index in range(len(self.variables)):
                conditional_mean = (
                    mean[index]
                    - (
                        (samples - mean) @ precision[index]
                        - precision[index, index] * (samples[:, index] - mean[index])
                    )
                    / precision[index, index]
                )
                samples[:, index] = truncnorm.rvs(
                    a=(lower[index] - conditional_mean) / conditional_deviation[index],
                    b=(upper[index] - conditional_mean) / conditional_deviation[index],
                    loc=conditional_mean,
                    scale=conditional_deviation[index],
                    size=amount,
                )
        return samples

    def __copy__(self) -> Self:
        return dataclasses.replace(self, untruncated=copy.copy(self.untruncated))

    def __deepcopy__(self, memo=None) -> Self:
        if memo is None:
            memo = {}
        id_self = id(self)
        if id_self in memo:
            return memo[id_self]
        result = dataclasses.replace(
            self,
            untruncated=copy.deepcopy(self.untruncated, memo),
            box=self.box.__deepcopy__(),
        )
        memo[id_self] = result
        return result
