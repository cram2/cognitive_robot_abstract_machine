from __future__ import annotations

import itertools
import math
from dataclasses import dataclass

import numpy as np
import numpy.typing as npt
from random_events.interval import singleton
from random_events.product_algebra import Event, SimpleEvent, VariableMap
from random_events.variable import Continuous, Variable
from scipy.stats import multivariate_normal, norm
from scipy.stats._multivariate import multivariate_normal_frozen
from typing_extensions import (
    Any,
    Dict,
    Iterable,
    Iterator,
    List,
    Optional,
    Self,
    Tuple,
)

from probabilistic_model.distributions.truncated_multivariate_gaussian import (
    TruncatedMultivariateGaussianDistribution,
)
from probabilistic_model.exceptions import (
    EventIsNotABoxError,
    ProbabilisticCircuitRequiredError,
    ShapeMismatchError,
    VariableNotInDistributionError,
)
from probabilistic_model.probabilistic_model import (
    CenterType,
    MomentType,
    OrderType,
    ProbabilisticModel,
)

# %% the covariance matrix


@dataclass
class Covariance:
    """
    A covariance matrix.

    A covariance matrix is symmetric, so only the entries on and below its diagonal are
    stored. Every row and column is addressed by its index.
    """

    lower_triangle: npt.NDArray
    """
    The entries of the covariance matrix on and below its diagonal, row by row.

    Its shape is ``(n * (n + 1) // 2,)`` for an ``n`` by ``n`` matrix.
    """

    def __post_init__(self):
        self.lower_triangle = np.asarray(self.lower_triangle, dtype=float)
        self.validate()

    def validate(self):
        """
        :raises ShapeMismatchError: If the entries are not the lower triangle of a
            square matrix.
        """
        entries = self.dimension * (self.dimension + 1) // 2
        if self.lower_triangle.shape != (entries,):
            raise ShapeMismatchError(self.lower_triangle.shape, (entries,))

    @classmethod
    def from_matrix(cls, matrix: npt.ArrayLike) -> Self:
        """
        :param matrix: A covariance matrix, of which only the entries on and below the
            diagonal are read.
        :return: The covariance.
        :raises ShapeMismatchError: If the matrix is not square.
        """
        matrix = np.atleast_2d(np.asarray(matrix, dtype=float))
        dimension = len(matrix)
        if matrix.shape != (dimension, dimension):
            raise ShapeMismatchError(matrix.shape, (dimension, dimension))
        return cls(lower_triangle=matrix[np.tril_indices(dimension)])

    @property
    def dimension(self) -> int:
        """
        :return: How many rows, and columns, the matrix has.
        """
        return int((math.isqrt(8 * len(self.lower_triangle) + 1) - 1) // 2)

    @property
    def matrix(self) -> npt.NDArray:
        """
        :return: The full, symmetric matrix.
        """
        rows, columns = np.tril_indices(self.dimension)
        matrix = np.empty((self.dimension, self.dimension))
        matrix[rows, columns] = self.lower_triangle
        matrix[columns, rows] = self.lower_triangle
        return matrix

    @property
    def variances(self) -> npt.NDArray:
        """
        :return: The diagonal of the matrix.
        """
        indices = np.arange(self.dimension)
        return self.lower_triangle[indices * (indices + 1) // 2 + indices]

    def between(self, first: int, second: int) -> float:
        """
        :param first: The index of a row.
        :param second: The index of a column.
        :return: The entry at that row and column.
        """
        row, column = max(first, second), min(first, second)
        return float(self.lower_triangle[row * (row + 1) // 2 + column])

    def marginal(self, indices: List[int]) -> Self:
        """
        :param indices: The rows and columns to keep, in the order to keep them in.
        :return: The covariance of only those.
        """
        return self.from_matrix(self.matrix[np.ix_(indices, indices)])

    def scaled(self, factors: npt.NDArray) -> Self:
        """
        :param factors: What to multiply each index by.
        :return: The covariance after every index is multiplied by its factor, which
            scales each entry once per index it relates.
        """
        rows, columns = np.tril_indices(self.dimension)
        return type(self)(
            lower_triangle=self.lower_triangle * factors[rows] * factors[columns]
        )


# %% a Gaussian over several variables at once


@dataclass
class MultivariateGaussianDistribution(ProbabilisticModel):
    """
    A multivariate Gaussian distribution over continuous random variables.

    Every tractable query is answered by :mod:`scipy.stats.multivariate_normal`, the
    same way :class:`~probabilistic_model.distributions.gaussian.GaussianDistribution`
    answers its own through :mod:`scipy.stats.norm`.
    """

    variables: Tuple[Continuous, ...]
    """
    The variables of the distribution.
    """

    mean: npt.NDArray
    """
    The mean.

    Its shape is ``(n,)`` for ``n`` variables, laid out in the order of the variables.
    """

    covariance: Covariance
    """
    The covariance.

    Its matrix has the shape ``(n, n)`` for ``n`` variables, with rows and columns laid
    out in the order of the variables.
    """

    def __post_init__(self):
        self.variables = tuple(self.variables)
        self.mean = np.asarray(self.mean, dtype=float)
        self.validate()

    def validate(self):
        """
        :raises ShapeMismatchError: If the mean or the covariance is not laid out by
            this distribution's variables.
        """
        amount = len(self.variables)
        if self.mean.shape != (amount,):
            raise ShapeMismatchError(self.mean.shape, (amount,))
        if self.covariance.dimension != amount:
            raise ShapeMismatchError(
                (self.covariance.dimension, self.covariance.dimension),
                (amount, amount),
            )

    @property
    def precision(self) -> npt.NDArray:
        """
        :return: The inverse of the covariance matrix, with rows and columns laid out
            in the order of the variables.
        """
        return np.linalg.inv(self.covariance.matrix)

    @property
    def support(self) -> Event:
        return self.universal_simple_event().as_composite_set()

    @property
    def scipy_distribution(self) -> multivariate_normal_frozen:
        """
        :return: The scipy distribution every tractable query is answered by.
        """
        return multivariate_normal(mean=self.mean, cov=self.covariance.matrix)

    # %% reading it by variable

    def index_of(self, variable: Continuous) -> int:
        """
        :param variable: The variable to locate.
        :return: Its index in the mean and in the covariance.
        :raises VariableNotInDistributionError: If this distribution is not over it.
        """
        try:
            return self.variables.index(variable)
        except ValueError as error:
            raise VariableNotInDistributionError(
                variable=variable, variables=list(self.variables)
            ) from error

    def covariance_between(self, first: Continuous, second: Continuous) -> float:
        """
        The covariance of two variables is an integral over both of them, and this
        distribution is one of the few for which it is tractable in closed form.

        :param first: One of the variables.
        :param second: The other one.
        :return: Their covariance.
        :raises VariableNotInDistributionError: If this distribution is not over either
            of them.
        """
        return self.covariance.between(self.index_of(first), self.index_of(second))

    # %% density and probability

    def log_likelihood(self, events: npt.NDArray) -> npt.NDArray:
        return np.atleast_1d(self.scipy_distribution.logpdf(events))

    def cumulative_distribution_function(self, events: npt.NDArray) -> npt.NDArray:
        return np.atleast_1d(self.scipy_distribution.cdf(events))

    def probability_of_simple_event(self, event: SimpleEvent) -> float:
        """
        The probability of an axis-aligned box under a correlated Gaussian has no closed
        form, so it is integrated numerically. A variable confined to several intervals
        makes several boxes, and their probabilities add.

        :param event: The box, or boxes, to measure.
        :return: How probable it is.
        """
        return float(
            sum(
                max(float(self.scipy_distribution.cdf(upper, lower_limit=lower)), 0.0)
                for lower, upper in self._bounds_of_boxes(event)
            )
        )

    def _bounds_of_boxes(
        self, event: SimpleEvent
    ) -> Iterator[Tuple[npt.NDArray, npt.NDArray]]:
        """
        :param event: The event to split into boxes.
        :return: The lower and the upper bounds of each box the event makes, laid out
            by this distribution's variables.
        """
        intervals_per_variable = [
            tuple(event[variable].simple_sets) for variable in self.variables
        ]
        for box in itertools.product(*intervals_per_variable):
            yield (
                np.array([interval.lower for interval in box]),
                np.array([interval.upper for interval in box]),
            )

    def log_mode(self) -> Tuple[Event, float]:
        """
        A Gaussian is most dense exactly at its mean.

        :return: The mean, and the log-density there.
        """
        mode = SimpleEvent.from_data(
            {
                variable: singleton(float(value))
                for variable, value in zip(self.variables, self.mean)
            }
        ).as_composite_set()
        return mode, float(self.log_likelihood(self.mean.reshape(1, -1))[0])

    # %% reading fewer variables

    def marginal(self, variables: Iterable[Variable]) -> Optional[Self]:
        """
        :param variables: The variables to keep. They are kept in this distribution's
            own order, whatever order they are asked for in, and those it is not over
            are ignored.
        :return: The Gaussian over only those variables, or nothing if none of them is
            one of this distribution's.
        """
        kept = set(variables)
        indices = [
            index for index, variable in enumerate(self.variables) if variable in kept
        ]
        if not indices:
            return None
        return self._marginal_over_variable_indices(indices)

    def _marginal_over_variable_indices(self, indices: List[int]) -> Self:
        """
        :param indices: The indices of the variables to keep, in this distribution's own
            order.
        :return: The Gaussian over those variables.
        """
        return type(self)(
            variables=tuple(self.variables[index] for index in indices),
            mean=self.mean[indices],
            covariance=self.covariance.marginal(indices),
        )

    # %% fixing variables at a value

    def log_conditional(
        self, point: Dict[Variable, Any]
    ) -> Tuple[Optional[ProbabilisticModel], float]:
        """
        Fix some of the variables at the values given and answer with the distribution
        over the rest, which stays Gaussian.

        :param point: What each fixed variable is known to be.
        :return: The distribution over whatever is left, and the log-density of the
            values given.
        :raises VariableNotInDistributionError: If a fixed variable is not one of this
            distribution's.
        :raises ProbabilisticCircuitRequiredError: If every variable is fixed, which
            leaves a product of Dirac impulses rather than a Gaussian.
        """
        fixed_indices = [self.index_of(variable) for variable in point]
        free_indices = [
            index
            for index in range(len(self.variables))
            if index not in set(fixed_indices)
        ]
        if not free_indices:
            raise ProbabilisticCircuitRequiredError(model=self)

        fixed_at = np.array([float(point[variable]) for variable in point])
        log_density = float(
            self._marginal_over_variable_indices(fixed_indices).log_likelihood(
                fixed_at.reshape(1, -1)
            )[0]
        )

        return (
            self._conditional_over_variable_indices(
                free_indices, fixed_indices, fixed_at
            ),
            log_density,
        )

    def _conditional_over_variable_indices(
        self,
        free_indices: List[int],
        fixed_indices: List[int],
        fixed_at: npt.NDArray,
    ) -> Self:
        """
        :param free_indices: The indices of the variables the answer is over, in this
            distribution's order.
        :param fixed_indices: The indices of the variables held at a value, in the order
            ``fixed_at`` uses.
        :param fixed_at: What those variables are held at.
        :return: The Gaussian over the free variables, conditioned on the fixed ones.
        """
        free = self._marginal_over_variable_indices(free_indices)
        matrix = self.covariance.matrix
        cross = matrix[np.ix_(free_indices, fixed_indices)]
        explained = cross @ np.linalg.inv(matrix[np.ix_(fixed_indices, fixed_indices)])
        return type(self)(
            variables=free.variables,
            mean=free.mean + explained @ (fixed_at - self.mean[fixed_indices]),
            covariance=Covariance.from_matrix(
                free.covariance.matrix - explained @ cross.T
            ),
        )

    def product_with_gaussian_likelihood(
        self, other: MultivariateGaussianDistribution
    ) -> Self:
        """
        Multiply this density by the density of another Gaussian over some of the same
        variables, and normalize the product.

        The product of two Gaussian densities is again a Gaussian density, over the
        variables of this distribution.

        :param other: The Gaussian to multiply by. Its variables must all be this
            distribution's.
        :return: The normalized product.
        :raises VariableNotInDistributionError: If ``other`` is over a variable this
            distribution is not over.
        """
        indices = [self.index_of(variable) for variable in other.variables]
        matrix = self.covariance.matrix
        cross = matrix[:, indices]
        gain = cross @ np.linalg.inv(
            matrix[np.ix_(indices, indices)] + other.covariance.matrix
        )
        return type(self)(
            variables=self.variables,
            mean=self.mean + gain @ (other.mean - self.mean[indices]),
            covariance=Covariance.from_matrix(matrix - gain @ cross.T),
        )

    # %% confining it to an event

    def log_truncated(
        self, event: Event, singleton_allowed: bool = False
    ) -> Tuple[Optional[ProbabilisticModel], float]:
        """
        Confine this Gaussian to a box.

        A Gaussian confined to anything other than the whole space is no longer a
        Gaussian, so this answers with a
        :class:`TruncatedMultivariateGaussianDistribution` rather than with one of its
        own kind.

        :param event: The box to confine it to.
        :param singleton_allowed: Ignored, since a Gaussian gives every single point
            probability zero.
        :return: The confined distribution and the log-probability of the box, or
            nothing at all if it cannot happen.
        :raises EventIsNotABoxError: If the event is not one simple interval per
            variable.
        """
        event.fill_missing_variables(set(self.variables))
        if not event.is_box():
            raise EventIsNotABoxError(model=self, event=event)
        [box] = event.simple_sets
        probability = self.probability_of_simple_event(box)
        if probability == 0.0:
            return None, -np.inf
        return (
            TruncatedMultivariateGaussianDistribution(untruncated=self, box=box),
            math.log(probability),
        )

    # %% moments

    def moment(self, order: OrderType, center: CenterType) -> MomentType:
        """
        Every moment asked for here is of one variable on its own, so each is answered
        by that variable's own marginal, through :mod:`scipy.stats.norm`.

        :param order: The order of the moment of each variable to answer for.
        :param center: What to take each of those moments about.
        :return: The moment of each variable asked for.
        """
        moments = VariableMap()
        for variable in order:
            index = self.index_of(variable)
            moments[variable] = float(
                norm(
                    loc=self.mean[index] - center[variable],
                    scale=math.sqrt(self.covariance.between(index, index)),
                ).moment(order[variable])
            )
        return moments

    # %% translation and scaling

    def apply_translation(self, translation: Dict[Variable, float]):
        """
        Move the mean, leaving the covariance as it is.

        :param translation: How far to move each variable; one left out does not move,
            and one this distribution is not over is ignored.
        """
        for index, variable in enumerate(self.variables):
            self.mean[index] += translation.get(variable, 0.0)

    def apply_scaling(self, scaling: Dict[Variable, float]):
        """
        Scale the variables, which scales the mean once and the covariance once per
        variable it relates.

        :param scaling: What to multiply each variable by; one left out keeps its size,
            and one this distribution is not over is ignored.
        """
        factors = np.array(
            [scaling.get(variable, 1.0) for variable in self.variables], dtype=float
        )
        self.mean = self.mean * factors
        self.covariance = self.covariance.scaled(factors)

    # %% sampling

    def sample(self, amount: int) -> npt.NDArray:
        return self.scipy_distribution.rvs(size=amount).reshape(
            amount, len(self.variables)
        )

    def __copy__(self) -> Self:
        return type(self)(
            variables=self.variables,
            mean=self.mean.copy(),
            covariance=Covariance(lower_triangle=self.covariance.lower_triangle.copy()),
        )

    def __deepcopy__(self, memo=None) -> Self:
        if memo is None:
            memo = {}
        id_self = id(self)
        if id_self in memo:
            return memo[id_self]
        result = type(self)(
            variables=tuple(
                variable.__class__(name=variable.name, domain=variable.domain)
                for variable in self.variables
            ),
            mean=self.mean.copy(),
            covariance=Covariance(lower_triangle=self.covariance.lower_triangle.copy()),
        )
        memo[id_self] = result
        return result
