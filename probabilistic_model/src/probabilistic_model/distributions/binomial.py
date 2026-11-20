"""
The binomial distribution of an integer variable.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy.typing as npt
from random_events.sigma_algebra import AbstractCompositeSet
from random_events.variable import Integer, Variable
from typing_extensions import Self

from probabilistic_model.distributions.distributions import IntegerDistribution
from probabilistic_model.exceptions import (
    SuccessProbabilityOutsideUnitIntervalError,
    UndefinedOperationError,
)
from probabilistic_model.query_results import LogTruncation
from probabilistic_model.utils import MissingDict

# %% binomial distribution


@dataclass(kw_only=True)
class BinomialDistribution(IntegerDistribution):
    """
    The distribution of :attr:`offset` plus the number of successes in
    :attr:`number_of_trials` independent trials that each succeed with
    :attr:`success_probability`.

    Its :attr:`probabilities` follow from these parameters and cannot be given.
    """

    number_of_trials: int
    """
    How many trials are counted.
    """

    success_probability: float
    """
    The probability of each trial to succeed.
    """

    offset: int = 0
    """
    The value of no success.
    """

    probabilities: MissingDict = field(
        init=False, default_factory=lambda: MissingDict(float)
    )
    """
    The probability of every value the distribution can take, see
    :meth:`probabilities_from_parameters`.
    """

    def __post_init__(self):
        """
        :raises SuccessProbabilityOutsideUnitIntervalError: If the success probability
            is not within [0, 1].
        """
        if not 0 <= self.success_probability <= 1:
            raise SuccessProbabilityOutsideUnitIntervalError(
                success_probabilities=[self.success_probability]
            )
        self.probabilities = self.probabilities_from_parameters()

    def probabilities_from_parameters(self) -> MissingDict:
        """
        :return: The probability of every value from :attr:`offset` to :attr:`offset`
            plus :attr:`number_of_trials` that the distribution can take.
        """
        probabilities = {
            self.offset + successes: math.comb(self.number_of_trials, successes)
            * self.success_probability**successes
            * (1 - self.success_probability) ** (self.number_of_trials - successes)
            for successes in range(self.number_of_trials + 1)
        }
        return MissingDict(
            float,
            {
                value: probability
                for value, probability in probabilities.items()
                if probability > 0
            },
        )

    @classmethod
    def from_location(
        cls, *, variable: Integer, location: int, standard_deviation: float
    ) -> Self:
        """
        :param variable: The variable the distribution is over.
        :param location: The value the distribution is symmetric around.
        :param standard_deviation: About how far the values spread around it.
        :return: The symmetric binomial distribution around the location whose standard
            deviation comes closest to the one given, a single value where that is
            below a half.
        """
        number_of_trials = 2 * round(2 * standard_deviation**2)
        return cls(
            variable=variable,
            number_of_trials=number_of_trials,
            success_probability=0.5,
            offset=location - number_of_trials // 2,
        )

    def to_integer_distribution(self) -> IntegerDistribution:
        """
        :return: An integer distribution with the probabilities of this one.
        """
        return IntegerDistribution(
            variable=self.variable,
            probabilities=MissingDict(float, self.probabilities),
        )

    def log_conditional_of_composite_set(
        self, event: AbstractCompositeSet
    ) -> LogTruncation[IntegerDistribution]:
        """
        Like :meth:`~probabilistic_model.distributions.distributions.DiscreteDistributio
        n.log_conditional_of_composite_set`, but the result is an integer distribution,
        since a binomial distribution restricted to some of its values is no binomial
        distribution.
        """
        return self.to_integer_distribution().log_conditional_of_composite_set(event)

    def apply_translation(self, translation: dict[Variable, int]):
        """
        Like :meth:`~probabilistic_model.distributions.distributions.IntegerDistribution
        .apply_translation`, by moving the :attr:`offset`.
        """
        self.offset += int(translation[self.variable])
        self.probabilities = self.probabilities_from_parameters()

    def apply_scaling(self, scaling: dict[Variable, int]):
        """
        Unlike :meth:`~probabilistic_model.distributions.distributions.IntegerDistributi
        on.apply_scaling`, a binomial distribution cannot be scaled in place, since the
        scaled one is no binomial distribution.

        :raises UndefinedOperationError: Always.
        """
        raise UndefinedOperationError(self)

    def fit(self, data: npt.NDArray) -> Self:
        """
        Unlike :meth:`~probabilistic_model.distributions.distributions.DiscreteDistribut
        ion.fit`, the frequencies of the data are no binomial distribution.

        :raises UndefinedOperationError: Always.
        """
        raise UndefinedOperationError(self)
