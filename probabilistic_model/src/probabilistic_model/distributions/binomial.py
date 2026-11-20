"""
The binomial distribution of an integer variable.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy.typing as npt
from random_events.interval import Interval, closed, singleton
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
        :return: The probability of every one of the :attr:`allowed_values` that the
            distribution can take.
        """
        allowed_values = self.allowed_values
        probabilities = {
            self.offset + successes: math.comb(self.number_of_trials, successes)
            * self.success_probability**successes
            * (1 - self.success_probability) ** (self.number_of_trials - successes)
            for successes in range(self.number_of_trials + 1)
            if self.offset + successes in allowed_values
        }
        total = sum(probabilities.values())
        return MissingDict(
            float,
            {
                value: probability / total
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

    @property
    def allowed_values(self) -> Interval:
        """
        :return: The values the distribution may take, those from :attr:`offset` to
            :attr:`offset` plus :attr:`number_of_trials`.
        """
        return closed(self.offset, self.offset + self.number_of_trials)

    def log_conditional_of_composite_set(
        self, event: AbstractCompositeSet
    ) -> LogTruncation[TruncatedBinomialDistribution]:
        """
        Like :meth:`DiscreteDistribution.log_conditional_of_composite_set`, but the
        result is the binomial distribution truncated to its allowed values in the
        event.
        """
        probability = sum(
            probability
            for value, probability in self.probabilities.items()
            if value in event
        )
        if probability == 0:
            return LogTruncation.impossible()
        return LogTruncation(
            TruncatedBinomialDistribution(
                variable=self.variable,
                number_of_trials=self.number_of_trials,
                success_probability=self.success_probability,
                offset=self.offset,
                truncation_interval=self.allowed_values & event,
            ),
            math.log(probability),
        )

    def apply_translation(self, translation: dict[Variable, int]):
        """
        Like :meth:`IntegerDistribution.apply_translation`, by moving the
        :attr:`offset`.
        """
        self.offset += int(translation[self.variable])
        self.probabilities = self.probabilities_from_parameters()

    def apply_scaling(self, scaling: dict[Variable, int]):
        """
        Unlike :meth:`IntegerDistribution.apply_scaling`, a binomial distribution cannot
        be scaled in place, since the scaled one is no binomial distribution.

        :raises UndefinedOperationError: Always.
        """
        raise UndefinedOperationError(self)

    def fit(self, data: npt.NDArray) -> Self:
        """
        Unlike :meth:`IntegerDistribution.fit`, the frequencies of the data are no
        binomial distribution.

        :raises UndefinedOperationError: Always.
        """
        raise UndefinedOperationError(self)


# %% truncated binomial distribution


@dataclass(kw_only=True)
class TruncatedBinomialDistribution(BinomialDistribution):
    """
    A binomial distribution restricted to some of its values and renormalized, as
    truncating a :class:`BinomialDistribution` to an event gives it.
    """

    truncation_interval: Interval
    """
    The values the distribution is restricted to.
    """

    @property
    def allowed_values(self) -> Interval:
        """
        Like :attr:`BinomialDistribution.allowed_values`, only those in the
        :attr:`truncation_interval`.
        """
        return super().allowed_values & self.truncation_interval

    def apply_translation(self, translation: dict[Variable, int]):
        """
        Like :meth:`BinomialDistribution.apply_translation`, moving the
        :attr:`truncation_interval` with the :attr:`offset`.
        """
        shift = int(translation[self.variable])
        truncation_interval = Interval()
        for value in self.probabilities:
            truncation_interval |= singleton(value + shift)
        self.truncation_interval = truncation_interval
        super().apply_translation(translation)
