"""
The binomial distribution of an integer variable.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from random_events.variable import Integer
from typing_extensions import Self

from probabilistic_model.distributions.distributions import IntegerDistribution
from probabilistic_model.utils import MissingDict

# %% binomial distribution


@dataclass(kw_only=True)
class BinomialDistribution(IntegerDistribution):
    """
    An integer distribution whose probabilities are those of the number of successes in
    independent trials of equal success probability, shifted by an offset.

    Its probabilities are all it keeps, so truncating, translating or copying it acts on
    them as on any integer distribution.
    """

    @classmethod
    def from_trials(
        cls,
        *,
        variable: Integer,
        number_of_trials: int,
        success_probability: float,
        offset: int = 0,
    ) -> Self:
        """
        :param variable: The variable the distribution is over.
        :param number_of_trials: How many trials are counted.
        :param success_probability: The probability of each trial to succeed.
        :param offset: The value of no success.
        :return: The distribution of ``offset`` plus the number of successes.
        """
        return cls(
            variable=variable,
            probabilities=MissingDict(
                float,
                {
                    offset + successes: math.comb(number_of_trials, successes)
                    * success_probability**successes
                    * (1 - success_probability) ** (number_of_trials - successes)
                    for successes in range(number_of_trials + 1)
                },
            ),
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
        return cls.from_trials(
            variable=variable,
            number_of_trials=number_of_trials,
            success_probability=0.5,
            offset=location - number_of_trials // 2,
        )
