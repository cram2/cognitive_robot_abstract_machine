"""
Tests of the binomial distribution of an integer variable.
"""

from __future__ import annotations

# %% imports
import math

import pytest
from random_events.interval import closed
from random_events.product_algebra import SimpleEvent
from random_events.variable import Integer

from probabilistic_model.distributions.binomial import BinomialDistribution

# %% fixtures


@pytest.fixture
def count() -> Integer:
    """
    An integer variable.
    """
    return Integer(name="count")


# %% the probabilities of successes


def test_the_probabilities_are_those_of_the_numbers_of_successes(count):
    """
    Each value has the probability of its number of successes above the offset.
    """
    distribution = BinomialDistribution.from_trials(
        variable=count, number_of_trials=4, success_probability=0.25, offset=10
    )
    assert distribution.probabilities[12] == pytest.approx(
        math.comb(4, 2) * 0.25**2 * 0.75**2
    )
    assert sum(distribution.probabilities.values()) == pytest.approx(1.0)


def test_a_distribution_around_a_location_is_centred_on_it_and_spread_as_asked(count):
    """
    The symmetric binomial around a location has its mean there and the standard
    deviation asked for, where that is a whole number of trials.
    """
    distribution = BinomialDistribution.from_location(
        variable=count, location=7, standard_deviation=2.0
    )
    mean = sum(value * p for value, p in distribution.probabilities.items())
    variance = sum(
        (value - mean) ** 2 * p for value, p in distribution.probabilities.items()
    )
    assert mean == pytest.approx(7)
    assert math.sqrt(variance) == pytest.approx(2.0)


def test_a_distribution_narrower_than_a_half_is_a_single_value(count):
    """
    Spread by less than a half, the distribution puts all its probability on its
    location.
    """
    distribution = BinomialDistribution.from_location(
        variable=count, location=3, standard_deviation=0.4
    )
    assert dict(distribution.probabilities) == {3: 1.0}


def test_a_truncated_distribution_keeps_the_probabilities_within_the_event(count):
    """
    Truncated to some of its values, the distribution keeps their probabilities,
    normalized.
    """
    distribution = BinomialDistribution.from_trials(
        variable=count, number_of_trials=4, success_probability=0.5
    )
    truncated = distribution.truncated(
        SimpleEvent.from_data({count: closed(0, 1)}).as_composite_set()
    ).distribution
    total = distribution.probabilities[0] + distribution.probabilities[1]
    assert dict(truncated.probabilities) == pytest.approx(
        {
            0: distribution.probabilities[0] / total,
            1: distribution.probabilities[1] / total,
        }
    )
