"""
Tests of the binomial distribution of an integer variable.
"""

from __future__ import annotations

# %% imports
import copy
import math

import numpy as np
import pytest
from random_events.interval import closed
from random_events.product_algebra import SimpleEvent
from random_events.variable import Integer

from probabilistic_model.distributions.binomial import (
    BinomialDistribution,
    TruncatedBinomialDistribution,
)
from probabilistic_model.exceptions import (
    SuccessProbabilityOutsideUnitIntervalError,
    UndefinedOperationError,
)

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
    distribution = BinomialDistribution(
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
    distribution = BinomialDistribution(
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


def test_a_truncated_distribution_is_a_truncated_binomial(count):
    """
    Restricted to some of its values, the distribution keeps its parameters and the
    values it may take.
    """
    distribution = BinomialDistribution(
        variable=count, number_of_trials=4, success_probability=0.5, offset=1
    )
    truncated = distribution.truncated(
        SimpleEvent.from_data({count: closed(0, 2)}).as_composite_set()
    ).distribution
    assert isinstance(truncated, TruncatedBinomialDistribution)
    assert (
        truncated.number_of_trials,
        truncated.success_probability,
        truncated.offset,
    ) == (
        distribution.number_of_trials,
        distribution.success_probability,
        distribution.offset,
    )
    assert truncated.allowed_values == closed(1, 2)


def test_a_distribution_truncated_to_a_union_keeps_the_probabilities_of_both(count):
    """
    Truncated to two intervals, the distribution keeps the probabilities of the values
    in either, normalized.
    """
    distribution = BinomialDistribution(
        variable=count, number_of_trials=4, success_probability=0.3
    )
    truncation = distribution.truncated(
        SimpleEvent.from_data({count: closed(0, 0) | closed(3, 4)}).as_composite_set()
    )
    kept = {value: distribution.probabilities[value] for value in (0, 3, 4)}
    assert truncation.probability == pytest.approx(sum(kept.values()))
    assert dict(truncation.distribution.probabilities) == pytest.approx(
        {value: p / sum(kept.values()) for value, p in kept.items()}
    )


def test_truncating_twice_restricts_to_both_events(count):
    """
    A truncated distribution truncated again may take only the values of both events.
    """
    distribution = BinomialDistribution(
        variable=count, number_of_trials=5, success_probability=0.4
    )
    once = distribution.truncated(
        SimpleEvent.from_data({count: closed(0, 3)}).as_composite_set()
    ).distribution
    twice = once.truncated(
        SimpleEvent.from_data({count: closed(2, 5)}).as_composite_set()
    ).distribution
    assert twice.allowed_values == closed(2, 3)


def test_a_translated_truncated_distribution_moves_its_values(count):
    """
    Translated, a truncated distribution moves the values it may take with its offset.
    """
    distribution = TruncatedBinomialDistribution(
        variable=count,
        number_of_trials=4,
        success_probability=0.5,
        truncation_interval=closed(1, 2),
    )
    original = dict(distribution.probabilities)
    distribution.apply_translation({count: 3})
    assert dict(distribution.probabilities) == pytest.approx(
        {value + 3: p for value, p in original.items()}
    )


# %% the parameters


def test_a_success_probability_outside_of_zero_and_one_is_refused(count):
    """
    A success probability that is no probability is refused.
    """
    with pytest.raises(SuccessProbabilityOutsideUnitIntervalError):
        BinomialDistribution(
            variable=count, number_of_trials=3, success_probability=1.2
        )


def test_a_copy_keeps_the_parameters_and_probabilities(count):
    """
    A deep copy is the same binomial distribution.
    """
    distribution = BinomialDistribution(
        variable=count, number_of_trials=3, success_probability=0.2, offset=4
    )
    copied = copy.deepcopy(distribution)
    assert (copied.number_of_trials, copied.success_probability, copied.offset) == (
        distribution.number_of_trials,
        distribution.success_probability,
        distribution.offset,
    )
    assert dict(copied.probabilities) == dict(distribution.probabilities)


def test_a_translation_moves_the_offset_and_the_probabilities(count):
    """
    Translated, the distribution moves its offset and every probability with it.
    """
    distribution = BinomialDistribution(
        variable=count, number_of_trials=3, success_probability=0.2, offset=4
    )
    original = dict(distribution.probabilities)
    distribution.apply_translation({count: 2})
    assert distribution.offset == 6
    assert dict(distribution.probabilities) == {
        value + 2: probability for value, probability in original.items()
    }


def test_a_scaled_or_fitted_binomial_is_refused(count):
    """
    Scaling or fitting the frequencies of data would give no binomial distribution.
    """
    distribution = BinomialDistribution(
        variable=count, number_of_trials=3, success_probability=0.2
    )
    with pytest.raises(UndefinedOperationError):
        distribution.apply_scaling({count: 2})
    with pytest.raises(UndefinedOperationError):
        distribution.fit(np.array([[1], [2]]))
