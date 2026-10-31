from __future__ import annotations

import numpy as np
import pytest
from random_events.interval import Bound, SimpleInterval
from random_events.variable import Continuous

from probabilistic_model.distributions.gaussian import GaussianDistribution
from probabilistic_model.distributions.uniform import UniformDistribution
from probabilistic_model.monte_carlo_estimator import MonteCarloEstimator

# %% fixtures


@pytest.fixture
def variable() -> Continuous:
    return Continuous("x")


@pytest.fixture(autouse=True)
def seeded_random_state() -> None:
    np.random.seed(69)


# %% total variation distance


def test_total_variation_distance_of_nearly_identical_models_is_accurate(
    variable: Continuous,
):
    model = GaussianDistribution(variable=variable, location=0.0, scale=1.0)
    other = GaussianDistribution(variable=variable, location=0.01, scale=1.0)

    # The densities intersect at the midpoint of the locations, so the event where the
    # model is more likely than the other is everything below it.
    midpoint = np.array([[(model.location + other.location) / 2]])
    exact_total_variation_distance = (
        model.cumulative_distribution_function(midpoint)[0]
        - other.cumulative_distribution_function(midpoint)[0]
    )

    estimate = MonteCarloEstimator(model, sample_size=1000).total_variation_distance(
        other
    )

    assert estimate == pytest.approx(exact_total_variation_distance, abs=1e-3)


def test_total_variation_distance_of_disjoint_models_is_one(variable: Continuous):
    model = UniformDistribution(
        variable=variable,
        interval=SimpleInterval.from_data(0, 1, Bound.CLOSED, Bound.OPEN),
    )
    other = UniformDistribution(
        variable=variable,
        interval=SimpleInterval.from_data(2, 3, Bound.CLOSED, Bound.OPEN),
    )

    assert MonteCarloEstimator(model).total_variation_distance(other) == 1


def test_total_variation_distance_of_identical_models_is_zero(variable: Continuous):
    model = GaussianDistribution(variable=variable, location=0.0, scale=1.0)

    assert MonteCarloEstimator(model).total_variation_distance(model) == 0
