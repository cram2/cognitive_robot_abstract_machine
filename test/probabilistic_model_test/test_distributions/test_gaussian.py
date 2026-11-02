from __future__ import annotations

import numpy as np
import plotly.graph_objects as go
import pytest
from scipy.stats import truncnorm
from random_events.interval import Bound, SimpleInterval, closed, reals, singleton
from random_events.product_algebra import SimpleEvent, VariableMap

from krrood.adapters.json_serializer import from_json, to_json
from probabilistic_model.distributions.distributions import DiracDeltaDistribution
from probabilistic_model.distributions.gaussian import (
    GaussianDistribution,
    TruncatedGaussianDistribution,
)

# %% fixtures


@pytest.fixture
def gaussian(x) -> GaussianDistribution:
    """
    A Gaussian with location two and scale two.
    """
    return GaussianDistribution(variable=x, location=2, scale=2)


@pytest.fixture
def truncated_gaussian(x) -> TruncatedGaussianDistribution:
    """
    A Gaussian with location two and scale two, truncated to [-2, 2].
    """
    return TruncatedGaussianDistribution(
        variable=x,
        interval=SimpleInterval.from_data(-2, 2, Bound.CLOSED, Bound.CLOSED),
        location=2,
        scale=2.0,
    )


@pytest.fixture
def standard_gaussian(x) -> GaussianDistribution:
    """
    The standard normal distribution.
    """
    return GaussianDistribution(variable=x, location=0, scale=1)


# %% every Gaussian


@pytest.mark.parametrize("distribution_fixture", ["gaussian", "truncated_gaussian"])
def test_samples_lie_in_the_support(distribution_fixture, request):
    distribution = request.getfixturevalue(distribution_fixture)
    samples = distribution.sample(100)
    assert samples.shape == (100, 1)
    likelihoods = distribution.likelihood(samples)
    assert likelihoods.shape == (100,)
    assert np.all(likelihoods > 0)


@pytest.mark.parametrize("distribution_fixture", ["gaussian", "truncated_gaussian"])
def test_serialization_round_trip(distribution_fixture, request):
    distribution = request.getfixturevalue(distribution_fixture)
    deserialized = from_json(to_json(distribution))
    assert isinstance(deserialized, type(distribution))
    assert deserialized == distribution


@pytest.mark.parametrize("distribution_fixture", ["gaussian", "truncated_gaussian"])
def test_plot_builds_a_figure(distribution_fixture, request):
    distribution = request.getfixturevalue(distribution_fixture)
    assert len(go.Figure(distribution.plot()).data) > 0


@pytest.mark.parametrize("distribution_fixture", ["gaussian", "truncated_gaussian"])
def test_truncation_to_an_interval_inside_the_support(distribution_fixture, request, x):
    distribution = request.getfixturevalue(distribution_fixture)
    event = SimpleEvent.from_data({x: closed(1, 2)}).as_composite_set()
    truncation = distribution.truncated(event)
    assert isinstance(truncation.distribution, TruncatedGaussianDistribution)
    cdf = distribution.cumulative_distribution_function(np.array([[1], [2]]))
    assert truncation.probability == pytest.approx(cdf[1] - cdf[0])
    assert truncation.distribution.lower == 1
    assert truncation.distribution.upper == 2


@pytest.mark.parametrize("distribution_fixture", ["gaussian", "truncated_gaussian"])
def test_mode_is_the_location(distribution_fixture, request):
    distribution = request.getfixturevalue(distribution_fixture)
    mode = distribution.univariate_log_mode().assignment
    assert mode == singleton(distribution.location)


# %% Gaussian


def test_gaussian_support_is_the_reals(gaussian):
    assert gaussian.univariate_support == reals()
    assert gaussian.probability(gaussian.support) == 1


def test_gaussian_raw_moments(gaussian):
    assert gaussian.raw_moment(0) == 1
    assert gaussian.raw_moment(1) == gaussian.location
    assert gaussian.raw_moment(2) == gaussian.location**2 + gaussian.scale**2


def test_gaussian_expectation_is_the_location(gaussian, x):
    expectation = gaussian.moment(VariableMap({x: 1}), VariableMap({x: 0}))
    assert expectation[x] == gaussian.location


@pytest.mark.parametrize("order, expected", [(2, 4), (3, 0), (4, 48)])
def test_gaussian_moment_around_the_expectation(gaussian, x, order, expected):
    expectation = VariableMap({x: gaussian.location})
    assert gaussian.moment(VariableMap({x: order}), expectation)[x] == expected


@pytest.mark.parametrize("order, expected", [(1, -0.5), (2, 4.25), (3, -6.125)])
def test_gaussian_moment_around_another_center(gaussian, x, order, expected):
    center = VariableMap({x: 2.5})
    assert gaussian.moment(VariableMap({x: order}), center)[x] == expected


def test_gaussian_variance(gaussian, x):
    assert gaussian.variance([x])[x] == gaussian.scale**2


def test_gaussian_truncation_to_a_singleton(standard_gaussian, x):
    event = SimpleEvent.from_data({x: singleton(0)}).as_composite_set()

    conditional = standard_gaussian.truncated(event).distribution
    assert conditional is None

    truncation = standard_gaussian.truncated(event, singleton_allowed=True)
    assert isinstance(truncation.distribution, DiracDeltaDistribution)
    assert truncation.distribution.location == 0.0
    assert truncation.probability == pytest.approx(
        np.exp(standard_gaussian.log_likelihood(np.array([[0]]))[0])
    )


# %% truncated Gaussian


def test_truncated_cumulative_distribution_function(truncated_gaussian):
    cdf = truncated_gaussian.cumulative_distribution_function(
        np.array([[0], [3], [-3]])
    )
    assert cdf[0] == pytest.approx(0.285, abs=1e-3)
    assert cdf[1] == 1
    assert cdf[2] == 0


def test_truncated_mode_likelihood(truncated_gaussian):
    log_likelihood = truncated_gaussian.univariate_log_mode().log_likelihood
    assert log_likelihood == truncated_gaussian.log_likelihood_without_bounds_check(
        np.array([[2]])
    )


def test_truncated_expectation(truncated_gaussian, x):
    expectation = truncated_gaussian.moment(VariableMap({x: 1}), VariableMap({x: 0}))
    assert expectation[x] == pytest.approx(0.5544205, abs=1e-7)


def test_truncated_first_moment_around_the_location(
    truncated_gaussian, standard_gaussian, x
):
    untruncated = GaussianDistribution(
        variable=x,
        location=truncated_gaussian.location,
        scale=truncated_gaussian.scale,
    )
    normalizing_constant = untruncated.probability(truncated_gaussian.support)
    alpha = (
        truncated_gaussian.lower - truncated_gaussian.location
    ) / truncated_gaussian.scale
    beta = (
        truncated_gaussian.upper - truncated_gaussian.location
    ) / truncated_gaussian.scale
    likelihood_alpha, likelihood_beta = standard_gaussian.likelihood(
        np.array([[alpha], [beta]])
    )
    expected = (
        -truncated_gaussian.scale
        * (likelihood_beta - likelihood_alpha)
        / normalizing_constant
    )

    center = VariableMap({x: truncated_gaussian.location})
    moment = truncated_gaussian.moment(VariableMap({x: 1}), center)
    assert moment[x] == pytest.approx(expected, abs=1e-7)


def test_truncated_moments_around_another_center(truncated_gaussian, x):
    center = VariableMap({x: 2.5})
    origin = VariableMap({x: 0})
    expectation = truncated_gaussian.moment(VariableMap({x: 1}), origin)[x]
    second_raw_moment = truncated_gaussian.moment(VariableMap({x: 2}), origin)[x]

    first_moment = truncated_gaussian.moment(VariableMap({x: 1}), center)[x]
    assert first_moment == pytest.approx(expectation - center[x], abs=1e-7)

    second_moment = truncated_gaussian.moment(VariableMap({x: 2}), center)[x]
    assert second_moment == pytest.approx(
        second_raw_moment + center[x] ** 2 - 2 * center[x] * expectation, abs=1e-7
    )


def test_truncated_truncation_to_an_interval_reaching_beyond_the_support(
    truncated_gaussian, x
):
    event = SimpleEvent.from_data({x: closed(-5, 1)}).as_composite_set()
    truncation = truncated_gaussian.truncated(event)
    assert truncation.distribution.interval == closed(-2, 1).simple_sets[0]
    points = np.linspace(-2, 1, 7).reshape(-1, 1)
    np.testing.assert_allclose(
        truncation.distribution.likelihood(points),
        truncated_gaussian.likelihood(points) / truncation.probability,
    )


def test_truncated_conditioning_on_the_mode(truncated_gaussian, x):
    mode = truncated_gaussian.mode().event
    conditional = truncated_gaussian.truncated(mode).distribution
    assert conditional is None

    location = mode.simple_sets[0][x].simple_sets[0].lower
    log_conditioning = truncated_gaussian.log_conditional({x: location})
    assert isinstance(log_conditioning.distribution, DiracDeltaDistribution)
    assert log_conditioning.log_likelihood > -np.inf


@pytest.mark.parametrize(
    "lower, upper, order, expected",
    [
        (0.5, np.inf, 1, 1.14),
        (0.5, np.inf, 2, 1.57),
        (0.5, np.inf, 3, 2.57),
        (-1, 1, 1, 0),
        (-1, 1, 2, 0.291),
        (-1, 1, 3, 0),
    ],
)
def test_truncated_standard_gaussian_raw_moments(
    standard_gaussian, x, lower, upper, order, expected
):
    truncated = standard_gaussian.truncated(
        SimpleEvent.from_data({x: closed(lower, upper)}).as_composite_set()
    ).distribution
    raw_moment = truncated.moment(VariableMap({x: order}), VariableMap({x: 0}))[x]
    assert raw_moment == pytest.approx(expected, abs=0.01)


def test_truncated_variance_is_precise_far_in_the_tail(x):
    distribution = TruncatedGaussianDistribution(
        variable=x, interval=SimpleInterval.from_data(5, 6), location=0, scale=1
    )
    # a standard normal, so the interval already is in scipy's standardized units
    lower, upper = distribution.lower, distribution.upper
    variance = distribution.moment(
        VariableMap({x: 2}), VariableMap({x: truncnorm.mean(lower, upper)})
    )[x]
    expected_variance = truncnorm.var(lower, upper)
    # far in the tail the variance is small compared to the squared mean, so this
    # tolerance fails any computation that cancels large raw moments
    assert variance == pytest.approx(expected_variance, abs=1e-10 * expected_variance)


@pytest.mark.parametrize(
    "lower, upper, scale",
    [
        (-1, np.inf, 1.0),
        (1, np.inf, 1.0),
        (-np.inf, -1, 1.0),
        (-0.62, 0.0, 0.5),
        (11, np.inf, 1.0),
    ],
)
def test_truncated_samples_average_to_the_expectation(x, lower, upper, scale):
    distribution = TruncatedGaussianDistribution(
        variable=x,
        interval=SimpleInterval.from_data(lower, upper),
        location=0,
        scale=scale,
    )
    samples = distribution.sample(1000)
    assert samples.shape == (1000, 1)
    assert np.all(distribution.likelihood(samples) > 0)
    assert samples.mean() == pytest.approx(distribution.expectation([x])[x], abs=0.1)
