from __future__ import annotations

import numpy as np
import plotly.graph_objects as go
import pytest
from random_events.interval import Bound, SimpleInterval, closed, singleton
from random_events.product_algebra import SimpleEvent, VariableMap

from probabilistic_model.distributions.distributions import DiracDeltaDistribution
from probabilistic_model.distributions.uniform import UniformDistribution

# %% fixtures


@pytest.fixture
def uniform(x) -> UniformDistribution:
    """
    A uniform distribution over [0, 2).
    """
    return UniformDistribution(
        variable=x, interval=SimpleInterval.from_data(0, 2, Bound.CLOSED, Bound.OPEN)
    )


# %% density and probability


def test_support_is_the_interval(uniform):
    assert uniform.univariate_support == uniform.interval.as_composite_set()
    assert uniform.probability(uniform.support) == 1


def test_likelihood(uniform):
    assert list(uniform.likelihood(np.array([[-1], [1], [2]]))) == [0, 0.5, 0]


def test_cumulative_distribution_function(uniform):
    cdf = uniform.cumulative_distribution_function(np.array([[-1], [1], [2]]))
    assert list(cdf) == [0, 0.5, 1]


def test_probability(uniform, x):
    event = SimpleEvent.from_data({x: closed(0, 1) | closed(1.5, 2)}).as_composite_set()
    assert uniform.probability(event) == 0.75


def test_mode_is_the_support(uniform):
    mode = uniform.mode()
    assert mode.event == uniform.support
    assert mode.likelihood == 0.5


def test_samples_have_the_density(uniform):
    samples = uniform.sample(100)
    assert len(samples) == 100
    assert np.all(uniform.likelihood(samples) == 0.5)


def test_moments(uniform, x):
    expectation = uniform.moment(VariableMap({x: 1}), VariableMap({x: 0}))
    assert expectation[x] == 1
    assert uniform.moment(VariableMap({x: 2}), expectation)[x] == 1 / 3


def test_plot_builds_a_figure(uniform):
    assert len(go.Figure(data=uniform.plot()).data) > 0


# %% truncation


def test_truncation_to_a_disjoint_event(uniform, x):
    event = SimpleEvent.from_data({x: closed(3, 4)}).as_composite_set()
    truncation = uniform.truncated(event)
    assert truncation.distribution is None
    assert truncation.probability == 0


def test_truncation_to_an_interval(uniform, x):
    event = SimpleEvent.from_data({x: closed(1, 2)}).as_composite_set()
    truncation = uniform.truncated(event)
    assert truncation.distribution == UniformDistribution(
        variable=x, interval=SimpleInterval.from_data(1, 2, Bound.CLOSED, Bound.OPEN)
    )
    assert truncation.probability == 0.5


def test_truncation_to_a_singleton(uniform, x):
    event = SimpleEvent.from_data({x: singleton(1)}).as_composite_set()

    truncation = uniform.truncated(event)
    assert truncation.distribution is None
    assert truncation.probability == 0.0

    truncation = uniform.truncated(event, singleton_allowed=True)
    assert isinstance(truncation.distribution, DiracDeltaDistribution)
    assert truncation.distribution.location == 1.0
    assert truncation.probability == pytest.approx(0.5)


def test_conditioning_on_a_point(uniform, x):
    conditioning = uniform.conditional({x: 1.0})
    assert isinstance(conditioning.distribution, DiracDeltaDistribution)
    assert conditioning.likelihood == 0.5


# %% translation


def test_translation_shifts_the_interval(x):
    distribution = UniformDistribution(
        variable=x, interval=SimpleInterval.from_data(0, 1)
    )
    distribution.apply_translation({x: 2.0})
    assert distribution == UniformDistribution(
        variable=x, interval=SimpleInterval.from_data(2, 3)
    )
