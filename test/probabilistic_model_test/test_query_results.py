"""
Tests for the results the queries of a probabilistic model answer with.
"""

from __future__ import annotations

import math

import pytest
from random_events.interval import closed, singleton
from random_events.product_algebra import SimpleEvent
from random_events.variable import Continuous

from probabilistic_model.distributions.uniform import UniformDistribution
from probabilistic_model.query_results import (
    AbstractMode,
    AbstractPosterior,
    Conditioning,
    LogConditioning,
    LogMode,
    LogTruncation,
    Mode,
    Truncation,
)

# %% fixtures


@pytest.fixture
def x() -> Continuous:
    return Continuous("x")


@pytest.fixture
def uniform(x: Continuous) -> UniformDistribution:
    return UniformDistribution(variable=x, interval=closed(0.0, 2.0).simple_sets[0])


# %% switching from log space


def test_an_exponentiated_log_mode_keeps_its_event_and_has_the_likelihood(x):
    event = SimpleEvent.from_data({x: singleton(1.0)}).as_composite_set()
    mode = LogMode(event, math.log(0.25)).exponentiated()
    assert mode == Mode(event, pytest.approx(0.25))


def test_an_exponentiated_log_truncation_has_the_probability(uniform):
    truncation = LogTruncation(uniform, math.log(0.5)).exponentiated()
    assert truncation.distribution is uniform
    assert truncation.probability == pytest.approx(0.5)


def test_an_exponentiated_log_conditioning_has_the_likelihood(uniform):
    conditioning = LogConditioning(uniform, math.log(0.5)).exponentiated()
    assert conditioning.distribution is uniform
    assert conditioning.likelihood == pytest.approx(0.5)


# %% impossible queries


def test_an_impossible_truncation_has_no_distribution_and_no_probability():
    truncation = Truncation.impossible()
    assert truncation.distribution is None
    assert truncation.probability == 0.0


def test_an_impossible_log_truncation_has_no_distribution_and_no_log_probability():
    truncation = LogTruncation.impossible()
    assert truncation.distribution is None
    assert truncation.log_probability == -math.inf
    assert truncation.exponentiated() == Truncation.impossible()


def test_an_impossible_conditioning_has_no_distribution_and_no_likelihood():
    conditioning = Conditioning.impossible()
    assert conditioning.distribution is None
    assert conditioning.likelihood == 0.0


def test_an_impossible_log_conditioning_has_no_distribution_and_no_log_likelihood():
    conditioning = LogConditioning.impossible()
    assert conditioning.distribution is None
    assert conditioning.log_likelihood == -math.inf
    assert conditioning.exponentiated() == Conditioning.impossible()


# %% shared interfaces


@pytest.mark.parametrize("mode_type", [Mode, LogMode])
def test_every_mode_is_an_abstract_mode(mode_type):
    assert issubclass(mode_type, AbstractMode)


@pytest.mark.parametrize(
    "posterior_type", [Truncation, LogTruncation, Conditioning, LogConditioning]
)
def test_every_impossible_posterior_has_no_distribution(posterior_type):
    assert issubclass(posterior_type, AbstractPosterior)
    assert posterior_type.impossible().distribution is None


# %% the queries of a model


def test_truncating_a_model_answers_with_a_truncation(uniform, x):
    event = SimpleEvent.from_data({x: closed(0.0, 1.0)}).as_composite_set()
    truncation = uniform.truncated(event)
    assert isinstance(truncation, Truncation)
    assert truncation.probability == pytest.approx(0.5)
    assert truncation.distribution.interval == closed(0.0, 1.0).simple_sets[0]


def test_truncating_a_model_to_an_impossible_event_has_no_distribution(uniform, x):
    event = SimpleEvent.from_data({x: closed(3.0, 4.0)}).as_composite_set()
    truncation = uniform.truncated(event)
    assert truncation == Truncation.impossible()


def test_conditioning_a_model_answers_with_a_conditioning(uniform, x):
    conditioning = uniform.conditional({x: 1.0})
    assert isinstance(conditioning, Conditioning)
    assert conditioning.likelihood == pytest.approx(0.5)


def test_the_mode_of_a_model_is_a_mode(uniform, x):
    mode = uniform.mode()
    assert isinstance(mode, Mode)
    assert mode.event == SimpleEvent.from_data({x: closed(0.0, 2.0)}).as_composite_set()
    assert mode.likelihood == pytest.approx(0.5)
