from __future__ import annotations

import json
import sys
from enum import IntEnum

import numpy as np
import plotly.graph_objects as go
import pytest
from random_events.interval import closed, open, open_closed, singleton
from random_events.product_algebra import SimpleEvent, VariableMap
from random_events.set import Set, SetElement
from random_events.variable import Integer, Symbolic

from krrood.adapters.json_serializer import from_json, to_json
from probabilistic_model.distributions.distributions import (
    DiracDeltaDistribution,
    IntegerDistribution,
    SymbolicDistribution,
    UnivariateDistribution,
)
from probabilistic_model.utils import (
    MissingDict,
    event_compatible_for_truncation_with_singletons,
)


class Letter(IntEnum):
    """
    The categories of a symbolic variable.
    """

    A = 0
    B = 1
    C = 2


# %% fixtures


@pytest.fixture
def integer_variable() -> Integer:
    """
    An integer variable.
    """
    return Integer("x")


@pytest.fixture
def letter_variable() -> Symbolic:
    """
    A symbolic variable over :class:`Letter`.
    """
    return Symbolic(name="x", domain=Set.from_iterable(Letter))


@pytest.fixture
def integer_distribution(integer_variable) -> IntegerDistribution:
    """
    A distribution over the integers 1, 2 and 4.
    """
    probabilities = MissingDict(float, {1: 4 / 20, 2: 5 / 20, 4: 11 / 20})
    return IntegerDistribution(variable=integer_variable, probabilities=probabilities)


@pytest.fixture
def letter_distribution(letter_variable) -> SymbolicDistribution:
    """
    A distribution over :class:`Letter` that never takes :attr:`Letter.C`.
    """
    probabilities = MissingDict(
        float, {hash(Letter.A): 7 / 20, hash(Letter.B): 13 / 20}
    )
    return SymbolicDistribution(variable=letter_variable, probabilities=probabilities)


@pytest.fixture
def dirac_delta(x) -> DiracDeltaDistribution:
    """
    A Dirac delta at zero with a density cap of two.
    """
    return DiracDeltaDistribution(variable=x, location=0, density_cap=2)


# %% every distribution


@pytest.mark.parametrize(
    "distribution_fixture",
    ["integer_distribution", "letter_distribution", "dirac_delta"],
)
def test_samples_lie_in_the_support(distribution_fixture, request):
    distribution: UnivariateDistribution = request.getfixturevalue(distribution_fixture)
    samples = distribution.sample(100)
    assert np.all(distribution.likelihood(samples) > 0)


@pytest.mark.parametrize(
    "distribution_fixture",
    ["integer_distribution", "letter_distribution", "dirac_delta"],
)
def test_plot_builds_a_figure(distribution_fixture, request):
    distribution: UnivariateDistribution = request.getfixturevalue(distribution_fixture)
    figure = go.Figure(distribution.plot(), distribution.plotly_layout())
    assert len(figure.data) > 0


@pytest.mark.parametrize(
    "distribution_fixture", ["integer_distribution", "dirac_delta"]
)
def test_serialization_round_trip(distribution_fixture, request):
    distribution: UnivariateDistribution = request.getfixturevalue(distribution_fixture)
    deserialized = from_json(to_json(distribution))
    assert isinstance(deserialized, type(distribution))
    assert deserialized == distribution


# %% integer distribution


def test_integer_likelihood(integer_distribution):
    likelihood = integer_distribution.likelihood(np.array([[1], [2], [3], [4]]))
    assert likelihood == pytest.approx([4 / 20, 5 / 20, 0, 11 / 20])


def test_integer_probability(integer_distribution, integer_variable):
    event = SimpleEvent.from_data({integer_variable: closed(1, 3)}).as_composite_set()
    assert integer_distribution.probability(event) == pytest.approx(9 / 20)


def test_integer_mode(integer_distribution, integer_variable):
    mode = integer_distribution.mode()
    assert mode.likelihood == pytest.approx(11 / 20)
    assert (
        mode.event
        == SimpleEvent.from_data({integer_variable: singleton(4)}).as_composite_set()
    )


def test_integer_truncation(integer_distribution, integer_variable):
    event = SimpleEvent.from_data(
        {integer_variable: closed(0, 1) | closed(3, 4)}
    ).as_composite_set()
    truncation = integer_distribution.truncated(event)
    assert truncation.probability == pytest.approx(15 / 20)
    assert truncation.distribution.probabilities[1] == pytest.approx(4 / 15)
    assert truncation.distribution.probabilities[4] == pytest.approx(11 / 15)


def test_integer_truncation_to_an_impossible_event(
    integer_distribution, integer_variable
):
    event = SimpleEvent.from_data({integer_variable: open(0, 1)}).as_composite_set()
    truncation = integer_distribution.truncated(event)
    assert truncation.distribution is None
    assert truncation.probability == 0


def test_integer_truncation_to_a_singleton(integer_variable):
    distribution = IntegerDistribution(
        variable=integer_variable, probabilities=MissingDict(float, {1: 0.3, 2: 0.7})
    )
    event = SimpleEvent.from_data({integer_variable: singleton(1)}).as_composite_set()
    truncation = distribution.truncated(event, singleton_allowed=True)
    assert isinstance(truncation.distribution, IntegerDistribution)
    assert truncation.distribution.probabilities[1] == 1.0
    assert truncation.probability == pytest.approx(0.3)


def test_integer_fit(integer_distribution):
    integer_distribution.fit([1, 2, 2, 2])
    assert integer_distribution.probabilities[1] == 1 / 4
    assert integer_distribution.probabilities[2] == 3 / 4


def test_integer_support(integer_distribution):
    assert integer_distribution.univariate_support == (
        singleton(1) | singleton(2) | singleton(4)
    )


def test_integer_support_without_probabilities_is_empty(integer_variable):
    distribution = IntegerDistribution(
        variable=integer_variable, probabilities=MissingDict(float)
    )
    assert distribution.univariate_support.is_empty()


def test_integer_cumulative_distribution_function(integer_distribution):
    cdf = integer_distribution.cumulative_distribution_function(
        np.array([[0], [1], [2], [3], [4]])
    )
    assert cdf == pytest.approx([0, 4 / 20, 9 / 20, 9 / 20, 1])


# %% symbolic distribution


def test_symbolic_category_unseen_before_serialization_has_no_probability_after_it(
    letter_distribution,
):
    """
    A distribution read back from JSON text keeps giving the values it never saw a
    probability of zero, as the distribution it was written from does.
    """
    deserialized = from_json(json.loads(json.dumps(to_json(letter_distribution))))
    assert deserialized.probabilities[hash(Letter.C)] == 0.0


def test_symbolic_mode(letter_distribution, letter_variable):
    mode = letter_distribution.mode()
    assert mode.likelihood == pytest.approx(13 / 20)
    assert (
        mode.event
        == SimpleEvent.from_data({letter_variable: Letter.B}).as_composite_set()
    )


def test_symbolic_probability(letter_distribution, letter_variable):
    event = SimpleEvent.from_data(
        {letter_variable: (Letter.A, Letter.C)}
    ).as_composite_set()
    assert letter_distribution.probability(event) == pytest.approx(7 / 20)


def test_symbolic_support(letter_distribution, letter_variable):
    all_elements = letter_variable.domain.simple_sets[0].all_elements
    assert letter_distribution.univariate_support == Set.from_simple_sets(
        SetElement.from_data(Letter.A, all_elements),
        SetElement.from_data(Letter.B, all_elements),
    )


def test_symbolic_fit(letter_distribution, letter_variable):
    letter_distribution.fit_from_indices([Letter.A, Letter.B, Letter.B, Letter.B])
    a = SimpleEvent.from_data({letter_variable: Letter.A}).as_composite_set()
    b = SimpleEvent.from_data({letter_variable: Letter.B}).as_composite_set()
    assert letter_distribution.probability(a) == 1 / 4
    assert letter_distribution.probability(b) == 3 / 4


def test_symbolic_truncation_to_a_singleton():
    variable = Symbolic("x", domain=Set.from_iterable(["a", "b"]))
    distribution = SymbolicDistribution(
        variable=variable,
        probabilities=MissingDict(float, {hash("a"): 0.4, hash("b"): 0.6}),
    )
    event = SimpleEvent.from_data({variable: "a"}).as_composite_set()
    truncation = distribution.truncated(event, singleton_allowed=True)
    assert isinstance(truncation.distribution, SymbolicDistribution)
    assert truncation.probability == pytest.approx(0.4)


def test_symbolic_likelihood_survives_a_category_whose_hash_exceeds_the_modulus():
    """
    A category still scores its true probability when its Python :func:`hash` falls
    outside :attr:`sys.hash_info.modulus`.
    """
    modulus = sys.hash_info.modulus
    category = next(
        candidate
        for candidate in (f"category_{index}" for index in range(1000))
        if abs(hash(candidate)) >= modulus
    )
    variable = Symbolic("category", domain=Set.from_iterable([category, "other"]))
    distribution = SymbolicDistribution(
        variable=variable, probabilities=MissingDict(float, {hash(category): 1.0})
    )
    log_likelihood = distribution.log_likelihood(np.array([[category]]))
    assert log_likelihood[0] == pytest.approx(0.0)


# %% Dirac delta distribution


def test_dirac_delta_likelihood(dirac_delta):
    assert list(dirac_delta.likelihood(np.array([[0], [1]]))) == [2, 0]


def test_dirac_delta_cumulative_distribution_function(dirac_delta):
    cdf = dirac_delta.cumulative_distribution_function(np.array([[-1], [0], [1]]))
    assert list(cdf) == [0, 1, 1]


def test_dirac_delta_probability_of_an_event_holding_the_location(dirac_delta, x):
    event = SimpleEvent.from_data({x: closed(0, 1) | closed(1.5, 2)}).as_composite_set()
    assert dirac_delta.probability(event) == 1


def test_dirac_delta_probability_of_an_event_beside_the_location(dirac_delta, x):
    event = SimpleEvent.from_data(
        {x: open_closed(dirac_delta.tolerance, 1)}
    ).as_composite_set()
    assert dirac_delta.probability(event) == 0.0


def test_dirac_delta_truncation_to_an_event_holding_the_location(dirac_delta, x):
    event = SimpleEvent.from_data({x: closed(-1, 2)}).as_composite_set()
    truncation = dirac_delta.truncated(event)
    assert truncation.distribution == dirac_delta
    assert truncation.probability == 1


def test_dirac_delta_truncation_to_an_impossible_event(dirac_delta, x):
    event = SimpleEvent.from_data({x: closed(1, 2)}).as_composite_set()
    truncation = dirac_delta.truncated(event)
    assert truncation.distribution is None
    assert truncation.probability == 0


def test_dirac_delta_truncation_to_its_own_support_keeps_the_distribution(x):
    """
    An interval holds its bounds with less precision than a location, so the support of
    a Dirac delta does not hold its own location exactly.

    Truncating to that support still keeps the distribution.
    """
    distribution = DiracDeltaDistribution(variable=x, location=3.427709925174713)
    support = SimpleEvent.from_data(
        {x: distribution.univariate_support}
    ).as_composite_set()
    assert distribution.univariate_support.simple_sets[0].lower != distribution.location
    assert distribution.probability(support) == 1.0
    truncation = distribution.truncated(support)
    assert truncation.distribution == distribution
    assert truncation.probability == 1.0


def test_dirac_delta_truncation_to_a_singleton(x):
    distribution = DiracDeltaDistribution(variable=x, location=1.0, density_cap=2.0)

    at_location = SimpleEvent.from_data({x: singleton(1.0)}).as_composite_set()
    truncation = distribution.truncated(at_location)
    assert truncation.distribution == distribution
    assert truncation.probability == pytest.approx(1.0)

    elsewhere = SimpleEvent.from_data({x: singleton(0.0)}).as_composite_set()
    truncation = distribution.truncated(elsewhere, singleton_allowed=True)
    assert truncation.distribution is None
    assert truncation.probability == 0.0


def test_dirac_delta_mode(dirac_delta):
    log_mode = dirac_delta.univariate_log_mode()
    assert log_mode.assignment == singleton(0)
    assert log_mode.log_likelihood == np.log(2)


def test_dirac_delta_samples_are_its_location(dirac_delta):
    assert np.all(dirac_delta.likelihood(dirac_delta.sample(100)) == 2)


def test_dirac_delta_moments(dirac_delta, x):
    expectation = dirac_delta.expectation([x])
    assert expectation[x] == 0
    assert dirac_delta.variance([x])[x] == 0
    assert dirac_delta.moment(VariableMap({x: 3}), expectation)[x] == 0


# %% events compatible with truncation to singletons


def test_an_event_of_singletons_is_compatible_with_truncation(x):
    event = SimpleEvent.from_data(
        {x: singleton(1.0) | singleton(2.0)}
    ).as_composite_set()
    assert event_compatible_for_truncation_with_singletons(event)


def test_an_event_mixing_singletons_and_intervals_is_incompatible_with_truncation(x):
    event = SimpleEvent.from_data(
        {x: singleton(3.0) | closed(0.0, 2.0)}
    ).as_composite_set()
    assert not event_compatible_for_truncation_with_singletons(event)


def test_symbolic_variables_do_not_affect_compatibility_with_truncation(x):
    y = Symbolic("y", domain=Set.from_iterable(["a", "b"]))
    event = SimpleEvent.from_data(
        {x: singleton(1.0) | singleton(2.0), y: ("a", "b")}
    ).as_composite_set()
    assert event_compatible_for_truncation_with_singletons(event)
