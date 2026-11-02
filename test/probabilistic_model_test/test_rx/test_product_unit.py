from __future__ import annotations

import copy

import numpy as np
import pytest
from random_events.interval import closed
from random_events.product_algebra import Event, SimpleEvent
from sortedcontainers import SortedSet

from krrood.adapters.json_serializer import from_json, to_json
from probabilistic_model.probabilistic_circuit.rx.probabilistic_circuit import (
    ProbabilisticCircuit,
    ProductUnit,
)

from ..circuit_building import uniform_leaf

# %% fixtures


@pytest.fixture
def product(x, y) -> ProbabilisticCircuit:
    """
    A product of a uniform distribution over *x* in [0, 1] and one over *y* in [3, 4].
    """
    circuit = ProbabilisticCircuit()
    product_unit = ProductUnit(probabilistic_circuit=circuit)
    product_unit.add_subcircuit(uniform_leaf(x, 0, 1, circuit))
    product_unit.add_subcircuit(uniform_leaf(y, 3, 4, circuit))
    return circuit


@pytest.fixture
def lower_quarter(x, y) -> Event:
    """
    The lower half of both factors' supports.
    """
    return SimpleEvent.from_data(
        {x: closed(0, 0.5), y: closed(3, 3.5)}
    ).as_composite_set()


# %% structure


def test_structure(product, x, y):
    assert len(list(product.nodes())) == 3
    assert len(product.edges()) == 2
    assert len(product.leaves) == 2
    assert product.variables == SortedSet([x, y])


def test_support_is_the_product_of_the_supports(product, x, y):
    assert (
        product.support
        == SimpleEvent.from_data({x: closed(0, 1), y: closed(3, 4)}).as_composite_set()
    )


def test_is_deterministic(product):
    assert product.is_deterministic()


# %% inference


def test_likelihood(product):
    assert product.likelihood(np.array([[0.5, 3.5]])) == 1


def test_probability(product, lower_quarter):
    assert product.probability(lower_quarter) == 0.25


def test_mode(product, x, y):
    mode = product.mode()
    assert mode.likelihood == 1
    assert (
        mode.event
        == SimpleEvent.from_data({x: closed(0, 1), y: closed(3, 4)}).as_composite_set()
    )


def test_distinct_samples_lie_in_the_support(product):
    samples = product.sample(100)
    assert np.all(product.likelihood(samples) > 0)
    assert len(np.unique(samples, axis=0)) == len(samples)


def test_expectation(product, x, y):
    expectation = product.expectation(product.variables)
    assert expectation[x] == 0.5
    assert expectation[y] == 3.5


def test_truncation(product, x):
    event = SimpleEvent.from_data({x: closed(0, 0.5)}).as_composite_set()
    truncation = product.truncated(event)
    assert truncation.probability == 0.5
    assert len(list(truncation.distribution.nodes())) == 3
    assert isinstance(truncation.distribution.root, ProductUnit)


def test_truncation_to_an_impossible_event(product, x):
    event = SimpleEvent.from_data({x: closed(1.5, 2)}).as_composite_set()
    truncation = product.truncated(event)
    assert truncation.probability == 0
    assert truncation.distribution is None


def test_marginal_keeps_the_factor_of_the_variable(product, x):
    marginal = product.marginal([x])
    assert len(list(marginal.nodes())) == 1
    assert marginal.variables == SortedSet([x])


def test_marginal_without_variables_is_none(product):
    assert product.marginal([]) is None


# %% copies


def test_serialization_keeps_the_probability(product, lower_quarter):
    deserialized = from_json(to_json(product))
    assert deserialized.probability(lower_quarter) == product.probability(lower_quarter)


def test_deep_copy_keeps_the_probability(product, lower_quarter):
    copied = copy.deepcopy(product)
    assert copied.probability(lower_quarter) == product.probability(lower_quarter)
