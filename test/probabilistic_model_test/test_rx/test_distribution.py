from __future__ import annotations

from enum import IntEnum

import numpy as np
import pytest
from random_events.interval import closed, singleton
from random_events.product_algebra import SimpleEvent
from random_events.set import Set
from random_events.variable import Integer, Symbolic

from probabilistic_model.distributions.distributions import (
    DiracDeltaDistribution,
    IntegerDistribution,
    SymbolicDistribution,
)
from probabilistic_model.probabilistic_circuit.rx.probabilistic_circuit import (
    ProbabilisticCircuit,
    SumUnit,
    UnivariateDiscreteLeaf,
    leaf,
)
from probabilistic_model.utils import MissingDict

from ..circuit_building import uniform_leaf


class Animal(IntEnum):
    CAT = 0
    DOG = 1
    FISH = 2


# %% continuous leaves


@pytest.fixture
def uniform_circuit(x) -> ProbabilisticCircuit:
    """
    A circuit of a single uniform leaf over [0, 1].
    """
    circuit = ProbabilisticCircuit()
    uniform_leaf(x, 0, 1, circuit)
    return circuit


def test_truncation_to_an_interval(uniform_circuit, x):
    event = SimpleEvent.from_data({x: closed(0.5, 2)}).as_composite_set()
    truncation = uniform_circuit.truncated(event)
    assert len(list(truncation.distribution.nodes())) == 1
    assert truncation.probability == 0.5
    assert truncation.distribution.root.distribution.univariate_support == closed(
        0.5, 1
    )


def test_truncation_to_a_singleton_outside_the_support(uniform_circuit, x):
    event = SimpleEvent.from_data({x: singleton(2)}).as_composite_set()
    truncated = uniform_circuit.truncated(event).distribution
    assert truncated is None


def test_truncation_to_a_singleton(uniform_circuit, x):
    event = SimpleEvent.from_data({x: singleton(0.5)}).as_composite_set()

    truncated = uniform_circuit.truncated(event).distribution
    assert truncated is None

    truncation = uniform_circuit.truncated(event, singleton_allowed=True)
    assert isinstance(truncation.distribution.root.distribution, DiracDeltaDistribution)
    assert truncation.distribution.root.distribution.location == 0.5
    assert truncation.probability == pytest.approx(1.0)


def test_conditioning_on_a_point(uniform_circuit, x):
    conditioning = uniform_circuit.conditional({x: 0.3})
    assert len(list(conditioning.distribution.nodes())) == 1
    assert conditioning.likelihood == 1.0
    assert conditioning.distribution.root.distribution.location == pytest.approx(0.3)


def test_truncation_to_a_union_of_intervals_and_a_singleton_mixes_the_intervals(
    uniform_circuit, x
):
    event = SimpleEvent.from_data(
        {x: closed(0.0, 0.2) | closed(0.5, 1.0) | singleton(0.3)}
    ).as_composite_set()
    truncated = uniform_circuit.truncated(event).distribution
    assert len(list(truncated.nodes())) == 3
    assert len(list(truncated.edges())) == 2
    assert isinstance(truncated.root, SumUnit)


def test_truncation_of_a_mixture_to_an_interval_and_a_singleton(x):
    circuit = ProbabilisticCircuit()
    root = SumUnit(probabilistic_circuit=circuit)
    root.add_subcircuit(uniform_leaf(x, 0, 1, circuit), np.log(0.5))
    root.add_subcircuit(uniform_leaf(x, 2, 3, circuit), np.log(0.5))
    event = SimpleEvent.from_data(
        {x: closed(0.5, 0.6) | singleton(2.5)}
    ).as_composite_set()

    truncation = circuit.truncated(event, singleton_allowed=True)

    # the interval holds 0.5 * 0.1 of the mass, the singleton a density of 0.5 * 1
    assert truncation.probability == pytest.approx(0.55)
    assert isinstance(truncation.distribution.root, SumUnit)
    assert sorted(np.exp(truncation.distribution.root.log_weights)) == pytest.approx(
        [0.05 / 0.55, 0.5 / 0.55]
    )


# %% discrete leaves


@pytest.fixture
def animal_circuit() -> ProbabilisticCircuit:
    """
    A circuit of a single leaf over :class:`Animal`.
    """
    variable = Symbolic(name="animal", domain=Set.from_iterable(Animal))
    probabilities = MissingDict(
        float, {hash(Animal.CAT): 0.1, hash(Animal.DOG): 0.2, hash(Animal.FISH): 0.7}
    )
    return leaf(
        SymbolicDistribution(variable=variable, probabilities=probabilities),
        ProbabilisticCircuit(),
    ).probabilistic_circuit


@pytest.fixture
def integer_circuit() -> ProbabilisticCircuit:
    """
    A circuit of a single leaf over the integers 0, 1 and 2.
    """
    probabilities = MissingDict(float, {0: 0.1, 1: 0.2, 2: 0.7})
    return leaf(
        IntegerDistribution(variable=Integer("x"), probabilities=probabilities),
        ProbabilisticCircuit(),
    ).probabilistic_circuit


def test_discrete_leaf_as_deterministic_sum(animal_circuit):
    probabilities = list(animal_circuit.root.distribution.probabilities.values())
    sum_unit = animal_circuit.root.as_deterministic_sum()
    assert isinstance(sum_unit, SumUnit)
    assert sum_unit == animal_circuit.root
    assert len(sum_unit.subcircuits) == 3
    assert np.allclose(sum_unit.log_weights[::-1], np.log(probabilities))


def test_discrete_leaf_from_a_deterministic_sum(integer_circuit):
    integer_circuit.root.as_deterministic_sum()
    result = UnivariateDiscreteLeaf.from_mixture(integer_circuit)
    assert isinstance(result.distribution, IntegerDistribution)
    assert np.allclose(
        list(result.distribution.probabilities.values()), [0.1, 0.2, 0.7]
    )
