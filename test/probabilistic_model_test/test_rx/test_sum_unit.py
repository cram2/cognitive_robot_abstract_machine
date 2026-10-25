from __future__ import annotations

import numpy as np
import pytest
from random_events.interval import closed
from random_events.product_algebra import Event, SimpleEvent
from sortedcontainers import SortedSet

from krrood.adapters.json_serializer import from_json, to_json
from probabilistic_model.probabilistic_circuit.rx.probabilistic_circuit import (
    LeafUnit,
    ProbabilisticCircuit,
    SumUnit,
)

from ..circuit_building import uniform_leaf

# %% fixtures


@pytest.fixture
def mixture(x) -> ProbabilisticCircuit:
    """
    A mixture of a uniform distribution over [0, 1] weighted 0.6 and one over [3, 4]
    weighted 0.4.
    """
    circuit = ProbabilisticCircuit()
    root = SumUnit(probabilistic_circuit=circuit)
    root.add_subcircuit(uniform_leaf(x, 0, 1, circuit), np.log(0.6))
    root.add_subcircuit(uniform_leaf(x, 3, 4, circuit), np.log(0.4))
    return circuit


@pytest.fixture
def lower_half_of_the_first_component(x) -> Event:
    """
    The lower half of the support of the first component.
    """
    return SimpleEvent.from_data({x: closed(0, 0.5)}).as_composite_set()


# %% structure


def test_structure(mixture, x):
    assert len(list(mixture.nodes())) == 3
    assert len(mixture.edges()) == 2
    assert mixture.variables == SortedSet([x])


def test_latent_variable_has_a_state_per_component(mixture):
    assert len(mixture.root.latent_variable.domain.simple_sets) == 2


def test_log_weighted_subcircuits(mixture):
    log_weights = [
        log_weight for log_weight, _ in mixture.root.log_weighted_subcircuits
    ]
    assert log_weights == [np.log(0.4), np.log(0.6)]


def test_support_is_the_union_of_the_supports(mixture, x):
    assert (
        mixture.support
        == SimpleEvent.from_data({x: closed(0, 1) | closed(3, 4)}).as_composite_set()
    )


def test_mixture_of_disjoint_supports_is_deterministic(mixture):
    assert mixture.is_deterministic()


def test_normalize(x):
    circuit = ProbabilisticCircuit()
    root = SumUnit(probabilistic_circuit=circuit)
    root.add_subcircuit(uniform_leaf(x, 0, 1, circuit), np.log(0.5))
    root.add_subcircuit(uniform_leaf(x, 3, 4, circuit), np.log(0.3))
    root.normalize()
    assert root.log_weights[1] == pytest.approx(np.log(0.5 / 0.8))
    assert root.log_weights[0] == pytest.approx(np.log(0.3 / 0.8))


def test_plot_has_traces(mixture):
    assert len(mixture.plot()) > 0


def test_mounting_a_circuit_brings_all_its_nodes(x):
    circuit = ProbabilisticCircuit()
    top = SumUnit(probabilistic_circuit=circuit)
    middle = SumUnit(probabilistic_circuit=circuit)
    bottom = uniform_leaf(x, 0, 1, circuit)
    circuit.add_edges_from([(top, middle, 1.0), (middle, bottom, 1.0)])
    target = SumUnit(probabilistic_circuit=ProbabilisticCircuit())
    target.probabilistic_circuit.mount(top)
    assert len(list(target.probabilistic_circuit.nodes())) == 4


# %% inference


def test_likelihood(mixture):
    assert mixture.likelihood(np.array([[0.5]])) == 0.6


def test_probability(mixture, x):
    event = SimpleEvent.from_data({x: closed(0, 3.5)}).as_composite_set()
    assert mixture.probability(event) == 0.8


def test_distinct_samples_lie_in_the_support(mixture):
    samples = mixture.sample(100)
    assert np.all(mixture.likelihood(samples) > 0)
    assert len(np.unique(samples, axis=0)) == len(samples)


def test_expectation(mixture, x):
    assert mixture.expectation(mixture.variables)[x] == 0.5 * 0.6 + 0.4 * 3.5


def test_marginal_over_all_variables_keeps_the_circuit(mixture, x):
    marginal = mixture.marginal([x])
    assert len(marginal.edges()) == len(mixture.edges())
    assert len(marginal.nodes()) == len(mixture.nodes())


def test_mode_is_the_support_of_the_heavier_component(mixture, x):
    mode = mixture.mode()
    assert mode.likelihood == 0.6
    assert mode.event == SimpleEvent.from_data({x: closed(0, 1)}).as_composite_set()


def test_truncation_to_one_component_leaves_its_leaf(
    mixture, lower_half_of_the_first_component
):
    truncation = mixture.truncated(lower_half_of_the_first_component)
    assert truncation.probability == pytest.approx(0.3)
    assert len(list(truncation.distribution.nodes())) == 1
    assert isinstance(truncation.distribution.root, LeafUnit)
    assert truncation.distribution.probability(lower_half_of_the_first_component) == 1


def test_truncation_to_an_impossible_event(mixture, x):
    event = SimpleEvent.from_data({x: closed(5, 6)}).as_composite_set()
    truncation = mixture.truncated(event)
    assert truncation.probability == 0.0
    assert truncation.distribution is None


def test_serialization_keeps_the_probability(
    mixture, lower_half_of_the_first_component
):
    deserialized = from_json(to_json(mixture))
    assert deserialized.probability(
        lower_half_of_the_first_component
    ) == mixture.probability(lower_half_of_the_first_component)
