from __future__ import annotations

from enum import IntEnum

import jax.numpy as jnp
import numpy as np
import optax
import pytest
from random_events.product_algebra import SimpleEvent
from random_events.set import Set
from random_events.variable import Continuous, Symbolic
from scipy.special import logsumexp
from sortedcontainers import SortedSet

from probabilistic_model.adapters.circuit_representations import CircuitRepresentations
from probabilistic_model.learning.region_graph.region_graph import RegionGraph
from probabilistic_model.probabilistic_circuit.jax.probabilistic_circuit import (
    ClassificationCircuit,
    DifferentiableLayeredCircuit,
)
from probabilistic_model.probabilistic_circuit.rx.probabilistic_circuit import (
    ProbabilisticCircuit,
    SumUnit,
    UnivariateContinuousLeaf,
    UnivariateDiscreteLeaf,
)


class Target(IntEnum):
    A = 0
    B = 1


@pytest.fixture
def target() -> Symbolic:
    """
    The variable a classification circuit predicts.
    """
    return Symbolic(name="target", domain=Set.from_iterable(Target))


def continuous_variables(number_of_variables: int, prefix: str = "") -> SortedSet:
    """
    :param number_of_variables: The number of variables.
    :param prefix: The prefix of their names, which are followed by their index.
    :return: The variables.
    """
    return SortedSet(
        [Continuous(f"{prefix}{index}") for index in range(number_of_variables)]
    )


# %% structure


def test_region_graph_has_a_region_per_split():
    region_graph = RegionGraph(
        continuous_variables(4), partitions=2, depth=1, repetitions=2
    ).create_random_region_graph()
    assert len(region_graph.nodes()) == 19


def test_circuit_of_a_region_graph_has_a_sum_unit_per_sum_node():
    region_graph = RegionGraph(
        continuous_variables(4), partitions=2, depth=1, repetitions=2
    ).create_random_region_graph()
    circuit = CircuitRepresentations().convert(
        region_graph.as_probabilistic_circuit(input_units=10, sum_units=5),
        ProbabilisticCircuit,
    )
    assert len([node for node in circuit.nodes() if isinstance(node, SumUnit)]) == 21


# %% learning


def test_learned_circuit_has_normalized_parameters(target):
    variables = continuous_variables(8) | SortedSet([target])
    region_graph = RegionGraph(
        variables, partitions=2, depth=2, repetitions=2
    ).create_random_region_graph()
    model = region_graph.as_probabilistic_circuit(input_units=5, sum_units=5)
    data = jnp.array(np.random.uniform(0, 1, (100, len(variables))))
    model.fit(data, epochs=10, optimizer=optax.adamw(0.01))

    circuit = CircuitRepresentations().convert(model, ProbabilisticCircuit)
    for node in circuit.nodes():
        if isinstance(node, SumUnit):
            assert logsumexp(node.log_weights) == pytest.approx(0.0)
        elif isinstance(node, UnivariateDiscreteLeaf):
            assert sum(node.distribution.probabilities) == pytest.approx(1.0)
        elif isinstance(node, UnivariateContinuousLeaf):
            assert node.distribution.scale > 0.0


def test_classification_circuit_predicts_a_distribution_over_the_target(target):
    features = continuous_variables(4, prefix="x")
    region_graph = RegionGraph(
        features, partitions=2, depth=1, repetitions=6, classes=2
    ).create_random_region_graph()
    model = region_graph.as_probabilistic_circuit(input_units=5, sum_units=5)
    assert isinstance(model, ClassificationCircuit)
    assert model.root.number_of_nodes == 2

    data = jnp.array(np.random.uniform(0, 1, (100, len(features))))
    labels = jnp.array(np.random.randint(2, size=len(data)))
    model.fit(data, labels=labels, epochs=10, optimizer=optax.adamw(0.01))
    joint = model.as_probabilistic_circuit(target)
    assert isinstance(joint, DifferentiableLayeredCircuit)
    assert joint.variables == features | SortedSet([target])

    circuit = CircuitRepresentations().convert(joint, ProbabilisticCircuit)
    assert circuit.is_decomposable()
    target_marginal = circuit.marginal(SortedSet([target]))
    total = sum(
        target_marginal.probability_of_simple_event(
            SimpleEvent.from_data({target: element})
        )
        for element in target.domain
    )
    assert total == pytest.approx(1.0)
