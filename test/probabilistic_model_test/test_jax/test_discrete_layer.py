from __future__ import annotations

from enum import IntEnum

import jax.numpy as jnp
import numpy as np
import pytest
from jax.experimental.sparse import BCOO
from random_events.set import Set
from random_events.variable import Symbolic
from sortedcontainers import SortedSet

from probabilistic_model.adapters.circuit_representations import CircuitRepresentations
from probabilistic_model.distributions.distributions import SymbolicDistribution
from probabilistic_model.probabilistic_circuit.jax.discrete_layer import (
    DifferentiableDiscreteLayer,
)
from probabilistic_model.probabilistic_circuit.jax.inner_layer import (
    DifferentiableSparseSumLayer,
)
from probabilistic_model.probabilistic_circuit.jax.probabilistic_circuit import (
    DifferentiableLayeredCircuit,
)
from probabilistic_model.probabilistic_circuit.rx.probabilistic_circuit import (
    ProbabilisticCircuit,
    SumUnit,
    UnivariateDiscreteLeaf,
    leaf,
)
from probabilistic_model.utils import MissingDict


class Animal(IntEnum):
    CAT = 0
    DOG = 1
    FISH = 2


# %% fixtures


@pytest.fixture
def animal() -> Symbolic:
    """
    A symbolic variable over :class:`Animal`.
    """
    return Symbolic(name="x", domain=Set.from_iterable(Animal))


@pytest.fixture
def unnormalized_masses() -> jnp.ndarray:
    """
    The unnormalized masses of two distributions over :class:`Animal`, one per row.
    """
    return jnp.array([[0.0, 1.0, 2.0], [3.0, 4.0, 0.0]])


@pytest.fixture
def discrete_layer(unnormalized_masses) -> DifferentiableDiscreteLayer:
    """
    A layer of the two distributions of :func:`unnormalized_masses`.
    """
    layer = DifferentiableDiscreteLayer(0, jnp.log(unnormalized_masses))
    layer.validate()
    return layer


# %% discrete layer


def test_log_normalization_constant(discrete_layer, unnormalized_masses):
    assert jnp.allclose(
        discrete_layer.log_normalization_constant,
        jnp.log(unnormalized_masses.sum(axis=1)),
        atol=1e-3,
    )


def test_log_likelihood_of_one_event(discrete_layer):
    log_likelihood = discrete_layer.log_likelihood_of_nodes_single(jnp.array([0.0]))
    assert jnp.allclose(log_likelihood, jnp.log(jnp.array([0.0, 3 / 7])), atol=1e-3)


def test_log_likelihood(discrete_layer, unnormalized_masses):
    log_likelihood = discrete_layer.log_likelihood_of_nodes(
        jnp.array([[0.0], [1.0], [2.0]])
    )
    expected = jnp.log(
        (unnormalized_masses / unnormalized_masses.sum(axis=1, keepdims=True)).T
    )
    assert log_likelihood.shape == (3, 2)
    assert jnp.allclose(log_likelihood, expected, atol=1e-3)


# %% conversion


def test_conversion_from_a_mixture_of_symbolic_leaves(
    discrete_layer, unnormalized_masses, animal
):
    circuit = ProbabilisticCircuit()
    mixture = SumUnit(probabilistic_circuit=circuit)
    for masses in reversed(unnormalized_masses.tolist()):
        probabilities = MissingDict(
            float, {hash(category): mass for category, mass in zip(Animal, masses)}
        )
        mixture.add_subcircuit(
            leaf(
                SymbolicDistribution(variable=animal, probabilities=probabilities),
                circuit,
            ),
            np.log(0.5),
        )

    layered = CircuitRepresentations().convert(circuit, DifferentiableLayeredCircuit)
    converted = layered.root.child_layers[0]
    assert isinstance(converted, DifferentiableDiscreteLayer)
    assert converted.variable == 0
    assert jnp.allclose(converted.log_probabilities, discrete_layer.log_probabilities)


def test_conversion_to_normalized_symbolic_leaves(discrete_layer, animal):
    mixture = DifferentiableSparseSumLayer(
        [discrete_layer],
        [BCOO((jnp.zeros(2), jnp.array([[0, 0], [0, 1]])), shape=(1, 2))],
    )
    circuit = CircuitRepresentations().convert(
        DifferentiableLayeredCircuit(SortedSet([animal]), mixture),
        ProbabilisticCircuit,
    )
    leaves = [
        node for node in circuit.nodes() if isinstance(node, UnivariateDiscreteLeaf)
    ]
    assert len(leaves) == discrete_layer.number_of_nodes
    for node in leaves:
        assert node.variable == animal
        assert sum(node.distribution.probabilities.values()) == pytest.approx(
            1.0, abs=1e-5
        )
