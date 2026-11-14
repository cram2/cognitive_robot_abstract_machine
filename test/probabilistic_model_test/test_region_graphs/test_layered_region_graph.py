"""
Tests for the layered circuits that region graphs build: their structure, their jax
parameters, training them and classifying with them.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import optax
import pytest

from probabilistic_model.learning.region_graph.exceptions import (
    ClassVariableMismatchError,
)
from probabilistic_model.learning.region_graph.region_graph import RegionGraph
from probabilistic_model.probabilistic_circuit.tensorized.inner_layer.edge_matrix import (
    DenseEdgeMatrix,
)
from probabilistic_model.probabilistic_circuit.tensorized.inner_layer.sum_layer import (
    SumLayer,
)
from probabilistic_model.probabilistic_circuit.tensorized.symbolic_encoding import (
    SymbolicEncoding,
)

from .test_region_graph import continuous_variables, target

# %% structure


@pytest.fixture
def region_graph() -> RegionGraph:
    """
    A random region graph over six continuous variables.
    """
    return RegionGraph(
        variables=continuous_variables(6), partitions=2, depth=2, repetitions=3
    ).create_random_region_graph()


def test_a_region_graph_builds_a_layered_circuit_with_jax_parameters(region_graph):
    """
    The layered circuit of a region graph has jax parameters and gives every row a
    finite log-likelihood.
    """
    circuit = region_graph.to_probabilistic_circuit(input_units=4, sum_units=3)
    assert all(isinstance(parameter, jax.Array) for parameter in circuit.parameters)
    data = jnp.asarray(np.random.default_rng(0).normal(size=(50, 6)))
    assert bool(jnp.all(jnp.isfinite(circuit.log_likelihood(data))))


def test_the_sum_layers_of_a_region_graph_are_fully_connected_and_dense(region_graph):
    """
    Every sum layer connects every unit to every unit of its children and stores its
    edges densely.
    """
    circuit = region_graph.to_probabilistic_circuit(input_units=4, sum_units=3)
    sum_layers = [layer for layer in circuit.layers if isinstance(layer, SumLayer)]
    assert sum_layers
    for layer in sum_layers:
        assert isinstance(layer.edges, DenseEdgeMatrix)
        assert bool(layer.edges.is_edge.all())


# %% learning


def test_fitting_a_layered_circuit_raises_the_log_likelihood_of_its_data(region_graph):
    """
    Fitting the layered circuit of a region graph raises the average log-likelihood of
    the data.
    """
    circuit = region_graph.to_probabilistic_circuit(input_units=4, sum_units=3)
    data = jnp.asarray(np.random.default_rng(0).normal(size=(500, 6)) * 0.5 + 1.0)
    before = float(jnp.mean(circuit.log_likelihood(data)))
    circuit.fit(data, epochs=100, optimizer=optax.adam(0.05))
    assert float(jnp.mean(circuit.log_likelihood(data))) > before + 0.5


# %% classification


def test_a_classifying_region_graph_needs_a_class_variable_of_its_classes(target):
    """
    A region graph with two classes cannot build a circuit without a class variable with
    two domain elements.
    """
    region_graph = RegionGraph(
        variables=continuous_variables(4), depth=1, repetitions=2, classes=3
    ).create_random_region_graph()
    with pytest.raises(ClassVariableMismatchError):
        region_graph.to_probabilistic_circuit(class_variable=target)


def test_a_trained_classifying_circuit_separates_two_classes(target):
    """
    Trained on two well separated classes, the class with the higher joint likelihood of
    a row is its class for almost every row.
    """
    features = continuous_variables(4, prefix="x")
    region_graph = RegionGraph(
        variables=features, partitions=2, depth=1, repetitions=4, classes=2
    ).create_random_region_graph()
    circuit = region_graph.to_probabilistic_circuit(
        input_units=3, sum_units=3, class_variable=target
    )
    generator = np.random.default_rng(0)
    labels = generator.integers(0, 2, 400)
    rows = generator.normal(size=(400, 4)) * 0.3 + (2.0 * labels - 1.0)[:, None]
    class_column = circuit.variables.index(target)
    data = np.insert(rows, class_column, labels, axis=1)
    # hashes do not survive single precision, so the rows stay numpy until encoded
    hashes = SymbolicEncoding(target).hashes
    data[:, class_column] = hashes[labels]
    circuit.fit(data, epochs=150, optimizer=optax.adam(0.05))

    joint = []
    for label in (0, 1):
        candidate = data.copy()
        candidate[:, class_column] = hashes[label]
        joint.append(np.asarray(circuit.log_likelihood(candidate)))
    predicted = np.argmax(np.stack(joint, axis=1), axis=1)
    assert (predicted == labels).mean() > 0.95
