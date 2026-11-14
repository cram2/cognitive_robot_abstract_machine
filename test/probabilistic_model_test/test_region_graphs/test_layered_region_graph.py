"""
Tests for the layered circuits that region graphs build: their structure, their jax
parameters, training them and classifying with them.
"""

from __future__ import annotations

from dataclasses import dataclass

import jax
import jax.numpy as jnp
import numpy as np
import optax
import pytest
import scipy.special
from array_api_compat import array_namespace
from random_events.variable import Symbolic

from probabilistic_model.learning.gradient_descent import CrossEntropy
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
from probabilistic_model.probabilistic_circuit.tensorized.layered_probabilistic_circuit import (
    LayeredProbabilisticCircuit,
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
    circuit.fit(data=data, epochs=100, optimizer=optax.adam(0.05))
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


@dataclass(kw_only=True)
class SeparatedClasses:
    """
    An untrained classifying region graph circuit and rows of two well separated
    classes.
    """

    circuit: LayeredProbabilisticCircuit
    """
    The circuit over the features and the class variable.
    """

    data: np.ndarray
    """
    The rows, whose class column holds the hash of the class.
    """

    labels: np.ndarray
    """
    The position of the class of every row in the domain of the class variable.
    """

    class_variable: Symbolic
    """
    The class variable.
    """

    def joint_log_likelihoods_of_classes(self) -> np.ndarray:
        """
        :return: The joint log-likelihood of every row with every class, one column per
            class.
        """
        class_column = self.circuit.variables.index(self.class_variable)
        hashes = SymbolicEncoding(self.class_variable).hashes
        joint = []
        for label in range(len(hashes)):
            candidate = self.data.copy()
            candidate[:, class_column] = hashes[label]
            joint.append(np.asarray(self.circuit.log_likelihood(candidate)))
        return np.stack(joint, axis=1)


@pytest.fixture
def separated_classes(target) -> SeparatedClasses:
    """
    A classifying circuit over four features and 400 rows of two classes centred at -1
    and 1.
    """
    features = continuous_variables(4, prefix="x")
    circuit = (
        RegionGraph(variables=features, partitions=2, depth=1, repetitions=4, classes=2)
        .create_random_region_graph()
        .to_probabilistic_circuit(input_units=3, sum_units=3, class_variable=target)
    )
    generator = np.random.default_rng(0)
    labels = generator.integers(0, 2, 400)
    rows = generator.normal(size=(400, 4)) * 0.3 + (2.0 * labels - 1.0)[:, None]
    class_column = circuit.variables.index(target)
    data = np.insert(rows, class_column, labels, axis=1)
    # hashes do not survive single precision, so the rows stay numpy until encoded
    data[:, class_column] = SymbolicEncoding(target).hashes[labels]
    return SeparatedClasses(
        circuit=circuit, data=data, labels=labels, class_variable=target
    )


def test_cross_entropy_is_the_negative_log_probability_of_the_class_given_the_rest(
    separated_classes,
):
    """
    The cross entropy is the average of ``-log p(class | features)``, by Bayes' rule
    over the joint likelihoods of the classes.
    """
    circuit = separated_classes.circuit
    joint = separated_classes.joint_log_likelihoods_of_classes()
    rows = np.arange(len(joint))
    expected = np.mean(
        scipy.special.logsumexp(joint, axis=1) - joint[rows, separated_classes.labels]
    )
    loss = CrossEntropy.from_class_variable(
        class_variable=separated_classes.class_variable, variables=circuit.variables
    )
    assert float(
        loss.evaluate(circuit.root, circuit.encoded(separated_classes.data))
    ) == pytest.approx(float(expected), rel=1e-5)


def test_the_cross_entropy_of_a_torch_circuit_is_the_one_of_its_jax_circuit(
    separated_classes,
):
    """
    The cross entropy computes the same value on torch parameters.
    """
    torch = pytest.importorskip("torch")
    circuit = separated_classes.circuit
    loss = CrossEntropy.from_class_variable(
        class_variable=separated_classes.class_variable, variables=circuit.variables
    )
    encoded = circuit.encoded(separated_classes.data)
    torch_circuit = circuit.to_namespace(array_namespace(torch.zeros(0)))
    assert float(
        loss.evaluate(torch_circuit.root, torch.asarray(encoded))
    ) == pytest.approx(float(loss.evaluate(circuit.root, encoded)), rel=1e-5)


@pytest.mark.parametrize("trained_discriminatively", [False, True])
def test_a_trained_classifying_circuit_separates_two_classes(
    separated_classes, trained_discriminatively
):
    """
    Trained on the joint likelihood or by cross entropy, the class with the higher joint
    likelihood of a row is its class for almost every row.
    """
    circuit = separated_classes.circuit
    loss = (
        CrossEntropy.from_class_variable(
            class_variable=separated_classes.class_variable,
            variables=circuit.variables,
        )
        if trained_discriminatively
        else None
    )
    circuit.fit(
        data=separated_classes.data,
        epochs=150,
        optimizer=optax.adam(0.05),
        loss=loss,
    )
    predicted = np.argmax(separated_classes.joint_log_likelihoods_of_classes(), axis=1)
    assert (predicted == separated_classes.labels).mean() > 0.95
