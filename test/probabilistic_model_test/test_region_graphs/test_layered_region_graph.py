"""
Tests for the layered circuits that region graphs build: their structure, their
parameters, training them with jax and classifying with them.
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
from random_events.interval import SimpleInterval
from sortedcontainers import SortedSet

from probabilistic_model.learning.gradient_descent import CrossEntropy
from probabilistic_model.learning.region_graph.exceptions import (
    PredictedVariableInRegionGraphError,
)
from probabilistic_model.learning.region_graph.region_graph import RegionGraph
from probabilistic_model.probabilistic_circuit.tensorized.folding import LayerFolding
from probabilistic_model.probabilistic_circuit.tensorized.inner_layer.edge_matrix import (
    BlockEdgeMatrix,
)
from probabilistic_model.probabilistic_circuit.tensorized.inner_layer.sum_layer import (
    SumLayer,
)
from probabilistic_model.probabilistic_circuit.tensorized.input_layer.gaussian_layer import (
    GaussianLayer,
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


def test_a_region_graph_builds_a_layered_circuit_with_numpy_parameters(region_graph):
    """
    The layered circuit of a region graph has numpy parameters and gives every row a
    finite log-likelihood.
    """
    circuit = region_graph.to_probabilistic_circuit(input_units=4, sum_units=3)
    assert all(isinstance(parameter, np.ndarray) for parameter in circuit.parameters)
    data = np.random.default_rng(0).normal(size=(50, 6))
    assert np.all(np.isfinite(circuit.log_likelihood(data)))


def test_the_circuit_of_a_region_graph_moves_to_jax_with_the_same_likelihoods(
    region_graph,
):
    """
    Moving the circuit of a region graph to jax keeps its log-likelihoods.
    """
    circuit = region_graph.to_probabilistic_circuit(input_units=4, sum_units=3)
    jax_circuit = circuit.to_namespace(jnp, floating_point_type=jnp.result_type(float))
    assert all(isinstance(parameter, jax.Array) for parameter in jax_circuit.parameters)
    data = np.random.default_rng(0).normal(size=(50, 6))
    np.testing.assert_allclose(
        np.asarray(jax_circuit.log_likelihood(jnp.asarray(data))),
        circuit.log_likelihood(data),
        rtol=1e-4,
    )


def test_region_graphs_with_the_same_seed_build_circuits_with_the_same_parameters():
    """
    The seed of a region graph fixes the initial parameters of its circuit.
    """
    circuits = [
        RegionGraph(variables=continuous_variables(6), depth=2, repetitions=3, seed=3)
        .create_random_region_graph()
        .to_probabilistic_circuit(input_units=4, sum_units=3)
        for _ in range(2)
    ]
    for first, second in zip(circuits[0].parameters, circuits[1].parameters):
        np.testing.assert_array_equal(first, second)


def test_the_initial_parameters_of_a_region_graph_circuit_lie_in_its_ranges():
    """
    The initial weights, means and standard deviations of a region graph circuit are
    drawn from the ranges of the region graph.
    """
    region_graph = RegionGraph(
        variables=continuous_variables(6),
        depth=2,
        repetitions=3,
        initial_weight_range=SimpleInterval.from_data(lower=2.0, upper=3.0),
        initial_location_range=SimpleInterval.from_data(lower=5.0, upper=6.0),
        initial_scale_range=SimpleInterval.from_data(lower=0.2, upper=0.4),
    ).create_random_region_graph()
    circuit = region_graph.to_probabilistic_circuit(input_units=4, sum_units=3)
    gaussians = [layer for layer in circuit.layers if isinstance(layer, GaussianLayer)]
    sum_layers = [layer for layer in circuit.layers if isinstance(layer, SumLayer)]
    locations = np.concatenate([layer.location for layer in gaussians])
    scales = np.concatenate([layer.scale for layer in gaussians])
    weights = np.exp(
        np.concatenate([layer.log_weights.ravel() for layer in sum_layers])
    )
    assert np.all((locations >= 5.0) & (locations < 6.0))
    assert np.all((scales >= 0.2) & (scales < 0.4))
    assert np.all((weights >= 2.0) & (weights < 3.0))


def test_the_circuit_of_a_region_graph_is_folded(region_graph):
    """
    A region graph builds its circuit folded: one layer per type of layer and height,
    every sum layer storing its edges as full blocks.
    """
    circuit = region_graph.to_probabilistic_circuit(input_units=4, sum_units=3)
    heights = LayerFolding(root=circuit.root).heights
    kinds = [(type(layer), heights[layer]) for layer in circuit.layers]
    sum_layers = [layer for layer in circuit.layers if isinstance(layer, SumLayer)]
    assert len(kinds) == len(set(kinds))
    assert sum_layers
    assert all(isinstance(layer.edges, BlockEdgeMatrix) for layer in sum_layers)


# %% learning


def test_fitting_a_layered_circuit_raises_the_log_likelihood_of_its_data(region_graph):
    """
    Fitting the layered circuit of a region graph raises the average log-likelihood of
    the data.
    """
    circuit = region_graph.to_probabilistic_circuit(
        input_units=4, sum_units=3
    ).to_namespace(jnp, floating_point_type=jnp.result_type(float))
    data = jnp.asarray(np.random.default_rng(0).normal(size=(500, 6)) * 0.5 + 1.0)
    before = float(jnp.mean(circuit.log_likelihood(data)))
    circuit.fit(data=data, epochs=100, optimizer=optax.adam(0.05))
    assert float(jnp.mean(circuit.log_likelihood(data))) > before + 0.5


# %% classification


def test_a_region_graph_cannot_predict_one_of_its_own_variables(target):
    """
    The predicted variable of a region graph circuit is not among the variables the
    region graph partitions.
    """
    region_graph = RegionGraph(
        variables=continuous_variables(3) | SortedSet([target]), depth=1, repetitions=2
    ).create_random_region_graph()
    with pytest.raises(PredictedVariableInRegionGraphError):
        region_graph.to_probabilistic_circuit(predicted_variable=target)


def test_the_root_region_has_a_sum_unit_per_category_of_the_predicted_variable(target):
    """
    The root mixes one category conditional unit per category of the predicted variable.
    """
    circuit = (
        RegionGraph(variables=continuous_variables(4), depth=1, repetitions=2)
        .create_random_region_graph()
        .to_probabilistic_circuit(predicted_variable=target)
    )
    [category_conditionals] = circuit.root.child_layers
    assert category_conditionals.number_of_nodes == len(target.domain.simple_sets)


@dataclass(kw_only=True)
class SeparatedCategories:
    """
    An untrained classifying region graph circuit and rows of two well separated
    categories.
    """

    circuit: LayeredProbabilisticCircuit
    """
    The circuit over the features and the predicted variable.
    """

    data: np.ndarray
    """
    The rows, whose predicted column holds the hash of the category.
    """

    labels: np.ndarray
    """
    The position of the category of every row in the domain of the predicted variable.
    """

    predicted_variable: Symbolic
    """
    The variable the circuit predicts.
    """

    def joint_log_likelihoods_of_categories(self) -> np.ndarray:
        """
        :return: The joint log-likelihood of every row with every category, one column
            per category.
        """
        predicted_column = self.circuit.variables.index(self.predicted_variable)
        hashes = SymbolicEncoding(self.predicted_variable).hashes
        joint = []
        for label in range(len(hashes)):
            candidate = self.data.copy()
            candidate[:, predicted_column] = hashes[label]
            joint.append(np.asarray(self.circuit.log_likelihood(candidate)))
        return np.stack(joint, axis=1)


@pytest.fixture
def separated_categories(target) -> SeparatedCategories:
    """
    A classifying circuit over four features and 400 rows of two categories centred at
    -1 and 1.
    """
    features = continuous_variables(4, prefix="x")
    circuit = (
        RegionGraph(variables=features, partitions=2, depth=1, repetitions=4)
        .create_random_region_graph()
        .to_probabilistic_circuit(input_units=3, sum_units=3, predicted_variable=target)
        .to_namespace(jnp, floating_point_type=jnp.result_type(float))
    )
    generator = np.random.default_rng(0)
    labels = generator.integers(0, 2, 400)
    rows = generator.normal(size=(400, 4)) * 0.3 + (2.0 * labels - 1.0)[:, None]
    predicted_column = circuit.variables.index(target)
    data = np.insert(rows, predicted_column, labels, axis=1)
    # hashes do not survive single precision, so the rows stay numpy until encoded
    data[:, predicted_column] = SymbolicEncoding(target).hashes[labels]
    return SeparatedCategories(
        circuit=circuit, data=data, labels=labels, predicted_variable=target
    )


def test_cross_entropy_is_the_negative_log_probability_of_the_category_given_the_rest(
    separated_categories,
):
    """
    The cross entropy is the average of ``-log p(category | features)``, by Bayes' rule
    over the joint likelihoods of the categories.
    """
    circuit = separated_categories.circuit
    joint = separated_categories.joint_log_likelihoods_of_categories()
    rows = np.arange(len(joint))
    expected = np.mean(
        scipy.special.logsumexp(joint, axis=1)
        - joint[rows, separated_categories.labels]
    )
    loss = CrossEntropy.from_predicted_variable(
        predicted_variable=separated_categories.predicted_variable,
        variables=circuit.variables,
    )
    assert float(
        loss.evaluate(circuit.root, circuit.encoded(separated_categories.data))
    ) == pytest.approx(float(expected), rel=1e-5)


def test_the_cross_entropy_of_a_torch_circuit_is_the_one_of_its_jax_circuit(
    separated_categories,
):
    """
    The cross entropy computes the same value on torch parameters.
    """
    torch = pytest.importorskip("torch")
    circuit = separated_categories.circuit
    loss = CrossEntropy.from_predicted_variable(
        predicted_variable=separated_categories.predicted_variable,
        variables=circuit.variables,
    )
    encoded = circuit.encoded(separated_categories.data)
    torch_circuit = circuit.to_namespace(array_namespace(torch.zeros(0)))
    assert float(
        loss.evaluate(torch_circuit.root, torch.asarray(encoded))
    ) == pytest.approx(float(loss.evaluate(circuit.root, encoded)), rel=1e-5)


@pytest.mark.parametrize("trained_discriminatively", [False, True])
def test_a_trained_classifying_circuit_separates_two_categories(
    separated_categories, trained_discriminatively
):
    """
    Trained on the joint likelihood or by cross entropy, the category with the higher
    joint likelihood of a row is its category for almost every row.
    """
    circuit = separated_categories.circuit
    loss = (
        CrossEntropy.from_predicted_variable(
            predicted_variable=separated_categories.predicted_variable,
            variables=circuit.variables,
        )
        if trained_discriminatively
        else None
    )
    circuit.fit(
        data=separated_categories.data,
        epochs=150,
        optimizer=optax.adam(0.05),
        loss=loss,
    )
    predicted = np.argmax(
        separated_categories.joint_log_likelihoods_of_categories(), axis=1
    )
    assert (predicted == separated_categories.labels).mean() > 0.95
