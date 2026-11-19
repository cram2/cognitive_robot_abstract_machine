"""
Tests for folding a layered circuit: the folded circuit answers every query as the
circuit it was folded from, in a few layers.
"""

from __future__ import annotations

# %% imports
import jax
import jax.numpy as jnp
import numpy as np
import pytest
from random_events.interval import closed
from random_events.product_algebra import SimpleEvent, VariableMap
from random_events.set import Set
from random_events.variable import Continuous, Symbolic
from sortedcontainers import SortedSet

from probabilistic_model.learning.region_graph.region_graph import RegionGraph
from probabilistic_model.probabilistic_circuit.tensorized.inner_layer.edge_matrix import (
    BlockEdgeMatrix,
)
from probabilistic_model.probabilistic_circuit.tensorized.inner_layer.sum_layer import (
    SumLayer,
)
from probabilistic_model.probabilistic_circuit.tensorized.layered_probabilistic_circuit import (
    LayeredProbabilisticCircuit,
)

from .test_leaf_variables_per_node import MixtureOverFourVariables

# %% circuits


@pytest.fixture
def variables() -> SortedSet:
    """
    :return: Three continuous variables and two symbolic ones.
    """
    return SortedSet(
        [Continuous(f"x{index}") for index in range(3)]
        + [
            Symbolic(
                name="y_colour", domain=Set.from_iterable(["red", "green", "blue"])
            ),
            Symbolic(name="y_size", domain=Set.from_iterable(["small", "large"])),
        ]
    )


@pytest.fixture
def random_network(variables: SortedSet) -> LayeredProbabilisticCircuit:
    """
    :return: A RAT-SPN over the variables.
    """
    np.random.seed(0)
    return (
        RegionGraph(variables=variables, depth=2, repetitions=3)
        .create_random_region_graph()
        .to_probabilistic_circuit(input_units=3, sum_units=3)
    )


@pytest.fixture(params=["random_network", "mixture_over_four_variables"])
def circuit(request) -> LayeredProbabilisticCircuit:
    """
    :return: A RAT-SPN, or a mixture whose leaf layers mix variables.
    """
    if request.param == "random_network":
        return request.getfixturevalue("random_network")
    return MixtureOverFourVariables(
        variables=SortedSet(
            [
                Continuous("x"),
                Continuous("y"),
                Symbolic(name="z_letter", domain=Set.from_iterable(["a", "b", "c"])),
                Symbolic(name="z_mark", domain=Set.from_iterable(["off", "on"])),
            ]
        )
    ).mixed


@pytest.fixture
def samples(circuit: LayeredProbabilisticCircuit) -> np.ndarray:
    """
    :return: Samples of the circuit.
    """
    np.random.seed(1)
    return circuit.sample(100)


@pytest.fixture
def event(circuit: LayeredProbabilisticCircuit) -> SimpleEvent:
    """
    :return: A box over the first continuous variable and two states of the first
        symbolic one.
    """
    continuous = next(v for v in circuit.variables if isinstance(v, Continuous))
    symbolic = next(v for v in circuit.variables if isinstance(v, Symbolic))
    elements = [simple_set.element for simple_set in symbolic.domain.simple_sets]
    return SimpleEvent.from_data(
        {
            continuous: closed(-1.0, 0.5),
            symbolic: Set.from_iterable([elements[0], elements[-1]]),
        }
    ).fill_missing_variables_pure(circuit.variables)


# %% structure


def test_a_folded_random_network_has_one_layer_per_type_and_height(random_network):
    """
    The 3 repetitions of a RAT-SPN fold into one layer per type of layer and height, the
    sums storing their edges as stacked blocks.
    """
    folded = random_network.folded()
    kinds = [type(layer) for layer in folded.layers]
    assert len(folded.layers) < len(random_network.layers)
    assert len(kinds) == len(
        {(type(layer), height) for layer, height in zip(folded.layers, kinds)}
    )
    assert all(
        isinstance(layer.edges, BlockEdgeMatrix)
        for layer in folded.layers
        if isinstance(layer, SumLayer)
    )


# %% queries agree with the circuit


def test_folded_log_likelihood(circuit, samples):
    """
    The folded circuit gives every sample the log-likelihood of the circuit.
    """
    np.testing.assert_allclose(
        circuit.folded().log_likelihood(samples), circuit.log_likelihood(samples)
    )


def test_folded_log_likelihood_in_jax(circuit, samples):
    """
    The folded circuit with jax parameters gives every sample the log-likelihood of the
    circuit.
    """
    with jax.enable_x64(True):
        folded = circuit.folded().to_namespace(jnp)
        np.testing.assert_allclose(
            np.asarray(folded.log_likelihood(jnp.asarray(samples))),
            circuit.log_likelihood(samples),
        )


def test_folded_probability_of_an_event(circuit, event):
    """
    The folded circuit gives a box the probability the circuit gives it.
    """
    assert circuit.folded().probability_of_simple_event(event) == pytest.approx(
        circuit.probability_of_simple_event(event)
    )


def test_folded_truncation(circuit, event, samples):
    """
    Truncating the folded circuit gives the probability and the distribution of the
    truncated circuit.
    """
    folded = circuit.folded().log_truncated(event.as_composite_set())
    truncated = circuit.log_truncated(event.as_composite_set())
    assert folded.log_probability == pytest.approx(truncated.log_probability)
    np.testing.assert_allclose(
        folded.distribution.log_likelihood(samples),
        truncated.distribution.log_likelihood(samples),
    )


def test_folded_conditioning(circuit, samples):
    """
    Conditioning the folded circuit on the first continuous and the first symbolic
    variable gives the likelihood and the distribution of the conditioned circuit.
    """
    continuous_column = next(
        column
        for column, variable in enumerate(circuit.variables)
        if isinstance(variable, Continuous)
    )
    symbolic_column = next(
        column
        for column, variable in enumerate(circuit.variables)
        if isinstance(variable, Symbolic)
    )
    point = {
        circuit.variables[continuous_column]: float(samples[0, continuous_column]),
        circuit.variables[symbolic_column]: circuit.variables[symbolic_column]
        .domain.simple_sets[0]
        .element,
    }
    folded = circuit.folded().log_conditional(point)
    conditioned = circuit.log_conditional(point)
    assert folded.log_likelihood == pytest.approx(conditioned.log_likelihood)
    conditioned_samples = samples.copy()
    conditioned_samples[:, continuous_column] = point[
        circuit.variables[continuous_column]
    ]
    conditioned_samples[:, symbolic_column] = hash(
        point[circuit.variables[symbolic_column]]
    )
    np.testing.assert_allclose(
        folded.distribution.log_likelihood(conditioned_samples),
        conditioned.distribution.log_likelihood(conditioned_samples),
    )


def test_folded_marginal(circuit, samples):
    """
    Marginalizing the folded circuit to every other variable gives the marginal of the
    circuit.
    """
    kept = list(circuit.variables)[1::2]
    columns = [circuit.variables.index(variable) for variable in kept]
    np.testing.assert_allclose(
        circuit.folded().marginal(kept).log_likelihood(samples[:, columns]),
        circuit.marginal(kept).log_likelihood(samples[:, columns]),
    )


def test_folded_moment(circuit):
    """
    The folded circuit has the means of the circuit.
    """
    continuous = [v for v in circuit.variables if isinstance(v, Continuous)]
    order = VariableMap({variable: 1 for variable in continuous})
    center = VariableMap({variable: 0.0 for variable in continuous})
    folded = circuit.folded().moment(order, center)
    expected = circuit.moment(order, center)
    assert [folded[v] for v in continuous] == pytest.approx(
        [expected[v] for v in continuous]
    )


def test_folded_samples_follow_the_circuit(circuit):
    """
    Samples of the folded circuit have the means of the circuit.
    """
    continuous = [v for v in circuit.variables if isinstance(v, Continuous)]
    columns = [circuit.variables.index(v) for v in continuous]
    np.random.seed(2)
    samples = circuit.folded().sample(20000)
    means = circuit.moment(
        VariableMap({variable: 1 for variable in continuous}),
        VariableMap({variable: 0.0 for variable in continuous}),
    )
    np.testing.assert_allclose(
        samples[:, columns].mean(axis=0), [means[v] for v in continuous], atol=0.1
    )
