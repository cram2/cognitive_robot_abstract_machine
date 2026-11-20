"""
Tests of the tensorized layer of binomial distributions.
"""

from __future__ import annotations

# %% imports
import copy

import jax
import jax.numpy as jnp
import numpy as np
import pandas as pd
import pytest
from random_events.interval import closed
from random_events.product_algebra import SimpleEvent, VariableMap
from random_events.variable import Integer
from sortedcontainers import SortedSet

from probabilistic_model.adapters.rustworkx_tensorized.rustworkx_to_tensorized import (
    RustworkxCircuitToLayeredCircuitConverter,
)
from probabilistic_model.distributions.binomial import (
    BinomialDistribution,
    TruncatedBinomialDistribution,
)
from probabilistic_model.exceptions import SuccessProbabilityOutsideUnitIntervalError
from probabilistic_model.learning.row_kernel_mixture import RowKernelMixture
from probabilistic_model.probabilistic_circuit.tensorized.input_layer.binomial_layer import (
    BinomialLayer,
)
from probabilistic_model.probabilistic_circuit.tensorized.moment_query import (
    MomentQuery,
)

# %% fixtures


@pytest.fixture
def count() -> Integer:
    """
    The integer variable of the layer.
    """
    return Integer(name="count")


@pytest.fixture
def distributions(count) -> list[BinomialDistribution]:
    """
    Binomial distributions of different numbers of trials, success probabilities and
    offsets, one of them certain to succeed.
    """
    return [
        BinomialDistribution(
            variable=count, number_of_trials=2, success_probability=0.3, offset=1
        ),
        BinomialDistribution(
            variable=count, number_of_trials=5, success_probability=0.5, offset=-1
        ),
        BinomialDistribution(
            variable=count, number_of_trials=3, success_probability=1.0, offset=0
        ),
    ]


@pytest.fixture
def layer(distributions) -> BinomialLayer:
    """
    The layer of the distributions.
    """
    return BinomialLayer.from_distributions(0, distributions)


@pytest.fixture
def values() -> np.ndarray:
    """
    Values of the variable below, within, between and above the supports of the nodes,
    as a column of events.
    """
    return np.array([-2.0, -1.0, 0.0, 1.0, 1.5, 2.0, 3.0, 4.0, 7.0]).reshape(-1, 1)


# %% the nodes are the binomial distributions


def test_the_nodes_are_the_distributions_the_layer_was_made_of(
    layer, distributions, count
):
    """
    Every node of the layer is the distribution it was made of.
    """
    assert [
        (node.number_of_trials, node.success_probability, node.offset)
        for node in layer.node_distributions(count)
    ] == [
        (each.number_of_trials, each.success_probability, each.offset)
        for each in distributions
    ]


def test_log_likelihoods_are_those_of_the_distributions(layer, distributions, values):
    """
    The log-likelihood of every node is that of its distribution, ``-inf`` outside of
    its support and between its values.
    """
    with np.errstate(divide="ignore"):
        expected = np.log(
            [
                [distribution.probabilities[value] for distribution in distributions]
                for value in values[:, 0]
            ]
        )
    assert np.allclose(layer.log_likelihood_of_nodes(values), expected)


def test_cumulative_distribution_is_that_of_the_distributions(
    layer, distributions, values
):
    """
    The cumulative distribution of every node is that of its distribution.
    """
    expected = np.stack(
        [
            distribution.cumulative_distribution_function(values)
            for distribution in distributions
        ],
        axis=1,
    )
    assert np.allclose(layer.cumulative_distribution_of_nodes(values), expected)


def test_probability_of_an_event_is_that_of_the_distributions(
    layer, distributions, count
):
    """
    The probability of an event under every node is that of its distribution.
    """
    event = SimpleEvent.from_data({count: closed(0, 1) | closed(3, 6)})
    assert np.allclose(
        layer.probability_of_simple_event_of_nodes(event, SortedSet([count])),
        [
            distribution.probability_of_simple_event(event)
            for distribution in distributions
        ],
    )


def test_moments_are_those_of_the_distributions(layer, distributions, count):
    """
    The second moment around a half of every node is that of its distribution.
    """
    query = MomentQuery.from_maps({count: 2}, {count: 0.5}, SortedSet([count]))
    assert np.allclose(
        layer.moment_of_nodes(query, SortedSet([count]))[:, 0],
        [
            distribution.moment(VariableMap({count: 2}), VariableMap({count: 0.5}))[
                count
            ]
            for distribution in distributions
        ],
    )


def test_samples_of_a_node_lie_in_its_support(layer, distributions, count):
    """
    Every sample of a node is one of the values its distribution can take.
    """
    np.random.seed(0)
    nodes = np.repeat(np.arange(len(distributions)), 50)
    samples = layer.sample_of_nodes(nodes, SortedSet([count]))
    for node, sample in zip(nodes, samples):
        assert distributions[node].probabilities[int(sample)] > 0


# %% truncating and conditioning restrict the binomials


@pytest.mark.parametrize(
    "assignment",
    [closed(1, 3), closed(-1, 0) | closed(3, 4)],
    ids=["interval", "union of intervals"],
)
def test_a_truncated_layer_holds_the_truncated_distributions(
    layer, distributions, count, assignment
):
    """
    Truncated to an assignment, every node becomes its distribution truncated to it, and
    the log-probabilities are those of the assignment under the nodes.
    """
    event = SimpleEvent.from_data({count: assignment}).as_composite_set()
    with np.errstate(divide="ignore"):
        truncation = layer.log_truncated_of_assignment(assignment, False)
        expected = [distribution.log_truncated(event) for distribution in distributions]

    assert type(truncation.layer) is BinomialLayer
    assert np.allclose(
        truncation.log_probabilities, [each.log_probability for each in expected]
    )
    for node, each in zip(truncation.layer.node_distributions(count), expected):
        if each.distribution is None:
            continue
        assert dict(node.probabilities) == pytest.approx(
            dict(each.distribution.probabilities)
        )


def test_a_truncated_layer_keeps_the_success_probabilities_as_parameters(
    layer, distributions
):
    """
    Truncated, the nodes keep their success probabilities as the parameters.
    """
    truncated = layer.log_truncated_of_assignment(closed(1, 3), False).layer
    [success_probability] = truncated.parameters
    assert success_probability.tolist() == [
        distribution.success_probability for distribution in distributions
    ]


def test_truncating_twice_restricts_to_both_assignments(layer, distributions, count):
    """
    Truncated to one assignment and then another, the nodes are restricted to the values
    of both.
    """
    once = layer.log_truncated_of_assignment(closed(0, 3), False).layer
    twice = once.log_truncated_of_assignment(closed(2, 6), False).layer
    event = SimpleEvent.from_data({count: closed(2, 3)}).as_composite_set()
    for node, distribution in zip(twice.node_distributions(count), distributions):
        assert dict(node.probabilities) == pytest.approx(
            dict(distribution.log_truncated(event).distribution.probabilities)
        )


def test_conditioning_on_a_value_keeps_only_that_value(layer, distributions, count):
    """
    Conditioned on a value, every node that can take it is certain of it, and the log-
    probabilities are the log-likelihoods of the value.
    """
    with np.errstate(divide="ignore"):
        conditioning = layer.log_conditional_of_value(2)
        expected = np.log(
            [distribution.probabilities[2] for distribution in distributions]
        )

    assert type(conditioning.layer) is BinomialLayer
    assert isinstance(
        conditioning.layer.node_distribution(1, count), TruncatedBinomialDistribution
    )
    assert np.allclose(conditioning.log_probabilities, expected)
    assert np.allclose(
        conditioning.layer.log_likelihood_of_nodes(np.array([[2.0]]))[0, :2], 0.0
    )


# %% the success probabilities are parameters


def test_the_success_probabilities_are_the_parameters(layer, distributions):
    """
    The parameters of the layer are the success probabilities of its nodes alone.
    """
    [success_probability] = layer.parameters
    assert success_probability.tolist() == [
        distribution.success_probability for distribution in distributions
    ]
    assert layer.number_of_own_parameters == len(distributions)


def test_the_gradient_of_the_log_likelihood_is_that_of_the_binomial(count):
    """
    The derivative of the log-likelihood by the success probability is that of the
    binomial, so that gradient descent can learn it.
    """
    trials, successes, probability = 6, 2, 0.4
    layer = BinomialLayer.from_distributions(
        0,
        [
            BinomialDistribution(
                variable=count, number_of_trials=trials, success_probability=probability
            )
        ],
    ).copy_with_parameters([jnp.asarray([probability])])

    def log_likelihood(success_probability: jax.Array) -> jax.Array:
        changed = layer.copy_with_parameters([success_probability])
        return changed.log_likelihood_of_nodes(jnp.asarray([[float(successes)]]))[0, 0]

    gradient = jax.grad(log_likelihood)(jnp.asarray([probability]))
    assert float(gradient[0]) == pytest.approx(
        successes / probability - (trials - successes) / (1 - probability), rel=1e-5
    )


def test_projected_success_probabilities_are_probabilities(layer):
    """
    Projected, a step past zero or one is brought back to it.
    """
    [projected] = layer.projected_parameters([np.array([-0.2, 0.5, 1.3])])
    assert projected.tolist() == [0.0, 0.5, 1.0]


def test_a_success_probability_outside_of_zero_and_one_is_refused(layer):
    """
    A success probability that is no probability makes the layer invalid.
    """
    invalid = layer.copy_with_parameters([np.array([0.3, 1.5, 0.2])])
    with pytest.raises(SuccessProbabilityOutsideUnitIntervalError):
        invalid.validate_own()


# %% layers in circuits


def test_a_translation_moves_the_offsets(layer, distributions):
    """
    Translated, every node keeps its trials and success probability and moves its
    offset.
    """
    translated = copy.deepcopy(layer)
    translated.apply_translation_own(np.array([3]))
    assert translated.offset.tolist() == [
        distribution.offset + 3 for distribution in distributions
    ]


def test_a_circuit_of_binomial_leaves_converts_to_binomial_layers():
    """
    A row kernel mixture with an integer column becomes a layered circuit of a binomial
    layer with the likelihoods of the circuit it was converted from.
    """
    rows = pd.DataFrame({"count": [1, 3, 5, 8], "angle": [0.1, 0.5, 2.0, 3.0]})
    circuit = RowKernelMixture(bandwidth=0.5).fit(rows)
    events = np.array([[0.2, 1], [1.0, 4], [3.0, 8], [0.0, 20]], dtype=float)

    layered = RustworkxCircuitToLayeredCircuitConverter.convert(circuit)

    assert any(isinstance(each, BinomialLayer) for each in layered.root.all_layers())
    assert np.allclose(layered.log_likelihood(events), circuit.log_likelihood(events))


def test_a_truncated_circuit_of_binomial_leaves_converts_to_binomial_layers():
    """
    Truncated to a union of intervals of its integer column, a row kernel mixture still
    converts to a layered circuit with its likelihoods.
    """
    rows = pd.DataFrame({"count": [1, 3, 5, 8], "angle": [0.1, 0.5, 2.0, 3.0]})
    circuit = RowKernelMixture(bandwidth=0.5).fit(rows)
    count = next(each for each in circuit.variables if each.name == "count")
    truncated = circuit.truncated(
        SimpleEvent.from_data({count: closed(0, 2) | closed(5, 6)}).as_composite_set()
    ).distribution
    events = np.array([[0.2, 1], [1.0, 4], [2.0, 5], [0.0, 20]], dtype=float)

    layered = RustworkxCircuitToLayeredCircuitConverter.convert(truncated)

    assert np.allclose(layered.log_likelihood(events), truncated.log_likelihood(events))
