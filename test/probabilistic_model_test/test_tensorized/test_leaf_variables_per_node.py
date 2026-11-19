"""
Tests for leaf layers whose nodes are over several variables: a circuit whose leaves mix
variables answers every query as the same circuit with one variable per leaf layer.
"""

from __future__ import annotations

# %% imports
from dataclasses import dataclass

import numpy as np
import pytest
from random_events.interval import closed
from random_events.product_algebra import SimpleEvent, VariableMap
from random_events.set import Set
from random_events.variable import Continuous, Symbolic
from sortedcontainers import SortedSet

from probabilistic_model.probabilistic_circuit.tensorized.exceptions import (
    LayerOverSeveralVariablesError,
    PartialMarginalOfLeafLayerError,
    UnsortedNodeVariablesError,
)
from probabilistic_model.probabilistic_circuit.tensorized.inner_layer.edge_matrix import (
    SparseEntries,
)
from probabilistic_model.probabilistic_circuit.tensorized.inner_layer.inner_layer_edge import (
    InnerLayerEdges,
)
from probabilistic_model.probabilistic_circuit.tensorized.inner_layer.product_layer import (
    ProductLayer,
)
from probabilistic_model.probabilistic_circuit.tensorized.inner_layer.sum_layer import (
    SumLayer,
)
from probabilistic_model.probabilistic_circuit.tensorized.input_layer.discrete_layer import (
    SymbolicLayer,
)
from probabilistic_model.probabilistic_circuit.tensorized.input_layer.gaussian_layer import (
    GaussianLayer,
)
from probabilistic_model.probabilistic_circuit.tensorized.input_layer.probability_table import (
    DenseProbabilityTable,
)
from probabilistic_model.probabilistic_circuit.tensorized.layered_probabilistic_circuit import (
    LayeredProbabilisticCircuit,
)

# %% circuits


@dataclass(kw_only=True)
class MixtureOverFourVariables:
    """
    A mixture of two products over two continuous and two symbolic variables, built once
    with one variable per leaf layer and once with leaf layers that mix variables.
    """

    variables: SortedSet
    """
    The variables x, y, letter and flag, in this order.
    """

    @property
    def gaussian_layers(self) -> list[GaussianLayer]:
        """
        :return: Two Gaussians over x and two over y.
        """
        return [
            GaussianLayer(
                np.zeros(2, dtype=np.int64), np.array([-1.0, 2.0]), np.array([0.5, 1.5])
            ),
            GaussianLayer(
                np.ones(2, dtype=np.int64), np.array([0.0, 1.0]), np.array([1.0, 2.0])
            ),
        ]

    @property
    def symbolic_layers(self) -> list[SymbolicLayer]:
        """
        :return: Two distributions over the three letters and two over the two flags.
        """
        return [
            SymbolicLayer(
                np.full(2, 2, dtype=np.int64),
                np.arange(3),
                DenseProbabilityTable(np.log([[0.2, 0.3, 0.5], [0.6, 0.3, 0.1]])),
            ),
            SymbolicLayer(
                np.full(2, 3, dtype=np.int64),
                np.arange(2),
                DenseProbabilityTable(np.log([[0.9, 0.1], [0.4, 0.6]])),
            ),
        ]

    def circuit_with_root_over(
        self, *, products: ProductLayer
    ) -> LayeredProbabilisticCircuit:
        """
        :param products: Two product nodes.
        :return: The circuit that mixes them with weights 0.3 and 0.7.
        """
        root = SumLayer.from_entries(
            [products],
            SparseEntries(np.log([0.3, 0.7]), np.zeros(2, dtype=int), np.arange(2)),
            1,
        )
        return LayeredProbabilisticCircuit(self.variables, root)

    @property
    def separate(self) -> LayeredProbabilisticCircuit:
        """
        :return: The circuit with one variable per leaf layer.
        """
        return self.circuit_with_root_over(
            products=ProductLayer.node_wise_product_of(
                self.gaussian_layers + self.symbolic_layers
            )
        )

    @property
    def mixed(self) -> LayeredProbabilisticCircuit:
        """
        :return: The circuit whose two leaf layers each hold the nodes of two variables.
        """
        leaves = [
            GaussianLayer.concatenate(self.gaussian_layers),
            SymbolicLayer.concatenate(self.symbolic_layers),
        ]
        # product i multiplies node i and node 2 + i of both leaf layers
        edges = InnerLayerEdges(
            np.tile([0, 1], 4),
            np.repeat([0, 1], 4),
            np.tile([0, 1, 2, 3], 2),
        )
        return self.circuit_with_root_over(
            products=ProductLayer.from_edges(leaves, edges, 2)
        )


@pytest.fixture
def mixture() -> MixtureOverFourVariables:
    """
    :return: The mixture over x, y, letter and flag.
    """
    return MixtureOverFourVariables(
        variables=SortedSet(
            [
                Continuous("x"),
                Continuous("y"),
                Symbolic(name="z_letter", domain=Set.from_iterable(["a", "b", "c"])),
                Symbolic(name="z_mark", domain=Set.from_iterable(["off", "on"])),
            ]
        )
    )


@pytest.fixture
def samples(mixture: MixtureOverFourVariables) -> np.ndarray:
    """
    :return: Samples of the mixture, symbolic values as hashes of domain elements.
    """
    np.random.seed(0)
    return mixture.separate.sample(200)


@pytest.fixture
def event(mixture: MixtureOverFourVariables) -> SimpleEvent:
    """
    :return: A box over x and the letters a and c.
    """
    x, _, letter, _ = mixture.variables
    return SimpleEvent.from_data(
        {
            x: closed(-1.5, 0.5),
            letter: Set.from_iterable([letter.domain.simple_sets[0].element, "c"]),
        }
    ).fill_missing_variables_pure(mixture.variables)


# %% the layer


def test_a_layer_over_several_variables_has_no_single_variable(
    mixture: MixtureOverFourVariables,
):
    """
    Asking the one variable of a layer whose nodes are over several raises.
    """
    layer = GaussianLayer.concatenate(mixture.gaussian_layers)
    with pytest.raises(LayerOverSeveralVariablesError):
        layer.variable


def test_a_layer_with_unsorted_node_variables_is_invalid():
    """
    A leaf layer whose nodes are not sorted by variable fails validation.
    """
    layer = GaussianLayer(np.array([1, 0]), np.zeros(2), np.ones(2))
    with pytest.raises(UnsortedNodeVariablesError):
        layer.validate()


def test_concatenated_symbolic_layers_pad_the_states_a_variable_lacks(
    mixture: MixtureOverFourVariables,
):
    """
    Joining symbolic layers over variables with different numbers of states gives the
    nodes of the smaller one probability zero for the states it lacks.
    """
    layer = SymbolicLayer.concatenate(mixture.symbolic_layers)
    np.testing.assert_array_equal(layer.states, np.arange(3))
    np.testing.assert_array_equal(np.exp(layer.table.log_probabilities)[2:, 2], [0, 0])


# %% queries agree with one variable per layer


def test_log_likelihood_of_mixed_leaves(mixture, samples):
    """
    The mixed circuit gives every sample the log-likelihood of the separate circuit.
    """
    np.testing.assert_allclose(
        mixture.mixed.log_likelihood(samples), mixture.separate.log_likelihood(samples)
    )


def test_probability_of_an_event_of_mixed_leaves(mixture, event):
    """
    The mixed circuit gives a box the probability of the separate circuit.
    """
    assert mixture.mixed.probability_of_simple_event(event) == pytest.approx(
        mixture.separate.probability_of_simple_event(event)
    )


def test_truncation_of_mixed_leaves(mixture, event, samples):
    """
    Truncating the mixed circuit gives the probability and the truncated distribution of
    the separate circuit.
    """
    mixed = mixture.mixed.log_truncated(event.as_composite_set())
    separate = mixture.separate.log_truncated(event.as_composite_set())
    assert mixed.log_probability == pytest.approx(separate.log_probability)
    np.testing.assert_allclose(
        mixed.distribution.log_likelihood(samples),
        separate.distribution.log_likelihood(samples),
    )


def test_conditioning_of_mixed_leaves(mixture, samples):
    """
    Conditioning the mixed circuit on every variable of a mixed leaf layer gives the
    likelihood and the conditional distribution of the separate circuit.
    """
    x, y, _, _ = mixture.variables
    point = {x: 0.3, y: -0.2}
    mixed = mixture.mixed.log_conditional(point)
    separate = mixture.separate.log_conditional(point)
    assert mixed.log_likelihood == pytest.approx(separate.log_likelihood)
    conditioned_samples = samples.copy()
    conditioned_samples[:, 0] = 0.3
    conditioned_samples[:, 1] = -0.2
    np.testing.assert_allclose(
        mixed.distribution.log_likelihood(conditioned_samples),
        separate.distribution.log_likelihood(conditioned_samples),
    )


def test_moment_of_mixed_leaves(mixture):
    """
    The mixed circuit has the means of the separate circuit.
    """
    x, y, _, _ = mixture.variables
    order = VariableMap({x: 1, y: 1})
    center = VariableMap({x: 0.0, y: 0.0})
    mixed = mixture.mixed.moment(order, center)
    separate = mixture.separate.moment(order, center)
    assert [mixed[x], mixed[y]] == pytest.approx([separate[x], separate[y]])


def test_support_of_mixed_leaves(mixture):
    """
    The mixed circuit has the support of the separate circuit.
    """
    assert mixture.mixed.support == mixture.separate.support


def test_samples_of_mixed_leaves_follow_the_circuit(mixture):
    """
    Samples of the mixed circuit have the means of the circuit.
    """
    x, y, _, _ = mixture.variables
    np.random.seed(1)
    samples = mixture.mixed.sample(20000)
    means = mixture.separate.moment(
        VariableMap({x: 1, y: 1}), VariableMap({x: 0.0, y: 0.0})
    )
    np.testing.assert_allclose(
        samples[:, :2].mean(axis=0), [means[x], means[y]], atol=0.05
    )


def test_a_marginal_that_splits_a_mixed_leaf_layer_raises(mixture):
    """
    Marginalizing a variable that a leaf layer shares with a kept one raises, since it
    would remove nodes its parents point to.
    """
    _, y, letter, mark = mixture.variables
    with pytest.raises(PartialMarginalOfLeafLayerError):
        mixture.mixed.marginal([y, letter, mark])
