"""
Tests of Chow-Liu trees and the hidden Chow-Liu tree circuits built on them.
"""

from __future__ import annotations

# %% imports
import numpy as np
import pytest
from random_events.variable import Continuous, Integer
from scipy.special import logsumexp
from sortedcontainers import SortedSet

from probabilistic_model.learning.structure.exceptions import (
    UnsupportedStructureVariableError,
)
from probabilistic_model.learning.structure.hidden_chow_liu_tree.hidden_chow_liu_tree import (
    ChowLiuTree,
    HiddenChowLiuTree,
)
from probabilistic_model.learning.structure.mutual_information import (
    BinnedMutualInformation,
    GaussianMutualInformation,
    MutualInformationEstimator,
    NearestNeighbourMutualInformation,
)
from probabilistic_model.probabilistic_circuit.tensorized.inner_layer.sum_layer import (
    SumLayer,
)
from .dependent_bits import DependentBits


@pytest.fixture
def bits() -> DependentBits:
    """
    Six binary variables in three pairs that copy each other.
    """
    return DependentBits()


# %% Chow-Liu tree


@pytest.mark.parametrize(
    "estimator",
    [BinnedMutualInformation(), NearestNeighbourMutualInformation()],
    ids=lambda estimator: type(estimator).__name__,
)
def test_chow_liu_tree_joins_the_most_dependent_pairs(
    estimator: MutualInformationEstimator, bits
):
    """
    The Chow-Liu tree connects the pairs of variables that copy each other, whichever
    estimator of the mutual information supports symbolic variables.
    """
    mutual_information = estimator.estimate(variables=bits.variables, data=bits.rows())

    tree = ChowLiuTree.from_mutual_information(mutual_information=mutual_information)

    edges = {
        frozenset((child, int(parent)))
        for child, parent in enumerate(tree.parents)
        if parent >= 0
    }
    assert bits.copied_pairs <= edges
    assert len(edges) == bits.number_of_variables - 1


def test_chow_liu_tree_of_a_chain_is_the_chain_rooted_at_the_first_variable():
    """
    Mutual information that is high only between neighbours gives the chain, ordered
    from the first variable.
    """
    mutual_information = np.array(
        [
            [0.0, 0.9, 0.1, 0.1],
            [0.9, 0.0, 0.8, 0.1],
            [0.1, 0.8, 0.0, 0.7],
            [0.1, 0.1, 0.7, 0.0],
        ]
    )

    tree = ChowLiuTree.from_mutual_information(mutual_information=mutual_information)

    assert tree.parents.tolist() == [-1, 0, 1, 2]
    assert tree.order.tolist() == [0, 1, 2, 3]


# %% hidden Chow-Liu tree circuits


def test_hidden_chow_liu_tree_is_a_normalized_decomposable_circuit(bits):
    """
    A hidden Chow-Liu tree circuit is decomposable and sums to one over every
    assignment.
    """
    circuit = HiddenChowLiuTree(
        variables=bits.variables, hidden_states=3
    ).build_circuit(data=bits.rows())

    total = np.exp(logsumexp(circuit.log_likelihood(events=bits.every_assignment())))
    assert circuit.is_decomposable()
    assert total == pytest.approx(1.0)


def test_hidden_chow_liu_tree_has_one_mixture_of_the_hidden_states_at_its_root(bits):
    """
    The root mixes the hidden states of the root variable's latent variable.
    """
    hidden_states = 5
    circuit = HiddenChowLiuTree(
        variables=bits.variables, hidden_states=hidden_states
    ).build_circuit(data=bits.rows())

    assert isinstance(circuit.root, SumLayer)
    assert circuit.root.number_of_nodes == 1
    assert circuit.root.edges.number_of_edges == hidden_states


@pytest.mark.parametrize(
    "estimator",
    [
        BinnedMutualInformation(),
        GaussianMutualInformation(),
        NearestNeighbourMutualInformation(),
    ],
    ids=lambda estimator: type(estimator).__name__,
)
def test_hidden_chow_liu_tree_over_continuous_variables_has_finite_likelihoods(
    estimator: MutualInformationEstimator,
):
    """
    Continuous variables hang below their latent variables as Gaussians, the tree
    spanned over the mutual information of the estimator.
    """
    generator = np.random.default_rng(0)
    first = generator.normal(size=300)
    rows = np.stack(
        [
            first,
            first + generator.normal(scale=0.1, size=300),
            generator.normal(size=300),
        ],
        axis=1,
    )
    variables = SortedSet([Continuous("a"), Continuous("b"), Continuous("c")])

    circuit = HiddenChowLiuTree(
        variables=variables, hidden_states=4, mutual_information=estimator
    ).build_circuit(data=rows)

    assert circuit.is_decomposable()
    assert np.isfinite(circuit.log_likelihood(events=rows)).all()


def test_same_seed_gives_the_same_circuit_and_another_seed_another(bits):
    """
    The random initial parameters are drawn from the seed.
    """
    rows = bits.rows()

    def log_likelihoods(seed: int) -> np.ndarray:
        return (
            HiddenChowLiuTree(variables=bits.variables, hidden_states=3, seed=seed)
            .build_circuit(data=rows)
            .log_likelihood(events=rows)
        )

    np.testing.assert_array_equal(log_likelihoods(0), log_likelihoods(0))
    assert not np.allclose(log_likelihoods(0), log_likelihoods(1))


def test_variable_without_input_distribution_is_refused():
    """
    A variable that is neither symbolic nor continuous has no input distribution.
    """
    variables = SortedSet([Continuous("a"), Integer("b")])
    rows = np.random.default_rng(0).integers(0, 3, size=(50, 2)).astype(float)

    with pytest.raises(UnsupportedStructureVariableError):
        HiddenChowLiuTree(variables=variables, hidden_states=2).build_circuit(data=rows)
