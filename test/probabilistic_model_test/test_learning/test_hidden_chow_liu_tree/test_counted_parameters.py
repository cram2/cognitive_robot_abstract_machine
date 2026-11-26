"""
Tests of the initial parameters of hidden Chow-Liu tree circuits that are counted from
rows completed with the state of every latent variable.
"""

from __future__ import annotations

# %% imports
import numpy as np
import pytest
from random_events.set import Set
from random_events.variable import Continuous, Symbolic
from scipy.special import logsumexp
from sortedcontainers import SortedSet

from probabilistic_model.learning.structure.exceptions import (
    LatentStatesMismatchError,
)
from probabilistic_model.learning.structure.hidden_chow_liu_tree.hidden_chow_liu_tree import (
    CountedParameters,
    HiddenChowLiuTree,
)
from probabilistic_model.probabilistic_circuit.tensorized.input_layer.gaussian_layer import (
    GaussianLayer,
)
from .dependent_bits import DependentBits


@pytest.fixture
def bits() -> DependentBits:
    """
    Six binary variables in three pairs that copy each other.
    """
    return DependentBits()


@pytest.fixture
def two_bits() -> DependentBits:
    """
    Two binary variables, the second copying the first.
    """
    return DependentBits(number_of_variables=2)


# %% counted symbolic probabilities and transitions


def test_symbolic_probabilities_are_the_smoothed_frequencies_under_each_latent_state(
    two_bits,
):
    """
    Each latent state's categorical distribution is the frequency of every value among
    the rows in that state, every count raised by the pseudocount.
    """
    data = two_bits.hashes_of_bits(bits=np.array([[0, 0], [0, 1], [1, 1], [1, 1]]))
    parameters = CountedParameters(
        variables=two_bits.variables,
        data=data,
        hidden_states=2,
        latent_states=np.array([[0, 0], [0, 0], [1, 1], [1, 0]]),
        pseudocount=1.0,
    )

    probabilities = parameters.symbolic_probabilities(variable_index=1)

    np.testing.assert_allclose(probabilities, [[2 / 5, 3 / 5], [1 / 3, 2 / 3]])


def test_transition_weights_count_the_pairs_of_parent_and_child_states(two_bits):
    """
    The weight of a transition is the number of rows in which the parent's latent
    variable has its state and the child's has its own, plus the pseudocount.
    """
    parameters = CountedParameters(
        variables=two_bits.variables,
        data=two_bits.rows()[:4],
        hidden_states=2,
        latent_states=np.array([[0, 0], [0, 1], [1, 1], [1, 1]]),
        pseudocount=0.5,
    )

    np.testing.assert_allclose(
        parameters.transition_weights(parent=0, child=1), [[1.5, 1.5], [0.5, 2.5]]
    )
    np.testing.assert_allclose(
        parameters.transition_weights(parent=None, child=0), [[2.5, 2.5]]
    )


# %% counted Gaussians


def test_gaussians_are_the_moments_of_the_rows_in_each_latent_state():
    """
    Each latent state's Gaussian has the mean and standard deviation of the rows in that
    state.
    """
    column = np.array([1.0, 3.0, 10.0, 14.0])
    parameters = CountedParameters(
        variables=SortedSet([Continuous("a")]),
        data=column[:, None],
        hidden_states=2,
        latent_states=np.array([[0], [0], [1], [1]]),
    )

    gaussians = parameters.gaussian_parameters(variable_index=0)

    np.testing.assert_allclose(gaussians.location, [2.0, 12.0])
    np.testing.assert_allclose(gaussians.scale, [1.0, 2.0])


def test_a_latent_state_without_rows_takes_the_moments_of_the_whole_column():
    """
    A latent state that no row is in gets the mean and standard deviation of the column,
    and a state with one row the column's standard deviation.
    """
    column = np.array([1.0, 3.0, 8.0])
    parameters = CountedParameters(
        variables=SortedSet([Continuous("a")]),
        data=column[:, None],
        hidden_states=3,
        latent_states=np.array([[0], [0], [1]]),
    )

    gaussians = parameters.gaussian_parameters(variable_index=0)

    np.testing.assert_allclose(gaussians.location, [2.0, 8.0, column.mean()])
    np.testing.assert_allclose(gaussians.scale, [1.0, column.std(), column.std()])


def test_a_latent_state_with_equal_values_keeps_the_minimum_scale():
    """
    The rows of a state that share one value give the smallest scale a Gaussian layer
    allows.
    """
    parameters = CountedParameters(
        variables=SortedSet([Continuous("a")]),
        data=np.array([[2.0], [2.0], [5.0], [7.0]]),
        hidden_states=2,
        latent_states=np.array([[0], [0], [1], [1]]),
    )

    gaussians = parameters.gaussian_parameters(variable_index=0)

    assert gaussians.scale[0] == GaussianLayer.minimum_scale


# %% misuse


@pytest.mark.parametrize(
    "latent_states",
    [np.zeros((4, 1), dtype=int), np.zeros((3, 2), dtype=int), np.full((4, 2), 2)],
    ids=["too few variables", "too few rows", "state out of range"],
)
def test_latent_states_that_do_not_fit_the_rows_are_refused(
    latent_states: np.ndarray, two_bits
):
    """
    The latent states need one state below the number of hidden states per row and
    variable.
    """
    with pytest.raises(LatentStatesMismatchError):
        CountedParameters(
            variables=two_bits.variables,
            data=two_bits.rows()[:4],
            hidden_states=2,
            latent_states=latent_states,
        )


# %% circuits from counted parameters


def test_counted_circuit_is_a_normalized_decomposable_circuit(bits):
    """
    A hidden Chow-Liu tree circuit counted from latent states sums to one over every
    assignment.
    """
    rows = bits.rows()
    latent_states = np.random.default_rng(0).integers(0, 3, size=rows.shape)

    circuit = HiddenChowLiuTree(
        variables=bits.variables, hidden_states=3
    ).build_circuit(data=rows, latent_states=latent_states)

    total = np.exp(logsumexp(circuit.log_likelihood(events=bits.every_assignment())))
    assert circuit.is_decomposable()
    assert total == pytest.approx(1.0)


def test_latent_states_that_copy_the_values_give_the_empirical_distribution():
    """
    When every latent variable's state is its variable's value, the counted circuit of a
    single variable reproduces the smoothed frequency of every value.
    """
    variables = SortedSet(
        [Symbolic(name="x", domain=Set.from_iterable(["0", "1", "2"]))]
    )
    states = np.array([0, 0, 0, 1, 2, 2])
    data = np.array([hash("0"), hash("1"), hash("2")], dtype=float)[states][:, None]
    pseudocount = 1.0

    circuit = HiddenChowLiuTree(
        variables=variables, hidden_states=3, pseudocount=pseudocount
    ).build_circuit(data=data, latent_states=states[:, None])

    state_counts = np.bincount(states, minlength=3)
    root_weights = state_counts + pseudocount
    value_probabilities = np.diag(state_counts) + pseudocount
    value_probabilities /= value_probabilities.sum(axis=1, keepdims=True)
    expected = root_weights / root_weights.sum() @ value_probabilities
    np.testing.assert_allclose(
        np.exp(circuit.log_likelihood(events=data[[0, 3, 4]])), expected
    )
