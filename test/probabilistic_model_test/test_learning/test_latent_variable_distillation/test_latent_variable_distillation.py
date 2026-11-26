"""
Tests of latent variable distillation: a masked autoencoder assigns the latent variables
of a hidden Chow-Liu tree circuit, whose parameters are counted from them.
"""

from __future__ import annotations

# %% imports
import numpy as np
import pytest
import torch
from random_events.variable import Continuous
from scipy.special import logsumexp
from sortedcontainers import SortedSet

from probabilistic_model.learning.structure.hidden_chow_liu_tree.hidden_chow_liu_tree import (
    HiddenChowLiuTree,
)
from probabilistic_model.learning.structure.latent_variable_distillation.latent_variable_distillation import (
    LatentVariableDistillation,
    MaskedAutoencoderTraining,
    TabularMaskedAutoencoder,
)
from ..test_hidden_chow_liu_tree.dependent_bits import DependentBits


@pytest.fixture
def bits() -> DependentBits:
    """
    Six binary variables in three pairs that copy each other.
    """
    return DependentBits()


@pytest.fixture
def short_training() -> MaskedAutoencoderTraining:
    """
    A small autoencoder trained for a few epochs.
    """
    return MaskedAutoencoderTraining(
        width=16, layers=1, heads=2, feedforward_width=32, epochs=3
    )


def distillation_of(
    *, bits: DependentBits, training: MaskedAutoencoderTraining, seed: int = 0
) -> LatentVariableDistillation:
    """
    :param bits: The dataset.
    :param training: The training of the autoencoder.
    :param seed: The seed of the structure and the autoencoder.
    :return: The distillation into a hidden Chow-Liu tree with three hidden states.
    """
    return LatentVariableDistillation(
        structure=HiddenChowLiuTree(
            variables=bits.variables, hidden_states=3, seed=seed
        ),
        autoencoder=training,
    )


# %% masked autoencoder


def test_autoencoder_encodes_one_token_per_variable():
    """
    The encoder gives every variable of every row one token of the embedding width.
    """
    autoencoder = TabularMaskedAutoencoder(
        number_of_states=[2, 0, 3], width=8, layers=1, heads=2, feedforward_width=16
    )
    values = torch.tensor([[0.0, 0.5, 2.0], [1.0, -1.0, 0.0]])

    embeddings = autoencoder.embeddings(
        values=values, masked=torch.zeros_like(values, dtype=torch.bool)
    )

    assert embeddings.shape == (2, 3, 8)


def test_training_lowers_the_reconstruction_loss(bits, short_training):
    """
    Training the autoencoder makes it reconstruct hidden values better.
    """
    distillation = distillation_of(bits=bits, training=short_training)
    values = distillation.autoencoder_inputs(data=bits.rows())
    masked = torch.rand(values.shape, generator=torch.Generator().manual_seed(1)) < 0.5
    untrained = short_training.untrained_autoencoder(
        number_of_states=distillation.number_of_states, seed=0
    )
    untrained_loss = untrained.reconstruction_loss(values=values, masked=masked).item()

    trained = short_training.trained_autoencoder(
        values=values, number_of_states=distillation.number_of_states, seed=0
    )

    trained.eval()
    with torch.no_grad():
        trained_loss = trained.reconstruction_loss(values=values, masked=masked).item()
    assert trained_loss < untrained_loss


# %% inputs of the autoencoder


def test_symbolic_values_become_state_indices_and_continuous_ones_are_standardized():
    """
    The autoencoder reads the state index of a symbolic value and the standard score of
    a continuous one.
    """
    bits = DependentBits(number_of_variables=2)
    continuous = np.array([1.0, 3.0, 5.0, 7.0])
    states = np.array([0, 1, 1, 0])
    variables = SortedSet([*bits.variables, Continuous("y")])
    data = np.column_stack(
        [bits.hashes_of_bits(bits=states), bits.hashes_of_bits(bits=states), continuous]
    )
    distillation = LatentVariableDistillation(
        structure=HiddenChowLiuTree(variables=variables, hidden_states=2)
    )

    values = distillation.autoencoder_inputs(data=data).numpy()

    np.testing.assert_array_equal(values[:, 0], states)
    np.testing.assert_allclose(
        values[:, 2], (continuous - continuous.mean()) / continuous.std(), rtol=1e-6
    )


# %% latent states and circuits


def test_latent_states_hold_one_cluster_per_row_and_variable(bits, short_training):
    """
    Every variable of every row is assigned one of the hidden states.
    """
    distillation = distillation_of(bits=bits, training=short_training)
    rows = bits.rows()

    latent_states = distillation.latent_states(data=rows)

    assert latent_states.shape == rows.shape
    assert set(np.unique(latent_states)) <= set(
        range(distillation.structure.hidden_states)
    )


def test_distilled_circuit_is_a_normalized_decomposable_circuit(bits, short_training):
    """
    The distilled hidden Chow-Liu tree circuit sums to one over every assignment.
    """
    circuit = distillation_of(bits=bits, training=short_training).build_circuit(
        data=bits.rows()
    )

    total = np.exp(logsumexp(circuit.log_likelihood(events=bits.every_assignment())))
    assert circuit.is_decomposable()
    assert total == pytest.approx(1.0)


def test_same_seed_distills_the_same_latent_states(bits, short_training):
    """
    The autoencoder and the clustering are drawn from the structure's seed.
    """
    rows = bits.rows()

    first = distillation_of(bits=bits, training=short_training).latent_states(data=rows)
    second = distillation_of(bits=bits, training=short_training).latent_states(
        data=rows
    )

    np.testing.assert_array_equal(first, second)


def test_clustering_on_fewer_rows_still_assigns_every_row(bits, short_training):
    """
    The clusters are fitted on a sample of the rows and every row is assigned to one.
    """
    distillation = distillation_of(bits=bits, training=short_training)
    distillation.clustering_rows = 50
    rows = bits.rows()

    latent_states = distillation.latent_states(data=rows)

    assert latent_states.shape == rows.shape
