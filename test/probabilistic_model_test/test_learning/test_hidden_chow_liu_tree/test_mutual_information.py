"""
Tests of the pairwise mutual information of discretized data.
"""

from __future__ import annotations

# %% imports
import numpy as np
import pytest
from random_events.set import Set
from random_events.variable import Continuous, Symbolic
from sortedcontainers import SortedSet

from probabilistic_model.learning.structure.mutual_information import (
    PairwiseMutualInformation,
)

# %% discretization


def test_continuous_column_with_many_values_is_cut_into_equally_populated_bins():
    """
    A continuous column of distinct values falls into bins of equally many rows.
    """
    mutual_information = PairwiseMutualInformation.from_rows(
        variables=SortedSet([Continuous("a")]),
        data=np.arange(100.0)[:, None],
        number_of_bins=4,
    )

    assert np.bincount(mutual_information.levels[:, 0]).tolist() == [25, 25, 25, 25]


def test_symbolic_column_keeps_a_level_per_state_however_many_states_it_has():
    """
    Every state of a symbolic variable is a level of its own, never a bin.
    """
    states = [str(index) for index in range(12)]
    variable = Symbolic(name="a", domain=Set.from_iterable(states))
    data = np.array([hash(state) for state in states], dtype=float)[:, None]

    mutual_information = PairwiseMutualInformation.from_rows(
        variables=SortedSet([variable]), data=data, number_of_bins=4
    )

    assert sorted(mutual_information.levels[:, 0].tolist()) == list(range(len(states)))


# %% mutual information


def test_mutual_information_of_a_copy_is_its_entropy_and_of_noise_is_small():
    """
    A column shares its entropy with its copy and nearly nothing with noise.
    """
    generator = np.random.default_rng(0)
    column = generator.integers(0, 2, 20000).astype(float)
    data = np.stack([column, column, generator.integers(0, 2, 20000)], axis=1)
    variables = SortedSet([Continuous("a"), Continuous("b"), Continuous("c")])

    mutual_information = PairwiseMutualInformation.from_rows(
        variables=variables, data=data, number_of_bins=2
    ).compute()

    frequency = column.mean()
    entropy = -(frequency * np.log(frequency) + (1 - frequency) * np.log(1 - frequency))
    assert mutual_information[0, 1] == pytest.approx(entropy)
    assert mutual_information[0, 2] < 1e-3
    np.testing.assert_allclose(mutual_information, mutual_information.T)
    np.testing.assert_array_equal(np.diag(mutual_information), 0.0)
