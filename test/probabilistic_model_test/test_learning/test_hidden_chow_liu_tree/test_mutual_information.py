"""
Tests of the pairwise mutual information of discretized data.
"""

from __future__ import annotations

# %% imports
import numpy as np
import pytest
from random_events.variable import Continuous
from sortedcontainers import SortedSet

from probabilistic_model.learning.structure.mutual_information import (
    DiscretizedData,
    PairwiseMutualInformation,
)

# %% discretization


def test_continuous_column_is_cut_into_equally_populated_bins():
    """
    A continuous column of distinct values falls into bins of equally many rows.
    """
    discretized = DiscretizedData(
        variables=SortedSet([Continuous("a")]),
        data=np.arange(100.0)[:, None],
        number_of_bins=4,
    )

    assert discretized.number_of_states.tolist() == [4]
    assert np.bincount(discretized.states[:, 0]).tolist() == [25, 25, 25, 25]


# %% mutual information


def test_mutual_information_of_a_copy_is_its_entropy_and_of_noise_is_small():
    """
    A column shares its entropy with its copy and nearly nothing with noise.
    """
    generator = np.random.default_rng(0)
    column = generator.integers(0, 2, 20000).astype(float)
    data = np.stack([column, column, generator.integers(0, 2, 20000)], axis=1)
    variables = SortedSet([Continuous("a"), Continuous("b"), Continuous("c")])

    mutual_information = PairwiseMutualInformation(
        discretized=DiscretizedData(variables=variables, data=data, number_of_bins=2)
    ).compute()

    frequency = column.mean()
    entropy = -(frequency * np.log(frequency) + (1 - frequency) * np.log(1 - frequency))
    assert mutual_information[0, 1] == pytest.approx(entropy)
    assert mutual_information[0, 2] < 1e-3
    np.testing.assert_allclose(mutual_information, mutual_information.T)
    np.testing.assert_array_equal(np.diag(mutual_information), 0.0)
