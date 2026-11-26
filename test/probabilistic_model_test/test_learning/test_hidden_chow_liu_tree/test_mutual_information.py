"""
Tests of the estimators of the pairwise mutual information.
"""

from __future__ import annotations

# %% imports
import numpy as np
import pytest
from random_events.set import Set
from random_events.variable import Continuous, Symbolic
from sortedcontainers import SortedSet

from probabilistic_model.learning.structure.exceptions import (
    UnsupportedMutualInformationVariableError,
)
from probabilistic_model.learning.structure.mutual_information import (
    BinnedMutualInformation,
    GaussianMutualInformation,
    MutualInformationEstimator,
    NearestNeighbourMutualInformation,
)
from .dependent_bits import DependentBits


@pytest.fixture
def correlated_gaussians() -> tuple[SortedSet, np.ndarray, float]:
    """
    Two standard normal columns of correlation 0.8 and an independent third one, with
    the mutual information of the first two.
    """
    correlation = 0.8
    rows = np.random.default_rng(0).multivariate_normal(
        mean=np.zeros(3),
        cov=[[1.0, correlation, 0.0], [correlation, 1.0, 0.0], [0.0, 0.0, 1.0]],
        size=5000,
    )
    variables = SortedSet([Continuous("a"), Continuous("b"), Continuous("c")])
    return variables, rows, -0.5 * np.log(1.0 - correlation**2)


def copy_and_noise() -> tuple[np.ndarray, float]:
    """
    :return: Rows of a fair bit, its copy and an independent bit, and the entropy of
        the first bit.
    """
    generator = np.random.default_rng(0)
    column = generator.integers(0, 2, 20000).astype(float)
    rows = np.stack([column, column, generator.integers(0, 2, 20000)], axis=1)
    frequency = column.mean()
    entropy = -(frequency * np.log(frequency) + (1 - frequency) * np.log(1 - frequency))
    return rows, entropy


# %% binned


def test_continuous_column_with_many_values_is_cut_into_equally_populated_bins():
    """
    A continuous column of distinct values falls into bins of equally many rows.
    """
    levels = BinnedMutualInformation(number_of_bins=4).discretize_rows(
        variables=SortedSet([Continuous("a")]), data=np.arange(100.0)[:, None]
    )

    assert np.bincount(levels[:, 0]).tolist() == [25, 25, 25, 25]


def test_symbolic_column_keeps_a_level_per_state_however_many_states_it_has():
    """
    Every state of a symbolic variable is a level of its own, never a bin.
    """
    states = [str(index) for index in range(12)]
    variable = Symbolic(name="a", domain=Set.from_iterable(states))
    data = np.array([hash(state) for state in states], dtype=float)[:, None]

    levels = BinnedMutualInformation(number_of_bins=4).discretize_rows(
        variables=SortedSet([variable]), data=data
    )

    assert sorted(levels[:, 0].tolist()) == list(range(len(states)))


def test_binned_mutual_information_of_a_copy_is_its_entropy_and_of_noise_is_small():
    """
    A column shares its entropy with its copy and nearly nothing with noise.
    """
    rows, entropy = copy_and_noise()
    variables = SortedSet([Continuous("a"), Continuous("b"), Continuous("c")])

    mutual_information = BinnedMutualInformation(number_of_bins=2).estimate(
        variables=variables, data=rows
    )

    assert mutual_information[0, 1] == pytest.approx(entropy)
    assert mutual_information[0, 2] < 1e-3


# %% Gaussian


def test_gaussian_mutual_information_is_that_of_the_correlation(correlated_gaussians):
    """
    The estimate of correlated Gaussians is close to their closed form.
    """
    variables, rows, expected = correlated_gaussians

    mutual_information = GaussianMutualInformation().estimate(
        variables=variables, data=rows
    )

    assert mutual_information[0, 1] == pytest.approx(expected, abs=0.02)
    assert mutual_information[0, 2] < 0.01


def test_gaussian_mutual_information_of_a_perfect_correlation_is_finite():
    """
    A column and its copy share the largest finite information, not an infinite one.
    """
    column = np.random.default_rng(0).normal(size=100)
    variables = SortedSet([Continuous("a"), Continuous("b")])

    mutual_information = GaussianMutualInformation().estimate(
        variables=variables, data=np.stack([column, column], axis=1)
    )

    assert np.isfinite(mutual_information).all()


def test_gaussian_mutual_information_refuses_symbolic_variables():
    """
    The correlation of a symbolic variable is not defined.
    """
    bits = DependentBits()

    with pytest.raises(UnsupportedMutualInformationVariableError):
        GaussianMutualInformation().estimate(variables=bits.variables, data=bits.rows())


# %% nearest neighbours


def test_nearest_neighbour_mutual_information_of_gaussians_is_close_to_the_closed_form(
    correlated_gaussians,
):
    """
    The estimate of correlated Gaussians is close to their closed form.
    """
    variables, rows, expected = correlated_gaussians

    mutual_information = NearestNeighbourMutualInformation().estimate(
        variables=variables, data=rows
    )

    assert mutual_information[0, 1] == pytest.approx(expected, abs=0.05)
    assert mutual_information[0, 2] < 0.05


def test_nearest_neighbour_mutual_information_of_a_symbolic_copy_is_its_entropy():
    """
    Between symbolic variables the estimate is the empirical mutual information.
    """
    rows, entropy = copy_and_noise()
    variables = DependentBits(number_of_variables=4).variables
    hashes = DependentBits.hashes_of_bits(bits=rows.astype(int))

    mutual_information = NearestNeighbourMutualInformation().estimate(
        variables=SortedSet(list(variables)[:3]), data=hashes
    )

    assert mutual_information[0, 1] == pytest.approx(entropy)
    assert mutual_information[0, 2] < 1e-3


# %% every estimator


@pytest.mark.parametrize(
    "estimator",
    [
        BinnedMutualInformation(),
        GaussianMutualInformation(),
        NearestNeighbourMutualInformation(),
    ],
    ids=lambda estimator: type(estimator).__name__,
)
def test_every_estimate_is_symmetric_with_a_zero_diagonal(
    estimator: MutualInformationEstimator, correlated_gaussians
):
    """
    Every estimator returns a symmetric matrix that is zero on the diagonal.
    """
    variables, rows, _ = correlated_gaussians

    mutual_information = estimator.estimate(variables=variables, data=rows)

    np.testing.assert_allclose(mutual_information, mutual_information.T)
    np.testing.assert_array_equal(np.diag(mutual_information), 0.0)
