from __future__ import annotations

import itertools
from dataclasses import replace

import numpy as np
import plotly.graph_objects as go
import pytest
from random_events.interval import SimpleInterval, closed, closed_open, reals
from random_events.variable import Continuous
from scipy.special import logsumexp, rel_entr

from krrood.adapters.json_serializer import from_json, to_json
from probabilistic_model.distributions.distributions import DiracDeltaDistribution
from probabilistic_model.distributions.uniform import UniformDistribution
from probabilistic_model.learning.nyga_induction import InductionStep, NygaInduction
from probabilistic_model.probabilistic_circuit.rx.probabilistic_circuit import (
    ProbabilisticCircuit,
    SumUnit,
)

from ..circuit_building import uniform_leaf


def induction_step_over(
    variable: Continuous, support: SimpleInterval = reals().simple_sets[0]
) -> InductionStep:
    """
    :param variable: The variable of the induction.
    :param support: The allowed span of the support of the induced distribution.
    :return: The first induction step over the equally weighted data 1, 2, 3, 4, 7, 9.
    """
    induction = NygaInduction(variable, minimum_samples_per_quantile=1, support=support)
    return induction.initial_induction_step(np.array([1, 2, 3, 4, 7, 9]))


@pytest.fixture
def induction_step(x) -> InductionStep:
    """
    The first induction step over six data points without a constrained support.
    """
    return induction_step_over(x)


@pytest.fixture
def spiked_data() -> np.ndarray:
    """
    Fifty copies of 0 followed by the values 1 to 10, so the unique values carry very
    different weights.
    """
    return np.array([0.0] * 50 + list(np.arange(1.0, 11.0)))


@pytest.fixture
def spiked_induction_step(x, spiked_data) -> InductionStep:
    """
    The first induction step over :func:`spiked_data`.
    """
    induction = NygaInduction(x, minimum_samples_per_quantile=1)
    return induction.initial_induction_step(spiked_data)


# %% connecting points and pieces


def test_connecting_points_of_the_whole_data_are_its_ends(induction_step):
    assert induction_step.left_connecting_point() == 1
    assert induction_step.right_connecting_point() == 9


def test_connecting_points_lie_between_neighbouring_data(induction_step):
    assert induction_step.left_connecting_point_from_index(3) == 3.5
    assert induction_step.right_connecting_point_from_index(5) == 8.0


def test_a_single_piece_spans_the_unconstrained_support(induction_step, x):
    assert induction_step.create_uniform_distribution() == UniformDistribution(
        variable=x, interval=reals().simple_sets[0]
    )


def test_a_single_piece_is_clipped_to_a_given_support(x):
    support = closed(0.0, 5.0).simple_sets[0]
    distribution = induction_step_over(x, support).create_uniform_distribution()
    assert distribution.interval == support


@pytest.mark.parametrize(
    "support",
    [reals().simple_sets[0], closed(-10.0, 20.0).simple_sets[0]],
    ids=["unconstrained", "wider than the data"],
)
def test_an_internal_piece_spans_its_connecting_points(x, support):
    distribution = induction_step_over(
        x, support
    ).create_uniform_distribution_from_indices(3, 5)
    assert distribution == UniformDistribution(
        variable=x, interval=closed_open(3.5, 8.0).simple_sets[0]
    )


def test_sum_of_weights(induction_step):
    assert induction_step.total_weights == pytest.approx(6)
    assert induction_step.sum_weights_from_indices(3, 5) == pytest.approx(2)


# %% splits


def split_side_log_likelihood(
    induction_step: InductionStep, split_index: int, is_left: bool
) -> float:
    """
    :return: The log-likelihood of one side of the split at the index.
    """
    begin_index, end_index = (
        (induction_step.begin_index, split_index)
        if is_left
        else (split_index, induction_step.end_index)
    )
    return float(
        induction_step.log_likelihood_of_segments(
            np.array([begin_index]), np.array([end_index])
        )[0]
    )


def test_log_likelihood_without_split(induction_step):
    assert induction_step.log_likelihood_without_split() == pytest.approx(
        -12.48, abs=0.01
    )


@pytest.mark.parametrize(
    "split_index, expected_left, expected_right",
    [
        (1, -1.1, -10.99),
        (2, -3.01, -9.11),
        (3, -4.83, -7.19),
        (4, -7.64, -4.7),
        (5, -10.64, -1.79),
    ],
)
def test_log_likelihood_of_the_sides_of_a_split(
    induction_step, split_index, expected_left, expected_right
):
    assert split_side_log_likelihood(
        induction_step, split_index, is_left=True
    ) == pytest.approx(expected_left, abs=0.01)
    assert split_side_log_likelihood(
        induction_step, split_index, is_left=False
    ) == pytest.approx(expected_right, abs=0.01)


def test_best_split(induction_step):
    _, index = induction_step.compute_best_split()
    assert index == 3


def test_best_split_of_too_few_samples_is_none(induction_step):
    induction_step.nyga_induction.minimum_samples_per_quantile = 4
    maximum, index = induction_step.compute_best_split()
    assert index is None
    assert maximum == -float("inf")


def test_best_split_after_an_earlier_split(induction_step):
    induction_step.begin_index = 3
    _, index = induction_step.compute_best_split()
    assert index == 5


@pytest.mark.parametrize(
    "construct, begin_index, end_index",
    [
        (InductionStep.construct_left_induction_step, 0, 1),
        (InductionStep.construct_right_induction_step, 1, 6),
    ],
    ids=["left", "right"],
)
def test_induction_steps_of_the_sides_share_the_data(
    induction_step, construct, begin_index, end_index
):
    side = construct(induction_step, 1)
    assert side.begin_index == begin_index
    assert side.end_index == end_index
    np.testing.assert_equal(side.data, induction_step.data)
    np.testing.assert_equal(side.cumulative_weights, induction_step.cumulative_weights)
    assert side.minimum_split_gain == induction_step.minimum_split_gain


# %% fitting


def test_fit_has_at_most_one_piece_per_quantile(induction_step):
    induction = induction_step.nyga_induction
    induction.minimum_samples_per_quantile = 20
    induction.split_penalty_factor = 0
    data = np.random.normal(0, 1, 500).tolist()
    circuit = induction.fit(data)
    assert len(circuit.root.subcircuits) <= len(data) // 20
    assert logsumexp(circuit.root.log_weights) == pytest.approx(0.0)


def test_fit_support_spans_the_data(induction_step, x):
    data = np.random.normal(0, 1, 100).tolist()
    support = induction_step.nyga_induction.fit(data).support
    assert len(support.simple_sets) == 1
    intervals = support.simple_sets[0][x].simple_sets
    assert intervals[0].lower == pytest.approx(min(data), abs=1e-4)
    assert intervals[-1].upper == pytest.approx(max(data), abs=1e-4)


def test_fit_of_singular_data_is_a_dirac_delta(induction_step):
    circuit = induction_step.nyga_induction.fit([1.0, 1.0])
    assert len(circuit.nodes()) == 1
    assert isinstance(circuit.root.distribution, DiracDeltaDistribution)


def test_serialization_round_trip(x):
    induction = NygaInduction(x)
    induction.fit(np.random.normal(0, 1, 100).tolist())
    deserialized = from_json(to_json(induction))
    assert isinstance(deserialized, NygaInduction)
    assert deserialized == induction


def test_from_mixture_of_overlapping_uniform_distributions(x):
    circuit = ProbabilisticCircuit()
    mixture = SumUnit(probabilistic_circuit=circuit)
    mixture.add_subcircuit(uniform_leaf(x, 0, 5, circuit), np.log(0.5))
    mixture.add_subcircuit(uniform_leaf(x, 2, 3, circuit), np.log(0.5))
    distribution = NygaInduction.from_uniform_mixture(circuit)

    expected_pieces = [
        UniformDistribution(variable=x, interval=closed_open(0, 2).simple_sets[0]),
        UniformDistribution(variable=x, interval=closed_open(2, 3).simple_sets[0]),
        UniformDistribution(variable=x, interval=closed(3, 5).simple_sets[0]),
    ]
    assert {leaf.distribution for leaf in distribution.leaves} == set(expected_pieces)
    np.testing.assert_allclose(distribution.root.log_weights, np.log([0.2, 0.6, 0.2]))


# %% a fitted distribution


@pytest.fixture
def standard_normal_data() -> np.ndarray:
    """
    Sorted samples of a standard normal distribution in single precision.
    """
    return np.sort(np.random.normal(0.0, 1.0, 1000).astype(np.float32))


@pytest.fixture
def fitted_induction(x, standard_normal_data) -> NygaInduction:
    """
    An induction fitted to :func:`standard_normal_data`.
    """
    induction = NygaInduction(x, minimum_samples_per_quantile=300)
    induction.fit(standard_normal_data)
    return induction


def test_fitted_distribution_is_deterministic(fitted_induction):
    assert fitted_induction.probabilistic_circuit.is_deterministic()


def test_fitted_distribution_supports_its_data(fitted_induction, standard_normal_data):
    log_likelihood = fitted_induction.probabilistic_circuit.log_likelihood(
        standard_normal_data.reshape(-1, 1)
    )
    assert log_likelihood.shape == (1000,)
    assert log_likelihood.min() > -np.inf


def test_plot_of_a_fitted_distribution_has_traces(fitted_induction):
    assert len(go.Figure(fitted_induction.probabilistic_circuit.plot()).data) > 0


# %% support


def test_fit_narrows_the_default_support_to_data_range_widened_by_tolerance(x):
    induction = NygaInduction(x, minimum_samples_per_quantile=1, split_penalty_factor=0)
    induction.fit([1.0, 2.0, 3.0])
    # a simple interval stores its bounds in single precision, so the bounds pick up
    # rounding noise around the scale of the tolerance itself
    assert induction.support.lower == pytest.approx(
        1.0 - induction.tolerance_at_extremes, abs=1e-5
    )
    assert induction.support.upper == pytest.approx(
        3.0 + induction.tolerance_at_extremes, abs=1e-5
    )


def test_fit_narrows_a_given_support_no_further_than_the_tolerance_window(x):
    # wide enough that the tolerance window around the data sits inside it
    induction = NygaInduction(
        x,
        support=closed(0.0, 10.0).simple_sets[0],
        minimum_samples_per_quantile=1,
        split_penalty_factor=0,
    )
    induction.fit([1.0, 2.0, 3.0])
    assert induction.support.lower == pytest.approx(
        1.0 - induction.tolerance_at_extremes, abs=1e-5
    )
    assert induction.support.upper == pytest.approx(
        3.0 + induction.tolerance_at_extremes, abs=1e-5
    )


def test_fit_narrows_a_given_support_to_stay_inside_it(x):
    # narrower than the tolerance window around the data on the upper side
    induction = NygaInduction(
        x,
        support=closed(0.0, 3.0000005).simple_sets[0],
        minimum_samples_per_quantile=1,
        split_penalty_factor=0,
    )
    induction.fit([1.0, 2.0, 3.0])
    widest_leaf = max(
        induction.probabilistic_circuit.leaves,
        key=lambda leaf: leaf.distribution.interval.upper,
    )
    assert widest_leaf.distribution.interval.upper == pytest.approx(3.0000005, abs=1e-5)


# %% duplicate values weigh in with their count


def test_best_split_isolates_a_heavily_weighted_value(spiked_induction_step):
    _, index = spiked_induction_step.compute_best_split()
    assert index == 1


def test_gain_of_a_split_is_the_weighted_divergence_of_its_two_sides(
    spiked_induction_step,
):
    step = spiked_induction_step
    split_indices = np.arange(step.begin_index + 1, step.end_index)
    begin_indices = np.full_like(split_indices, step.begin_index)
    end_indices = np.full_like(split_indices, step.end_index)
    gains = (
        step.log_likelihood_of_segments(begin_indices, split_indices)
        + step.log_likelihood_of_segments(split_indices, end_indices)
        - step.log_likelihood_without_split()
    )

    left_end = step.right_connecting_point_from_index(split_indices)
    share_of_weight = step.sum_weights_from_indices(begin_indices, split_indices) / (
        step.total_weights
    )
    share_of_width = (left_end - step.left_connecting_point()) / (
        step.right_connecting_point() - step.left_connecting_point()
    )
    bernoulli_divergence = rel_entr(share_of_weight, share_of_width) + rel_entr(
        1 - share_of_weight, 1 - share_of_width
    )
    np.testing.assert_allclose(gains, step.total_weights * bernoulli_divergence)


def test_weights_of_duplicate_values_add_up(x):
    induction = NygaInduction(x, minimum_samples_per_quantile=1, split_penalty_factor=0)
    weighted = induction.initial_induction_step(
        np.array([2.0, 0.0, 0.0, 1.0]), np.array([1.0, 3.0, 1.0, 2.0])
    )
    expanded = induction.initial_induction_step(
        np.array([2.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0])
    )
    np.testing.assert_equal(weighted.data, expanded.data)
    np.testing.assert_allclose(weighted.cumulative_weights, expanded.cumulative_weights)


# %% stopping criterion


def test_minimum_split_gain_of_unit_weights_is_the_bayesian_information_criterion(
    x,
):
    induction = NygaInduction(x, split_penalty_factor=1.5)
    assert induction.minimum_split_gain(np.ones(10)) == pytest.approx(
        induction.split_penalty_factor * np.log(10)
    )


def test_minimum_split_gain_scales_with_the_weights(x):
    induction = NygaInduction(x)
    weights = np.random.uniform(0.1, 1.0, 50)
    assert induction.minimum_split_gain(7 * weights) == pytest.approx(
        7 * induction.minimum_split_gain(weights)
    )


@pytest.mark.parametrize("split_penalty_factor, expected", [(1.0, False), (0.0, True)])
def test_split_has_to_outweigh_the_penalty(x, split_penalty_factor, expected):
    induction = NygaInduction(
        x, minimum_samples_per_quantile=1, split_penalty_factor=split_penalty_factor
    )
    step = induction.initial_induction_step(np.array([1, 2, 3, 4, 7, 9]))
    maximum_log_likelihood, _ = step.compute_best_split()
    assert step.improvement_is_good_enough(maximum_log_likelihood) == expected


def test_fit_of_evenly_spaced_data_is_one_piece(x):
    induction = NygaInduction(x, minimum_samples_per_quantile=1)
    circuit = induction.fit(np.linspace(0.0, 1.0, 1000))
    assert len(circuit.root.subcircuits) == 1


# %% copies


def test_empty_copy_keeps_every_parameter(x):
    induction = NygaInduction(
        x,
        split_penalty_factor=2.0,
        minimum_samples_per_quantile=3,
        tolerance_at_extremes=1e-3,
        support=closed(0.0, 5.0).simple_sets[0],
        maximum_unique_values_for_optimal_partition=10,
    )
    induction.fit(np.random.uniform(1.0, 4.0, 50))
    copy = induction.empty_copy()
    assert copy == induction
    assert copy.probabilistic_circuit.nodes() == []


# %% globally optimal partition


def penalized_log_likelihood(step: InductionStep, boundaries: list[int]) -> float:
    """
    :return: The log likelihood of the partition of the step's data at the boundaries,
        minus the step's minimum split gain for every split.
    """
    log_likelihood = step.log_likelihood_of_segments(
        np.array(boundaries[:-1]), np.array(boundaries[1:])
    ).sum()
    return float(log_likelihood - step.minimum_split_gain * (len(boundaries) - 2))


def best_penalized_log_likelihood_by_enumeration(step: InductionStep) -> float:
    """
    :return: The highest :func:`penalized_log_likelihood` of every partition of the
        step's data whose segments each hold at least the minimal number of samples.
    """
    inner_indices = range(step.begin_index + 1, step.end_index)
    partitions = [
        [step.begin_index, *cuts, step.end_index]
        for number_of_cuts in range(len(inner_indices) + 1)
        for cuts in itertools.combinations(inner_indices, number_of_cuts)
    ]
    admissible = [
        boundaries
        for boundaries in partitions
        if len(boundaries) == 2
        or min(np.diff(boundaries)) >= step.minimum_samples_per_quantile
    ]
    return max(penalized_log_likelihood(step, boundaries) for boundaries in admissible)


@pytest.mark.parametrize("minimum_samples_per_quantile", [1, 2])
@pytest.mark.parametrize("split_penalty_factor", [0.0, 0.3, 1.0])
def test_optimal_segment_boundaries_maximize_the_penalized_likelihood(
    x, minimum_samples_per_quantile, split_penalty_factor
):
    induction = NygaInduction(
        x,
        minimum_samples_per_quantile=minimum_samples_per_quantile,
        split_penalty_factor=split_penalty_factor,
    )
    step = induction.initial_induction_step(np.random.randint(0, 13, 40).astype(float))
    boundaries = step.optimal_segment_boundaries()
    assert penalized_log_likelihood(step, boundaries) == pytest.approx(
        best_penalized_log_likelihood_by_enumeration(step)
    )


@pytest.fixture
def unevenly_spaced_data() -> np.ndarray:
    """
    Data on which splitting greedily misses the partition with the highest penalized
    likelihood at a split penalty factor of 0.1.
    """
    return np.array([2.0, 6.0, 10.0, 20.0, 30.0, 36.0, 37.0, 40.0, 50.0])


def fitted_pieces(induction: NygaInduction, data: np.ndarray) -> set:
    """
    :return: The uniform distributions of the circuit the induction fits to the data.
    """
    return {leaf.distribution for leaf in induction.fit(data).leaves}


def optimal_pieces(induction: NygaInduction, data: np.ndarray) -> set:
    """
    :return: The uniform distributions over the parts of the optimal partition of the
        data, within the support the induction narrowed while fitting.
    """
    step = induction.initial_induction_step(data)
    return {
        step.create_uniform_distribution_from_indices(begin_index, end_index)
        for begin_index, end_index in itertools.pairwise(
            step.optimal_segment_boundaries()
        )
    }


def test_fit_partitions_optimally_up_to_the_limit(x, unevenly_spaced_data):
    induction = NygaInduction(
        x,
        minimum_samples_per_quantile=1,
        split_penalty_factor=0.1,
        maximum_unique_values_for_optimal_partition=len(unevenly_spaced_data),
    )
    pieces = fitted_pieces(induction, unevenly_spaced_data)
    assert pieces == optimal_pieces(induction, unevenly_spaced_data)


def test_fit_splits_greedily_above_the_limit(x, unevenly_spaced_data):
    above_the_limit = NygaInduction(
        x,
        minimum_samples_per_quantile=1,
        split_penalty_factor=0.1,
        maximum_unique_values_for_optimal_partition=len(unevenly_spaced_data) - 1,
    )
    pieces = fitted_pieces(above_the_limit, unevenly_spaced_data)
    always_greedy = replace(
        above_the_limit, maximum_unique_values_for_optimal_partition=0
    )
    assert pieces == fitted_pieces(always_greedy, unevenly_spaced_data)
    assert pieces != optimal_pieces(above_the_limit, unevenly_spaced_data)


def test_optimally_partitioned_fit_is_a_deterministic_distribution(x):
    induction = NygaInduction(x, minimum_samples_per_quantile=5)
    data = np.concatenate(
        [np.random.normal(0.0, 1.0, 300), np.random.normal(6.0, 0.3, 300)]
    )
    circuit = induction.fit(data)
    assert len(data) <= induction.maximum_unique_values_for_optimal_partition
    assert circuit.is_deterministic()
    assert logsumexp(circuit.root.log_weights) == pytest.approx(0.0)
