from __future__ import annotations

import numpy as np
import plotly.graph_objects as go
import pytest
from random_events.interval import SimpleInterval, closed, closed_open, reals
from random_events.variable import Continuous
from scipy.special import logsumexp

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
    sorted_data = np.array([1, 2, 3, 4, 7, 9])
    weights = np.ones(len(sorted_data))
    return InductionStep(
        sorted_data,
        np.append(0, np.cumsum(weights)),
        np.append(0, np.cumsum(np.log(weights))),
        0,
        len(sorted_data),
        NygaInduction(
            variable,
            min_samples_per_quantile=1,
            min_likelihood_improvement=0.01,
            support=support,
        ),
    )


@pytest.fixture
def induction_step(x) -> InductionStep:
    """
    The first induction step over six data points without a constrained support.
    """
    return induction_step_over(x)


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
    assert induction_step.sum_weights() == pytest.approx(6)
    assert induction_step.sum_weights_from_indices(3, 5) == pytest.approx(2)


# %% splits


def split_side_log_likelihood(
    induction_step: InductionStep,
    split_index: int,
    connecting_point: float,
    is_left: bool,
) -> float:
    """
    :return: The log-likelihood of one side of the split at the index.
    """
    split_indices = np.array([split_index])
    split_values = (
        induction_step.data[split_indices - 1] + induction_step.data[split_indices]
    ) / 2
    return induction_step._vectorized_log_likelihood_of_split_side(
        split_indices, split_values, connecting_point, is_left
    )[0]


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
        induction_step, split_index, 1, is_left=True
    ) == pytest.approx(expected_left, abs=0.01)
    assert split_side_log_likelihood(
        induction_step, split_index, 9, is_left=False
    ) == pytest.approx(expected_right, abs=0.01)


def test_best_split(induction_step):
    _, index = induction_step.compute_best_split()
    assert index == 3


def test_best_split_of_too_few_samples_is_none(induction_step):
    induction_step.nyga_induction.min_samples_per_quantile = 4
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
    np.testing.assert_equal(
        side.cumulative_log_weights, induction_step.cumulative_log_weights
    )


# %% fitting


def test_fit_has_at_most_one_piece_per_quantile(induction_step):
    induction = induction_step.nyga_induction
    induction.min_samples_per_quantile = 20
    induction.min_likelihood_improvement = 0
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
    induction = NygaInduction(x, min_likelihood_improvement=0.01)
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
    induction = NygaInduction(
        x, min_likelihood_improvement=0.001, min_samples_per_quantile=300
    )
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
    induction = NygaInduction(
        x, min_samples_per_quantile=1, min_likelihood_improvement=0
    )
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
        min_samples_per_quantile=1,
        min_likelihood_improvement=0,
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
        min_samples_per_quantile=1,
        min_likelihood_improvement=0,
    )
    induction.fit([1.0, 2.0, 3.0])
    widest_leaf = max(
        induction.probabilistic_circuit.leaves,
        key=lambda leaf: leaf.distribution.interval.upper,
    )
    assert widest_leaf.distribution.interval.upper == pytest.approx(3.0000005, abs=1e-5)
