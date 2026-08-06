import pytest

from experiments.experiment_definitions import (
    PercentageBound,
    VolumeBound,
)

# %% PercentageBound


def test_ratio_of_pairs_worst_case_ends():
    numerator = VolumeBound(lower=8.0, upper=10.0)
    denominator = VolumeBound(lower=20.0, upper=40.0)

    bound = PercentageBound.ratio_of(numerator, denominator)

    # lower: smallest numerator over largest denominator; upper: largest numerator
    # over smallest denominator.
    assert bound.lower == pytest.approx(100.0 * 8.0 / 40.0)
    assert bound.upper == pytest.approx(100.0 * 10.0 / 20.0)


def test_ratio_of_clips_at_one_hundred_percent():
    numerator = VolumeBound(lower=9.0, upper=10.0)
    denominator = VolumeBound(lower=9.0, upper=10.0)

    bound = PercentageBound.ratio_of(numerator, denominator)

    assert bound.upper == 100.0


def test_ratio_of_a_fully_covered_exact_match_is_exactly_one_hundred_percent():
    exact = VolumeBound(lower=5.0, upper=5.0)

    bound = PercentageBound.ratio_of(exact, exact)

    assert bound.lower == pytest.approx(100.0)
    assert bound.upper == pytest.approx(100.0)
