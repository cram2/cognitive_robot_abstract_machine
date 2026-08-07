import pytest

from experiments.experiment_definitions import (
    IncompatibleUnitConversionError,
    MeanAndStandardDeviation,
    PercentageBound,
    Unit,
    VolumeBound,
)

# %% MeanAndStandardDeviation


def test_str_has_no_suffix_for_an_untagged_unit():
    value = MeanAndStandardDeviation.from_measurements([1.0, 3.0])

    assert str(value) == "2.0 ± 1.41"


def test_str_appends_the_unit_suffix():
    value = MeanAndStandardDeviation.from_measurements([1.0, 3.0], unit=Unit.SECONDS)

    assert str(value) == "2.0 ± 1.41 s"


def test_converting_seconds_to_milliseconds_scales_by_a_thousand():
    seconds = MeanAndStandardDeviation.from_measurements(
        [0.01, 0.03], unit=Unit.SECONDS
    )

    milliseconds = seconds.to(Unit.MILLISECONDS)

    assert milliseconds.mean == seconds.mean * 1000
    assert milliseconds.standard_deviation == seconds.standard_deviation * 1000
    assert milliseconds.unit is Unit.MILLISECONDS


def test_converting_an_untagged_value_is_rejected():
    value = MeanAndStandardDeviation.from_measurements([1.0, 3.0])

    with pytest.raises(IncompatibleUnitConversionError):
        value.to(Unit.MILLISECONDS)


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
