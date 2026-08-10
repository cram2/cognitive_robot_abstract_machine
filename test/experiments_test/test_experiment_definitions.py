"""
Bounding a measured quantity, and turning experiment results into the table a scientific
article presents.

The rows here are built directly rather than measured, so each test pins one property of
the presentation -- which columns a row has, and how a value of a given type is written
-- independently of what any experiment happens to measure.
"""

from __future__ import annotations

import enum
import json
import pathlib
from dataclasses import dataclass

import pytest

from experiments.experiment_definitions import (
    ExperimentResult,
    ExperimentsTable,
    MeanAndStandardDeviation,
    PercentageBound,
    RowIsNotAnExperimentResult,
    RowsOfDifferingTypes,
    TypstRenderer,
    VolumeBound,
)


class MeasuredQuality(enum.Enum):
    """
    A property a row reports by name rather than by value.
    """

    FULLY_ESTABLISHED = "established by the experiment"
    NOT_ESTABLISHED = "left open by the experiment"


@dataclass
class NestedMeasurement(ExperimentResult):
    """
    A result reported as part of another result rather than as a row of its own.
    """

    trials: int
    """
    Number of trials the measurement was taken over.
    """


@dataclass
class MeasurementRow(ExperimentResult):
    """
    A row reporting a named quality, a quantity, and a quantity that may not have been
    established.
    """

    quality: MeasuredQuality
    """
    The quality the row reports.
    """

    score: float
    """
    The quantity measured.
    """

    unestablished_score: float | None
    """
    A quantity the experiment may have failed to establish.
    """


def rendered(*rows: ExperimentResult) -> str:
    """
    :param rows: The results to present.
    :return: The Typst markup presenting them as a table.
    """
    return TypstRenderer(ExperimentsTable(list(rows))).render_table()


def row(
    quality: MeasuredQuality = MeasuredQuality.FULLY_ESTABLISHED,
    score: float = 1.0,
    unestablished_score: float | None = 1.0,
) -> MeasurementRow:
    """
    :param quality: The quality the row reports.
    :param score: The quantity measured.
    :param unestablished_score: The quantity that may not have been established.
    :return: A row reporting those values.
    """
    return MeasurementRow(
        quality=quality, score=score, unestablished_score=unestablished_score
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


# %% which columns a row has


def test_quantity_that_may_be_unestablished_is_a_column():
    """
    An experiment that cannot establish a quantity still reports the column, so a result
    is free to say a measurement is missing rather than inventing a value for it.
    """
    assert MeasurementRow.get_column_names() == [
        "quality",
        "score",
        "unestablished_score",
    ]


def test_result_reported_within_another_contributes_its_own_columns():
    """
    A result held by another is presented as part of the same row, so a table stays flat
    however the results are composed.
    """

    @dataclass
    class ComposedRow(ExperimentResult):
        """
        A row holding another result.
        """

        nested: NestedMeasurement
        """
        The result reported as part of this one.
        """

        score: float
        """
        The quantity measured.
        """

    assert ComposedRow.get_column_names() == ["trials", "score"]


# %% how a value is written


def test_named_quality_is_written_as_a_label():
    """
    A reader of the table sees the quality's name, not the notation the experiment
    happens to hold it in.
    """
    assert "[Fully Established]" in rendered(row())


def test_quantity_is_written_to_two_decimals():
    """
    A measured quantity is reported at the precision a reader can act on, rather than at
    the precision the arithmetic happened to produce.
    """
    assert "[75.71]" in rendered(row(score=75.70977917981072))


def test_unestablished_quantity_is_written_as_absent():
    """
    A quantity the experiment did not establish is marked absent, so it is never read as
    a measurement that came out at zero.
    """
    markup = rendered(row(unestablished_score=None))

    assert "[--]" in markup
    assert "[0.0]" not in markup


def test_table_presented_to_a_reader_carries_its_caption():
    """
    Every table a reader sees explains what it shows, so the figure holds the caption
    around the same table the renderer produces.
    """
    renderer = TypstRenderer(ExperimentsTable([row()]))

    figure = renderer.render_figure("What the experiment measured.")

    assert renderer.render_table() in figure
    assert "caption: [What the experiment measured.]" in figure


def test_count_keeps_its_exact_value():
    """
    A count is exact, so it is written as it is rather than as a rounded quantity.
    """
    assert "[7]" in rendered(NestedMeasurement(trials=7))


# %% recording the results alongside the table


def test_manifest_records_every_row_in_a_readable_form(tmp_path: pathlib.Path):
    """
    The manifest records what the table presents, with values written in a form that
    survives being read back rather than in the notation the experiment held them in.
    """
    table = ExperimentsTable(
        [row(score=0.5), row(quality=MeasuredQuality.NOT_ESTABLISHED)]
    )

    manifest_path = table.write_manifest(tmp_path, "results.json")

    recorded = json.loads(manifest_path.read_text())
    assert [each["quality"] for each in recorded] == [
        "FULLY_ESTABLISHED",
        "NOT_ESTABLISHED",
    ]
    assert recorded[0]["score"] == 0.5


def test_manifest_records_a_nested_result_flat(tmp_path: pathlib.Path):
    """
    The manifest records the columns the table presents, so a result reported within another
    appears under its own columns rather than nested inside the field holding it.
    """

    @dataclass
    class ComposedRow(ExperimentResult):
        """
        A row holding another result.
        """

        nested: NestedMeasurement
        """
        The result reported as part of this one.
        """

        score: float
        """
        The quantity measured.
        """

    manifest_path = ExperimentsTable(
        [ComposedRow(nested=NestedMeasurement(trials=7), score=0.5)]
    ).write_manifest(tmp_path, "results.json")

    [recorded] = json.loads(manifest_path.read_text())
    assert recorded == {"trials": 7, "score": 0.5}


def test_manifest_keeps_an_unestablished_measurement_distinguishable(
    tmp_path: pathlib.Path,
):
    """
    A measurement the experiment did not establish is recorded as absent, so reading the
    manifest back never turns it into a value that was measured.
    """
    manifest_path = ExperimentsTable([row(unestablished_score=None)]).write_manifest(
        tmp_path, "results.json"
    )

    [recorded] = json.loads(manifest_path.read_text())
    assert recorded["unestablished_score"] is None


def test_manifest_records_a_measurement_reported_as_a_spread(tmp_path: pathlib.Path):
    """
    A value a row reports through a class of its own is recorded by its parts, so the
    manifest holds what was measured rather than the notation it was held in.
    """

    @dataclass
    class SpreadRow(ExperimentResult):
        """
        A row reporting a measurement as a mean and a spread around it.
        """

        duration: MeanAndStandardDeviation
        """
        The measurement and how much it varied.
        """

    manifest_path = ExperimentsTable(
        [
            SpreadRow(
                duration=MeanAndStandardDeviation(mean=1.5, standard_deviation=0.25)
            )
        ]
    ).write_manifest(tmp_path, "results.json")

    [recorded] = json.loads(manifest_path.read_text())
    assert recorded["duration"] == {"mean": 1.5, "standard_deviation": 0.25}


def test_manifest_records_several_measurements_of_one_column(tmp_path: pathlib.Path):
    """
    A column holding several measurements records each of them, so a row reporting a
    series is read back as that series.
    """

    @dataclass
    class SeriesRow(ExperimentResult):
        """
        A row reporting the measurements a trial produced.
        """

        durations: list[float]
        """
        Every measurement taken.
        """

    manifest_path = ExperimentsTable([SeriesRow(durations=[0.5, 1.5])]).write_manifest(
        tmp_path, "results.json"
    )

    [recorded] = json.loads(manifest_path.read_text())
    assert recorded["durations"] == [0.5, 1.5]


# %% tables the renderer refuses


def test_rows_of_differing_types_are_refused():
    """
    A table's columns come from its row type, so rows of different types have no common
    set of columns and are refused rather than presented under the first row's headers.
    """
    with pytest.raises(RowsOfDifferingTypes):
        ExperimentsTable([row(), NestedMeasurement(trials=1)])


def test_rows_that_are_not_results_are_refused():
    """
    A row is refused when it reports no columns at all, rather than failing later while
    the table is being written.
    """
    with pytest.raises(RowIsNotAnExperimentResult):
        ExperimentsTable([MeasuredQuality.FULLY_ESTABLISHED])
