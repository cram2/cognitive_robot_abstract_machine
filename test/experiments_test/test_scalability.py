import statistics

import plotly.graph_objects as go
import pytest

from experiments.experiment_definitions import (
    ExperimentsTable,
    MeanAndStandardDeviation,
)
from experiments.ormatic_experiments.scalability import (
    ORMaticScalabilityAggregateResult,
    build_cram_class_sets,
    ormatic_scalability_experiment,
    plot_scalability,
    run_scalability_experiment,
)


@pytest.fixture(scope="module")
def small_class_set():
    all_classes, alternative_mappings, type_mappings = build_cram_class_sets()
    mandatory = {am.original_class() for am in alternative_mappings}
    extras = set(list(all_classes - mandatory)[:5])
    return mandatory | extras, alternative_mappings, type_mappings


def test_experiment_timings_are_non_negative(small_class_set):
    classes, alternative_mappings, type_mappings = small_class_set
    result = ormatic_scalability_experiment(
        classes, alternative_mappings, type_mappings
    )
    assert result.total_duration >= 0
    assert result.class_diagram_creation_duration >= 0
    assert result.ormatic_reasoning_duration >= 0
    assert result.writing_to_file_duration >= 0


def _make_table() -> ExperimentsTable:
    rows = [
        ORMaticScalabilityAggregateResult(
            class_drop_probability=p,
            number_of_classes=MeanAndStandardDeviation(10 + i * 5, 1.0),
            number_of_associations=MeanAndStandardDeviation(5 + i * 2, 0.5),
            number_of_inheritances=MeanAndStandardDeviation(3 + i, 0.3),
            total_duration=MeanAndStandardDeviation(0.1 + i * 0.05, 0.01),
            class_diagram_creation_duration=MeanAndStandardDeviation(0.02, 0.001),
            ormatic_reasoning_duration=MeanAndStandardDeviation(0.05, 0.005),
            writing_to_file_duration=MeanAndStandardDeviation(0.03, 0.002),
        )
        for i, p in enumerate([0.0, 0.3, 0.6])
    ]
    return ExperimentsTable(rows)


def test_plot_returns_figure():
    assert isinstance(plot_scalability(_make_table()), go.Figure)
