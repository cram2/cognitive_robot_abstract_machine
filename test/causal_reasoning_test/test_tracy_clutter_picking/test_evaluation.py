"""
Tests for the whole comparison and its Markdown rendering, on synthetic attempts.
"""

from __future__ import annotations

import numpy as np
import pytest

from experiments.causal_reasoning.tracy_clutter_picking.dataset import ClutterPickDataset
from experiments.causal_reasoning.tracy_clutter_picking.evaluation import (
    Refusal,
    describe_region,
    evaluate,
)
from experiments.causal_reasoning.tracy_clutter_picking.queries import (
    CrowdingCausesLift,
    FrictionCausesLift,
    query_catalogue,
)
from experiments.causal_reasoning.tracy_clutter_picking.report import (
    MarkdownReport,
    Verdict,
)
from experiments.causal_reasoning.tracy_clutter_picking.synthetic import (
    synthetic_clutter_pick_scenes,
)
from random_events.interval import closed
from random_events.product_algebra import SimpleEvent
from random_events.set import Set
from random_events.variable import Continuous, Symbolic

RECORDED_NEIGHBOUR_COUNT = 5


@pytest.fixture(scope="module")
def cases():
    return [
        FrictionCausesLift(RECORDED_NEIGHBOUR_COUNT),
        CrowdingCausesLift(RECORDED_NEIGHBOUR_COUNT + 2),
    ]


@pytest.fixture(scope="module")
def report(cases):
    dataset = ClutterPickDataset(
        synthetic_clutter_pick_scenes(
            np.random.default_rng(0),
            scene_count=120,
            object_count=RECORDED_NEIGHBOUR_COUNT + 1,
        )
    )
    return evaluate(dataset, min_samples_per_leaf=15, cases=cases)


# %% describing regions


def test_describe_region_writes_a_point_as_its_value():
    variable = Continuous("x")
    event = SimpleEvent.from_data({variable: closed(0.5, 0.5)}).as_composite_set()
    assert describe_region(event, variable) == "0.5"


def test_describe_region_writes_a_range_as_its_bounds():
    variable = Continuous("x")
    event = SimpleEvent.from_data({variable: closed(0.5, 1.5)}).as_composite_set()
    assert describe_region(event, variable) == "[0.5, 1.5]"


def test_describe_region_writes_symbols_by_name():
    variable = Symbolic("side", domain=Set.from_iterable(["along", "across"]))
    event = SimpleEvent.from_data(
        {variable: Set.from_iterable(["along"])}
    ).as_composite_set()
    assert describe_region(event, variable) == "along"


# %% the comparison


def test_catalogue_asks_each_kind_of_cause_at_the_recorded_size_and_at_others():
    catalogue = query_catalogue(RECORDED_NEIGHBOUR_COUNT)
    counts = [case.neighbour_count for case in catalogue]
    assert counts.count(RECORDED_NEIGHBOUR_COUNT) == 3
    assert len(set(counts)) == 3


def test_report_splits_the_dataset(report):
    assert report.training_scene_count == 96
    assert report.test_scene_count == 24
    assert report.recorded_neighbour_count == RECORDED_NEIGHBOUR_COUNT


def test_report_records_every_question_for_every_pipeline(report, cases):
    assert [pipeline.name for pipeline in report.pipelines] == [
        "relational circuit",
        "flat-table tree",
    ]
    for pipeline in report.pipelines:
        assert [outcome.case for outcome in pipeline.outcomes] == cases


def test_only_the_relational_pipeline_answers_about_another_clutter_size(report):
    relational, flat = report.pipelines
    assert relational.outcomes[1].answered
    assert flat.outcomes[1].refusal == Refusal.SCHEMA_MISMATCH


def test_shared_coverage_likelihood_is_reported_per_pipeline(report):
    assert set(report.shared_coverage_log_likelihoods) == {
        pipeline.name for pipeline in report.pipelines
    }


# %% rendering


def test_markdown_report_names_every_question_and_marks_the_verdicts(report, cases):
    markdown = MarkdownReport(report).render()
    for case in cases:
        assert case.question in markdown
    assert Verdict.ANSWERED in markdown
    assert f"{Verdict.REFUSED}: {Refusal.SCHEMA_MISMATCH}" in markdown
    for pipeline in report.pipelines:
        assert pipeline.name in markdown


def test_every_outcome_is_timed_asked_again(report):
    for pipeline in report.pipelines:
        for outcome in pipeline.outcomes:
            assert outcome.repeat_duration >= 0


def test_markdown_report_puts_an_answer_into_words(report):
    markdown = MarkdownReport(report).render()
    relational_answer = report.pipelines[0].outcomes[0]
    assert (
        relational_answer.case.describe_cause(
            relational_answer.most_effective.cause_region
        )
        in markdown
    )
    assert relational_answer.case.effect in markdown
    assert "## What the results show" in markdown
