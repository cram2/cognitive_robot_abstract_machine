"""
Tests for the whole comparison and its Markdown rendering, on synthetic attempts.
"""

from __future__ import annotations

import experiments.orm.ormatic_interface  # noqa: F401  # registers the DAO classes
import numpy as np
import pytest

from experiments.causal_reasoning.comparison.evaluation import (
    InterventionalEffect,
    Refusal,
    evaluate,
    scaling_study,
)
from experiments.causal_reasoning.comparison.report import MarkdownReport, Verdict
from experiments.causal_reasoning.tracy_clutter_picking.dataset import (
    ClutterPickDataset,
)
from experiments.causal_reasoning.tracy_clutter_picking.queries import (
    ENVIRONMENT,
    ClosingAxisSideCausesDisturbance,
    CrowdingCausesLift,
    FrictionCausesLift,
    query_catalogue,
)
from experiments.causal_reasoning.tracy_clutter_picking.run_pipeline import (
    attempts_of_size,
    tracy_experiment,
)
from experiments.causal_reasoning.tracy_clutter_picking.domain import FrictionLadder
from experiments.causal_reasoning.tracy_clutter_picking.exceptions import (
    UnreadableQuestionError,
)
from experiments.causal_reasoning.tracy_clutter_picking.synthetic import (
    ClutterTruth,
    MechanismSetting,
    synthetic_clutter_pick_scenes,
)

LEAF_SHARE = 0.15
"""
The share of its training rows a leaf may hold in these tests.
"""


@pytest.fixture(scope="module")
def recorded_neighbour_count() -> int:
    """
    How many neighbours the synthetic attempts are recorded with.
    """
    return 5


@pytest.fixture(scope="module")
def experiment(recorded_neighbour_count):
    return tracy_experiment(recorded_neighbour_count)


@pytest.fixture(scope="module")
def cases(recorded_neighbour_count):
    return [
        FrictionCausesLift(open_part_count=recorded_neighbour_count),
        FrictionCausesLift(open_part_count=recorded_neighbour_count + 2),
        ClosingAxisSideCausesDisturbance(
            open_part_count=recorded_neighbour_count + 2,
            neighbour_index=recorded_neighbour_count + 1,
        ),
    ]


@pytest.fixture(scope="module")
def report(experiment, cases, recorded_neighbour_count):
    dataset = ClutterPickDataset(
        synthetic_clutter_pick_scenes(
            np.random.default_rng(0),
            scene_count=120,
            object_count=recorded_neighbour_count + 1,
        )
    ).examples()
    return evaluate(
        experiment.comparison,
        dataset,
        cases,
        min_samples_per_leaf=LEAF_SHARE,
        min_region_support=1,
    )


def _adjusted_by_region(outcome):
    return {
        effect.cause_region: effect.adjusted_probability for effect in outcome.effects
    }


# %% the comparison


def test_catalogue_asks_each_kind_of_cause_at_the_recorded_size_and_at_others(
    recorded_neighbour_count,
):
    catalogue = query_catalogue(recorded_neighbour_count)
    counts = [case.neighbour_count for case in catalogue]
    assert counts.count(recorded_neighbour_count) == 4
    assert len(set(counts)) == 3
    assert len({case.name for case in catalogue}) == len(catalogue)


def test_the_crowding_question_is_asked_adjusted_and_unadjusted(
    recorded_neighbour_count,
):
    adjustments = [
        case.confounders
        for case in query_catalogue(recorded_neighbour_count)
        if isinstance(case, CrowdingCausesLift)
        and case.neighbour_count == recorded_neighbour_count
    ]
    assert adjustments == [(ENVIRONMENT,), ()]


def test_the_unadjusted_crowding_question_says_so(recorded_neighbour_count):
    unadjusted = CrowdingCausesLift(
        open_part_count=recorded_neighbour_count, confounders=()
    )
    assert unadjusted.name == "crowding_causes_lift_5_neighbours_unadjusted"
    assert unadjusted.question.endswith("with nothing adjusted for?")


def test_report_splits_the_dataset(report):
    assert report.training_example_count == 96
    assert report.test_example_count == 24


def test_report_records_every_question_for_every_pipeline(report, cases):
    assert [pipeline.name for pipeline in report.pipelines] == [
        "relational circuit",
        "propositional tree",
        "unrolled tree",
        "scalars-only tree",
        "regression adjustment",
        "neural adjustment",
    ]
    for pipeline in report.pipelines:
        assert [outcome.case for outcome in pipeline.outcomes] == cases


def test_the_flat_tables_cannot_tell_clutter_sizes_apart(report):
    """
    A flat table ignores the parts a query leaves open, so it answers a question about
    a larger clutter with the numbers it has for the recorded one; the relational
    circuit grounds itself for the queried clutter.
    """
    relational, *flat = report.pipelines
    assert relational.outcomes[1].answered
    for pipeline in flat:
        recorded, larger = pipeline.outcomes[:2]
        assert larger.answered
        assert _adjusted_by_region(larger) == pytest.approx(
            _adjusted_by_region(recorded)
        )


def test_only_the_estimators_without_positions_answer_about_a_twelfth_neighbour(
    report,
):
    """
    A query about a neighbour beyond the recorded clutter size has no column in any
    flat table, so every estimator that addresses a part by position refuses it. The
    relational circuit grounds itself for the queried neighbours and the neural
    estimator pools them, so neither is bound by a position and both answer.
    """
    answered = {
        pipeline.name: pipeline.outcomes[2].answered for pipeline in report.pipelines
    }
    assert answered == {
        "relational circuit": True,
        "propositional tree": False,
        "unrolled tree": False,
        "scalars-only tree": False,
        "regression adjustment": False,
        "neural adjustment": True,
    }
    for name in ("propositional tree", "unrolled tree", "regression adjustment"):
        assert report.pipeline(name).outcomes[2].refusal == Refusal.SCHEMA_MISMATCH


def test_every_circuit_answers_about_friction_at_the_recorded_size(report):
    for pipeline in report.pipelines:
        assert pipeline.outcomes[0].answered


# %% the scaling study


def test_the_scaling_study_measures_the_part_modelling_pipelines(experiment):
    study = scaling_study(
        experiment.comparison,
        attempts_of_size,
        experiment.scaling.case,
        sizes=(2, 4),
        example_count=40,
        min_samples_per_leaf=LEAF_SHARE,
        plain_min_samples_per_leaf=LEAF_SHARE,
    )
    assert study.sizes == [2, 4]
    assert study.pipeline_names == ["relational circuit", "unrolled tree"]
    for name in study.pipeline_names:
        for size in study.sizes:
            assert study.point(name, size).fit.size.node_count > 0


# %% rendering


def test_markdown_report_names_every_question_and_marks_the_verdicts(
    experiment, report, cases
):
    markdown = MarkdownReport(
        experiment.comparison.domain, experiment.text, report
    ).render()
    for case in cases:
        assert case.question in markdown
    assert Verdict.ANSWERED in markdown
    assert f"{Verdict.REFUSED}: {Refusal.SCHEMA_MISMATCH}" in markdown
    for pipeline in report.pipelines:
        assert pipeline.name in markdown


def test_markdown_report_puts_an_answer_into_words(experiment, report):
    markdown = MarkdownReport(
        experiment.comparison.domain, experiment.text, report
    ).render()
    relational_answer = report.pipelines[0].outcomes[0]
    assert (
        relational_answer.case.describe_cause(
            relational_answer.most_effective.cause_region
        )
        in markdown
    )
    assert relational_answer.case.effect in markdown
    assert "## What the results show" in markdown


# %% the known truth


def test_the_mechanism_reads_the_two_queries_it_forces(recorded_neighbour_count):
    truth = ClutterTruth()
    for case in (
        FrictionCausesLift(open_part_count=recorded_neighbour_count),
        CrowdingCausesLift(open_part_count=recorded_neighbour_count),
    ):
        assert truth._cause_of(case) in {"friction", "crowding"}
    with pytest.raises(UnreadableQuestionError):
        truth._cause_of(
            ClosingAxisSideCausesDisturbance(
                open_part_count=recorded_neighbour_count, neighbour_index=0
            )
        )


def test_the_truth_rises_with_the_friction(recorded_neighbour_count):
    """
    The mechanism gives the hold a share of the friction, so forcing a higher level
    can only raise the probability of a lift.
    """
    truth = ClutterTruth(layout_count=400)
    setting = MechanismSetting(0.25)
    case = FrictionCausesLift(open_part_count=recorded_neighbour_count)
    probabilities = [
        truth.probability(setting, case, _region(str(level)))
        for level in sorted(FrictionLadder().levels)
    ]
    assert probabilities == sorted(probabilities)
    assert probabilities[-1] > probabilities[0]


def test_more_adjacent_neighbours_lower_the_truth(recorded_neighbour_count):
    truth = ClutterTruth(layout_count=400)
    setting = MechanismSetting(0.25)
    case = CrowdingCausesLift(open_part_count=recorded_neighbour_count)
    probabilities = [
        truth.probability(setting, case, _region(str(count))) for count in range(4)
    ]
    assert probabilities == sorted(probabilities, reverse=True)


def test_removing_the_confounding_leaves_the_crowding_truth_alone(
    recorded_neighbour_count,
):
    """
    The environment drives the friction only when the confounding is on, and the
    crowding is forced in both, so the two settings differ only through the friction
    the layouts were given.
    """
    truth = ClutterTruth(layout_count=400)
    case = CrowdingCausesLift(open_part_count=recorded_neighbour_count)
    confounded = truth.probability(MechanismSetting(0.25), case, _region("0"))
    unconfounded = truth.probability(
        MechanismSetting(0.25, confounded=False), case, _region("0")
    )
    assert unconfounded > confounded


def _region(name: str) -> InterventionalEffect:
    return InterventionalEffect(
        cause_region=name,
        region_probability=1.0,
        naive_probability=0.5,
        adjusted_probability=0.5,
        support_count=10,
    )
