"""
Tests for the whole comparison and its Markdown rendering, on synthetic molecules.
"""

from __future__ import annotations

import math

import experiments.orm.ormatic_interface  # noqa: F401  # registers the DAO classes
import numpy as np
import pytest
from random_events.interval import closed
from random_events.product_algebra import SimpleEvent
from random_events.set import Set
from random_events.variable import Continuous, Symbolic

from experiments.causal_reasoning.comparison.domain import ExampleView
from experiments.causal_reasoning.comparison.evaluation import (
    InterventionalEffect,
    Refusal,
    describe_region,
    evaluate,
    learning_curve,
    permutation_study,
    regions_partition_the_cause,
    split_study,
)
from experiments.causal_reasoning.comparison.report import MarkdownReport, Verdict
from experiments.causal_reasoning.mutagenesis.dataset import (
    mutagenesis_dataset,
    mutagenicity_summaries,
    synthetic_mutagenesis_molecules,
)
from experiments.causal_reasoning.mutagenesis.queries import (
    ATOM_COUNT,
    INDICATOR,
    BranchingAtomsCauseTerminalAtom,
    CountCausesMutagenicity,
    ElementCausesTerminalAtom,
    IndicatorCausesElement,
    IndicatorCausesMutagenicity,
    query_catalogue,
)
from experiments.causal_reasoning.mutagenesis.run_pipeline import (
    mutagenesis_experiment,
)

LEAF_SHARE = 0.15
"""
The share of its training rows a leaf may hold in these tests.
"""


@pytest.fixture(scope="module")
def experiment():
    return mutagenesis_experiment()


@pytest.fixture(scope="module")
def cases():
    return [
        CountCausesMutagenicity(
            statistic_name="chlorine_count", count_noun="chlorine atoms"
        ),
        BranchingAtomsCauseTerminalAtom(),
    ]


@pytest.fixture(scope="module")
def dataset():
    return mutagenesis_dataset(
        synthetic_mutagenesis_molecules(
            np.random.default_rng(0), molecule_count=120, atom_count=3, bond_count=4
        )
    )


@pytest.fixture(scope="module")
def report(experiment, cases, dataset):
    return evaluate(
        experiment.comparison,
        dataset,
        cases,
        min_samples_per_leaf=LEAF_SHARE,
        min_region_support=1,
    )


@pytest.fixture(scope="module")
def permutations(experiment, dataset):
    return permutation_study(
        experiment.comparison,
        dataset,
        [BranchingAtomsCauseTerminalAtom()],
        ordering_count=2,
        min_samples_per_leaf=LEAF_SHARE,
        min_region_support=1,
    )


@pytest.fixture(scope="module")
def splits(experiment, cases, dataset):
    return split_study(
        experiment.comparison,
        dataset,
        cases,
        random_seeds=(0, 1),
        min_samples_per_leaf=LEAF_SHARE,
        min_region_support=1,
    )


@pytest.fixture(scope="module")
def curve(experiment, dataset):
    return learning_curve(
        experiment.comparison, dataset, train_fractions=(0.4, 0.8), random_seeds=(0,)
    )


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
    variable = Symbolic("element", domain=Set.from_iterable(["c", "h"]))
    event = SimpleEvent.from_data(
        {variable: Set.from_iterable(["c"])}
    ).as_composite_set()
    assert describe_region(event, variable) == "c"


# %% checking the regions


def _effect_in_region(region: str, probability: float) -> InterventionalEffect:
    return InterventionalEffect(
        cause_region=region,
        region_probability=probability,
        naive_probability=0.5,
        adjusted_probability=0.5,
        support_count=10,
    )


def test_regions_whose_probabilities_sum_to_one_partition_the_cause():
    effects = [_effect_in_region("c", 0.7), _effect_in_region("h", 0.3)]
    assert regions_partition_the_cause(effects, tolerance=1e-6)


def test_overlapping_regions_do_not_partition_the_cause():
    effects = [
        _effect_in_region("c", 0.7),
        _effect_in_region("c", 0.7),
        _effect_in_region("h", 0.3),
    ]
    assert not regions_partition_the_cause(effects, tolerance=1e-6)


# %% the dataset


def test_shuffling_the_parts_keeps_every_molecule(dataset):
    shuffled = dataset.with_shuffled_parts(np.random.default_rng(0))
    assert len(shuffled.examples) == len(dataset.examples)
    for original, reordered in zip(dataset.examples, shuffled.examples):
        assert reordered.mutagenic == original.mutagenic
        assert sorted(atom.charge for atom in reordered.atoms) == sorted(
            atom.charge for atom in original.atoms
        )
        assert len(reordered.bonds) == len(original.bonds)


def test_mutagenic_rate_by_groups_the_molecules(dataset):
    rates = dataset.effect_rate_by(lambda molecule: molecule.mutagenic)
    assert set(rates) == {False, True}
    assert rates[True].rate == 1.0
    assert rates[False].rate == 0.0
    assert rates[True].example_count + rates[False].example_count == len(
        dataset.examples
    )


def test_the_mutagenicity_summaries_cover_every_molecule(dataset):
    summaries = mutagenicity_summaries(dataset)
    assert list(summaries) == ["ind1", "branching atoms", "aromatic bonds", "atoms"]
    for rates in summaries.values():
        assert sum(rate.example_count for rate in rates.values()) == len(
            dataset.examples
        )


# %% the comparison


def test_catalogue_asks_molecule_level_causes_then_atom_level_questions():
    catalogue = query_catalogue()
    kinds = [type(case) for case in catalogue]
    assert kinds == [CountCausesMutagenicity] * 9 + [
        IndicatorCausesMutagenicity,
        IndicatorCausesMutagenicity,
        IndicatorCausesElement,
        BranchingAtomsCauseTerminalAtom,
        ElementCausesTerminalAtom,
    ]
    assert len({case.name for case in catalogue}) == len(catalogue)


def test_every_count_is_asked_under_every_adjustment():
    adjustments = {
        case.confounders
        for case in query_catalogue()
        if isinstance(case, CountCausesMutagenicity)
    }
    assert adjustments == {(INDICATOR,), (ATOM_COUNT,), (INDICATOR, ATOM_COUNT)}


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


def test_the_estimators_without_the_atoms_refuse_the_question_about_one_atom(report):
    """
    A question whose effect lives on an atom needs the atoms. The relational circuit
    grounds them, the unrolled tree holds them by position and the neural estimator
    pools them; the tables that summarise them away refuse.
    """
    answered = {
        pipeline.name: pipeline.outcomes[1].answered for pipeline in report.pipelines
    }
    assert answered == {
        "relational circuit": True,
        "propositional tree": False,
        "unrolled tree": True,
        "scalars-only tree": False,
        "regression adjustment": False,
        "neural adjustment": True,
    }
    for name in ("propositional tree", "scalars-only tree", "regression adjustment"):
        assert report.pipeline(name).outcomes[1].refusal == Refusal.SCHEMA_MISMATCH


def test_shared_coverage_likelihood_is_reported_per_view(report):
    shared = report.shared_coverage_log_likelihoods
    assert set(shared[ExampleView.SCALARS]) == {
        pipeline.name
        for pipeline in report.pipelines
        if pipeline.likelihoods[ExampleView.SCALARS] is not None
    }
    assert not {"regression adjustment", "neural adjustment"} & set(
        shared[ExampleView.SCALARS]
    )
    assert set(shared[ExampleView.WHOLE]) == {"relational circuit", "unrolled tree"}


# %% the studies


def test_reordering_the_atoms_leaves_the_relational_answer_alone(permutations):
    [relational] = [
        question
        for question in permutations.questions
        if question.pipeline_name == "relational circuit"
    ]
    assert len(relational.answered) == permutations.ordering_count
    assert relational.largest_adjusted_difference == pytest.approx(0.0, abs=1e-9)
    assert permutations.largest_likelihood_drop("relational circuit") == (
        pytest.approx(0.0, abs=1e-9)
    )


def test_the_permutation_study_covers_the_pipelines_modelling_parts(permutations):
    assert {question.pipeline_name for question in permutations.questions} == {
        "relational circuit",
        "unrolled tree",
    }
    assert set(permutations.whole_example_likelihoods) == {
        "relational circuit",
        "unrolled tree",
    }


def test_the_split_study_reports_every_split(splits, cases):
    assert [report.random_seed for report in splits.reports] == [0, 1]
    assert (
        len(splits.shared_log_likelihood("relational circuit", ExampleView.SCALARS))
        == 2
    )
    assert len(splits.outcomes("unrolled tree", 0)) == 2


def test_the_learning_curve_measures_every_pipeline_at_every_size(curve):
    assert curve.train_fractions == [0.4, 0.8]
    assert curve.pipeline_names == [
        "relational circuit",
        "propositional tree",
        "unrolled tree",
        "scalars-only tree",
    ]
    for name in curve.pipeline_names:
        for train_fraction in curve.train_fractions:
            [point] = curve.points_of(name, train_fraction)
            assert point.likelihoods[ExampleView.SCALARS] is not None


# %% rendering


def test_markdown_report_names_every_question_and_marks_the_verdicts(
    experiment, report, cases, permutations, splits, curve
):
    markdown = MarkdownReport(
        experiment.comparison.domain,
        experiment.text,
        report,
        permutations=permutations,
        splits=splits,
        curve=curve,
    ).render()
    assert "## Does the order of the parts matter?" in markdown
    assert "## Over several splits" in markdown
    assert "## How much training data it takes" in markdown
    for case in cases:
        assert case.question in markdown
    assert Verdict.ANSWERED in markdown
    assert f"{Verdict.REFUSED}: {Refusal.SCHEMA_MISMATCH}" in markdown
    for pipeline in report.pipelines:
        assert pipeline.name in markdown


def test_every_circuit_outcome_is_timed_asked_again(report):
    """
    Every circuit is asked again once its models are fitted; the estimators that are
    not circuits fit one model per question, so they have nothing to time a second
    time.
    """
    for pipeline in report.pipelines:
        for outcome in pipeline.outcomes:
            if pipeline.name in ("regression adjustment", "neural adjustment"):
                assert math.isnan(outcome.repeat_duration)
            else:
                assert outcome.repeat_duration >= 0


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
