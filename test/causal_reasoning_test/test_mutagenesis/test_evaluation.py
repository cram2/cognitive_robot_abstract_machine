"""
Tests for the whole comparison and its Markdown rendering, on synthetic molecules.
"""

from __future__ import annotations

import numpy as np
import pytest

from experiments.causal_reasoning.mutagenesis.dataset import (
    MutagenesisDataset,
    synthetic_mutagenesis_molecules,
)
from experiments.causal_reasoning.mutagenesis.evaluation import (
    InterventionalEffect,
    Refusal,
    describe_region,
    evaluate,
    learning_curve,
    permutation_study,
    regions_partition_the_cause,
    split_study,
)
from experiments.causal_reasoning.mutagenesis.flat_table import MoleculeView
from experiments.causal_reasoning.mutagenesis.queries import (
    BranchingAtomsCauseTerminalAtom,
    CountCausesMutagenicity,
    ElementCausesTerminalAtom,
    IndicatorCausesElement,
    IndicatorCausesMutagenicity,
    query_catalogue,
)
from experiments.causal_reasoning.mutagenesis.report import (
    MarkdownReport,
    Verdict,
)
from random_events.interval import closed
from random_events.product_algebra import SimpleEvent
from random_events.set import Set
from random_events.variable import Continuous, Symbolic


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
    return MutagenesisDataset(
        synthetic_mutagenesis_molecules(
            np.random.default_rng(0), molecule_count=120, atom_count=3, bond_count=4
        )
    )


@pytest.fixture(scope="module")
def report(cases, dataset):
    return evaluate(dataset, min_samples_per_leaf=15, cases=cases)


@pytest.fixture(scope="module")
def permutations(dataset):
    return permutation_study(
        dataset,
        ordering_count=2,
        min_samples_per_leaf=15,
        cases=[BranchingAtomsCauseTerminalAtom()],
    )


@pytest.fixture(scope="module")
def splits(cases, dataset):
    return split_study(
        dataset, random_seeds=(0, 1), min_samples_per_leaf=15, cases=cases
    )


@pytest.fixture(scope="module")
def curve(dataset):
    return learning_curve(dataset, train_fractions=(0.4, 0.8), random_seeds=(0,))


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
    assert len(shuffled.molecules) == len(dataset.molecules)
    for original, reordered in zip(dataset.molecules, shuffled.molecules):
        assert reordered.mutagenic == original.mutagenic
        assert sorted(atom.charge for atom in reordered.atoms) == sorted(
            atom.charge for atom in original.atoms
        )
        assert len(reordered.bonds) == len(original.bonds)


def test_mutagenic_rate_by_groups_the_molecules(dataset):
    rates = dataset.mutagenic_rate_by(lambda molecule: molecule.mutagenic)
    assert set(rates) == {False, True}
    assert rates[True].rate == 1.0
    assert rates[False].rate == 0.0
    assert rates[True].molecule_count + rates[False].molecule_count == len(
        dataset.molecules
    )


# %% the comparison


def test_catalogue_asks_molecule_level_causes_then_atom_level_questions():
    catalogue = query_catalogue()
    kinds = [type(case) for case in catalogue]
    assert kinds == [
        CountCausesMutagenicity,
        CountCausesMutagenicity,
        CountCausesMutagenicity,
        IndicatorCausesMutagenicity,
        IndicatorCausesMutagenicity,
        IndicatorCausesElement,
        BranchingAtomsCauseTerminalAtom,
        ElementCausesTerminalAtom,
    ]
    assert len({case.name for case in catalogue}) == len(catalogue)


def test_report_splits_the_dataset(report):
    assert report.training_molecule_count == 96
    assert report.test_molecule_count == 24


def test_report_records_every_question_for_every_pipeline(report, cases):
    assert [pipeline.name for pipeline in report.pipelines] == [
        "relational circuit",
        "propositional tree",
        "unrolled tree",
        "scalars-only tree",
    ]
    for pipeline in report.pipelines:
        assert [outcome.case for outcome in pipeline.outcomes] == cases


def test_the_propositional_tree_alone_refuses_the_question_about_one_atom(report):
    relational, propositional, unrolled, scalars_only = report.pipelines
    assert relational.outcomes[1].answered
    assert unrolled.outcomes[1].answered
    assert propositional.outcomes[1].refusal == Refusal.SCHEMA_MISMATCH
    assert scalars_only.outcomes[1].refusal == Refusal.SCHEMA_MISMATCH


def test_shared_coverage_likelihood_is_reported_per_view(report):
    shared = report.shared_coverage_log_likelihoods
    assert set(shared[MoleculeView.SCALARS]) == {
        pipeline.name for pipeline in report.pipelines
    }
    assert set(shared[MoleculeView.WHOLE_MOLECULE]) == {
        "relational circuit",
        "unrolled tree",
    }


# %% the studies


def test_reordering_the_atoms_leaves_the_relational_answer_alone(permutations):
    [relational] = [
        question
        for question in permutations.questions
        if question.pipeline_name == "relational circuit"
    ]
    assert len(relational.answered) == permutations.ordering_count
    assert relational.largest_adjusted_difference == pytest.approx(0.0, abs=1e-9)
    assert permutations.largest_likelihood_difference("relational circuit") == (
        pytest.approx(0.0, abs=1e-9)
    )


def test_the_permutation_study_covers_the_pipelines_modelling_parts(permutations):
    assert {question.pipeline_name for question in permutations.questions} == {
        "relational circuit",
        "unrolled tree",
    }
    assert set(permutations.whole_molecule_likelihoods) == {
        "relational circuit",
        "unrolled tree",
    }


def test_the_split_study_reports_every_split(splits, cases):
    assert [report.random_seed for report in splits.reports] == [0, 1]
    assert (
        len(splits.shared_log_likelihood("relational circuit", MoleculeView.SCALARS))
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
            assert point.likelihoods[MoleculeView.SCALARS] is not None


# %% rendering


def test_markdown_report_names_every_question_and_marks_the_verdicts(
    report, cases, permutations, splits, curve
):
    markdown = MarkdownReport(
        report, permutations=permutations, splits=splits, curve=curve
    ).render()
    assert "## Does the order of the atoms matter?" in markdown
    assert "## Over several splits" in markdown
    assert "## How much training data it takes" in markdown
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
