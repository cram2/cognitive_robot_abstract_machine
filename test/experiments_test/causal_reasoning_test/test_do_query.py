"""
Tests for the do() questions put to the Mutagenesis circuit.

Everything here fits on synthetic molecules rather than the CTU database, so the whole
module keeps running in CI without network access.
"""

from __future__ import annotations

import experiments.orm.ormatic_interface  # type: ignore  # noqa: F401
import numpy as np
import pytest
from krrood.entity_query_language.factories import variable
from typing_extensions import List

from experiments.causal_reasoning.mutagenesis.dataset import (
    synthetic_mutagenesis_molecules,
)
from experiments.causal_reasoning.mutagenesis.do_query import (
    CountCausesMutagenicity,
    DoQueryAnswer,
    IndicatorCausesMutagenicity,
    MoleculeAttribute,
    MoleculeCount,
    MutagenesisDoQuery,
    MutagenesisQuestion,
    question_catalogue,
)
from experiments.causal_reasoning.mutagenesis.domain import (
    MutagenesisMolecule,
    MutagenesisMoleculeAggregations,
)


@pytest.fixture(scope="module")
def probability_tolerance() -> float:
    """
    How far outside ``[0, 1]`` a probability read off a circuit may land through
    floating-point arithmetic alone.
    """
    return 1e-9


@pytest.fixture(scope="module")
def molecule_count() -> int:
    """
    Enough molecules for a count to take several values, few enough to fit quickly.
    """
    return 40


@pytest.fixture(scope="module")
def molecules(molecule_count: int) -> List[MutagenesisMolecule]:
    """
    Synthetic molecules, the same ones for every test in the module.
    """
    return synthetic_mutagenesis_molecules(
        np.random.default_rng(0), molecule_count=molecule_count
    )


# %% what a molecule offers as a cause


def test_every_count_is_named_after_its_own_aggregation() -> None:
    """
    A count names the variable the circuit fits for the statistic of the same name, so
    stratifying by it reaches the right column.
    """
    aggregations = variable(MutagenesisMoleculeAggregations)

    assert MoleculeCount.BRANCHING_ATOMS.circuit_variable_name == (
        aggregations.branching_atom_count()._name_
    )
    assert MoleculeCount.ATOMS.circuit_variable_name == aggregations.atom_count()._name_


def test_every_attribute_is_named_after_its_own_field() -> None:
    assert MoleculeAttribute.INDICATOR_1.circuit_variable_name == (
        variable(MutagenesisMolecule).indicator_1._name_
    )


def test_every_count_and_attribute_reads_as_words() -> None:
    assert all(count.noun for count in MoleculeCount)
    assert all(attribute.noun for attribute in MoleculeAttribute)


# %% the size of a molecule


def test_the_atom_count_is_how_many_atoms_the_molecule_holds(
    molecules: List[MutagenesisMolecule],
) -> None:
    """
    The statistic counts every atom, whatever its element, so it is the length of the
    molecule's own atom list.
    """
    for molecule in molecules:
        assert MutagenesisMoleculeAggregations(instance=molecule).atom_count() == len(
            molecule.atoms
        )


# %% how a question describes itself


def test_a_question_is_named_after_its_cause_and_what_it_adjusts_for() -> None:
    question = CountCausesMutagenicity(count=MoleculeCount.CHLORINE_ATOMS)

    assert question.name.startswith(MoleculeCount.CHLORINE_ATOMS.value)
    assert question.name.endswith(MoleculeAttribute.INDICATOR_1.value)
    assert question.adjustment_variable_names == (
        MoleculeAttribute.INDICATOR_1.circuit_variable_name,
    )


def test_a_question_stratifies_the_fit_by_its_own_cause() -> None:
    question = CountCausesMutagenicity(count=MoleculeCount.AROMATIC_BONDS)

    assert question.cause_variable_name == (
        MoleculeCount.AROMATIC_BONDS.circuit_variable_name
    )


def test_the_indicator_question_adjusts_for_the_size_of_the_molecule() -> None:
    question = IndicatorCausesMutagenicity()

    assert question.cause_variable_name == (
        MoleculeAttribute.INDICATOR_1.circuit_variable_name
    )
    assert question.adjustment_variable_names == (
        MoleculeCount.ATOMS.circuit_variable_name,
    )


def test_the_catalogue_asks_each_question_once() -> None:
    names = [question.name for question in question_catalogue()]

    assert len(names) == len(set(names))


# %% asking one question of a fitted circuit


@pytest.fixture(scope="module")
def branching_atoms_answer(molecules: List[MutagenesisMolecule]) -> DoQueryAnswer:
    """
    The answer to the branching-atom-count question, read once.
    """
    return MutagenesisDoQuery(
        question=CountCausesMutagenicity(count=MoleculeCount.BRANCHING_ATOMS)
    ).run(molecules)


def test_the_answer_reports_the_question_it_was_asked(
    branching_atoms_answer: DoQueryAnswer, molecule_count: int
) -> None:
    question = CountCausesMutagenicity(count=MoleculeCount.BRANCHING_ATOMS)

    assert branching_atoms_answer.asked == question.asked
    assert branching_atoms_answer.training_example_count == molecule_count


def test_one_region_per_value_the_molecules_take(
    branching_atoms_answer: DoQueryAnswer, molecules: List[MutagenesisMolecule]
) -> None:
    """
    The cause is split into one region per value the training molecules give it, read
    off the same statistic the circuit was fitted on.
    """
    taken = {
        MutagenesisMoleculeAggregations(instance=molecule).branching_atom_count()
        for molecule in molecules
    }

    assert len(branching_atoms_answer.regions) == len(taken)


def test_the_regions_hold_the_whole_population_between_them(
    branching_atoms_answer: DoQueryAnswer,
) -> None:
    total = sum(region.probability for region in branching_atoms_answer.regions)

    assert total == pytest.approx(1.0)


def test_every_answer_on_a_region_is_a_probability(
    branching_atoms_answer: DoQueryAnswer, probability_tolerance: float
) -> None:
    for region in branching_atoms_answer.regions:
        assert (
            -probability_tolerance
            <= region.conditioned_probability
            <= (1.0 + probability_tolerance)
        )
        assert (
            -probability_tolerance
            <= region.adjusted_probability
            <= (1.0 + probability_tolerance)
        )


def test_the_shift_from_adjusting_is_the_distance_between_the_two_answers(
    branching_atoms_answer: DoQueryAnswer,
) -> None:
    [region, *_] = branching_atoms_answer.regions

    assert region.shift_from_adjusting == pytest.approx(
        region.adjusted_probability - region.conditioned_probability
    )
    assert branching_atoms_answer.largest_shift_from_adjusting == pytest.approx(
        max(abs(one.shift_from_adjusting) for one in branching_atoms_answer.regions)
    )


# %% adjusting for nothing changes nothing


def test_adjusting_for_nothing_leaves_the_conditioned_answer_alone(
    molecules: List[MutagenesisMolecule],
) -> None:
    """
    With no confounder to adjust for, the adjusted circuit is the conditioned one, so
    every region's two answers agree exactly.
    """
    answer = MutagenesisDoQuery(
        question=CountCausesMutagenicity(
            count=MoleculeCount.BRANCHING_ATOMS, adjusted_for=()
        )
    ).run(molecules)

    assert answer.largest_shift_from_adjusting == pytest.approx(0.0)


# %% every question in the catalogue can actually be asked


@pytest.mark.parametrize(
    "question", question_catalogue(), ids=lambda question: question.name
)
def test_every_question_of_the_catalogue_is_answered(
    question: MutagenesisQuestion,
    molecules: List[MutagenesisMolecule],
    probability_tolerance: float,
) -> None:
    """
    Each question has to ground, adjust and come back with regions that hold the whole
    population: a cause or a confounder the fitted circuit names differently would fail
    here rather than in a run.
    """
    answer = MutagenesisDoQuery(question=question).run(molecules)

    assert answer.regions
    assert sum(region.probability for region in answer.regions) == pytest.approx(1.0)
    for region in answer.regions:
        assert (
            -probability_tolerance
            <= region.conditioned_probability
            <= (1.0 + probability_tolerance)
        )
        assert (
            -probability_tolerance
            <= region.adjusted_probability
            <= (1.0 + probability_tolerance)
        )
