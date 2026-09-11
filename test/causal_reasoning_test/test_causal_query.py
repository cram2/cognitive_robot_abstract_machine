"""
Validation of causal-query grounding against the CTU Mutagenesis dataset
(https://relational.fel.cvut.cz/dataset/Mutagenesis), covering plan step 5:
registering branching-atom count as a cause of mutagenicity and comparing naive
conditioning against backdoor adjustment.
"""

from __future__ import annotations

import numpy as np
import pytest

from experiments.causal_reasoning.mutagenesis.causal_query import (
    BranchingAtomCountCausalQuery,
)
from experiments.causal_reasoning.mutagenesis.dataset import (
    fetch_mutagenesis_molecules,
    is_mutagenesis_dataset_reachable,
    synthetic_mutagenesis_molecules,
)
from experiments.causal_reasoning.mutagenesis.domain import (
    MutagenesisAtom,
    MutagenesisElement,
    MutagenesisMolecule,
    MutagenesisMoleculeAggregations,
)

requires_mutagenesis_dataset = pytest.mark.skipif(
    not is_mutagenesis_dataset_reachable(),
    reason="CTU relational-dataset repository is not reachable from this environment",
)


# %% synthetic-data pipeline (no network access, runs in CI)


def test_synthetic_causal_circuit_is_support_deterministic():
    """
    Regression test, paired with the live-dataset tests below: stratifying the class
    circuit by branching-atom count used to fail support-determinism verification once a
    partition was large and varied enough for JointProbabilityTree to split it further
    on other variables, because both grounding and verification computed marginals
    through a path that flattens nested SumUnits and erases which partition a further-
    split branch actually belongs to.
    """
    molecules = synthetic_mutagenesis_molecules(
        np.random.default_rng(0), molecule_count=60, atom_count=3, bond_count=4
    )
    result = BranchingAtomCountCausalQuery().run(molecules, atom_count=2, bond_count=1)
    assert result.support_determinism_verified


def _molecule_with_branching_atom_count(
    branching_atom_count: int, indicator_1: bool, mutagenic: bool
) -> MutagenesisMolecule:
    atoms = [
        MutagenesisAtom(
            element=MutagenesisElement.CARBON, atom_type=1, charge=0.0, bond_count=3
        )
        for _ in range(branching_atom_count)
    ] + [
        MutagenesisAtom(
            element=MutagenesisElement.CARBON, atom_type=1, charge=0.0, bond_count=1
        )
    ]
    return MutagenesisMolecule(
        indicator_1=indicator_1,
        logp=0.0,
        lumo=0.0,
        mutagenic=mutagenic,
        atoms=atoms,
        bonds=[],
    )


def test_classify_by_branching_atom_count_and_indicator_1_votes_per_pair():
    """
    Two molecules share a branching-atom count but differ on ``ind1``, splitting them
    into their own perfectly separable groups; voting on branching-atom count alone
    would tie 2-2 and misclassify half of them, so this is a regression test that the
    ``ind1`` split, not just the branching-atom-count tally, drives the prediction.
    """
    molecules = [
        _molecule_with_branching_atom_count(3, indicator_1=True, mutagenic=True),
        _molecule_with_branching_atom_count(3, indicator_1=True, mutagenic=True),
        _molecule_with_branching_atom_count(3, indicator_1=False, mutagenic=False),
        _molecule_with_branching_atom_count(3, indicator_1=False, mutagenic=False),
    ]

    confusion_matrix = (
        BranchingAtomCountCausalQuery.classify_by_branching_atom_count_and_indicator_1(
            molecules
        )
    )

    assert confusion_matrix.true_positive == 2
    assert confusion_matrix.true_negative == 2
    assert confusion_matrix.false_positive == 0
    assert confusion_matrix.false_negative == 0
    assert confusion_matrix.total == 4
    assert confusion_matrix.accuracy == pytest.approx(1.0)


# %% live-dataset pipeline (real CTU Mutagenesis data, skipped without network access)


@pytest.fixture(scope="module")
def mutagenesis_molecules():
    return fetch_mutagenesis_molecules()


@pytest.fixture(scope="module")
def causal_query_result(mutagenesis_molecules):
    return BranchingAtomCountCausalQuery().run(mutagenesis_molecules, atom_count=2)


@requires_mutagenesis_dataset
def test_causal_circuit_is_support_deterministic(causal_query_result):
    assert causal_query_result.support_determinism_verified


@requires_mutagenesis_dataset
def test_every_distinct_branching_atom_count_is_reported(
    causal_query_result, mutagenesis_molecules
):
    """
    Every distinct branching-atom-count value present in the training population must
    survive grounding and registration, not just the dominant one.
    """
    expected_counts = sorted(
        {
            MutagenesisMoleculeAggregations(instance=molecule).branching_atom_count()
            for molecule in mutagenesis_molecules
        }
    )
    reported_counts = [
        effect.branching_atom_count for effect in causal_query_result.effects
    ]
    assert reported_counts == sorted(reported_counts)
    assert len(reported_counts) == len(set(reported_counts))
    assert reported_counts == expected_counts


@requires_mutagenesis_dataset
def test_region_probabilities_sum_to_one(causal_query_result):
    total = sum(effect.region_probability for effect in causal_query_result.effects)
    assert total == pytest.approx(1.0, abs=0.01)


@requires_mutagenesis_dataset
def test_effect_probabilities_are_valid_probabilities(causal_query_result):
    for effect in causal_query_result.effects:
        assert 0.0 <= effect.naive_probability_mutagenic <= 1.0
        assert 0.0 <= effect.adjusted_probability_mutagenic <= 1.0


@requires_mutagenesis_dataset
def test_confusion_matrix_covers_every_molecule(mutagenesis_molecules):
    confusion_matrix = (
        BranchingAtomCountCausalQuery.classify_by_branching_atom_count_and_indicator_1(
            mutagenesis_molecules
        )
    )
    assert confusion_matrix.total == len(mutagenesis_molecules)
