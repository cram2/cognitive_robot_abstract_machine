"""
Tests for the two causal-query pipelines on synthetic molecules.
"""

from __future__ import annotations

import experiments.orm.ormatic_interface  # noqa: F401  # registers the DAO classes
import numpy as np
import pytest

from krrood.entity_query_language.backends import ProbabilisticBackend
from krrood.entity_query_language.factories import cause

from experiments.causal_reasoning.comparison.domain import AbsentPart, ExampleView
from experiments.causal_reasoning.comparison.evaluation import QuestionAsker, Refusal
from experiments.causal_reasoning.comparison.exceptions import (
    FlatTableSchemaMismatchError,
    OneCausePerQueryError,
    PipelineNotFittedError,
)
from experiments.causal_reasoning.comparison.flat_table import (
    FlatTable,
    PartAttribute,
    Schema,
    TableLayout,
)
from experiments.causal_reasoning.comparison.pipelines import (
    CauseStratification,
    FlatTablePipeline,
    RelationalPipeline,
)
from experiments.causal_reasoning.comparison.queries import example_query, part_query
from experiments.causal_reasoning.mutagenesis.dataset import (
    synthetic_mutagenesis_molecules,
)
from experiments.causal_reasoning.mutagenesis.domain import (
    MutagenesisAtom,
    MutagenesisBond,
    MutagenesisElement,
    PartField,
    molecule_domain,
)
from experiments.causal_reasoning.mutagenesis.queries import (
    BranchingAtomsCauseTerminalAtom,
    CountCausesMutagenicity,
    ElementCausesTerminalAtom,
    IndicatorCausesElement,
    IndicatorCausesMutagenicity,
)

LEAF_SHARE = 0.125
"""
The share of its training rows a leaf may hold in these tests: fifteen of the hundred
and twenty molecules fitted on.
"""


@pytest.fixture(scope="module")
def atom_count() -> int:
    """
    How many atoms each synthetic molecule has.
    """
    return 3


@pytest.fixture(scope="module")
def molecules(atom_count):
    return synthetic_mutagenesis_molecules(
        np.random.default_rng(0),
        molecule_count=150,
        atom_count=atom_count,
        bond_count=4,
    )


@pytest.fixture(scope="module")
def schema():
    return Schema(molecule_domain())


@pytest.fixture(scope="module")
def relational_pipeline(molecules):
    pipeline = RelationalPipeline(
        domain=molecule_domain(), min_samples_per_leaf=LEAF_SHARE
    )
    pipeline.fit(molecules[:120])
    return pipeline


@pytest.fixture(scope="module")
def flat_table_pipeline(molecules):
    pipeline = FlatTablePipeline(
        domain=molecule_domain(), min_samples_per_leaf=LEAF_SHARE
    )
    pipeline.fit(molecules[:120])
    return pipeline


@pytest.fixture(scope="module")
def unrolled_pipeline(molecules, schema):
    pipeline = FlatTablePipeline(
        domain=molecule_domain(),
        layout=TableLayout.UNROLLED,
        part_widths=FlatTable.unrolled_for(schema, molecules[:120]).part_widths,
        min_samples_per_leaf=LEAF_SHARE,
    )
    pipeline.fit(molecules[:120])
    return pipeline


@pytest.fixture(scope="module")
def scalars_only_pipeline(molecules):
    pipeline = FlatTablePipeline(
        domain=molecule_domain(),
        layout=TableLayout.SCALARS,
        min_samples_per_leaf=LEAF_SHARE,
    )
    pipeline.fit(molecules[:120])
    return pipeline


# %% flat table


def test_flat_table_columns_are_named_like_eql_variables(schema):
    assert schema.scalar_column("mutagenic") == "MutagenesisMolecule.mutagenic"
    assert (
        schema.part_column(PartAttribute("atoms", 2, "element"))
        == "MutagenesisMolecule.atoms[2].element"
    )
    assert schema.aggregation_columns == (
        "MutagenesisMoleculeAggregations.atom_count()",
        "MutagenesisMoleculeAggregations.chlorine_count()",
        "MutagenesisMoleculeAggregations.branching_atom_count()",
        "MutagenesisMoleculeAggregations.double_bond_count()",
        "MutagenesisMoleculeAggregations.aromatic_bond_count()",
    )


def test_propositional_row_holds_scalars_and_counts_only(molecules, schema, atom_count):
    molecule = molecules[0]
    row = FlatTable(schema).row(molecule)
    assert row[schema.scalar_column("logp")] == molecule.logp
    assert row[schema.aggregation_column("chlorine_count")] == atom_count
    assert set(row) == set(FlatTable(schema).columns)
    assert not any(schema.part_attribute(column) for column in row)


def test_scalars_only_row_holds_no_counts(molecules, schema):
    row = FlatTable(schema, TableLayout.SCALARS).row(molecules[0])
    assert set(row) == set(schema.scalar_columns)


def test_unrolled_row_lists_every_part_by_position(molecules, schema, atom_count):
    table = FlatTable.unrolled_for(schema, molecules)
    molecule = molecules[0]
    row = table.row(molecule)
    assert table.part_widths == {"atoms": atom_count, "bonds": 4}
    for index, atom in enumerate(molecule.atoms):
        assert (
            row[schema.part_column(PartAttribute("atoms", index, "element"))]
            == atom.element
        )
    assert row[schema.part_column(PartAttribute("bonds", 3, "bond_type"))] == (
        molecule.bonds[3].bond_type
    )
    assert set(row) == set(table.columns)


def test_unrolled_row_pads_the_positions_a_molecule_does_not_fill(schema):
    small, large = synthetic_mutagenesis_molecules(
        np.random.default_rng(3), molecule_count=2, atom_count=2, bond_count=2
    )
    large.atoms.append(large.atoms[0])
    table = FlatTable.unrolled_for(schema, [small, large])
    row = table.row(small)
    assert table.fits(small) and table.fits(large)
    assert row[schema.part_column(PartAttribute("atoms", 2, "element"))] == (
        AbsentPart.ABSENT
    )
    assert row[schema.part_column(PartAttribute("atoms", 2, "charge"))] == (
        table.padding.real
    )
    assert row[schema.part_column(PartAttribute("atoms", 2, "bond_count"))] == (
        table.padding.integer
    )


def test_unrolled_table_rejects_a_molecule_wider_than_itself(molecules, schema):
    table = FlatTable.unrolled_for(schema, molecules)
    wider = synthetic_mutagenesis_molecules(
        np.random.default_rng(4), molecule_count=1, atom_count=5, bond_count=4
    )[0]
    assert not table.fits(wider)
    with pytest.raises(FlatTableSchemaMismatchError):
        table.row(wider)


def test_each_layout_offers_the_views_it_holds(molecules, schema):
    assert (
        FlatTable(schema, TableLayout.SCALARS).columns_of(
            ExampleView.SCALARS_AND_COUNTS
        )
        is None
    )
    assert FlatTable(schema).columns_of(ExampleView.WHOLE) is None
    unrolled = FlatTable.unrolled_for(schema, molecules)
    assert unrolled.columns_of(ExampleView.WHOLE) == unrolled.columns


def test_schema_tells_a_part_attribute_from_a_molecule_variable(schema):
    assert schema.part_attribute(
        "MutagenesisMolecule.bonds[3].bond_type"
    ) == PartAttribute("bonds", 3, "bond_type")
    assert schema.part_attribute(schema.scalar_column("lumo")) is None
    assert schema.part_attribute(schema.aggregation_column("chlorine_count")) is None


# %% stratification per cause


def test_a_molecule_level_cause_stratifies_the_class_circuit(schema):
    stratification = CauseStratification.for_variable(
        schema.aggregation_column("branching_atom_count"), schema
    )
    assert stratification.class_columns == [
        schema.aggregation_column("branching_atom_count")
    ]
    assert stratification.part_attributes == {}


def test_an_atom_cause_stratifies_the_atom_template(schema):
    stratification = CauseStratification.for_variable(
        schema.part_column(PartAttribute("atoms", 1, "element")), schema
    )
    assert stratification.class_columns is None
    assert stratification.part_attributes == {"atoms": ["element"]}


def test_unfitted_pipeline_refuses_to_serve_a_model():
    with pytest.raises(PipelineNotFittedError):
        RelationalPipeline(domain=molecule_domain()).registry_for(None)


def test_two_causes_in_one_query_are_rejected(relational_pipeline):
    query = example_query(
        molecule_domain(),
        {
            PartField.ATOMS: [part_query(MutagenesisAtom)],
            PartField.BONDS: [part_query(MutagenesisBond)],
        },
        indicator_1=cause,
        chlorine_count=cause,
    )
    query.causes_effect(query.variable.mutagenic == True)
    with pytest.raises(OneCausePerQueryError):
        ProbabilisticBackend(model_registry=relational_pipeline.registry).rank_causes(
            query
        )


def test_flat_table_pipeline_cannot_fit_a_model_for_an_atom_cause(
    flat_table_pipeline, schema
):
    with pytest.raises(FlatTableSchemaMismatchError):
        flat_table_pipeline.registry_for(
            schema.part_column(PartAttribute("atoms", 0, "element"))
        )


# %% answering the questions


@pytest.fixture(scope="module")
def asker():
    return QuestionAsker(random_seed=0, min_region_support=1)


@pytest.fixture(scope="module")
def chlorine_causes_mutagenicity():
    return CountCausesMutagenicity(
        statistic_name="chlorine_count", count_noun="chlorine atoms"
    )


def _adjusted_by_region(outcome):
    return {
        effect.cause_region: effect.adjusted_probability for effect in outcome.effects
    }


@pytest.mark.parametrize(
    "pipeline_name", ["relational_pipeline", "flat_table_pipeline"]
)
def test_both_pipelines_find_chlorine_the_cause_of_mutagenicity(
    request, asker, pipeline_name, atom_count, chlorine_causes_mutagenicity
):
    """
    The synthetic molecules are mutagenic exactly when every atom is chlorine, so the
    adjusted effect must be certain at the full chlorine count and impossible at none
    for either pipeline.
    """
    pipeline = request.getfixturevalue(pipeline_name)
    outcome = asker.ask(pipeline, chlorine_causes_mutagenicity)

    assert outcome.answered
    adjusted = _adjusted_by_region(outcome)
    assert set(adjusted) == {str(atom_count), "0"}
    assert adjusted[str(atom_count)] == pytest.approx(1.0)
    assert adjusted["0"] == pytest.approx(0.0)


@pytest.mark.parametrize(
    "pipeline_name", ["relational_pipeline", "flat_table_pipeline"]
)
def test_both_pipelines_answer_about_the_indicator(request, asker, pipeline_name):
    pipeline = request.getfixturevalue(pipeline_name)
    outcome = asker.ask(
        pipeline,
        IndicatorCausesMutagenicity(
            confounder_name="branching_atom_count", confounder_noun="branching count"
        ),
    )

    assert outcome.answered
    assert set(_adjusted_by_region(outcome)) == {"True", "False"}


def test_relational_pipeline_answers_about_one_atom(relational_pipeline, asker):
    outcome = asker.ask(
        relational_pipeline,
        IndicatorCausesElement(element=MutagenesisElement.CHLORINE),
    )
    assert outcome.answered
    assert set(_adjusted_by_region(outcome)) == {"True", "False"}


def test_relational_pipeline_answers_a_count_cause_of_an_atom_effect(
    relational_pipeline, asker, atom_count
):
    outcome = asker.ask(relational_pipeline, BranchingAtomsCauseTerminalAtom())
    assert outcome.answered
    assert set(_adjusted_by_region(outcome)) == {
        str(count) for count in range(atom_count + 1)
    }


def test_flat_table_pipeline_refuses_a_question_about_one_atom(
    flat_table_pipeline, asker
):
    outcome = asker.ask(flat_table_pipeline, BranchingAtomsCauseTerminalAtom())
    assert not outcome.answered
    assert outcome.refusal == Refusal.SCHEMA_MISMATCH


def test_flat_table_pipeline_refuses_an_atom_cause(flat_table_pipeline, asker):
    outcome = asker.ask(flat_table_pipeline, ElementCausesTerminalAtom())
    assert not outcome.answered
    assert outcome.refusal == Refusal.SCHEMA_MISMATCH


def test_unrolled_pipeline_answers_about_the_atom_at_a_position(
    unrolled_pipeline, asker, atom_count
):
    outcome = asker.ask(unrolled_pipeline, BranchingAtomsCauseTerminalAtom())
    assert outcome.answered
    assert set(_adjusted_by_region(outcome)) == {
        str(count) for count in range(atom_count + 1)
    }


def test_unrolled_pipeline_answers_an_atom_cause(unrolled_pipeline, asker):
    outcome = asker.ask(unrolled_pipeline, ElementCausesTerminalAtom())
    assert outcome.answered
    assert set(_adjusted_by_region(outcome)) == {"cl", "h"}


def test_scalars_only_pipeline_refuses_a_count_cause(
    scalars_only_pipeline, asker, chlorine_causes_mutagenicity
):
    outcome = asker.ask(scalars_only_pipeline, chlorine_causes_mutagenicity)
    assert outcome.refusal == Refusal.SCHEMA_MISMATCH


def test_scalars_only_pipeline_answers_a_scalar_question(scalars_only_pipeline, asker):
    outcome = asker.ask(
        scalars_only_pipeline,
        IndicatorCausesMutagenicity(confounder_name="logp", confounder_noun="logp"),
    )
    assert outcome.answered
    assert set(_adjusted_by_region(outcome)) == {"True", "False"}


def test_an_effect_that_never_occurs_is_reported(relational_pipeline, asker):
    """
    No synthetic atom is carbon, so no region of the cause gives the effect any
    probability.
    """
    outcome = asker.ask(
        relational_pipeline, IndicatorCausesElement(element=MutagenesisElement.CARBON)
    )
    assert outcome.refusal == Refusal.EFFECT_NEVER_OCCURS


def test_each_cause_gets_its_own_model(
    relational_pipeline, schema, asker, chlorine_causes_mutagenicity
):
    """
    Every distinct cause asked about fitted one further model, on top of the plain one.
    """
    asker.ask(relational_pipeline, chlorine_causes_mutagenicity)
    asker.ask(
        relational_pipeline,
        IndicatorCausesMutagenicity(confounder_name="logp", confounder_noun="logp"),
    )

    assert relational_pipeline.fit_report.model_count == 1 + len(
        relational_pipeline.cause_models
    )
    assert set(relational_pipeline.cause_models) >= {
        schema.aggregation_column("chlorine_count"),
        schema.scalar_column("indicator_1"),
    }


# %% likelihood


@pytest.mark.parametrize(
    "pipeline_name",
    [
        "relational_pipeline",
        "flat_table_pipeline",
        "unrolled_pipeline",
        "scalars_only_pipeline",
    ],
)
def test_every_pipeline_scores_the_scalars(request, molecules, pipeline_name):
    pipeline = request.getfixturevalue(pipeline_name)
    report = pipeline.log_likelihood(molecules[120:], ExampleView.SCALARS)
    assert report.example_count == 30
    assert 0 < report.covered_example_count <= report.example_count
    assert np.isfinite(report.mean_log_likelihood)


def test_plain_models_on_the_same_columns_are_the_same_tree(
    relational_pipeline, flat_table_pipeline, unrolled_pipeline, molecules
):
    """
    The relational circuit's class-level circuit and the propositional tree are fitted
    on the same rows with the same settings, so they score held-out molecules alike on
    the scalars and counts.
    """
    relational = relational_pipeline.log_likelihood(
        molecules[120:], ExampleView.SCALARS_AND_COUNTS
    )
    flat = flat_table_pipeline.log_likelihood(
        molecules[120:], ExampleView.SCALARS_AND_COUNTS
    )
    assert np.allclose(relational.log_likelihoods, flat.log_likelihoods)


def test_only_the_pipelines_modelling_parts_score_whole_molecules(
    relational_pipeline,
    flat_table_pipeline,
    unrolled_pipeline,
    scalars_only_pipeline,
    molecules,
):
    held_out = molecules[120:]
    assert flat_table_pipeline.log_likelihood(held_out, ExampleView.WHOLE) is None
    assert (
        scalars_only_pipeline.log_likelihood(held_out, ExampleView.SCALARS_AND_COUNTS)
        is None
    )
    for pipeline in (relational_pipeline, unrolled_pipeline):
        assert pipeline.models_parts
        whole = pipeline.log_likelihood(held_out, ExampleView.WHOLE)
        assert whole.example_count == 30
        assert 0 < whole.covered_example_count
        assert np.isfinite(whole.mean_log_likelihood)


def test_unrolled_pipeline_cannot_score_a_molecule_wider_than_its_table(
    unrolled_pipeline,
):
    wider = synthetic_mutagenesis_molecules(
        np.random.default_rng(4), molecule_count=3, atom_count=5, bond_count=4
    )
    report = unrolled_pipeline.log_likelihood(wider, ExampleView.WHOLE)
    assert report.covered_example_count == 0
