"""
Tests for the causal-query pipelines on synthetic attempts.
"""

from __future__ import annotations

import experiments.orm.ormatic_interface  # noqa: F401  # registers the DAO classes
import numpy as np
import pytest
from krrood.entity_query_language.backends import ProbabilisticBackend
from krrood.entity_query_language.factories import cause

from experiments.causal_reasoning.comparison.domain import ExampleView
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
from experiments.causal_reasoning.tracy_clutter_picking.domain import (
    ClutteredObject,
    FrictionLadder,
    PartField,
    attempt_domain,
)
from experiments.causal_reasoning.tracy_clutter_picking.queries import (
    ClosingAxisSideCausesDisturbance,
    CrowdingCausesLift,
    FrictionCausesLift,
)
from experiments.causal_reasoning.tracy_clutter_picking.synthetic import (
    synthetic_clutter_pick_scenes,
)

LEAF_SHARE = 0.125
"""
The share of its training rows a leaf may hold in these tests: fifteen of the hundred
and twenty attempts fitted on.
"""


@pytest.fixture(scope="module")
def recorded_neighbour_count() -> int:
    """
    Neighbours per synthetic attempt; fewer than the mock's ten cartons, to keep the
    tests quick.
    """
    return 5


@pytest.fixture(scope="module")
def scenes(recorded_neighbour_count):
    return synthetic_clutter_pick_scenes(
        np.random.default_rng(0),
        scene_count=150,
        object_count=recorded_neighbour_count + 1,
    )


@pytest.fixture(scope="module")
def schema():
    return Schema(attempt_domain())


@pytest.fixture(scope="module")
def relational_pipeline(scenes):
    pipeline = RelationalPipeline(
        domain=attempt_domain(), min_samples_per_leaf=LEAF_SHARE
    )
    pipeline.fit(scenes[:120])
    return pipeline


@pytest.fixture(scope="module")
def flat_table_pipeline(scenes, schema):
    pipeline = FlatTablePipeline(
        domain=attempt_domain(),
        layout=TableLayout.UNROLLED,
        part_widths=FlatTable.unrolled_for(schema, scenes[:120]).part_widths,
        min_samples_per_leaf=LEAF_SHARE,
    )
    pipeline.fit(scenes[:120])
    return pipeline


# %% flat table


def test_flat_table_columns_are_named_like_eql_variables(schema):
    assert schema.scalar_column("lifted") == "ClutterPickScene.lifted"
    assert (
        schema.part_column(PartAttribute("neighbours", 2, "x"))
        == "ClutterPickScene.neighbours[2].x"
    )
    assert schema.aggregation_columns == (
        "ClutterPickSceneAggregations.crowding_count()",
    )


def test_flat_table_row_keeps_every_attribute(scenes, schema):
    scene = scenes[0]
    table = FlatTable.unrolled_for(schema, scenes)
    row = table.row(scene)
    assert (
        row[schema.scalar_column("friction_coefficient")] == scene.friction_coefficient
    )
    assert (
        row[schema.part_column(PartAttribute("neighbours", 3, "distance_band"))]
        == scene.neighbours[3].distance_band
    )
    assert set(row) == set(table.columns)


def test_flat_table_rejects_a_scene_of_more_neighbours(
    scenes, schema, recorded_neighbour_count
):
    narrow = FlatTable(
        schema,
        TableLayout.UNROLLED,
        part_widths={PartField.NEIGHBOURS: recorded_neighbour_count - 1},
    )
    with pytest.raises(FlatTableSchemaMismatchError):
        narrow.row(scenes[0])


# %% stratification per cause


def test_a_scene_level_cause_stratifies_the_class_circuit(schema):
    stratification = CauseStratification.for_variable(
        schema.scalar_column("friction_coefficient"), schema
    )
    assert stratification.class_columns == [
        schema.scalar_column("friction_coefficient")
    ]
    assert stratification.part_attributes == {}


def test_a_neighbour_cause_stratifies_the_neighbour_template(schema):
    stratification = CauseStratification.for_variable(
        schema.part_column(PartAttribute("neighbours", 4, "closing_axis_side")), schema
    )
    assert stratification.class_columns is None
    assert stratification.part_attributes == {"neighbours": ["closing_axis_side"]}


def test_unfitted_pipeline_refuses_to_serve_a_model():
    with pytest.raises(PipelineNotFittedError):
        RelationalPipeline(domain=attempt_domain()).registry_for(None)


def test_two_causes_in_one_query_are_rejected(
    relational_pipeline, recorded_neighbour_count
):
    query = example_query(
        attempt_domain(),
        {
            PartField.NEIGHBOURS: [
                part_query(ClutteredObject) for _ in range(recorded_neighbour_count)
            ]
        },
        friction_coefficient=cause,
        crowding_count=cause,
    )
    query.causes_effect(query.variable.lifted == True)
    with pytest.raises(OneCausePerQueryError):
        ProbabilisticBackend(model_registry=relational_pipeline.registry).rank_causes(
            query
        )


# %% answering the questions


@pytest.fixture(scope="module")
def asker():
    return QuestionAsker(random_seed=0, min_region_support=1)


def _adjusted_by_region(outcome):
    return {
        effect.cause_region: effect.adjusted_probability for effect in outcome.effects
    }


@pytest.mark.parametrize(
    "pipeline_name", ["relational_pipeline", "flat_table_pipeline"]
)
def test_both_pipelines_find_the_highest_friction_the_surest_lift(
    request, asker, pipeline_name, recorded_neighbour_count
):
    """
    The synthetic attempts hold the target for certain at the highest friction level
    and almost never at the lowest, so the adjusted effect must be highest at the top
    of the ladder and lowest at its bottom for either pipeline.
    """
    pipeline = request.getfixturevalue(pipeline_name)
    outcome = asker.ask(
        pipeline, FrictionCausesLift(open_part_count=recorded_neighbour_count)
    )

    assert outcome.answered
    adjusted = _adjusted_by_region(outcome)
    assert max(adjusted, key=adjusted.get) == f"{FrictionLadder().highest:g}"
    assert min(adjusted, key=adjusted.get) == f"{FrictionLadder().lowest:g}"


@pytest.mark.parametrize(
    "pipeline_name", ["relational_pipeline", "flat_table_pipeline"]
)
def test_both_pipelines_find_an_empty_neighbourhood_the_surest_lift(
    request, asker, pipeline_name, recorded_neighbour_count
):
    pipeline = request.getfixturevalue(pipeline_name)
    outcome = asker.ask(
        pipeline, CrowdingCausesLift(open_part_count=recorded_neighbour_count)
    )

    assert outcome.answered
    adjusted = _adjusted_by_region(outcome)
    assert max(adjusted, key=adjusted.get) == "0"


def test_relational_pipeline_answers_about_a_clutter_of_another_size(
    relational_pipeline, asker, recorded_neighbour_count
):
    outcome = asker.ask(
        relational_pipeline,
        FrictionCausesLift(open_part_count=recorded_neighbour_count + 2),
    )
    assert outcome.answered
    assert len(outcome.effects) == len(FrictionLadder().levels)


def test_flat_table_pipeline_cannot_tell_clutter_sizes_apart(
    flat_table_pipeline, asker, recorded_neighbour_count
):
    recorded = asker.ask(
        flat_table_pipeline,
        FrictionCausesLift(open_part_count=recorded_neighbour_count),
    )
    larger = asker.ask(
        flat_table_pipeline,
        FrictionCausesLift(open_part_count=recorded_neighbour_count + 2),
    )
    assert larger.answered
    assert _adjusted_by_region(larger) == pytest.approx(_adjusted_by_region(recorded))


def test_flat_table_pipeline_refuses_a_neighbour_beyond_its_positions(
    flat_table_pipeline, asker, recorded_neighbour_count
):
    outcome = asker.ask(
        flat_table_pipeline,
        ClosingAxisSideCausesDisturbance(
            open_part_count=recorded_neighbour_count + 2,
            neighbour_index=recorded_neighbour_count + 1,
        ),
    )
    assert not outcome.answered
    assert outcome.refusal == Refusal.SCHEMA_MISMATCH


def test_relational_pipeline_answers_a_question_about_one_neighbour(
    relational_pipeline, asker, recorded_neighbour_count
):
    outcome = asker.ask(
        relational_pipeline,
        ClosingAxisSideCausesDisturbance(
            open_part_count=recorded_neighbour_count, neighbour_index=2
        ),
    )
    assert outcome.answered
    assert set(_adjusted_by_region(outcome)) == {"along", "across"}


def test_each_cause_gets_its_own_model(
    relational_pipeline, schema, asker, recorded_neighbour_count
):
    """
    Every distinct cause asked about fitted one further model, on top of the plain one.
    """
    asker.ask(
        relational_pipeline,
        FrictionCausesLift(open_part_count=recorded_neighbour_count),
    )
    asker.ask(
        relational_pipeline,
        CrowdingCausesLift(open_part_count=recorded_neighbour_count),
    )

    assert relational_pipeline.fit_report.model_count == 1 + len(
        relational_pipeline.cause_models
    )
    assert set(relational_pipeline.cause_models) >= {
        schema.scalar_column("friction_coefficient"),
        schema.aggregation_column("crowding_count"),
    }


# %% likelihood


@pytest.mark.parametrize(
    "pipeline_name", ["relational_pipeline", "flat_table_pipeline"]
)
def test_held_out_likelihood_covers_some_attempts(request, scenes, pipeline_name):
    pipeline = request.getfixturevalue(pipeline_name)
    report = pipeline.log_likelihood(scenes[120:], ExampleView.WHOLE)
    assert report.example_count == 30
    assert 0 < report.covered_example_count <= report.example_count
    assert np.isfinite(report.mean_log_likelihood)


def test_flat_table_pipeline_cannot_score_a_larger_clutter(
    flat_table_pipeline, recorded_neighbour_count
):
    larger = synthetic_clutter_pick_scenes(
        np.random.default_rng(1),
        scene_count=5,
        object_count=recorded_neighbour_count + 2,
    )
    report = flat_table_pipeline.log_likelihood(larger, ExampleView.WHOLE)
    assert report.covered_example_count == 0
