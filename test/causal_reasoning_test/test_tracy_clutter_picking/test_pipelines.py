"""
Tests for the two causal-query pipelines on synthetic attempts.
"""

from __future__ import annotations

import numpy as np
import pytest

from experiments.causal_reasoning.tracy_clutter_picking.domain import FrictionLadder
from experiments.causal_reasoning.tracy_clutter_picking.evaluation import (
    QuestionAsker,
    Refusal,
)
from experiments.causal_reasoning.tracy_clutter_picking.exceptions import (
    FlatTableSchemaMismatchError,
    OneCausePerQueryError,
    PipelineNotFittedError,
)
from experiments.causal_reasoning.tracy_clutter_picking.flat_table import FlatTable, SceneSchema
from experiments.causal_reasoning.tracy_clutter_picking.pipelines import (
    CauseStratification,
    FlatTablePipeline,
    RelationalPipeline,
)
from experiments.causal_reasoning.tracy_clutter_picking.queries import (
    ClosingAxisSideCausesDisturbance,
    CrowdingCausesLift,
    FrictionCausesLift,
    neighbour_query,
    scene_query,
)
from experiments.causal_reasoning.tracy_clutter_picking.synthetic import (
    synthetic_clutter_pick_scenes,
)
from krrood.entity_query_language.backends import ProbabilisticBackend
from krrood.entity_query_language.factories import cause

RECORDED_NEIGHBOUR_COUNT = 5
"""
Neighbours per synthetic attempt; fewer than the mock's ten cartons, to keep the tests
quick.
"""


@pytest.fixture(scope="module")
def scenes():
    return synthetic_clutter_pick_scenes(
        np.random.default_rng(0),
        scene_count=150,
        object_count=RECORDED_NEIGHBOUR_COUNT + 1,
    )


@pytest.fixture(scope="module")
def relational_pipeline(scenes):
    pipeline = RelationalPipeline(min_samples_per_leaf=15)
    pipeline.fit(scenes[:120])
    return pipeline


@pytest.fixture(scope="module")
def flat_table_pipeline(scenes):
    pipeline = FlatTablePipeline(
        neighbour_count=RECORDED_NEIGHBOUR_COUNT, min_samples_per_leaf=15
    )
    pipeline.fit(scenes[:120])
    return pipeline


# %% flat table


@pytest.fixture(scope="module")
def schema():
    return SceneSchema()


def test_flat_table_columns_are_named_like_eql_variables(schema):
    assert schema.scene_column("lifted") == "ClutterPickScene.lifted"
    assert schema.neighbour_column(2, "x") == "ClutterPickScene.neighbours[2].x"
    assert schema.aggregation_columns == (
        "ClutterPickSceneAggregations.crowding_count()",
    )


def test_flat_table_row_keeps_every_attribute(scenes, schema):
    scene = scenes[0]
    row = FlatTable(RECORDED_NEIGHBOUR_COUNT).row(scene)
    assert (
        row[schema.scene_column("friction_coefficient")] == scene.friction_coefficient
    )
    assert (
        row[schema.neighbour_column(3, "distance_band")]
        == scene.neighbours[3].distance_band
    )
    assert set(row) == set(FlatTable(RECORDED_NEIGHBOUR_COUNT).columns)


def test_flat_table_rejects_a_scene_of_another_neighbour_count(scenes):
    with pytest.raises(FlatTableSchemaMismatchError):
        FlatTable(RECORDED_NEIGHBOUR_COUNT + 1).row(scenes[0])


# %% stratification per cause


def test_a_scene_level_cause_stratifies_the_class_circuit(schema):
    stratification = CauseStratification.for_variable(
        schema.scene_column("friction_coefficient")
    )
    assert stratification.class_columns == [schema.scene_column("friction_coefficient")]
    assert stratification.neighbour_attributes is None


def test_a_neighbour_cause_stratifies_the_neighbour_template(schema):
    stratification = CauseStratification.for_variable(
        schema.neighbour_column(4, "closing_axis_side")
    )
    assert stratification.class_columns is None
    assert stratification.neighbour_attributes == ["closing_axis_side"]


def test_unfitted_pipeline_refuses_to_serve_a_model():
    with pytest.raises(PipelineNotFittedError):
        RelationalPipeline().registry_for(None)


def test_two_causes_in_one_query_are_rejected(relational_pipeline):
    query = scene_query(
        [neighbour_query() for _ in range(RECORDED_NEIGHBOUR_COUNT)],
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
    return QuestionAsker(random_seed=0)


def _adjusted_by_region(outcome):
    return {
        effect.cause_region: effect.adjusted_probability for effect in outcome.effects
    }


@pytest.mark.parametrize(
    "pipeline_name", ["relational_pipeline", "flat_table_pipeline"]
)
def test_both_pipelines_find_the_highest_friction_the_surest_lift(
    request, asker, pipeline_name
):
    """
    The synthetic attempts hold the target for certain at the highest friction level
    and almost never at the lowest, so the adjusted effect must be highest at the top
    of the ladder and lowest at its bottom for either pipeline.
    """
    pipeline = request.getfixturevalue(pipeline_name)
    outcome = asker.ask(pipeline, FrictionCausesLift(RECORDED_NEIGHBOUR_COUNT))

    assert outcome.answered
    adjusted = _adjusted_by_region(outcome)
    assert max(adjusted, key=adjusted.get) == f"{FrictionLadder().highest:g}"
    assert min(adjusted, key=adjusted.get) == f"{FrictionLadder().lowest:g}"


@pytest.mark.parametrize(
    "pipeline_name", ["relational_pipeline", "flat_table_pipeline"]
)
def test_both_pipelines_find_an_empty_neighbourhood_the_surest_lift(
    request, asker, pipeline_name
):
    pipeline = request.getfixturevalue(pipeline_name)
    outcome = asker.ask(pipeline, CrowdingCausesLift(RECORDED_NEIGHBOUR_COUNT))

    assert outcome.answered
    adjusted = _adjusted_by_region(outcome)
    assert max(adjusted, key=adjusted.get) == "0"


def test_relational_pipeline_answers_about_a_clutter_of_another_size(
    relational_pipeline, asker
):
    outcome = asker.ask(
        relational_pipeline, FrictionCausesLift(RECORDED_NEIGHBOUR_COUNT + 2)
    )
    assert outcome.answered
    assert len(outcome.effects) == len(FrictionLadder().levels)


def test_flat_table_pipeline_refuses_a_clutter_of_another_size(
    flat_table_pipeline, asker
):
    outcome = asker.ask(
        flat_table_pipeline, FrictionCausesLift(RECORDED_NEIGHBOUR_COUNT + 2)
    )
    assert not outcome.answered
    assert outcome.refusal == Refusal.SCHEMA_MISMATCH


def test_relational_pipeline_answers_a_question_about_one_neighbour(
    relational_pipeline, asker
):
    outcome = asker.ask(
        relational_pipeline,
        ClosingAxisSideCausesDisturbance(RECORDED_NEIGHBOUR_COUNT, neighbour_index=2),
    )
    assert outcome.answered
    assert set(_adjusted_by_region(outcome)) == {"along", "across"}


def test_each_cause_gets_its_own_model(relational_pipeline, schema, asker):
    """
    Every distinct cause asked about fitted one further model, on top of the plain one.
    """
    asker.ask(relational_pipeline, FrictionCausesLift(RECORDED_NEIGHBOUR_COUNT))
    asker.ask(relational_pipeline, CrowdingCausesLift(RECORDED_NEIGHBOUR_COUNT))

    assert relational_pipeline.fit_report.model_count == 1 + len(
        relational_pipeline.cause_models
    )
    assert set(relational_pipeline.cause_models) >= {
        schema.scene_column("friction_coefficient"),
        schema.aggregation_column("crowding_count"),
    }


# %% likelihood


@pytest.mark.parametrize(
    "pipeline_name", ["relational_pipeline", "flat_table_pipeline"]
)
def test_held_out_likelihood_covers_some_attempts(request, scenes, pipeline_name):
    pipeline = request.getfixturevalue(pipeline_name)
    report = pipeline.log_likelihood(scenes[120:])
    assert report.scene_count == 30
    assert 0 < report.covered_scene_count <= report.scene_count
    assert np.isfinite(report.mean_log_likelihood)


def test_flat_table_pipeline_cannot_score_a_clutter_of_another_size(
    flat_table_pipeline,
):
    other_size = synthetic_clutter_pick_scenes(
        np.random.default_rng(1), scene_count=5, object_count=RECORDED_NEIGHBOUR_COUNT
    )
    report = flat_table_pipeline.log_likelihood(other_size)
    assert report.covered_scene_count == 0
