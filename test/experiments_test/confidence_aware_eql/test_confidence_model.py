"""Exercise the confidence model on mesh files this repository already ships.

The robocasa suite alongside this one covers the same pipeline on downloaded assets
and skips wherever they are absent. These meshes stand in for real objects so the
pipeline itself - the aggregation statistics, the fitted circuit, the calibrated
threshold - is covered on every run.
"""

from __future__ import annotations

import copy
from pathlib import Path

import pytest

from experiments.confidence_aware_eql.confidence_model import (
    AmbiguousThresholdError,
    ConfidenceModel,
    UnmodeledClassError,
)
from experiments.confidence_aware_eql.feature_pipeline import (
    Feature,
    ObjectClass,
    ObjectShape,
    ObjectShapeAggregations,
    extract_feature_dataframe,
)
from semantic_digital_twin.datastructures.prefixed_name import PrefixedName
from semantic_digital_twin.semantic_annotations.semantic_annotations import (
    Cup,
    Plate,
    Pot,
)
from semantic_digital_twin.world_description.geometry import Mesh
from semantic_digital_twin.world_description.shape_collection import ShapeCollection
from semantic_digital_twin.world_description.world_entity import (
    Body,
    SemanticAnnotation,
)
from typing_extensions import List, Type

# %% stand-in objects

MESH_DIRECTORY = (
    Path(__file__).resolve().parents[3] / "coraplex" / "resources" / "objects"
)

CUP_MESHES = (
    "jeroen_cup.stl",
    "Static_MilkPitcher.stl",
    "milk.stl",
    "breakfast_cereal.stl",
)
"""Watertight meshes standing in for the familiar instances of one class."""

POT_MESHES = ("bread.stl", "big-knife.stl", "whisk.stl")
"""Watertight meshes standing in for the familiar instances of another class."""


def annotated_object(
    annotation_class: Type[SemanticAnnotation], name: str, mesh_file: str
) -> SemanticAnnotation:
    """Build an object of ``annotation_class`` whose collision geometry is one mesh."""
    body = Body(name=PrefixedName(name))
    body.collision = ShapeCollection([Mesh(filename=str(MESH_DIRECTORY / mesh_file))])
    return annotation_class(name=PrefixedName(name), root=body)


@pytest.fixture(scope="module")
def cups() -> List[SemanticAnnotation]:
    """The familiar instances of the first class."""
    return [
        annotated_object(Cup, f"cup_{index}", mesh_file)
        for index, mesh_file in enumerate(CUP_MESHES)
    ]


@pytest.fixture(scope="module")
def pots() -> List[SemanticAnnotation]:
    """The familiar instances of the second class."""
    return [
        annotated_object(Pot, f"pot_{index}", mesh_file)
        for index, mesh_file in enumerate(POT_MESHES)
    ]


@pytest.fixture(scope="module")
def model(cups, pots) -> ConfidenceModel:
    """A confidence model fitted on both classes of familiar instances."""
    return ConfidenceModel.fit_from_instances(cups + pots)


# %% what the object description carries


def test_object_description_carries_the_collision_shapes(cups):
    """An object is described by the shapes of its own collision geometry."""
    described = ObjectShape.from_annotation(cups[0])
    assert described.shapes == list(cups[0].root.collision)


# %% what the model learns from


def test_geometric_features_are_the_aggregation_statistics(cups):
    """Every feature but the class is a statistic the aggregation class declares."""
    statistics = {
        statistic.__name__
        for statistic in ObjectShapeAggregations.aggregation_features_of_field(
            ObjectShapeAggregations.aggregated_field
        )
    }
    assert set(extract_feature_dataframe(cups)) == {Feature.CLASS} | statistics


def test_class_is_a_feature_of_its_own(cups):
    """An object's class is carried as a feature alongside the geometric statistics."""
    dataframe = extract_feature_dataframe(cups)
    assert list(dataframe[Feature.CLASS]) == [ObjectClass.CUP] * len(cups)


def test_one_model_is_fitted_per_class(model, cups, pots):
    """Instances of different classes are never pooled into one distribution."""
    assert set(model.models_by_class) == {type(cups[0]), type(pots[0])}


# %% judging familiarity


def test_familiar_object_is_accepted(model, cups):
    """An object the model was fitted on is judged familiar."""
    assert model.is_familiar(cups[0])


def test_every_training_instance_is_accepted(model, cups, pots):
    """The calibrated threshold admits the instances it was calibrated on."""
    assert all(model.is_familiar(instance) for instance in cups + pots)


def test_object_wearing_another_class_geometry_is_rejected(model, cups, pots):
    """An object whose collision geometry belongs to another class is judged unfamiliar.

    Its class still selects its own model, so the geometry is scored against the
    distribution it does not belong to - the out-of-distribution case the model
    exists to catch.
    """
    disguised = copy.deepcopy(cups[-1])
    disguised.root.collision = pots[0].root.collision
    assert not model.is_familiar(disguised)


# %% recognising objects it was not trained on


@pytest.fixture(scope="module")
def model_without_the_first_cup(cups, pots) -> ConfidenceModel:
    """A model fitted on every familiar instance except the first cup."""
    return ConfidenceModel.fit_from_instances(cups[1:] + pots)


def test_unseen_object_inside_the_learnt_range_is_accepted(
    model_without_the_first_cup, cups
):
    """An object the model never saw is familiar when its features fall in range.

    The first cup's volume and aspect ratio both lie between the other cups', so it
    belongs to the region they describe. A class circuit whose leaves each cover a
    single instance instead collapses onto those instances' exact values, which
    judges every unseen object unfamiliar and leaves the model unable to recognise
    anything it was not trained on.
    """
    assert model_without_the_first_cup.is_familiar(cups[0])


# %% thresholds


def test_threshold_of_a_class_admits_its_own_instances(model, cups):
    """An instance's score is at or above the threshold of its own class."""
    assert model.log_likelihood_of(cups[0]) >= model.threshold_for(cups[0])


def test_each_class_is_calibrated_on_its_own_distribution(model, cups, pots):
    """Each class's threshold comes from that class's own instances.

    Scoring an instance against a circuit already conditioned on the very statistics
    being scored makes every instance equally certain, which collapses both classes
    onto one threshold; a threshold that still tells the classes apart shows the
    score reflects the fitted distribution rather than the instance itself.
    """
    assert model.threshold_for(cups[0]) != model.threshold_for(pots[0])


def test_threshold_is_ambiguous_across_several_classes(model):
    """Reading one threshold from a model fitted on several classes is refused."""
    with pytest.raises(AmbiguousThresholdError):
        model.threshold


def test_scoring_an_unfitted_class_is_refused(model):
    """An instance of a class the model never saw cannot be scored."""
    with pytest.raises(UnmodeledClassError):
        model.log_likelihood_of(annotated_object(Plate, "plate", CUP_MESHES[0]))
