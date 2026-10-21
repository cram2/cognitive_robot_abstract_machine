from __future__ import annotations

from py_trees.common import Status
from py_trees.composites import Sequence
from semantic_digital_twin.world import World

from robokudo.annotators.core import BaseAnnotator
from robokudo.camera_placement import CameraPlacement, SharedWorldCameraPlacement
from robokudo.pipeline import Pipeline


class ReadsItsWorld(BaseAnnotator):
    """
    An annotator that only needs the world it perceives into.
    """

    def __init__(self) -> None:
        super().__init__(name="ReadsItsWorld")

    def update(self) -> Status:
        return Status.SUCCESS


# %% world of a pipeline


class TestWorldOfAPipeline:
    def test_pipeline_without_a_world_perceives_into_an_empty_world_of_its_own(
        self,
    ) -> None:
        assert Pipeline("Pipeline").world.is_empty()

    def test_pipeline_perceives_into_the_given_world(self) -> None:
        world = World()

        assert Pipeline("Pipeline", world=world).world is world

    def test_pipeline_perceives_into_a_world_handed_to_it_later(self) -> None:
        pipeline = Pipeline("Pipeline")
        world = World()

        pipeline.world = world

        assert pipeline.world is world

    def test_nested_pipeline_perceives_into_the_world_of_the_outer_one(self) -> None:
        world = World()
        outer = Pipeline("Outer", world=world)
        inner = Pipeline("Inner", world=World())
        outer.add_child(inner)

        assert inner.world is world


# %% camera placement of a pipeline


class TestCameraPlacementOfAPipeline:
    def test_pipeline_without_a_camera_placement_places_into_its_own_world(
        self,
    ) -> None:
        pipeline = Pipeline("Pipeline")
        world = World()
        pipeline.world = world

        camera_placement = pipeline.camera_placement

        assert type(camera_placement) is CameraPlacement
        assert camera_placement.world is world

    def test_pipeline_places_with_the_given_camera_placement(self) -> None:
        camera_placement = SharedWorldCameraPlacement(World())
        pipeline = Pipeline("Pipeline")

        pipeline.camera_placement = camera_placement

        assert pipeline.camera_placement is camera_placement

    def test_nested_pipeline_places_with_the_camera_placement_of_the_outer_one(
        self,
    ) -> None:
        camera_placement = SharedWorldCameraPlacement(World())
        outer = Pipeline("Outer")
        outer.camera_placement = camera_placement
        inner = Pipeline("Inner")
        outer.add_child(inner)

        assert inner.camera_placement is camera_placement


# %% world of an annotator


class TestWorldOfAnAnnotator:
    def test_annotator_perceives_into_the_world_of_its_pipeline(self) -> None:
        world = World()
        annotator = ReadsItsWorld()
        Pipeline("Pipeline", world=world).add_child(annotator)

        assert annotator.world is world

    def test_annotator_below_a_sequence_perceives_into_the_world_of_its_pipeline(
        self,
    ) -> None:
        world = World()
        annotator = ReadsItsWorld()
        sequence = Sequence(name="Sequence", memory=True)
        sequence.add_child(annotator)
        Pipeline("Pipeline", world=world).add_child(sequence)

        assert annotator.world is world
