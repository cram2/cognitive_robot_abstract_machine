"""
Tests for the continuous Montessori perception pipeline and the query interface it
answers through.
"""

from __future__ import annotations

import math

import cv2
import numpy as np
import pytest

from experiments.montessori.perception.detections import (
    MontessoriScene,
    MontessoriShapeDetection,
    ShapeSortingHoleDetection,
)
from experiments.montessori.perception.pipeline import MontessoriPerceptionPipeline
from experiments.montessori.perception.scene_source import FixedScene, PerceivedObjects
from experiments.montessori.semantics import MontessoriShapeCategory
from krrood.entity_query_language.factories import a, the

from .dataset import montessori_scene_fixtures
from .dataset.montessori_scene_renderer import MontessoriSceneRenderer, PlacedPiece

pytest_plugins = [montessori_scene_fixtures.__name__]
"""
The rendered scene and pipeline fixtures the tests below ask for by name.
"""

# %% the pipeline


def test_pipeline_finds_every_hole_the_board_has(
    scene: MontessoriScene, renderer: MontessoriSceneRenderer
):
    assert scene.board is not None
    assert len(scene.holes) == len(renderer.hole_footprints())


def test_pipeline_puts_each_hole_within_three_millimetres_of_its_true_centre(
    scene: MontessoriScene, renderer: MontessoriSceneRenderer
):
    detected = [tuple(hole.pose.to_position().to_np()[:2]) for hole in scene.holes]

    for footprint in renderer.hole_footprints():
        expected_x, expected_y = renderer.hole_center(footprint)
        nearest = min(math.hypot(x - expected_x, y - expected_y) for x, y in detected)
        assert nearest == pytest.approx(0.0, abs=0.003)


def test_pipeline_reports_hole_centres_on_the_board_lid(
    scene: MontessoriScene, renderer: MontessoriSceneRenderer
):
    for hole in scene.holes:
        assert float(hole.pose.to_position().to_np()[2]) == pytest.approx(
            renderer.lid_height
        )


def test_pipeline_recognises_the_shape_of_the_widest_holes(
    scene: MontessoriScene, renderer: MontessoriSceneRenderer
):
    expected = {
        footprint.category
        for footprint in renderer.hole_footprints()
        if min(footprint.size.x, footprint.size.y) > 0.02
    }

    assert expected <= {hole.category for hole in scene.holes}


def test_pipeline_finds_each_loose_piece_where_it_stands(
    scene: MontessoriScene, placed_pieces: list[PlacedPiece]
):
    detected = [tuple(piece.pose.to_position().to_np()[:2]) for piece in scene.shapes]

    for placed in placed_pieces:
        nearest = min(math.hypot(x - placed.x, y - placed.y) for x, y in detected)
        assert nearest == pytest.approx(0.0, abs=0.006)


def test_pipeline_cancels_the_parallax_that_stretches_a_piece(
    scene: MontessoriScene, renderer: MontessoriSceneRenderer, placed_pieces
):
    [cube] = [
        placed
        for placed in placed_pieces
        if placed.category is MontessoriShapeCategory.CUBE
    ]
    [true_footprint] = [
        footprint
        for footprint in renderer.hole_footprints()
        if footprint.category is MontessoriShapeCategory.CUBE
    ]
    nearest = min(
        scene.shapes,
        key=lambda piece: math.hypot(
            float(piece.pose.to_position().to_np()[0]) - cube.x,
            float(piece.pose.to_position().to_np()[1]) - cube.y,
        ),
    )

    assert nearest.footprint.length == pytest.approx(
        max(true_footprint.size.x, true_footprint.size.y), abs=0.008
    )


def test_pipeline_does_not_report_the_board_lid_as_a_loose_piece(
    scene: MontessoriScene,
):
    assert scene.board is not None
    for piece in scene.shapes:
        position = piece.pose.to_position().to_np()
        assert not scene.board.encloses(float(position[0]), float(position[1]))


def test_pipeline_reports_no_board_when_none_is_in_view(
    pipeline: MontessoriPerceptionPipeline, renderer: MontessoriSceneRenderer
):
    empty = renderer.render([])
    empty.color[:, :] = cv2.cvtColor(
        np.full((1, 1, 3), (30, 13, 156), dtype=np.uint8), cv2.COLOR_HSV2BGR
    )[0, 0]

    assert pipeline.detect(empty).board is None


# %% querying it


def test_a_query_over_perceived_objects_runs_perception_to_answer_itself(
    scene: MontessoriScene,
):
    class CountingSource(FixedScene):
        looks: int = 0

        def scene(self) -> MontessoriScene:
            self.looks += 1
            return self.captured

    source = CountingSource(captured=scene)
    perceived = PerceivedObjects(source=source)
    query = a(MontessoriShapeDetection).from_(perceived)

    assert source.looks == 0
    results = query.tolist()
    assert source.looks == 1
    assert len(results) == len(scene.shapes)


def test_a_query_selects_a_hole_by_the_shape_it_takes(scene: MontessoriScene):
    perceived = PerceivedObjects(source=FixedScene(captured=scene))

    holes = (
        a(ShapeSortingHoleDetection)(category=MontessoriShapeCategory.CUBE)
        .from_(perceived)
        .tolist()
    )

    assert holes
    for hole in holes:
        assert hole.category is MontessoriShapeCategory.CUBE


def test_a_query_answers_a_pose_a_plan_can_reach_for(scene: MontessoriScene):
    perceived = PerceivedObjects(source=FixedScene(captured=scene))
    [expected] = [
        hole
        for hole in scene.holes
        if hole.category is MontessoriShapeCategory.TRIANGULAR_PRISM
    ][:1]

    hole = the(
        a(ShapeSortingHoleDetection)(category=MontessoriShapeCategory.TRIANGULAR_PRISM)
        .from_(perceived)
        .expression
    ).tolist()[0]

    assert hole.pose.to_position().to_np() == pytest.approx(
        expected.pose.to_position().to_np()
    )


def test_a_query_over_one_kind_does_not_return_the_other(scene: MontessoriScene):
    perceived = PerceivedObjects(source=FixedScene(captured=scene))

    pieces = a(MontessoriShapeDetection).from_(perceived).tolist()

    assert pieces
    assert all(not isinstance(piece, ShapeSortingHoleDetection) for piece in pieces)


# %% how tall a piece is taken to stand


def test_a_piece_the_depth_image_cannot_resolve_stands_at_its_nominal_height(
    pipeline: MontessoriPerceptionPipeline, scene: MontessoriScene
):
    nominal = pipeline.piece_detector.piece_height

    for piece in scene.shapes:
        assert piece.height == pytest.approx(nominal)
        assert piece.surface_height == pytest.approx(pipeline.table_height)
        assert piece.top_height == pytest.approx(pipeline.table_height + nominal)


def test_a_hole_has_no_thickness_to_stand_above_its_own_surface(
    scene: MontessoriScene,
):
    hole = scene.holes[0]

    assert hole.top_height == hole.surface_height
