"""
Tests for recognising which known piece an outline is, which way it is turned, and what
colour it was seen in.
"""

from __future__ import annotations

import math

import cv2
import numpy as np
import pytest
from typing_extensions import List, Tuple

from experiments.montessori.perception.edges import EdgeDistances
from experiments.montessori.perception.detections import MontessoriScene
from experiments.montessori.perception.orthophoto import Orthophoto, WorkspaceRegion
from experiments.montessori.perception.piece_matcher import PieceMatcher
from experiments.montessori.perception.pipeline import (
    MontessoriPerceptionPipeline,
    SurfaceColors,
)
from experiments.montessori.pieces import (
    CYAN_HUE,
    HUE_RANGE,
    KNOWN_PIECES,
    KNOWN_PIECE_BY_CATEGORY,
    YELLOW_HUE,
    KnownPiece,
    hue_distance,
)
from experiments.montessori.semantics import MontessoriShapeCategory
from semantic_digital_twin.world_description.geometry import Color

from .dataset import montessori_scene_fixtures
from .dataset.montessori_scene_renderer import MontessoriSceneRenderer, PlacedPiece

pytest_plugins = [montessori_scene_fixtures.__name__]
"""
The rendered scene and pipeline fixtures the tests below ask for by name.
"""

# %% reading a piece's colour


def _painted(hue_saturation_value: np.ndarray) -> Orthophoto:
    """
    A rectified view of one flat patch painted a given colour.

    :param hue_saturation_value: The colour of every pixel, as a hue-saturation-value
        image.
    """
    return Orthophoto(
        image=cv2.cvtColor(hue_saturation_value, cv2.COLOR_HSV2BGR),
        region=WorkspaceRegion(
            minimum_x=0.0,
            maximum_x=0.001 * hue_saturation_value.shape[1],
            minimum_y=0.0,
            maximum_y=0.001 * hue_saturation_value.shape[0],
        ),
        plane_height=0.0,
    )


def test_a_region_is_read_as_the_colour_of_its_own_pixels():
    painted = np.zeros((8, 8, 3), dtype=np.uint8)
    painted[:, :4] = (CYAN_HUE, 200, 200)
    painted[:, 4:] = (YELLOW_HUE, 200, 200)
    region = np.zeros((8, 8), dtype=np.uint8)
    region[:, :4] = 255

    assert SurfaceColors().measure_hue(_painted(painted), region) == CYAN_HUE


def test_a_washed_out_region_carries_no_colour_to_read():
    colors = SurfaceColors()
    painted = np.zeros((8, 8, 3), dtype=np.uint8)
    painted[:, :] = (CYAN_HUE, colors.minimum_hue_saturation - 1, 250)

    assert colors.measure_hue(_painted(painted), np.full((8, 8), 255, np.uint8)) is None


def test_a_piece_is_coloured_the_pure_form_of_the_hue_it_was_measured_at():
    scarlet = KnownPiece(
        category=MontessoriShapeCategory.CUBE,
        outline=np.zeros((0, 2)),
        height=0.03,
        hue=0,
        rotation_period=None,
    )

    assert scarlet.color == Color(1.0, 0.0, 0.0)


def test_two_pieces_measured_at_one_hue_are_coloured_alike():
    cube = KNOWN_PIECE_BY_CATEGORY[MontessoriShapeCategory.CUBE]
    cylinder = KNOWN_PIECE_BY_CATEGORY[MontessoriShapeCategory.CYLINDER]
    rectangular_prism = KNOWN_PIECE_BY_CATEGORY[
        MontessoriShapeCategory.RECTANGULAR_PRISM
    ]

    assert cube.color == cylinder.color
    assert cube.color != rectangular_prism.color


def test_hue_is_measured_the_short_way_round_the_colour_circle():
    assert hue_distance(2, HUE_RANGE - 3) == 5
    assert hue_distance(20, 25) == 5


# %% recognising a piece and how it is turned

TURNABLE_PIECES = [piece for piece in KNOWN_PIECES if piece.rotation_period is not None]
"""
The pieces a turn can be told on at all, so the ones a period means something for.
"""

DRAWN_TABLE_COLOR = (60, 60, 60)
"""
Blue, green and red of the bare table a test draws a piece's top face onto.
"""

DRAWN_PIECE_COLOR = (200, 220, 230)
"""
Blue, green and red of the top face a test draws.
"""

CLEAN_FIT_AGREEMENT = 0.75
"""
How well a piece laid exactly over the outline it was drawn at agrees with it.

Short of one because an edge found in a millimetre-resolution picture lies about a pixel
off the line that drew it, which costs a share of every point of the outline.
"""


def _piece_id(piece: KnownPiece) -> str:
    return str(piece.category)


def _drawn(
    outline: np.ndarray, center: Tuple[float, float] = (0.0, 0.0)
) -> EdgeDistances:
    """
    The edges of a rectified view holding one outline drawn on a bare table.

    :param outline: The outline to draw, as ``(n, 2)`` ``(x, y)`` points in metres about
        its own centre.
    :param center: Where in the world frame to draw it, in metres.
    :return: The edges seen in that view.
    """
    reach = 0.1
    region = WorkspaceRegion(
        minimum_x=center[0] - reach,
        maximum_x=center[0] + reach,
        minimum_y=center[1] - reach,
        maximum_y=center[1] + reach,
    )
    image = np.zeros((region.height_in_pixels, region.width_in_pixels, 3), np.uint8)
    image[:, :] = DRAWN_TABLE_COLOR
    corners = np.stack(
        [
            (outline[:, 0] + center[0] - region.minimum_x) / region.resolution,
            (outline[:, 1] + center[1] - region.minimum_y) / region.resolution,
        ],
        axis=1,
    )
    cv2.fillPoly(image, [np.round(corners).astype(np.int32)], DRAWN_PIECE_COLOR)
    return EdgeDistances.of(Orthophoto(image=image, region=region, plane_height=0.0))


@pytest.mark.parametrize("piece", KNOWN_PIECES, ids=_piece_id)
def test_each_known_piece_is_recognised_from_its_own_outline(piece: KnownPiece):
    matcher = PieceMatcher()
    placed = math.radians(17)

    match = matcher.match(_drawn(piece.turned_outline(placed)), (0.0, 0.0), piece.hue)

    assert match.piece.category is piece.category
    assert match.outline_agreement > CLEAN_FIT_AGREEMENT
    assert match.yaw == pytest.approx(
        piece.smallest_equivalent_turn(placed), abs=matcher.angle_step
    )


@pytest.mark.parametrize("piece", TURNABLE_PIECES, ids=_piece_id)
def test_a_piece_turned_by_its_own_period_looks_untouched(piece: KnownPiece):
    matcher = PieceMatcher()

    match = matcher.match(
        _drawn(piece.turned_outline(piece.rotation_period)), (0.0, 0.0), piece.hue
    )

    assert match.piece.category is piece.category
    assert match.yaw == pytest.approx(0.0, abs=matcher.angle_step)


def test_an_orientation_is_reported_as_the_smallest_turn_that_reaches_it():
    cube = KNOWN_PIECE_BY_CATEGORY[MontessoriShapeCategory.CUBE]
    matcher = PieceMatcher()
    placed = cube.rotation_period - math.radians(10)

    match = matcher.match(_drawn(cube.turned_outline(placed)), (0.0, 0.0), cube.hue)

    assert match.yaw == pytest.approx(math.radians(-10), abs=matcher.angle_step)


@pytest.mark.parametrize("piece", KNOWN_PIECES, ids=_piece_id)
def test_a_piece_is_found_where_it_stands_and_not_where_it_was_looked_for(
    piece: KnownPiece,
):
    """
    A piece read together with its reflection is seeded from the middle of the two, so
    the fit has to walk to the piece itself.
    """
    matcher = PieceMatcher()
    stands_at = (0.6, 0.2)
    looked_for = (stands_at[0] - 0.012, stands_at[1] + 0.009)

    match = matcher.match(_drawn(piece.outline, stands_at), looked_for, piece.hue)

    assert match.center == pytest.approx(stands_at, abs=matcher.step)


def test_a_piece_is_never_recognised_as_one_of_the_other_colour():
    cylinder = KNOWN_PIECE_BY_CATEGORY[MontessoriShapeCategory.CYLINDER]
    matcher = PieceMatcher()
    edges = _drawn(cylinder.outline)

    recognised = matcher.match(edges, (0.0, 0.0), cylinder.hue)
    seen_yellow = matcher.match(edges, (0.0, 0.0), YELLOW_HUE)

    assert recognised.piece.category is MontessoriShapeCategory.CYLINDER
    assert seen_yellow is None or seen_yellow.piece.hue == YELLOW_HUE


def test_a_colour_no_piece_wears_leaves_nothing_to_recognise():
    cube = KNOWN_PIECE_BY_CATEGORY[MontessoriShapeCategory.CUBE]
    unworn = (CYAN_HUE + YELLOW_HUE) // 2

    assert PieceMatcher().match(_drawn(cube.outline), (0.0, 0.0), unworn) is None


def test_edges_no_known_piece_follows_are_refused():
    reach = max(piece.radius for piece in KNOWN_PIECES) * 1.5
    sprawl = np.array(
        [[-reach, -reach], [reach, -reach], [reach, reach], [-reach, reach]]
    )

    assert PieceMatcher().match(_drawn(sprawl), (0.0, 0.0), CYAN_HUE) is None


def test_an_outline_with_no_colour_to_read_is_recognised_by_its_shape_alone():
    triangle = KNOWN_PIECE_BY_CATEGORY[MontessoriShapeCategory.TRIANGULAR_PRISM]

    match = PieceMatcher().match(_drawn(triangle.outline), (0.0, 0.0), None)

    assert match.piece.category is MontessoriShapeCategory.TRIANGULAR_PRISM


def test_a_reflection_around_a_piece_does_not_move_where_it_is_recognised():
    """
    The table throws a diffuse copy of each piece back at the camera, which segmenting
    by colour takes in along with the piece; the edges the fit follows are the piece's
    own.
    """
    cube = KNOWN_PIECE_BY_CATEGORY[MontessoriShapeCategory.CUBE]
    stands_at = (0.6, 0.2)
    edges = _drawn(cube.outline, stands_at)

    match = PieceMatcher().match(edges, (stands_at[0] - 0.015, stands_at[1]), cube.hue)

    assert match.piece.category is MontessoriShapeCategory.CUBE
    assert match.center == pytest.approx(stands_at, abs=0.002)


@pytest.mark.parametrize("piece", TURNABLE_PIECES, ids=_piece_id)
def test_a_piece_is_detected_at_the_angle_it_was_placed(
    renderer: MontessoriSceneRenderer,
    pipeline: MontessoriPerceptionPipeline,
    piece: KnownPiece,
):
    placed = math.radians(25)
    frame = renderer.render([PlacedPiece(piece.category, x=0.58, y=0.15, yaw=placed)])

    [detected] = pipeline.detect(frame).shapes

    assert detected.category is piece.category
    assert detected.yaw == pytest.approx(
        piece.smallest_equivalent_turn(placed), abs=math.radians(4)
    )


REFLECTION_SPREAD = 0.015
"""
How far this lab's table smears a piece's colour around it, in metres.

Measured off the outlines colour alone gives on the real table, where a twenty by forty
millimetre piece came out forty-seven by fifty-one; drawing the smear this wide brings
the rendered outlines to the same size.
"""


def test_a_piece_is_recognised_through_the_reflection_the_table_throws(
    pipeline: MontessoriPerceptionPipeline, placed_pieces: List[PlacedPiece]
):
    reflecting = MontessoriSceneRenderer(reflection_spread=REFLECTION_SPREAD)

    scene = pipeline.detect(reflecting.render(placed_pieces))

    assert {detected.category for detected in scene.shapes} == {
        placed.category for placed in placed_pieces
    }


def test_a_piece_is_reported_where_it_stands_and_not_where_its_reflection_reaches(
    pipeline: MontessoriPerceptionPipeline, placed_pieces: List[PlacedPiece]
):
    reflecting = MontessoriSceneRenderer(reflection_spread=REFLECTION_SPREAD)

    scene = pipeline.detect(reflecting.render(placed_pieces))

    stands_at = {placed.category: (placed.x, placed.y) for placed in placed_pieces}
    for detected in scene.shapes:
        assert detected.pose.to_position().to_np()[:2] == pytest.approx(
            stands_at[detected.category], abs=0.003
        )


def test_a_piece_only_half_in_view_is_not_reported(
    renderer: MontessoriSceneRenderer, placed_pieces: List[PlacedPiece]
):
    cut_through_a_piece = MontessoriPerceptionPipeline(
        region=WorkspaceRegion(
            minimum_x=0.35,
            maximum_x=1.35,
            minimum_y=placed_pieces[0].y,
            maximum_y=0.75,
        ),
        table_height=renderer.table_height,
        board_height=renderer.board_height,
    )

    scene = cut_through_a_piece.detect(renderer.render(placed_pieces))

    assert placed_pieces[0].category not in {
        detected.category for detected in scene.shapes
    }


def test_a_cleanly_seen_piece_reports_how_closely_it_fitted(scene: MontessoriScene):
    for detected in scene.shapes:
        assert detected.outline_agreement > PieceMatcher().minimum_agreement
