"""
Tests for the views perception takes of a frame: rectified onto a plane, cut down to the
workspace, and drawn on.
"""

from __future__ import annotations

import math

import cv2
import numpy as np
import pytest
from typing_extensions import List, Tuple

from experiments.montessori.perception.camera import CameraIntrinsics, RgbdFrame
from experiments.montessori.perception.colors import (
    HOLE_COLOR,
    PIECE_COLOR,
    DetectionColor,
)
from experiments.montessori.perception.detections import MontessoriScene
from experiments.montessori.perception.exceptions import WorkspaceOutOfView
from experiments.montessori.perception.orthophoto import (
    OrthophotoProjector,
    WorkspaceRegion,
)
from experiments.montessori.perception.overlay import (
    CameraView,
    DetectionOverlay,
    RectifiedView,
    project_to_pixels,
)
from experiments.montessori.perception.pipeline import MontessoriPerceptionPipeline
from experiments.montessori.pieces import KnownPiece
from experiments.montessori.semantics import MontessoriShapeCategory

from .dataset import montessori_scene_fixtures
from .dataset.montessori_scene_renderer import (
    MontessoriSceneRenderer,
    PlacedPiece,
    piece_color,
)

pytest_plugins = [montessori_scene_fixtures.__name__]
"""
The rendered scene and pipeline fixtures the tests below ask for by name.
"""

# %% rectification


def test_rectified_plane_puts_a_known_point_back_where_it_came_from(
    renderer: MontessoriSceneRenderer,
):
    region = WorkspaceRegion(
        minimum_x=0.35, maximum_x=1.35, minimum_y=-0.45, maximum_y=0.75
    )
    frame = renderer.render([])
    world_x, world_y = 0.83, 0.12

    homography = OrthophotoProjector.pixel_T_region(frame, renderer.lid_height)
    pixel = homography @ np.array([world_x, world_y, 1.0])
    recovered = np.linalg.inv(homography) @ pixel

    assert recovered[0] / recovered[2] == pytest.approx(world_x, abs=1e-9)
    assert recovered[1] / recovered[2] == pytest.approx(world_y, abs=1e-9)


def test_a_hole_lands_at_its_own_world_position_in_the_rectified_lid(
    renderer: MontessoriSceneRenderer,
):
    region = WorkspaceRegion(
        minimum_x=0.35, maximum_x=1.35, minimum_y=-0.45, maximum_y=0.75
    )
    orthophoto = OrthophotoProjector(region=region).project(
        renderer.render([]), renderer.lid_height
    )
    [widest] = sorted(
        renderer.hole_footprints(), key=lambda hole: -hole.size.x * hole.size.y
    )[:1]
    expected_x, expected_y = renderer.hole_center(widest)

    darkness = cv2.cvtColor(orthophoto.image, cv2.COLOR_BGR2HSV)[:, :, 2]
    hole_mask = ((darkness > 0) & (darkness < 120)).astype(np.uint8)
    contours, _ = cv2.findContours(
        hole_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    centers = [orthophoto.contour_center(contour) for contour in contours]

    assert min(
        math.hypot(x - expected_x, y - expected_y) for x, y in centers
    ) == pytest.approx(0.0, abs=0.003)


# %% drawing the detections onto the frame


@pytest.fixture
def frame(
    renderer: MontessoriSceneRenderer, placed_pieces: list[PlacedPiece]
) -> RgbdFrame:
    return renderer.render(placed_pieces)


def looking_straight_down(height: float) -> np.ndarray:
    """
    A camera hanging at a height above the world origin, pointing down.

    :param height: How far above the origin it hangs, in metres.
    :return: Its pose as a 4x4 homogeneous transformation.
    """
    pose = np.eye(4)
    pose[:3, :3] = np.array([[1.0, 0.0, 0.0], [0.0, -1.0, 0.0], [0.0, 0.0, -1.0]])
    pose[:3, 3] = (0.0, 0.0, height)
    return pose


def frame_seen_from_above(height: float = 1.0) -> RgbdFrame:
    """
    An empty frame taken by a camera hanging above the world origin.

    :param height: How far above the origin the camera hangs, in metres.
    """
    return RgbdFrame(
        color=np.zeros((480, 640, 3), dtype=np.uint8),
        depth=np.full((480, 640), height, dtype=np.float32),
        intrinsics=CameraIntrinsics(100.0, 100.0, 320.0, 240.0),
        reference_frame_T_camera=looking_straight_down(height),
    )


def test_the_point_under_the_camera_lands_on_its_principal_point():
    pixels = project_to_pixels(frame_seen_from_above(), np.array([[0.0, 0.0]]), 0.0)

    assert pixels[0] == pytest.approx([320.0, 240.0])


def test_a_point_in_space_lands_where_its_own_plane_puts_it():
    frame = frame_seen_from_above()
    standing_at = 0.3

    pixels = frame.project(np.array([[0.1, 0.2, standing_at]]))

    assert pixels[0] == pytest.approx(
        project_to_pixels(frame, np.array([[0.1, 0.2]]), standing_at)[0]
    )


def test_a_point_beside_the_camera_axis_lands_beside_the_principal_point():
    pixels = project_to_pixels(frame_seen_from_above(), np.array([[0.1, 0.0]]), 0.0)

    assert pixels[0] == pytest.approx([330.0, 240.0])


def test_a_piece_rests_half_its_height_below_its_own_pose(scene: MontessoriScene):
    piece = scene.shapes[0]

    assert piece.surface_height == pytest.approx(
        float(piece.pose.to_position().to_np()[2]) - piece.height / 2
    )


def test_a_hole_lies_on_the_surface_its_pose_names(scene: MontessoriScene):
    hole = scene.holes[0]

    assert hole.surface_height == pytest.approx(
        float(hole.pose.to_position().to_np()[2])
    )


def test_drawing_nothing_leaves_the_frame_as_it_was():
    frame = frame_seen_from_above()

    drawn = DetectionOverlay().draw(CameraView(frame), MontessoriScene())

    assert np.array_equal(drawn, frame.color)


def test_the_overlay_leaves_the_frame_it_drew_from_untouched(
    frame: RgbdFrame, scene: MontessoriScene
):
    before = frame.color.copy()

    DetectionOverlay().draw(CameraView(frame), scene)

    assert np.array_equal(frame.color, before)


def test_each_kind_of_detection_is_drawn_in_its_own_colour(
    frame: RgbdFrame, scene: MontessoriScene
):
    drawn = DetectionOverlay().draw(CameraView(frame), scene)

    for color in (PIECE_COLOR, HOLE_COLOR):
        assert (drawn == np.array(color.to_bgr(), dtype=np.uint8)).all(axis=2).any()


def _extent_drawn_in(
    image: np.ndarray, color: DetectionColor
) -> Tuple[int, int, int, int]:
    """
    The bounding box of everything one colour was drawn in.

    :param image: The image that was drawn on.
    :param color: The colour to look for.
    :return: Its leftmost, topmost, rightmost and bottommost pixel.
    """
    marked = (image == np.array(color.to_bgr(), dtype=np.uint8)).all(axis=2)
    rows, columns = np.nonzero(marked)
    return columns.min(), rows.min(), columns.max(), rows.max()


def test_a_standing_piece_is_boxed_around_the_top_face_the_camera_sees(
    renderer: MontessoriSceneRenderer, pipeline: MontessoriPerceptionPipeline
):
    frame = renderer.render([PlacedPiece(MontessoriShapeCategory.CUBE, x=0.58, y=0.15)])
    [piece] = pipeline.detect(frame).shapes
    view = CameraView(frame)

    drawn = DetectionOverlay().draw(view, MontessoriScene(shapes=[piece]))

    left, top, right, bottom = _extent_drawn_in(drawn, PIECE_COLOR)
    for height in (piece.surface_height, piece.top_height):
        pixels = view.to_pixels(piece.outline, height)
        assert pixels[:, 0].min() >= left
        assert pixels[:, 0].max() <= right
        assert pixels[:, 1].min() >= top
        assert pixels[:, 1].max() <= bottom


# %% cutting the view down to the workspace


def _colored_pixels(image: np.ndarray, piece: KnownPiece) -> int:
    """
    How many pixels of an image the renderer drew in one piece's own colour.

    :param image: The image to count in, blue/green/red.
    :param piece: The piece whose colour is counted.
    """
    wanted = cv2.cvtColor(
        np.array([[piece_color(piece)]], dtype=np.uint8), cv2.COLOR_HSV2BGR
    )[0, 0]
    return int((image == wanted).all(axis=2).sum())


def test_the_view_is_cut_down_to_the_workspace(
    renderer: MontessoriSceneRenderer,
    pipeline: MontessoriPerceptionPipeline,
    placed_pieces: List[PlacedPiece],
):
    frame = renderer.render(placed_pieces)

    clipped = pipeline.workspace.clip(frame.color, frame)

    assert clipped.shape[0] < frame.height
    assert clipped.shape[1] < frame.width


def test_everything_standing_in_the_workspace_stays_in_view(
    renderer: MontessoriSceneRenderer,
    pipeline: MontessoriPerceptionPipeline,
    placed_pieces: List[PlacedPiece],
):
    frame = renderer.render(placed_pieces)

    clipped = pipeline.workspace.clip(frame.color, frame)

    for placed in placed_pieces:
        assert _colored_pixels(clipped, placed.known_piece) == _colored_pixels(
            frame.color, placed.known_piece
        )


def test_the_depth_image_is_cut_down_the_same_way_as_the_colour_one(
    renderer: MontessoriSceneRenderer,
    pipeline: MontessoriPerceptionPipeline,
    placed_pieces: List[PlacedPiece],
):
    frame = renderer.render(placed_pieces)
    workspace = pipeline.workspace

    assert (
        workspace.clip(frame.depth, frame).shape
        == workspace.clip(frame.color, frame).shape[:2]
    )


def test_a_camera_that_is_not_looking_at_the_workspace_has_nothing_to_show(
    renderer: MontessoriSceneRenderer, pipeline: MontessoriPerceptionPipeline
):
    elsewhere = looking_straight_down(pipeline.table_height + 1.0)
    elsewhere[0, 3] = 100.0
    frame = RgbdFrame(
        color=np.zeros((64, 64, 3), dtype=np.uint8),
        depth=np.zeros((64, 64), dtype=np.float32),
        intrinsics=renderer.intrinsics,
        reference_frame_T_camera=elsewhere,
    )

    with pytest.raises(WorkspaceOutOfView):
        pipeline.workspace.clip(frame.color, frame)


# %% the rectified view


def test_a_point_on_the_rectified_plane_lands_on_the_pixel_that_samples_it(
    frame: RgbdFrame, pipeline: MontessoriPerceptionPipeline
):
    orthophoto = pipeline.rectify_table(frame)
    x, y = 0.6, 0.2

    [pixel] = RectifiedView(frame, orthophoto).to_pixels(
        np.array([[x, y]]), orthophoto.plane_height
    )

    assert orthophoto.region.to_world_position(*pixel) == pytest.approx((x, y))


def test_a_point_above_the_rectified_plane_lands_further_from_the_camera_axis(
    frame: RgbdFrame, pipeline: MontessoriPerceptionPipeline
):
    orthophoto = pipeline.rectify_table(frame)
    view = RectifiedView(frame, orthophoto)
    below_camera = np.array(frame.reference_frame_T_camera[:2, 3]).reshape(1, 2)
    point = below_camera + np.array([[0.2, 0.0]])

    on_plane, above_plane = (
        view.to_pixels(point, height)[0]
        for height in (orthophoto.plane_height, orthophoto.plane_height + 0.03)
    )
    [nadir] = view.to_pixels(below_camera, orthophoto.plane_height)

    assert np.linalg.norm(above_plane - nadir) > np.linalg.norm(on_plane - nadir)


def test_the_rectified_view_draws_on_the_rectified_image(
    frame: RgbdFrame, pipeline: MontessoriPerceptionPipeline, scene: MontessoriScene
):
    orthophoto = pipeline.rectify_table(frame)

    drawn = DetectionOverlay().draw(RectifiedView(frame, orthophoto), scene)

    assert drawn.shape == orthophoto.image.shape
    assert (drawn == np.array(PIECE_COLOR.to_bgr(), dtype=np.uint8)).all(axis=2).any()
