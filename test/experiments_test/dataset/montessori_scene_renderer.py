"""
Render a Montessori scene of known geometry into a camera image, so perception can be
tested against ground truth it was never told.

The camera looks at the table from off to one side, the way the real one does, so a
piece standing on the table shows its sides as well as its top and its silhouette comes
out stretched -- the very effect
:meth:`~experiments.montessori.perception.pipeline.LoosePieceDetector.detect` has to
cancel.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import cv2
import numpy as np
from typing_extensions import List, Sequence, Tuple

from experiments.montessori.hole_geometry import HoleFootprint, detect_hole_footprints
from experiments.montessori.perception.camera import CameraIntrinsics, RgbdFrame
from experiments.montessori.pieces import KNOWN_PIECE_BY_CATEGORY, KnownPiece
from experiments.montessori.semantics import MontessoriShapeCategory
from semantic_digital_twin.spatial_types.spatial_types import (
    HomogeneousTransformationMatrix,
)

# %% how the scene looks

TABLE_COLOR = (30, 13, 156)
"""
Hue, saturation and value of the bare metal table, measured off the real one.
"""

LID_COLOR = (19, 53, 204)
"""
Hue, saturation and value of the board's wooden lid, measured off the real one.
"""

HOLE_COLOR = (17, 149, 79)
"""
Hue, saturation and value seen through a hole in the lid, measured off the real board.
"""

PIECE_SATURATION = 115
"""
Saturation of a loose piece, measured off the real ones.
"""

PIECE_BRIGHTNESS = 220
"""
Value of a loose piece's lit top face, measured off the real ones.
"""

PIECE_SIDE_BRIGHTNESS = 150
"""
Value of a loose piece's shaded sides, measured off the real ones.

The light comes from above, so a piece's sides read markedly darker than the face it
turns to it, and the crease between the two is an edge perception can find.
"""


def piece_color(piece: KnownPiece) -> Tuple[int, int, int]:
    """
    Hue, saturation and value one loose piece's top face is drawn in.

    Each piece carries its own measured hue, since that is what tells the two colours in
    this set apart.

    :param piece: The piece to colour.
    """
    return (piece.hue, PIECE_SATURATION, PIECE_BRIGHTNESS)


def piece_side_color(piece: KnownPiece) -> Tuple[int, int, int]:
    """
    Hue, saturation and value one loose piece's sides are drawn in.

    :param piece: The piece to colour.
    """
    return (piece.hue, PIECE_SATURATION, PIECE_SIDE_BRIGHTNESS)


# %% what is in the scene


@dataclass(frozen=True)
class PlacedPiece:
    """
    One loose piece standing on the table, at a position the test chose.
    """

    category: MontessoriShapeCategory
    """
    Which shape it is.
    """

    x: float
    """
    Where its centre stands along the world frame's x-axis, in metres.
    """

    y: float
    """
    Where its centre stands along the world frame's y-axis, in metres.
    """

    yaw: float = 0.0
    """
    How far it is turned about the world frame's z-axis, in radians.
    """

    @property
    def known_piece(self) -> KnownPiece:
        """
        The piece of this kind the physical set contains, which fixes its outline, its
        height and its colour.
        """
        return KNOWN_PIECE_BY_CATEGORY[self.category]


@dataclass
class MontessoriSceneRenderer:
    """
    Draws a Montessori scene of known geometry as the camera would see it.
    """

    table_height: float = 0.88
    """
    Height of the table's top surface above the world frame's origin, in metres.
    """

    board_height: float = 0.08
    """
    How far the board's lid stands above the table, in metres.
    """

    reflection_spread: float = 0.0
    """
    How far, in metres, the table smears a piece's own colour across itself around the
    piece, which is what a polished metal table does and a matte one does not.

    The smear carries the piece's colour and fades away with no boundary anywhere, so it
    joins the piece in anything segmented by colour while leaving no edge of its own.
    """

    board_x: float = 0.83
    """
    Where the board's centre stands along the world frame's x-axis, in metres.
    """

    board_y: float = 0.12
    """
    Where the board's centre stands along the world frame's y-axis, in metres.
    """

    image_width: int = 1920
    """
    Width of the rendered image, matching the real camera.
    """

    image_height: int = 1080
    """
    Height of the rendered image, matching the real camera.
    """

    intrinsics: CameraIntrinsics = field(
        default_factory=lambda: CameraIntrinsics(
            focal_length_x=1117.02,
            focal_length_y=1116.93,
            principal_point_x=950.57,
            principal_point_y=527.61,
        )
    )
    """
    The real camera's own intrinsics.
    """

    camera_pose: Tuple[float, ...] = (
        0.4089,
        -0.0473,
        1.8153,
        -0.7024,
        0.7053,
        -0.0710,
        0.0644,
    )
    """
    Where the real camera stands, as a position and a quaternion in the world frame.
    """

    @property
    def lid_height(self) -> float:
        """
        Height of the board's lid above the world frame's origin, in metres.
        """
        return self.table_height + self.board_height

    @property
    def world_T_camera(self) -> np.ndarray:
        """
        The camera's pose as a 4x4 homogeneous transformation.
        """
        return HomogeneousTransformationMatrix.from_xyz_quaternion(
            *self.camera_pose
        ).to_np()

    def hole_footprints(self) -> List[HoleFootprint]:
        """
        The holes cut into the board this renderer draws, read from the board's own
        mesh.
        """
        return detect_hole_footprints()

    def hole_center(self, footprint: HoleFootprint) -> Tuple[float, float]:
        """
        Where a hole's centre lies in the world frame.

        :param footprint: The hole, positioned relative to the board's own centre.
        """
        return (self.board_x + footprint.center.x, self.board_y + footprint.center.y)

    def render(self, pieces: Sequence[PlacedPiece]) -> RgbdFrame:
        """
        Draw the scene as the camera sees it.

        :param pieces: The loose pieces to place on the table.
        :return: The frame, with a depth image left empty since the real table is too
            reflective to return one.
        """
        canvas = np.zeros((self.image_height, self.image_width, 3), dtype=np.uint8)
        canvas[:, :] = TABLE_COLOR
        self._draw_board(canvas)
        for piece in pieces:
            self._draw_reflection(canvas, piece)
        for piece in pieces:
            self._draw_piece(canvas, piece)
        return RgbdFrame(
            color=cv2.cvtColor(canvas, cv2.COLOR_HSV2BGR),
            depth=np.zeros((self.image_height, self.image_width), dtype=np.float32),
            intrinsics=self.intrinsics,
            reference_frame_T_camera=self.world_T_camera,
        )

    def _draw_board(self, canvas: np.ndarray) -> None:
        """
        Draw the board's lid and the holes cut through it.

        :param canvas: The hue-saturation-value image to draw into.
        """
        footprints = self.hole_footprints()
        half_width = max(abs(point[0]) for point in self._lid_corners(footprints))
        half_length = max(abs(point[1]) for point in self._lid_corners(footprints))
        lid = [
            (self.board_x + sign_x * half_width, self.board_y + sign_y * half_length)
            for sign_x, sign_y in ((-1, -1), (1, -1), (1, 1), (-1, 1))
        ]
        self._fill(canvas, lid, self.lid_height, LID_COLOR)
        for footprint in footprints:
            center = self.hole_center(footprint)
            self._fill(
                canvas,
                [
                    (center[0] + point.x, center[1] + point.y)
                    for point in footprint.boundary
                ],
                self.lid_height,
                HOLE_COLOR,
            )

    @staticmethod
    def _lid_corners(
        footprints: Sequence[HoleFootprint],
    ) -> List[Tuple[float, float]]:
        """
        Corners of a lid drawn just large enough to hold every hole with a margin.

        :param footprints: The holes the lid must hold.
        """
        margin = 0.02
        return [
            (
                footprint.center.x + sign_x * (footprint.size.x / 2 + margin),
                footprint.center.y + sign_y * (footprint.size.y / 2 + margin),
            )
            for footprint in footprints
            for sign_x, sign_y in ((-1, -1), (1, -1), (1, 1), (-1, 1))
        ]

    def _draw_reflection(self, canvas: np.ndarray, piece: PlacedPiece) -> None:
        """
        Smear one piece's colour across the table around it.

        The smear takes the piece's own hue and fades from the piece's brightness back
        into the table's, so it has no edge for a fit to catch on.

        :param canvas: The hue-saturation-value image to draw into.
        :param piece: The piece whose colour is smeared.
        """
        if self.reflection_spread <= 0.0:
            return
        known = piece.known_piece
        boundary = self._piece_boundary(piece)
        spread = self._pixels_across(boundary, self.reflection_spread)
        cast = np.zeros(canvas.shape[:2], dtype=np.uint8)
        cv2.fillPoly(
            cast, [self._project(boundary, self.table_height).astype(np.int32)], 255
        )
        cast = cv2.GaussianBlur(
            cv2.dilate(cast, np.ones((spread, spread), np.uint8)),
            (2 * spread + 1, 2 * spread + 1),
            0,
        )
        share = (cast.astype(float) / 255)[..., None]
        smeared = np.array(piece_side_color(known), dtype=float)
        lit = cast > 0
        canvas[lit] = (
            (1 - share) * np.array(TABLE_COLOR, dtype=float) + share * smeared
        ).astype(np.uint8)[lit]
        canvas[lit, 0] = known.hue

    def _pixels_across(
        self, boundary: Sequence[Tuple[float, float]], reach: float
    ) -> int:
        """
        How many pixels of the camera image a distance on the table covers, measured
        where a piece stands.

        :param boundary: The piece's world-frame ``(x, y)`` outline.
        :param reach: The distance to convert, in metres.
        :return: The distance in pixels, at least one.
        """
        here = self._project(boundary[:1], self.table_height)
        there = self._project(
            [(boundary[0][0] + reach, boundary[0][1])], self.table_height
        )
        return max(1, int(round(float(np.linalg.norm(there - here)))))

    def _draw_piece(self, canvas: np.ndarray, piece: PlacedPiece) -> None:
        """
        Draw one loose piece as the camera sees it: the sides standing on its base, and
        its lit top face over them.

        :param canvas: The hue-saturation-value image to draw into.
        :param piece: The piece to draw.
        """
        known = piece.known_piece
        boundary = self._piece_boundary(piece)
        base = self._project(boundary, self.table_height)
        top = self._project(boundary, self.table_height + known.height)
        silhouette = cv2.convexHull(np.vstack([base, top]).astype(np.int32))
        cv2.fillPoly(canvas, [silhouette], piece_side_color(known))
        cv2.fillPoly(canvas, [top.astype(np.int32)], piece_color(known))

    @staticmethod
    def _piece_boundary(piece: PlacedPiece) -> List[Tuple[float, float]]:
        """
        A piece's own outline in the world frame, as the physical piece was measured and
        turned to where it was placed.

        :param piece: The piece to outline.
        """
        turned = piece.known_piece.turned_outline(piece.yaw)
        return [(piece.x + x, piece.y + y) for x, y in turned]

    def _fill(
        self,
        canvas: np.ndarray,
        boundary: Sequence[Tuple[float, float]],
        height: float,
        color: Tuple[int, int, int],
    ) -> None:
        """
        Fill a flat polygon lying at a given height.

        :param canvas: The hue-saturation-value image to draw into.
        :param boundary: The polygon's world-frame ``(x, y)`` corners.
        :param height: Height of the plane it lies on, in metres.
        :param color: Hue, saturation and value to fill it with.
        """
        cv2.fillPoly(canvas, [self._project(boundary, height).astype(np.int32)], color)

    def _project(
        self, boundary: Sequence[Tuple[float, float]], height: float
    ) -> np.ndarray:
        """
        Project world-frame points on a horizontal plane into the camera image.

        :param boundary: The points' world-frame ``(x, y)``.
        :param height: Height of the plane they lie on, in metres.
        :return: The pixels they land on, shape ``(n, 2)``.
        """
        camera_T_world = np.linalg.inv(self.world_T_camera)
        world_points = np.array([[x, y, height, 1.0] for x, y in boundary], dtype=float)
        camera_points = (camera_T_world @ world_points.T)[:3]
        pixels = self.intrinsics.to_matrix() @ camera_points
        return (pixels[:2] / pixels[2]).T
