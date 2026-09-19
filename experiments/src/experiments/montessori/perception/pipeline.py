"""
Find the Montessori board, its holes, and the loose pieces in one RGB-D frame.

The scene splits into two horizontal surfaces, and each is rectified and searched on its
own: the loose pieces rest on the table, while the holes are cut into the board's raised
lid. Rectifying each surface at its own height is what keeps a hole's centre accurate --
read off the table's plane instead, the lid's outline is magnified by the ratio of the
two camera distances, which walks the outer holes several centimetres away from where
they are.

Colour says where to look and edges say what is there. The pieces are brightly coloured
and the holes are dark openings in pale wood, both of which separate cleanly by
saturation and brightness, but the table under them is polished metal that throws a
coloured reflection of every piece back at the camera, so a coloured region is only ever
a place worth searching. Which piece stands there is settled by fitting the pieces this
set contains to the edges seen around that place. Depth is asked how tall a piece stands
and usually cannot say: the same reflections leave the depth image with large dropouts
and centimetre-scale noise, far too coarse to measure a thirty millimetre piece.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import cv2
import numpy as np
from typing_extensions import List, Optional, Tuple

from experiments.montessori.perception.camera import RgbdFrame
from experiments.montessori.perception.detections import (
    MontessoriBoardDetection,
    MontessoriDetection,
    MontessoriScene,
    MontessoriShapeDetection,
    ShapeSortingHoleDetection,
)
from experiments.montessori.perception.edges import EdgeDistances
from experiments.montessori.perception.footprint import (
    CrossSectionClassifier,
    Footprint,
    FootprintClassifier,
)
from experiments.montessori.perception.orthophoto import (
    Orthophoto,
    OrthophotoProjector,
    WorkspaceBox,
    WorkspaceRegion,
)
from experiments.montessori.perception.piece_matcher import PieceMatcher
from experiments.montessori.pieces import HUE_RANGE, HUE_TOLERANCE, PIECE_HUES
from experiments.montessori.world import BOARD_SCALE
from semantic_digital_twin.spatial_types.spatial_types import Pose
from semantic_digital_twin.world_description.world_entity import (
    KinematicStructureEntity,
)

# %% segmentation


@dataclass(frozen=True)
class SurfaceColors:
    """
    How the scene's surfaces separate in hue-saturation-value, measured off the table
    this runs on.

    The table is bare metal and so is almost colourless, while the wooden board and the
    plastic pieces both carry real colour; the holes are openings that fall into shadow
    and so are much darker than the lid around them.
    """

    minimum_saturation: int = 45
    """
    Saturation at or above which a pixel belongs to an object rather than to the bare
    table.

    The table reads around 13 and drops towards 6 under a specular highlight, while the
    palest thing standing on it, the board's own wood, reads around 53. Sitting just
    under that keeps the whole board and every piece while rejecting the coloured fringe
    that the specular streaks running down the table leave along their edges, which
    otherwise merges into a piece standing near one and inflates its outline.
    """

    minimum_value: int = 60
    """
    Brightness at or above which a coloured pixel belongs to an object at all, rather
    than to unlit background.

    Deliberately low: how dark a hole reads against its own lid varies with the light
    falling on the board, so the holes are cut out by comparing them against that lid
    (see :meth:`dark_mask`) rather than against a fixed brightness.
    """

    def surface_mask(self, orthophoto: Orthophoto) -> np.ndarray:
        """
        Mark the coloured surfaces in a rectified image, separating them from the bare
        table.

        :param orthophoto: The rectified image to segment.
        :return: A ``uint8`` mask, 255 on a surface and 0 elsewhere.
        """
        hue_saturation_value = cv2.cvtColor(orthophoto.image, cv2.COLOR_BGR2HSV)
        mask = (
            (hue_saturation_value[:, :, 1] >= self.minimum_saturation)
            & (hue_saturation_value[:, :, 2] >= self.minimum_value)
            & orthophoto.observed
        )
        return mask.astype(np.uint8) * 255

    minimum_hue_saturation: int = 30
    """
    Saturation at or above which a pixel's hue is worth reading.

    Lower than :attr:`minimum_saturation`, which has to tell a coloured surface from a
    colourless one on its own: a pixel held against the pieces' own colours need only be
    colourful enough for its hue to mean something, and the pale pieces in this set wash
    out towards white where they catch the light.
    """

    def piece_mask(self, orthophoto: Orthophoto) -> np.ndarray:
        """
        Mark the pixels wearing one of the loose pieces' own colours.

        A piece is picked out by the colour it is rather than by standing out from the
        surface under it, because that surface is whatever the table happens to be
        covered with: bare metal has no colour to be told apart from, but paper laid
        over it to stop the reflections has plenty.

        :param orthophoto: The rectified image to segment.
        :return: A ``uint8`` mask, 255 on a piece's colour and 0 elsewhere.
        """
        hue_saturation_value = cv2.cvtColor(orthophoto.image, cv2.COLOR_BGR2HSV)
        hue = hue_saturation_value[:, :, 0].astype(int)
        apart = np.stack([np.abs(hue - worn) for worn in PIECE_HUES], axis=0)
        wears_one = np.minimum(apart, HUE_RANGE - apart).min(axis=0) <= HUE_TOLERANCE
        mask = (
            wears_one
            & (hue_saturation_value[:, :, 1] >= self.minimum_hue_saturation)
            & (hue_saturation_value[:, :, 2] >= self.minimum_value)
            & orthophoto.observed
        )
        return mask.astype(np.uint8) * 255

    def measure_hue(self, orthophoto: Orthophoto, region: np.ndarray) -> Optional[int]:
        """
        Read the colour one region of a rectified image is.

        Only the coloured pixels are read: a specular highlight washes a lit face towards
        white, where the hue a pixel reports is noise. A piece's reflection in the table
        carries the piece's own colour, so a region that has taken some of that in still
        reports the colour of the piece.

        :param orthophoto: The rectified image the region was found in.
        :param region: Mask of the region to read, 255 inside it and 0 outside.
        :return: The middle of its coloured pixels' hue, or None where it has none.
        """
        hue_saturation_value = cv2.cvtColor(orthophoto.image, cv2.COLOR_BGR2HSV)
        colored = (region > 0) & (
            hue_saturation_value[:, :, 1] >= self.minimum_hue_saturation
        )
        if not colored.any():
            return None
        return int(np.median(hue_saturation_value[:, :, 0][colored]))

    @staticmethod
    def dark_mask(orthophoto: Orthophoto, region: np.ndarray) -> np.ndarray:
        """
        Mark the darker part of one surface, which for the board's lid is its holes.

        The split is chosen by Otsu's method over that surface alone, so it follows
        however brightly the board happens to be lit instead of assuming a level.

        :param orthophoto: The rectified image the surface was found in.
        :param region: Mask of the surface to split, 255 inside it and 0 outside.
        :return: A ``uint8`` mask, 255 on the darker part of the surface and 0
            elsewhere.
        """
        brightness = cv2.cvtColor(orthophoto.image, cv2.COLOR_BGR2HSV)[:, :, 2]
        inside = brightness[region > 0]
        if inside.size == 0:
            return np.zeros_like(region)
        threshold, _ = cv2.threshold(
            inside, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU
        )
        return (((brightness < threshold) & (region > 0)).astype(np.uint8)) * 255


_CLOSING_KERNEL = np.ones((5, 5), np.uint8)
"""
Structuring element that bridges the one- and two-pixel gaps a rectified edge picks up,
without closing the smallest hole on the board.
"""

_OPENING_KERNEL = np.ones((3, 3), np.uint8)
"""
Structuring element that removes isolated speckles left by sensor noise.
"""


def _filled(contour: np.ndarray, orthophoto: Orthophoto) -> np.ndarray:
    """
    The area one contour encloses, as a mask over the image it was found in.

    :param contour: The contour, in rectified pixels.
    :param orthophoto: The rectified view it was found in.
    :return: A ``uint8`` mask, 255 inside the contour and 0 outside.
    """
    region = np.zeros(orthophoto.image.shape[:2], dtype=np.uint8)
    cv2.drawContours(region, [contour], -1, 255, cv2.FILLED)
    return region


def _wholly_within(contour: np.ndarray, orthophoto: Orthophoto) -> bool:
    """
    Whether a contour lies inside the rectified view rather than running off its edge.

    :param contour: The contour, in rectified pixels.
    :param orthophoto: The rectified view it was found in.
    """
    height, width = orthophoto.image.shape[:2]
    pixels = contour.reshape(-1, 2)
    return bool(
        (pixels[:, 0] > 0).all()
        and (pixels[:, 0] < width - 1).all()
        and (pixels[:, 1] > 0).all()
        and (pixels[:, 1] < height - 1).all()
    )


def _clean(mask: np.ndarray) -> np.ndarray:
    """
    Bridge the gaps a rectified edge picks up and drop isolated speckles.

    :param mask: The mask to clean.
    :return: The cleaned mask.
    """
    closed = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, _CLOSING_KERNEL)
    return cv2.morphologyEx(closed, cv2.MORPH_OPEN, _OPENING_KERNEL)


# %% size expectations


@dataclass(frozen=True)
class SizeRange:
    """
    The range of outline areas a kind of thing is allowed to have.
    """

    minimum_area: float
    """
    Smallest area, in square metres, an outline may have and still be considered.
    """

    maximum_area: float
    """
    Largest area, in square metres, an outline may have and still be considered.
    """

    def admits(self, footprint: Footprint) -> bool:
        """
        Whether an outline's area falls in this range.

        :param footprint: The measured outline.
        """
        return self.minimum_area <= footprint.area <= self.maximum_area


LOOSE_PIECE_SIZE = SizeRange(minimum_area=0.0002, maximum_area=0.004)
"""
Area a loose Montessori piece's footprint may cover, spanning roughly a fourteen to a
sixty millimetre square -- wide enough for every piece in the set and narrow enough to
reject a hand reaching into the scene.
"""

HOLE_SIZE = SizeRange(minimum_area=0.00006, maximum_area=0.003)
"""
Area a hole in the board's lid may cover.

The lower bound sits below the narrowest hole, the disk's five by forty-eight millimetre
slot, and above the slivers of shadow that fall along the lid's own edges.
"""


# %% detectors


@dataclass
class BoardDetector:
    """
    Finds the shape-sorting board and the holes in its lid, in a view rectified onto
    that lid.

    The board is picked out by the holes themselves rather than by being the largest
    thing in view: an arm reaching over the table is both larger and just as strongly
    coloured, but only the board is a surface with several openings cut through it.
    """

    classifier: FootprintClassifier = field(default_factory=CrossSectionClassifier)
    """
    Decides which shape each hole is cut for.
    """

    colors: SurfaceColors = field(default_factory=SurfaceColors)
    """
    How the lid and its holes separate by colour.
    """

    hole_size: SizeRange = HOLE_SIZE
    """
    Area a hole's outline may cover.
    """

    minimum_hole_count: int = 3
    """
    How many holes a surface must have cut through it to be taken for the board.
    """

    minimum_lid_area: float = 0.01
    """
    Area, in square metres, a surface must cover before its holes are looked for.

    The board's lid covers about three hundred square centimetres; this keeps the search
    off the loose pieces, whose own shading would otherwise split into hole-sized dark
    patches.
    """

    board_footprint: Tuple[float, float] = (float(BOARD_SCALE.x), float(BOARD_SCALE.y))
    """
    How large the board's lid is, in metres, taken from the same board the demo builds
    its world from.

    Sets how far apart two holes may lie and still belong to the same board.
    """

    def detect(
        self,
        orthophoto: Orthophoto,
        reference_frame: Optional[KinematicStructureEntity],
    ) -> Optional[MontessoriBoardDetection]:
        """
        Find the board in a view rectified onto its lid.

        :param orthophoto: The rectified view of the lid's plane.
        :param reference_frame: Frame the resulting poses are expressed in.
        :return: The board and its holes, or None if no surface in view had enough holes
            cut through it.
        """
        mask = _clean(self.colors.surface_mask(orthophoto))
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        best: List[ShapeSortingHoleDetection] = []
        for contour in contours:
            footprint = Footprint.from_contour(contour, orthophoto.region.resolution)
            if footprint.area < self.minimum_lid_area:
                continue
            holes = self._board_sized_cluster(
                self._holes_within(contour, orthophoto, reference_frame)
            )
            if len(holes) > len(best):
                best = holes
        if len(best) < self.minimum_hole_count:
            return None
        return self._board_around(best, orthophoto, reference_frame)

    def _holes_within(
        self,
        surface: np.ndarray,
        orthophoto: Orthophoto,
        reference_frame: Optional[KinematicStructureEntity],
    ) -> List[ShapeSortingHoleDetection]:
        """
        Recognise the holes cut through one candidate surface.

        The surface is filled in first, so that a hole is a dark patch within a solid
        region rather than a gap that the surface's own outline has to enclose; a hole
        broken open at the board's edge would otherwise be missed entirely.

        :param surface: The candidate surface's own contour, in rectified pixels.
        :param orthophoto: The rectified view it was found in.
        :param reference_frame: Frame the resulting poses are expressed in.
        :return: One detection per hole that is the right size and a recognisable shape.
        """
        region = _filled(surface, orthophoto)
        # Left unopened on purpose: the narrowest hole on the board is five millimetres
        # across, which the speckle-removing kernel would eat entirely. Slivers are
        # rejected by area instead, in :attr:`hole_size`.
        dark = self.colors.dark_mask(orthophoto, region)
        contours, _ = cv2.findContours(dark, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        holes = []
        for contour in contours:
            footprint = Footprint.from_contour(contour, orthophoto.region.resolution)
            if not self.hole_size.admits(footprint):
                continue
            category = self.classifier.classify(footprint)
            if category is None:
                continue
            holes.append(
                ShapeSortingHoleDetection(
                    pose=self._pose(contour, orthophoto, reference_frame, footprint),
                    footprint=footprint,
                    outline=_to_world_outline(contour, orthophoto),
                    category=category,
                )
            )
        return holes

    def _board_sized_cluster(
        self, holes: List[ShapeSortingHoleDetection]
    ) -> List[ShapeSortingHoleDetection]:
        """
        Keep only the holes that could belong to one board.

        A surface that has merged with an arm reaching over the table carries dark
        patches scattered far beyond any board, so the holes are grouped by how close
        together they lie and only the largest group that still fits on a board is kept.

        :param holes: Every hole-shaped patch found on one surface.
        :return: The largest group of them that fits within one board's lid.
        """
        if not holes:
            return []
        centers = np.array([_position_of(hole) for hole in holes])
        groups = [self._grow_from(seed, holes, centers) for seed in centers]
        return max(groups, key=len)

    def _grow_from(
        self,
        seed: np.ndarray,
        holes: List[ShapeSortingHoleDetection],
        centers: np.ndarray,
    ) -> List[ShapeSortingHoleDetection]:
        """
        Grow a group outwards from one hole, taking in the next nearest hole for as long
        as the group still fits on a single lid.

        :param seed: World-frame ``(x, y)`` of the hole to grow from.
        :param holes: Every hole-shaped patch found on one surface.
        :param centers: Those holes' world-frame ``(x, y)``, in the same order.
        :return: The grown group.
        """
        order = np.argsort(np.linalg.norm(centers - seed, axis=1))
        group: List[ShapeSortingHoleDetection] = []
        for index in order:
            candidate = group + [holes[index]]
            if self._fits_on_a_lid(candidate):
                group = candidate
        return group

    def _fits_on_a_lid(self, holes: List[ShapeSortingHoleDetection]) -> bool:
        """
        Whether a group of holes is packed tightly enough to have come from one lid.

        :param holes: The group to check.
        """
        centers = np.array([_position_of(hole) for hole in holes], dtype=np.float32)
        if len(centers) < 2:
            return len(centers) > 0
        (_, _), (first_side, second_side), _ = cv2.minAreaRect(centers)
        span = sorted((first_side, second_side))
        return all(
            measured <= allowed
            for measured, allowed in zip(span, sorted(self.board_footprint))
        )

    def _board_around(
        self,
        holes: List[ShapeSortingHoleDetection],
        orthophoto: Orthophoto,
        reference_frame: Optional[KinematicStructureEntity],
    ) -> MontessoriBoardDetection:
        """
        Build the board that carries a group of holes.

        The lid's own edges are not used: they run into whatever the board is standing
        against, and the holes are laid out symmetrically about the lid's centre anyway,
        so the board is reported as a lid-sized rectangle centred on and aligned with
        its holes.

        :param holes: The holes the board carries.
        :param orthophoto: The rectified view of the lid's plane.
        :param reference_frame: Frame the resulting pose is expressed in.
        :return: The board.
        """
        centers = np.array([_position_of(hole) for hole in holes], dtype=np.float32)
        center = centers.mean(axis=0)
        (_, _), (_, _), angle_in_degrees = cv2.minAreaRect(centers)
        yaw = math.radians(angle_in_degrees)
        width, length = sorted(self.board_footprint)
        outline = cv2.boxPoints(
            ((float(center[0]), float(center[1])), (width, length), angle_in_degrees)
        )
        return MontessoriBoardDetection(
            pose=Pose.from_xyz_rpy(
                float(center[0]),
                float(center[1]),
                orthophoto.plane_height,
                yaw=yaw,
                reference_frame=reference_frame,
            ),
            footprint=Footprint(
                area=width * length,
                width=width,
                length=length,
                fill_ratio=1.0,
                corner_count=4,
                yaw=(yaw + math.pi / 2) % math.pi - math.pi / 2,
            ),
            outline=np.asarray(outline, dtype=float),
            holes=holes,
            lid_height=orthophoto.plane_height,
        )

    @staticmethod
    def _pose(
        contour: np.ndarray,
        orthophoto: Orthophoto,
        reference_frame: Optional[KinematicStructureEntity],
        footprint: Optional[Footprint] = None,
    ) -> Pose:
        """
        Where a contour's centre sits on the rectified plane.

        :param contour: The contour, in rectified pixels.
        :param orthophoto: The rectified view it was found in.
        :param reference_frame: Frame the pose is expressed in.
        :param footprint: The contour's own measurements, remeasured if not given.
        :return: The pose of the contour's centre, on the rectified plane.
        """
        if footprint is None:
            footprint = Footprint.from_contour(contour, orthophoto.region.resolution)
        x, y = orthophoto.contour_center(contour)
        return Pose.from_xyz_rpy(
            x,
            y,
            orthophoto.plane_height,
            yaw=footprint.yaw,
            reference_frame=reference_frame,
        )


@dataclass
class LoosePieceDetector:
    """
    Finds the loose Montessori pieces in a view rectified onto the table they rest on.
    """

    matcher: PieceMatcher = field(default_factory=PieceMatcher)
    """
    Recognises which piece stands where, and how far it is turned.
    """

    colors: SurfaceColors = field(default_factory=SurfaceColors)
    """
    How a piece separates from the bare table by colour.
    """

    piece_size: SizeRange = LOOSE_PIECE_SIZE
    """
    Area a piece's outline may cover.
    """

    piece_height: float = 0.03
    """
    Roughly how tall a loose piece stands, in metres, used to cancel the parallax that
    would otherwise stretch its outline (see :meth:`detect`) and reported as a piece's
    own height wherever the depth image cannot resolve it.

    The pieces in this set stand between twenty and thirty millimetres tall, and the
    cancellation is forgiving of the difference.
    """

    def detect(
        self,
        orthophoto: Orthophoto,
        top_orthophoto: Orthophoto,
        frame: RgbdFrame,
        reference_frame: Optional[KinematicStructureEntity],
        board: Optional[MontessoriBoardDetection],
    ) -> List[MontessoriShapeDetection]:
        """
        Find the pieces lying on the table.

        Colour says where to look and the edges say what is there. A piece seen from off
        to one side hides none of itself but shows its sides as well as its top, so its
        silhouette rectified onto the table is the piece's footprint together with its
        top face pushed away from the point directly below the camera -- on this table
        that stretches a thirty millimetre piece by nearly twenty. Rectifying onto the
        piece's top instead pushes the *base* the other way, towards that point, so what
        the two silhouettes agree on is roughly the footprint, and roughly is as far as
        colour gets on a table that reflects each piece back at the camera.

        The piece itself is then found by fitting the known pieces to the edges of the
        view rectified onto their tops, where a piece's own top face lies at exactly its
        footprint, undistorted and sharply bounded.

        An outline that reaches the edge of the region is passed over: only part of it
        was seen, so how large it is and what shape it has were never measured.

        :param orthophoto: The rectified view of the table's plane.
        :param top_orthophoto: The rectified view of the plane a piece's top stands on.
        :param frame: The camera data, for measuring how tall each piece stands.
        :param reference_frame: Frame the resulting poses are expressed in.
        :param board: The board, whose own outline is excluded so that its lid is not
            reported as a pile of loose pieces; None if it was not in view.
        :return: One detection per recognised piece.
        """
        mask = _clean(
            cv2.bitwise_and(
                self.colors.piece_mask(orthophoto),
                self.colors.piece_mask(top_orthophoto),
            )
        )
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        edges = EdgeDistances.of(top_orthophoto)

        pieces = []
        for contour in contours:
            footprint = Footprint.from_contour(contour, orthophoto.region.resolution)
            if not self.piece_size.admits(footprint):
                continue
            if not _wholly_within(contour, orthophoto):
                continue
            x, y = orthophoto.contour_center(contour)
            if board is not None and board.encloses(x, y):
                continue
            match = self.matcher.match(
                edges,
                (x, y),
                self.colors.measure_hue(orthophoto, _filled(contour, orthophoto)),
            )
            if match is None:
                continue
            height = _measure_height(contour, orthophoto, frame, self.piece_height)
            pieces.append(
                MontessoriShapeDetection(
                    pose=Pose.from_xyz_rpy(
                        match.center[0],
                        match.center[1],
                        orthophoto.plane_height + height / 2,
                        yaw=match.yaw,
                        reference_frame=reference_frame,
                    ),
                    footprint=footprint,
                    outline=match.piece.turned_outline(match.yaw)
                    + np.asarray(match.center),
                    category=match.piece.category,
                    height=height,
                    outline_agreement=match.outline_agreement,
                )
            )
        return pieces


# %% measuring how tall a piece stands

_MINIMUM_DEPTH_SAMPLES = 50
"""
How many depth readings a piece's top must return before its height is trusted.

Below this the reflective table around it dominates whatever the sensor did return.
"""

_HEIGHT_SAMPLE_STRIDE = 3
"""
Sample every third rectified pixel of a piece's interior, which keeps the reprojection
cheap while still leaving hundreds of readings for a piece of any usable size.
"""


def _measure_height(
    contour: np.ndarray,
    orthophoto: Orthophoto,
    frame: RgbdFrame,
    nominal_height: float,
) -> float:
    """
    Measure how far a piece's top surface stands above the plane it was rectified on.

    Reprojects the piece's interior back into the camera, takes the median of whatever
    depth readings came back, and turns that into a height along the world frame's
    z-axis.

    :param contour: The piece's outline, in rectified pixels.
    :param orthophoto: The rectified view it was found in.
    :param frame: The camera data to read depth from.
    :param nominal_height: Height to report where the depth image cannot answer.
    :return: The height in metres.
    """
    rows, columns = np.nonzero(_filled(contour, orthophoto))
    rows = rows[::_HEIGHT_SAMPLE_STRIDE]
    columns = columns[::_HEIGHT_SAMPLE_STRIDE]
    if rows.size == 0:
        return nominal_height

    pixel_T_region = OrthophotoProjector.pixel_T_region(frame, orthophoto.plane_height)
    homography = pixel_T_region @ orthophoto.region.region_T_pixel
    rectified = np.stack([columns, rows, np.ones_like(rows)], axis=0).astype(float)
    projected = homography @ rectified
    camera_pixels = np.round(projected[:2] / projected[2]).T

    depths = frame.depth_at(camera_pixels)
    if depths.size < _MINIMUM_DEPTH_SAMPLES:
        return nominal_height

    projected_center = homography @ np.array([columns.mean(), rows.mean(), 1.0])
    center_pixel = projected_center[:2] / projected_center[2]
    [camera_P_top] = frame.intrinsics.deproject(
        center_pixel.reshape(1, 2), np.array([float(np.median(depths))])
    )
    reference_frame_P_top = frame.reference_frame_T_camera @ np.append(
        camera_P_top, 1.0
    )
    measured = float(reference_frame_P_top[2]) - orthophoto.plane_height
    return measured if measured > 0.0 else nominal_height


def _position_of(detection: MontessoriDetection) -> np.ndarray:
    """
    A detection's world-frame ``(x, y)``.

    :param detection: The detection to read.
    """
    return detection.pose.to_position().to_np()[:2].astype(float)


def _to_world_outline(contour: np.ndarray, orthophoto: Orthophoto) -> np.ndarray:
    """
    Express a rectified contour as world-frame points on the plane it was found on.

    :param contour: The contour, in rectified pixels.
    :param orthophoto: The rectified view it was found in.
    :return: The outline as ``(n, 2)`` world-frame ``(x, y)`` points.
    """
    pixels = contour.reshape(-1, 2).astype(float)
    return np.stack(
        orthophoto.region.to_world_position(pixels[:, 0], pixels[:, 1]), axis=1
    )


# %% the pipeline


@dataclass
class MontessoriPerceptionPipeline:
    """
    Turns one RGB-D frame into everything recognised in the Montessori scene.
    """

    region: WorkspaceRegion
    """
    The patch of the table perception looks at.
    """

    table_height: float
    """
    Height of the table's top surface above the world frame's origin, in metres.

    Read off the robot's own model rather than fitted to the depth image, which this
    reflective table is too noisy to support.
    """

    board_height: float
    """
    How far the board's lid stands above the table, in metres.

    Must match the physical board, since it sets the plane the holes are rectified onto
    and so how far apart their centres come out.
    """

    reference_frame: Optional[KinematicStructureEntity] = None
    """
    Frame the detections' poses are expressed in, which must be the frame the camera's
    own pose was given in.
    """

    board_detector: BoardDetector = field(default_factory=BoardDetector)
    """
    Finds the board and its holes.
    """

    piece_detector: LoosePieceDetector = field(default_factory=LoosePieceDetector)
    """
    Finds the loose pieces on the table.
    """

    headroom: float = 0.15
    """
    How far above the table, in metres, the view still holds something worth seeing.

    Wide enough for the board's lid and for a piece held over it on its way in; anything
    higher is a passing arm or the room behind the table.
    """

    @property
    def lid_height(self) -> float:
        """
        Height of the board's lid above the world frame's origin, in metres.
        """
        return self.table_height + self.board_height

    @property
    def workspace(self) -> WorkspaceBox:
        """
        The space this pipeline looks at: its own patch of table, and the room above it
        a piece or the board can stand in.
        """
        return WorkspaceBox(
            region=self.region,
            minimum_height=self.table_height,
            maximum_height=self.table_height + self.headroom,
        )

    def rectify_table(self, frame: RgbdFrame) -> Orthophoto:
        """
        Rectify a frame onto the table the loose pieces rest on, which is the plane
        their outlines are measured in.

        :param frame: The camera data to rectify.
        :return: The table's top-down view.
        """
        return OrthophotoProjector(region=self.region).project(frame, self.table_height)

    def detect(self, frame: RgbdFrame) -> MontessoriScene:
        """
        Recognise everything in one frame.

        :param frame: The camera data to search.
        :return: The pieces, the board, and its holes.
        """
        projector = OrthophotoProjector(region=self.region)
        board = self.board_detector.detect(
            projector.project(frame, self.lid_height), self.reference_frame
        )
        pieces = self.piece_detector.detect(
            self.rectify_table(frame),
            projector.project(
                frame, self.table_height + self.piece_detector.piece_height
            ),
            frame,
            self.reference_frame,
            board,
        )
        return MontessoriScene(shapes=pieces, board=board)
