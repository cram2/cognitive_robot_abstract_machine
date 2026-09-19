"""
Where the sharp edges lie in a rectified view, and how far a point on that plane lies
from the nearest one.

This is what a piece is recognised against on a mirror-finish table. The table throws a
diffuse reflection of each piece back at the camera, coloured like the piece and large
enough to swallow it: segmenting by colour there gives an outline half again the size of
the piece that cast it. That reflection has no sharp boundary anywhere, while the piece
itself is bounded by three -- the edge of its top face, the crease where that face meets
its sides, and the line where its sides meet the table -- so the edges say where the
piece is even where its colour does not.
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from experiments.montessori.perception.orthophoto import Orthophoto, WorkspaceRegion

# %% finding the edges

SMOOTHING_WIDTH = 3
"""
Width, in rectified pixels, of the blur applied before the edges are found, which stops
the sensor noise on a bare metal table from being read as edges of its own.
"""

WEAK_EDGE_STEP = 30
"""
Brightness step, out of 255, a pixel must stand out by to continue an edge already
found.
"""

STRONG_EDGE_STEP = 90
"""
Brightness step, out of 255, a pixel must stand out by to start an edge.

Low enough to keep the crease between a piece's lit top face and its shaded sides, which
is the boundary that says how far the piece reaches; a reflection's own gradients are
spread over centimetres and stay far below it.
"""

# %% how far from an edge


@dataclass(frozen=True)
class EdgeDistances:
    """
    How far every point of a rectified plane lies from the nearest edge seen in it.
    """

    distances: np.ndarray
    """
    Distance to the nearest edge, in metres, one entry per rectified pixel.
    """

    region: WorkspaceRegion
    """
    The patch of the plane the distances cover.
    """

    @classmethod
    def of(cls, orthophoto: Orthophoto) -> EdgeDistances:
        """
        Find the edges in a rectified view and measure the distance to them.

        :param orthophoto: The rectified view to read.
        """
        brightness = cv2.cvtColor(orthophoto.image, cv2.COLOR_BGR2GRAY)
        smoothed = cv2.GaussianBlur(brightness, (SMOOTHING_WIDTH, SMOOTHING_WIDTH), 0)
        edges = cv2.Canny(smoothed, WEAK_EDGE_STEP, STRONG_EDGE_STEP)
        return cls(
            distances=cv2.distanceTransform(255 - edges, cv2.DIST_L2, 3)
            * orthophoto.region.resolution,
            region=orthophoto.region,
        )

    def distance_to_edge(self, points: np.ndarray) -> np.ndarray:
        """
        How far world-frame points lie from the nearest edge.

        Points outside the region are answered with the distance at the nearest point
        inside it, so an outline reaching over the edge of the view is not rewarded for
        it.

        :param points: World-frame ``(x, y)`` points, shape ``(..., 2)``.
        :return: The distances in metres, shape ``(...)``.
        """
        columns = np.clip(
            np.round((points[..., 0] - self.region.minimum_x) / self.region.resolution),
            0,
            self.distances.shape[1] - 1,
        ).astype(int)
        rows = np.clip(
            np.round((points[..., 1] - self.region.minimum_y) / self.region.resolution),
            0,
            self.distances.shape[0] - 1,
        ).astype(int)
        return self.distances[rows, columns]

    def agreement(self, outline: np.ndarray, reach: float) -> np.ndarray:
        """
        How well outlines follow the edges that were seen.

        Each of an outline's points counts for one where it lies on an edge and for
        nothing where the nearest edge is further off than ``reach``, so the result is
        the share of the outline the picture agrees with.

        :param outline: World-frame ``(x, y)`` points along one or more outlines, shape
            ``(..., points, 2)``.
        :param reach: How far, in metres, an edge may lie from the outline and still
            count for anything.
        :return: The agreement of each outline, between zero and one, shape ``(...)``.
        """
        return np.clip(1.0 - self.distance_to_edge(outline) / reach, 0.0, None).mean(
            axis=-1
        )
