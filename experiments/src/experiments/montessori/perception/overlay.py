"""
Draw what perception found onto an image it was found in, so a detection can be checked
against the pixels it came from without leaving the camera window.

The same drawing serves the camera's own image and the top-down view rectified from it:
the two differ only in where a point on a horizontal plane lands, which is what a
:class:`DetectionView` answers.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass

import cv2
import numpy as np

from experiments.montessori.perception.camera import RgbdFrame
from experiments.montessori.perception.colors import (
    BOARD_COLOR,
    HOLE_COLOR,
    LABEL_COLOR,
    PIECE_COLOR,
    DetectionColor,
)
from experiments.montessori.perception.detections import (
    MontessoriDetection,
    MontessoriScene,
)
from experiments.montessori.perception.orthophoto import Orthophoto, OrthophotoProjector

# %% where a detection falls in an image


def _map_plane_points(homography: np.ndarray, points: np.ndarray) -> np.ndarray:
    """
    Send points on a horizontal plane through a homography.

    :param homography: The 3x3 mapping to apply.
    :param points: The points as ``(n, 2)`` ``(x, y)`` pairs.
    :return: The mapped points, as ``(n, 2)`` ``(x, y)`` pairs.
    """
    on_plane = np.asarray(points, dtype=float).reshape(-1, 2)
    mapped = np.column_stack([on_plane, np.ones(len(on_plane))]) @ homography.T
    return mapped[:, :2] / mapped[:, 2:3]


def project_to_pixels(
    frame: RgbdFrame, points: np.ndarray, height: float
) -> np.ndarray:
    """
    Where points on a horizontal plane fall in the image that saw them.

    :param frame: The camera data, carrying the camera's own pose and intrinsics.
    :param points: The points as ``(n, 2)`` world-frame ``(x, y)`` pairs.
    :param height: Height of the plane they lie on, in metres.
    :return: The pixels they fall on, as ``(n, 2)`` ``(x, y)`` pairs.
    """
    return _map_plane_points(OrthophotoProjector.pixel_T_region(frame, height), points)


class DetectionView(ABC):
    """
    An image detections can be drawn on, together with where a point on a horizontal
    plane falls in it.
    """

    @abstractmethod
    def to_image(self) -> np.ndarray:
        """
        :return: A copy of the image itself, to be drawn on.
        """

    @abstractmethod
    def to_pixels(self, points: np.ndarray, height: float) -> np.ndarray:
        """
        Where points on a horizontal plane fall in this image.

        :param points: The points as ``(n, 2)`` world-frame ``(x, y)`` pairs.
        :param height: Height of the plane they lie on, in metres.
        :return: The pixels they fall on, as ``(n, 2)`` ``(x, y)`` pairs.
        """


@dataclass(frozen=True)
class CameraView(DetectionView):
    """
    The camera's colour image as it was taken.
    """

    frame: RgbdFrame
    """
    The camera data, carrying the image, the camera's own pose and its intrinsics.
    """

    def to_image(self) -> np.ndarray:
        return self.frame.color.copy()

    def to_pixels(self, points: np.ndarray, height: float) -> np.ndarray:
        return project_to_pixels(self.frame, points, height)


@dataclass(frozen=True)
class RectifiedView(DetectionView):
    """
    The metric top-down view rectified from a frame onto one horizontal plane.

    Only that plane is rectified, so anything standing above it is drawn where the
    camera saw it, and a detection's box carries the same parallax the rectified image
    itself shows.
    """

    frame: RgbdFrame
    """
    The camera data the view was rectified from.
    """

    orthophoto: Orthophoto
    """
    The rectified view itself.
    """

    def to_image(self) -> np.ndarray:
        return self.orthophoto.image.copy()

    def to_pixels(self, points: np.ndarray, height: float) -> np.ndarray:
        camera_pixel_T_view = (
            OrthophotoProjector.pixel_T_region(self.frame, self.orthophoto.plane_height)
            @ self.orthophoto.region.region_T_pixel
        )
        return _map_plane_points(
            np.linalg.inv(camera_pixel_T_view)
            @ OrthophotoProjector.pixel_T_region(self.frame, height),
            points,
        )


# %% drawing them


@dataclass
class DetectionOverlay:
    """
    Draws each detection's outline box, its centre and its name onto a view.

    The box is the smallest upright rectangle holding the whole of the detection's own
    measured body -- its outline drawn both at the surface it rests on and at its own
    top -- so a piece standing above the surface is boxed together with the top face the
    camera sees pushed off to one side of it.
    """

    line_width: int = 2
    """
    Thickness of a drawn box, in pixels.
    """

    center_radius: int = 4
    """
    Radius of the dot marking a detection's centre, in pixels.
    """

    label_height: float = 0.6
    """
    Size the name is written at, as OpenCV's own multiple of its base font.
    """

    label_offset: int = 6
    """
    How far above its box a name is written, in pixels.
    """

    def draw(self, view: DetectionView, scene: MontessoriScene) -> np.ndarray:
        """
        Draw everything one look at the scene found.

        :param view: The view the detections are drawn on.
        :param scene: The detections to draw.
        :return: A copy of the view's image with the detections on it.
        """
        image = view.to_image()
        for piece in scene.shapes:
            self._draw_detection(image, view, piece, PIECE_COLOR)
        for hole in scene.holes:
            self._draw_detection(image, view, hole, HOLE_COLOR)
        if scene.board is not None:
            self._draw_detection(image, view, scene.board, BOARD_COLOR)
        return image

    def _draw_detection(
        self,
        image: np.ndarray,
        view: DetectionView,
        detection: MontessoriDetection,
        color: DetectionColor,
    ) -> None:
        """
        Draw one detection's box, centre and name.

        The centre is drawn on the surface the detection rests on, which is where a
        caller asking for its position is told it stands.

        :param image: The image to draw on, changed in place.
        :param view: The view the detection is drawn on.
        :param detection: The detection to draw.
        :param color: The colour to draw it in.
        """
        body = np.vstack(
            [
                view.to_pixels(detection.outline, height)
                for height in (detection.surface_height, detection.top_height)
            ]
        )
        left, top, width, height = cv2.boundingRect(
            body.astype(np.float32).reshape(-1, 1, 2)
        )
        cv2.rectangle(
            image,
            (left, top),
            (left + width, top + height),
            color.to_bgr(),
            self.line_width,
        )
        [center] = view.to_pixels(
            detection.pose.to_position().to_np()[:2].reshape(1, 2),
            detection.surface_height,
        )
        cv2.circle(
            image,
            (round(center[0]), round(center[1])),
            self.center_radius,
            color.to_bgr(),
            cv2.FILLED,
        )
        cv2.putText(
            image,
            detection.label,
            (left, top - self.label_offset),
            cv2.FONT_HERSHEY_SIMPLEX,
            self.label_height,
            LABEL_COLOR.to_bgr(),
            self.line_width,
            cv2.LINE_AA,
        )
