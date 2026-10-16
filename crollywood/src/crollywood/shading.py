"""
Areas of a frame shaded towards black, so that what is written over them reads on any
picture.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray
from PIL import Image, ImageDraw
from semantic_digital_twin.world_description.geometry import PlanarBoundingBox

from crollywood.exceptions import DarknessOutOfRange
from crollywood.frame import Frame, PixelBox

# %% a shade


@dataclass(frozen=True)
class Shade(ABC):
    """
    How an area of a frame is shaded towards black.
    """

    darkness: float
    """
    How dark the shade is where it is deepest, from zero, which leaves the picture, to
    one, black.
    """

    def __post_init__(self) -> None:
        """
        :raises DarknessOutOfRange: If the darkness lies outside zero to one.
        """
        if not 0.0 <= self.darkness <= 1.0:
            raise DarknessOutOfRange(darkness=self.darkness)

    @abstractmethod
    def over(self, frame: Frame, area: PlanarBoundingBox) -> Frame:
        """
        :param frame: The frame.
        :param area: The area of the frame shaded.
        :return: A copy of the frame with the area shaded.
        """

    @staticmethod
    def _darkened(pixels: NDArray[np.uint8], weight: NDArray[np.float32]) -> Frame:
        """
        :param pixels: The pixels.
        :param weight: How far each pixel moves towards black, from zero to one.
        :return: The pixels moved that far towards black.
        """
        darkened = pixels.astype(np.float32) * (1.0 - weight)
        return Frame(np.clip(darkened + 0.5, 0, 255).astype(np.uint8))


# %% the kinds of shade


@dataclass(frozen=True)
class EvenShade(Shade):
    """
    A shade equally dark over its whole area, its corners rounded.
    """

    corner_radius: int = 0
    """
    Pixels the corners of the area are rounded by.
    """

    def over(self, frame: Frame, area: PlanarBoundingBox) -> Frame:
        """
        :param frame: The frame.
        :param area: The area of the frame shaded.
        :return: A copy of the frame with the area shaded evenly.
        """
        height, width = frame.pixels.shape[:2]
        mask = Image.new("L", (width, height), 0)
        box = PixelBox.nearest(area)
        ImageDraw.Draw(mask).rounded_rectangle(
            (box.left, box.top, box.right - 1, box.bottom - 1),
            radius=self.corner_radius,
            fill=255,
        )
        weight = np.asarray(mask, dtype=np.float32)[..., None] / 255.0 * self.darkness
        return self._darkened(frame.pixels, weight)


@dataclass(frozen=True)
class GradientShade(Shade):
    """
    A shade that deepens from nothing at the top edge of its area to its full darkness
    part of the way down, and stays at it below.
    """

    deepening_share: float = 1 / 3
    """
    The share of the area's height, from its top edge, over which the shade deepens.
    """

    def over(self, frame: Frame, area: PlanarBoundingBox) -> Frame:
        """
        :param frame: The frame.
        :param area: The area of the frame shaded.
        :return: A copy of the frame with the area shaded, deeper towards its bottom.
        """
        band = PixelBox.nearest(area).inside(frame.resolution)
        if band.is_empty:
            return Frame(frame.pixels.copy())
        rows = np.arange(band.height, dtype=np.float32)
        deepening_rows = max(band.height * self.deepening_share, 1.0)
        weight = np.clip(rows / deepening_rows, 0.0, 1.0) * self.darkness
        result = frame.pixels.copy()
        inside = (slice(band.top, band.bottom), slice(band.left, band.right))
        result[inside] = self._darkened(result[inside], weight[:, None, None]).pixels
        return Frame(result)
