"""
The pictures a video is made of, and drawing on them.

A frame is rows of columns of red, green and blue bytes. Colours are those of
:class:`~semantic_digital_twin.world_description.geometry.Color`, positions are
:class:`~semantic_digital_twin.spatial_types.spatial_types.Point2` and rectangles are
:class:`~semantic_digital_twin.world_description.geometry.PlanarBoundingBox`, all in
pixels from the frame's top left corner: ``x`` runs along the columns and ``y`` down the
rows, so a box's ``min_x`` is its left edge and its ``min_y`` its top edge.
"""

from __future__ import annotations

from dataclasses import astuple, dataclass

import cv2
import numpy as np
from numpy.typing import NDArray
from semantic_digital_twin.spatial_types.spatial_types import (
    HomogeneousTransformationMatrix,
    Point2,
)
from semantic_digital_twin.world_description.geometry import (
    Bounds,
    Color,
    PlanarBoundingBox,
)

# %% whole pixels


@dataclass(frozen=True)
class Pixel:
    """
    The pixel at a column and row of a frame.
    """

    column: int
    """
    Pixels from the frame's left edge.
    """

    row: int
    """
    Pixels from the frame's top edge.
    """

    @classmethod
    def nearest(cls, point: Point2) -> Pixel:
        """
        :param point: A position in pixels.
        :return: The pixel it falls on.
        """
        return cls(round(float(point.x)), round(float(point.y)))


@dataclass
class PixelBox(Bounds[Pixel]):
    """
    A box of whole pixels of a frame, from its top left pixel to its bottom right one.
    """

    @classmethod
    def between(cls, left: int, top: int, right: int, bottom: int) -> PixelBox:
        """
        :param left: The column of its left edge.
        :param top: The row of its top edge.
        :param right: The column just right of it.
        :param bottom: The row just below it.
        :return: The box of the pixels between those edges.
        """
        return cls(Pixel(left, top), Pixel(right - 1, bottom - 1))

    @classmethod
    def nearest(cls, box: PlanarBoundingBox) -> PixelBox:
        """
        :param box: A box in pixels.
        :return: The whole pixels it covers, its edges rounded to the nearest.
        """
        return cls.between(
            round(box.min_x), round(box.min_y), round(box.max_x), round(box.max_y)
        )

    @property
    def left(self) -> int:
        """
        The column of its left edge.
        """
        return self.lower.column

    @property
    def top(self) -> int:
        """
        The row of its top edge.
        """
        return self.lower.row

    @property
    def right(self) -> int:
        """
        The column just right of the box.
        """
        return self.upper.column + 1

    @property
    def bottom(self) -> int:
        """
        The row just below the box.
        """
        return self.upper.row + 1

    @property
    def width(self) -> int:
        """
        How many columns wide it is.
        """
        return self.right - self.left

    @property
    def height(self) -> int:
        """
        How many rows tall it is.
        """
        return self.bottom - self.top

    @property
    def is_empty(self) -> bool:
        """
        Whether it covers no pixel.
        """
        return self.width <= 0 or self.height <= 0

    def inside(self, resolution: Resolution) -> PixelBox:
        """
        :param resolution: The size of a frame.
        :return: The part of the box inside a frame that size; empty if none is.
        """
        return PixelBox.between(
            max(self.left, 0),
            max(self.top, 0),
            min(self.right, resolution.width),
            min(self.bottom, resolution.height),
        )


# %% the size of a frame


@dataclass(frozen=True)
class Resolution:
    """
    The size every frame of a video has, and the room its scenes leave around what they
    show.
    """

    width: int
    """
    Columns of pixels.
    """

    height: int
    """
    Rows of pixels.
    """

    margin: int = 32
    """
    The margin every frame keeps all round, in pixels.
    """

    subtitle_band_share: float = 0.14
    """
    The share of the frame's height, at its bottom, left clear for the captions: two
    rows of them on a fixed baseline, with their margin.
    """

    @classmethod
    def standard(cls) -> Resolution:
        """
        :return: 720 rows by 1280 columns: sharp enough to read the smallest letters and
            small enough to encode minutes of video in a few megabytes.
        """
        return cls(width=1280, height=720)

    @property
    def shape(self) -> tuple[int, int, int]:
        """
        The array shape of one frame this size.
        """
        return (self.height, self.width, 3)

    @property
    def stage_height(self) -> float:
        """
        How far down a scene may draw: the rows above the band left for subtitles.
        """
        return self.height * (1.0 - self.subtitle_band_share)

    @property
    def whole(self) -> PlanarBoundingBox:
        """
        The box covering a frame of this size.
        """
        return Frame.box(0, 0, self.width, self.height)

    @classmethod
    def of(cls, frame: Frame) -> Resolution:
        """
        :param frame: A frame.
        :return: Its size.
        """
        height, width = frame.pixels.shape[:2]
        return cls(width=width, height=height)

    def blank(self, grey: int = 255) -> Frame:
        """
        :param grey: The grey every pixel is set to.
        :return: A frame this size of one grey.
        """
        return Frame(np.full(self.shape, grey, dtype=np.uint8))


# %% a frame


@dataclass(frozen=True, eq=False)
class Frame:
    """
    One picture of the video: red, green and blue bytes, rows by columns by channel.

    Drawing returns a new frame and leaves this one as it was.
    """

    pixels: NDArray[np.uint8]
    """
    The bytes, rows by columns by channel.
    """

    @property
    def resolution(self) -> Resolution:
        """
        The size of this frame.
        """
        return Resolution.of(self)

    @classmethod
    def of_color(cls, resolution: Resolution, color: Color) -> Frame:
        """
        :param resolution: The size.
        :param color: What every pixel is set to.
        :return: A frame that size of one colour.
        """
        return cls(
            np.full(resolution.shape, tuple(color.to_rgb_bytes()), dtype=np.uint8)
        )

    @staticmethod
    def box(left: float, top: float, width: float, height: float) -> PlanarBoundingBox:
        """
        :param left: Pixels from the frame's left edge to the box's.
        :param top: Pixels from the frame's top edge to the box's.
        :param width: How many columns wide it is.
        :param height: How many rows tall it is.
        :return: The box in pixels from the frame's top left corner.
        """
        return PlanarBoundingBox(
            min_x=left,
            min_y=top,
            max_x=left + width,
            max_y=top + height,
            origin=HomogeneousTransformationMatrix(),
        )

    @staticmethod
    def fitting(room: PlanarBoundingBox, aspect: float) -> PlanarBoundingBox:
        """
        :param room: The space available.
        :param aspect: Width over height.
        :return: The largest box of that aspect centred in the room.
        """
        room_width, room_height = room.max_x - room.min_x, room.max_y - room.min_y
        if room_width / room_height > aspect:
            height = room_height
            width = height * aspect
        else:
            width = room_width
            height = width / aspect
        return Frame.box(
            room.min_x + (room_width - width) / 2,
            room.min_y + (room_height - height) / 2,
            width,
            height,
        )

    def pasted(self, picture: Frame, into: PlanarBoundingBox) -> Frame:
        """
        :param picture: What to paste.
        :param into: Where, and how big; what leaves the frame is cropped.
        :return: A copy with the picture resized into the box.
        """
        result = self.pixels.copy()
        box = PixelBox.nearest(into)
        shown = box.inside(self.resolution)
        if box.is_empty or shown.is_empty:
            return Frame(result)
        resized = cv2.resize(
            picture.pixels, (box.width, box.height), interpolation=cv2.INTER_AREA
        )
        result[shown.top : shown.bottom, shown.left : shown.right] = resized[
            shown.top - box.top : shown.bottom - box.top,
            shown.left - box.left : shown.right - box.left,
        ]
        return Frame(result)

    def fitted(self, picture: Frame, into: PlanarBoundingBox) -> Frame:
        """
        :param picture: What to paste.
        :param into: The room it may take.
        :return: A copy with the picture pasted as large as it goes into the room
            without distorting it, centred there.
        """
        height, width = picture.pixels.shape[:2]
        return self.pasted(picture, self.fitting(into, width / height))

    def dimmed(self, by: float) -> Frame:
        """
        :param by: How far towards white, from zero to one.
        :return: A copy faded towards white.
        """
        faded = self.pixels.astype(np.float32) * (1.0 - by) + 255.0 * by
        return Frame(np.clip(faded + 0.5, 0, 255).astype(np.uint8))

    def blended(self, other: Frame, weight: float) -> Frame:
        """
        :param other: The frame shown when the weight is one.
        :param weight: How much of the other frame shows, from zero to one.
        :return: A mix of this frame and the other.
        """
        mixed = (
            self.pixels.astype(np.float32) * (1.0 - weight)
            + other.pixels.astype(np.float32) * weight
        )
        return Frame(np.clip(mixed + 0.5, 0, 255).astype(np.uint8))

    def framed(
        self, around: PlanarBoundingBox, color: Color, thickness: int = 3
    ) -> Frame:
        """
        :param around: The box.
        :param color: What the outline is drawn in.
        :param thickness: Pixels the outline is thick.
        :return: A copy with the box outlined.
        """
        result = self.pixels.copy()
        box = PixelBox.nearest(around)
        cv2.rectangle(
            result,
            (box.left, box.top),
            (box.right, box.bottom),
            tuple(color.to_rgb_bytes()),
            thickness,
        )
        return Frame(result)

    def filled(self, around: PlanarBoundingBox, color: Color) -> Frame:
        """
        :param around: The box.
        :param color: What it is painted in.
        :return: A copy with the box painted over.
        """
        result = self.pixels.copy()
        box = PixelBox.nearest(around)
        cv2.rectangle(
            result,
            (box.left, box.top),
            (box.right, box.bottom),
            tuple(color.to_rgb_bytes()),
            -1,
        )
        return Frame(result)

    def lined(
        self, start: Point2, end: Point2, color: Color, thickness: int = 2
    ) -> Frame:
        """
        :param start: Where the line starts.
        :param end: Where it ends.
        :param color: What it is drawn in.
        :param thickness: Pixels it is thick.
        :return: A copy with a straight line drawn on it.
        """
        result = self.pixels.copy()
        cv2.line(
            result,
            astuple(Pixel.nearest(start)),
            astuple(Pixel.nearest(end)),
            tuple(color.to_rgb_bytes()),
            thickness,
            cv2.LINE_AA,
        )
        return Frame(result)

    def arrowed(
        self,
        start: Point2,
        end: Point2,
        color: Color,
        thickness: int = 2,
        head_length: float = 12.0,
    ) -> Frame:
        """
        :param start: Where the arrow starts.
        :param end: Where its head is.
        :param color: What it is drawn in.
        :param thickness: Pixels it is thick.
        :param head_length: Pixels the head is long, at most half the arrow.
        :return: A copy with an arrow drawn on it.
        """
        result = self.pixels.copy()
        length = max(
            (
                (float(end.x) - float(start.x)) ** 2
                + (float(end.y) - float(start.y)) ** 2
            )
            ** 0.5,
            1.0,
        )
        cv2.arrowedLine(
            result,
            astuple(Pixel.nearest(start)),
            astuple(Pixel.nearest(end)),
            tuple(color.to_rgb_bytes()),
            thickness,
            cv2.LINE_AA,
            tipLength=min(head_length / length, 0.5),
        )
        return Frame(result)
