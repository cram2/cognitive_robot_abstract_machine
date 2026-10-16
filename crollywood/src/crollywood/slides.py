"""
Slides, shown for as long as they are held: a single picture, a table of data, and a
card of a heading and a line.
"""

from __future__ import annotations

from abc import ABC
from dataclasses import dataclass, field
from datetime import timedelta

from semantic_digital_twin.spatial_types.spatial_types import Point2

from crollywood.frame import Frame, Resolution
from crollywood.lettering import FontFace
from crollywood.theme import VideoTheme
from crollywood.typesetting import Anchor, TextSize
from crollywood.video import Video

# %% a slide


@dataclass
class HeldSlide(Video, ABC):
    """
    A video that lasts as long as it is held, rather than as long as what it shows, so
    it may be held longer to make room for what is said over it.
    """

    held_for: timedelta = field(kw_only=True)
    """
    How long it is held.
    """

    @property
    def duration(self) -> timedelta:
        """
        How long the slide is held.
        """
        return self.held_for


@dataclass
class SingleFrameSlide(HeldSlide):
    """
    One picture held for a while.
    """

    picture: Frame
    """
    The picture.
    """

    def frame_at(self, moment: timedelta) -> Frame:
        """
        :param moment: A moment of the slide, from its start.
        :return: The picture, the same at every moment.
        """
        return self.picture


# %% a table of data


@dataclass(frozen=True)
class TabularDataRow:
    """
    One row of a table: its name and its value.
    """

    row_name: str
    """
    What the row is of, on its left.
    """

    value: str
    """
    Its value, on its right, written with its unit if it has one.
    """


@dataclass
class TabularDataSlide(HeldSlide):
    """
    Data as a table of named rows: a plain title over it, one hairline over each row,
    the values bold and right-aligned, centred on the page with nothing else on it.
    """

    title: str
    """
    What the table is of, over it.
    """

    rows: tuple[TabularDataRow, ...]
    """
    The rows.
    """

    held_for: timedelta = field(default=timedelta(seconds=4), kw_only=True)
    """
    How long the table is held.
    """

    resolution: Resolution = field(default_factory=Resolution.standard)
    """
    The size of the slide.
    """

    theme: VideoTheme = field(default_factory=VideoTheme.neutral)
    """
    The colours and fonts it is drawn in.
    """

    table_width: int = 900
    """
    Pixels the table is wide.
    """

    row_pitch: int = 64
    """
    Pixels from one row of the table to the next.
    """

    title_above_table: int = 50
    """
    Pixels the table's title stands above its first rule.
    """

    cell_inset: int = 8
    """
    Pixels between a cell's text and the table's edge.
    """

    def frame_at(self, moment: timedelta) -> Frame:
        """
        :param moment: A moment of the slide, from its start.
        :return: The table, the same at every moment.
        """
        frame = self.theme.page_of(self.resolution)
        left = self.resolution.width / 2 - self.table_width / 2
        right = left + self.table_width
        height = self.row_pitch * len(self.rows)
        top = (self.resolution.stage_height - height) / 2 + self.title_above_table / 2
        frame = self.theme.typesetting(TextSize.BODY, FontFace.BOLD).written(
            frame,
            self.title,
            Point2(self.resolution.width / 2, top - self.title_above_table),
            Anchor.CENTRE_MIDDLE,
        )
        row_name = self.theme.typesetting(TextSize.BODY)
        value = self.theme.typesetting(TextSize.BODY, FontFace.BOLD)
        for number, row in enumerate(self.rows):
            y = top + number * self.row_pitch
            frame = frame.lined(
                Point2(left, y), Point2(right, y), self.theme.hairline, thickness=2
            )
            middle = y + self.row_pitch / 2
            frame = row_name.written(
                frame,
                row.row_name,
                Point2(left + self.cell_inset, middle),
                Anchor.LEFT_MIDDLE,
            )
            frame = value.written(
                frame,
                row.value,
                Point2(right - self.cell_inset, middle),
                Anchor.RIGHT_MIDDLE,
            )
        return frame.lined(
            Point2(left, top + height),
            Point2(right, top + height),
            self.theme.hairline,
            thickness=2,
        )


# %% a card of text


@dataclass
class TextCard(HeldSlide):
    """
    A heading in bold with one line under it, centred on the page: a title card, or an
    end card with a link.
    """

    heading: str
    """
    What is written in bold.
    """

    line: str
    """
    What is written under it.
    """

    held_for: timedelta = field(default=timedelta(seconds=5), kw_only=True)
    """
    How long the card is held.
    """

    resolution: Resolution = field(default_factory=Resolution.standard)
    """
    The size of the slide.
    """

    theme: VideoTheme = field(default_factory=VideoTheme.neutral)
    """
    The colours and fonts it is drawn in.
    """

    line_gap: int = 54
    """
    Pixels from the heading's middle to the line's middle.
    """

    def frame_at(self, moment: timedelta) -> Frame:
        """
        :param moment: A moment of the slide, from its start.
        :return: The card, the same at every moment.
        """
        frame = self.theme.page_of(self.resolution)
        centre = self.resolution.width / 2
        middle = self.resolution.stage_height / 2
        frame = self.theme.typesetting(TextSize.BODY, FontFace.BOLD).written(
            frame,
            self.heading,
            Point2(centre, middle - self.line_gap / 2),
            Anchor.CENTRE_MIDDLE,
        )
        return self.theme.typesetting(TextSize.BODY).written(
            frame,
            self.line,
            Point2(centre, middle + self.line_gap / 2),
            Anchor.CENTRE_MIDDLE,
        )
