"""
Writing text on a frame: where it goes, what it is written in and how it breaks.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import IntEnum, StrEnum

from PIL import Image, ImageDraw
import numpy as np
from semantic_digital_twin.spatial_types.spatial_types import Point2
from semantic_digital_twin.world_description.geometry import Color

from crollywood.frame import Frame
from crollywood.lettering import FontFace, Typeface


class TextSize(IntEnum):
    """
    The heights of letters a video writes in, in pixels of the video's own size.
    """

    LABEL = 20
    """
    Labels: the smallest size the video writes in.
    """

    BODY = 28
    """
    Captions, body text and table cells.
    """

    CLAIM = 40
    """
    A title or a claim: the largest size.
    """


class Anchor(StrEnum):
    """
    Which point of a line of text is put where it is written, as the drawing library
    names anchors.
    """

    LEFT_MIDDLE = "lm"
    """
    The left end, halfway up the letters.
    """

    CENTRE_MIDDLE = "mm"
    """
    The centre, halfway up the letters.
    """

    RIGHT_MIDDLE = "rm"
    """
    The right end, halfway up the letters.
    """

    LEFT_TOP = "lt"
    """
    The left end, at the top of the letters.
    """

    CENTRE_TOP = "mt"
    """
    The centre, at the top of the letters.
    """

    RIGHT_TOP = "rt"
    """
    The right end, at the top of the letters.
    """

    @property
    def horizontal(self) -> str:
        """
        The drawing library's code for the horizontal part of the anchor.
        """
        return self.value[0]

    @property
    def is_horizontally_centred(self) -> bool:
        """
        Whether the anchor is the centre of the line.
        """
        return self.value[0] == "m"

    @property
    def is_vertically_middle(self) -> bool:
        """
        Whether the anchor is halfway up the letters.
        """
        return self.value[1] == "m"


@dataclass(frozen=True)
class Typesetting:
    """
    How a line of text is written on a frame.
    """

    size: int = TextSize.BODY
    """
    The height of the letters, in pixels.
    """

    face: FontFace = FontFace.REGULAR
    """
    The face they are set in.
    """

    color: Color = field(default_factory=Color.BLACK)
    """
    What they are written in.
    """

    typeface: Typeface = field(default_factory=Typeface)
    """
    The fonts the faces are set in.
    """

    @property
    def row_spacing(self) -> float:
        """
        Pixels between the rows of text that break over several lines.
        """
        return self.size * 0.35

    def written(
        self,
        frame: Frame,
        text: str,
        at: Point2,
        anchor: Anchor = Anchor.LEFT_MIDDLE,
    ) -> Frame:
        """
        :param frame: The frame written on.
        :param text: What to write; new lines break it.
        :param at: Where the anchor goes, in pixels.
        :param anchor: Which point of the text goes there.
        :return: A copy of the frame with the text written on it.
        """
        picture = Image.fromarray(frame.pixels)
        drawing = ImageDraw.Draw(picture)
        font = self.typeface.font(self.face, self.size)
        position = (float(at.x), float(at.y))
        anchor_value = anchor.value
        if "\n" in text:
            # several rows hang from their top; a middle anchor is met by raising them
            # by half their height
            anchor_value = anchor.horizontal + "a"
            if anchor.is_vertically_middle:
                _, top, _, bottom = drawing.multiline_textbbox(
                    position,
                    text,
                    font=font,
                    anchor=anchor_value,
                    spacing=self.row_spacing,
                )
                position = (position[0], position[1] - (bottom - top) / 2)
        drawing.multiline_text(
            position,
            text,
            fill=tuple(self.color.to_rgb_bytes()),
            font=font,
            anchor=anchor_value,
            align="center" if anchor.is_horizontally_centred else "left",
            spacing=self.row_spacing,
        )
        return Frame(np.array(picture))

    def width_of(self, text: str) -> float:
        """
        :param text: One line.
        :return: How many pixels wide it is set.
        """
        return self.typeface.font(self.face, self.size).getlength(text)

    def wrapped(self, text: str, width: float) -> str:
        """
        :param text: What to break.
        :param width: The widest a line may be, in pixels.
        :return: The text broken into lines no wider than given, at spaces.
        """
        lines: list[str] = []
        line = ""
        for word in text.split():
            candidate = word if not line else f"{line} {word}"
            if self.width_of(candidate) <= width or not line:
                line = candidate
            else:
                lines.append(line)
                line = word
        if line:
            lines.append(line)
        return "\n".join(lines)
