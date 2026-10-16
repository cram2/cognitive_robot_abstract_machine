"""
Footage played back in a video: a recording read from its file, framed, filling the
frame, sped up and saying by how much, with a second view of the same stretch small in
its corner.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import timedelta

from krrood.patterns.role import Role
from semantic_digital_twin.spatial_types.spatial_types import Point2

from crollywood.frame import Frame
from crollywood.lettering import FontFace
from crollywood.theme import VideoTheme
from crollywood.typesetting import Anchor, TextSize, Typesetting
from crollywood.video import Video

# %% what of a picture is shown


@dataclass(frozen=True)
class Framing:
    """
    The margins cut off a camera picture before it is shown, in the picture's pixels.

    It is applied to the drawn picture, after any findings are drawn on it, so nothing
    drawn in the camera's pixel coordinates has to move.
    """

    top: int = 0
    """
    Rows cut off the top.
    """

    left: int = 0
    """
    Columns cut off the left.
    """

    right: int = 0
    """
    Columns cut off the right.
    """

    bottom: int = 0
    """
    Rows cut off the bottom.
    """

    def of(self, picture: Frame) -> Frame:
        """
        :param picture: A picture.
        :return: What is left of it inside the margins.
        """
        height, width = picture.pixels.shape[:2]
        return Frame(
            picture.pixels[
                self.top : height - self.bottom, self.left : width - self.right
            ]
        )


# %% a stretch of a recording


@dataclass
class ViewOfTheRun:
    """
    One camera's view of a stretch of a run.
    """

    video: Video
    """
    What the camera recorded.
    """

    starts_at: timedelta
    """
    Where in the video the stretch starts: the moment that matches the stretch's start
    in every other viewpoint of it.
    """

    framing: Framing = field(default_factory=Framing)
    """
    What of each picture is shown.
    """

    def frame_at(self, into: timedelta) -> Frame:
        """
        :param into: How far into the stretch.
        :return: What the camera saw then, framed.
        """
        return self.framing.of(self.video.frame_at(self.starts_at + into))


# %% badges on footage


@dataclass(eq=False)
class FootageFrame(Role[Frame]):
    """
    A frame of footage, which carries badges: short labels on dark boxes, in white
    letters whatever the theme.
    """

    theme: VideoTheme = field(default_factory=VideoTheme.neutral)
    """
    The colours the badge is drawn in.
    """

    badge_height: int = 36
    """
    Pixels a badge is tall.
    """

    badge_padding: int = 14
    """
    Pixels between a badge's edge and its text.
    """

    def badged(self, text: str, at: Point2, anchor: Anchor = Anchor.LEFT_TOP) -> Frame:
        """
        :param text: What the badge says.
        :param at: Where its corner goes.
        :param anchor: Which corner: the left top, the right top or the right middle
            standing for the right bottom.
        :return: A copy of the frame with the badge on it.
        """
        lettering = Typesetting(
            size=TextSize.LABEL, face=FontFace.BOLD, color=self.theme.footage_text
        )
        width = lettering.width_of(text) + 2 * self.badge_padding
        x, y = float(at.x), float(at.y)
        if anchor is Anchor.LEFT_TOP:
            badge = Frame.box(x, y, width, self.badge_height)
        elif anchor is Anchor.RIGHT_TOP:
            badge = Frame.box(x - width, y, width, self.badge_height)
        else:
            badge = Frame.box(
                x - width, y - self.badge_height, width, self.badge_height
            )
        frame = self.role_taker.filled(badge, self.theme.footage_badge)
        centre = Point2(
            (badge.min_x + badge.max_x) / 2, (badge.min_y + badge.max_y) / 2
        )
        return lettering.written(frame, text, centre, Anchor.CENTRE_MIDDLE)

    def speed_badged(self, speed: float) -> Frame:
        """
        :param speed: How many recorded seconds pass per second played.
        :return: A copy of the frame with the speed-up in its top right corner.
        """
        resolution = self.role_taker.resolution
        return self.badged(
            f"×{speed:g}",
            Point2(resolution.width - resolution.margin, resolution.margin),
            Anchor.RIGHT_TOP,
        )
