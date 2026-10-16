"""
Footage played back: a stretch of a run filling the frame, and a title over it.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import timedelta

from semantic_digital_twin.spatial_types.spatial_types import Point2
from semantic_digital_twin.world_description.geometry import PlanarBoundingBox

from crollywood.footage import FootageFrame, ViewOfTheRun
from crollywood.frame import Frame, Resolution
from crollywood.lettering import FontFace
from crollywood.shading import EvenShade, Shade
from crollywood.theme import VideoTheme
from crollywood.timeline import FrameFade
from crollywood.typesetting import Anchor, TextSize, Typesetting
from crollywood.video import Video

# %% the frame filled with footage


@dataclass(frozen=True)
class InsetStill:
    """
    A picture the inset shows before it runs, such as what a camera saw at one moment
    with what was found on it drawn, and for how long it is shown.
    """

    picture: Frame
    """
    The picture.
    """

    shown_for: timedelta
    """
    How long it is shown, from the footage's start.
    """


@dataclass
class FootagePlayback(Video):
    """
    A stretch of a run filling the whole frame, played faster than recorded with the
    speed-up badged in the top right corner, and optionally a second view of the same
    stretch small in the bottom right corner, in step.
    """

    full_view: ViewOfTheRun
    """
    The camera that fills the frame.
    """

    length: timedelta
    """
    How much of the recording the stretch runs for.
    """

    speed: float = 2.0
    """
    How many recorded seconds pass per second played.
    """

    inset: ViewOfTheRun | None = None
    """
    The second view, or None to show the one camera.
    """

    inset_still: InsetStill | None = None
    """
    What the inset shows before it runs, if anything.
    """

    resolution: Resolution = field(default_factory=Resolution.standard)
    """
    The size it is drawn at: the video's own, the footage filling it.
    """

    inset_width: int = 300
    """
    Pixels wide the second view is in the corner of the footage, at the video's size.
    """

    theme: VideoTheme = field(default_factory=VideoTheme.neutral)
    """
    The colours of the badge and of the frame around the inset.
    """

    @property
    def duration(self) -> timedelta:
        """
        How long the stretch plays at its speed.
        """
        return self.length / self.speed

    @property
    def inset_panel(self) -> PlanarBoundingBox:
        """
        Where the inset lies: the bottom right corner of the footage, above the band the
        captions are read on.
        """
        sample = self.inset.frame_at(timedelta(0))
        sample_height, sample_width = sample.pixels.shape[:2]
        height = self.inset_width * sample_height / sample_width
        return Frame.box(
            self.resolution.width - self.resolution.margin - self.inset_width,
            self.resolution.stage_height - self.resolution.margin / 2 - height,
            self.inset_width,
            height,
        )

    def inset_picture_at(self, moment: timedelta) -> Frame:
        """
        :param moment: A moment of the footage played back.
        :return: What the inset shows then: what it holds first, else the second view.
        """
        if self.inset_still is not None and moment < self.inset_still.shown_for:
            return self.inset_still.picture
        return self.inset.frame_at(moment * self.speed)

    def frame_at(self, moment: timedelta) -> Frame:
        """
        :param moment: A moment of the footage played back, from its start.
        :return: The footage then, with the inset in its corner and the speed badged.
        """
        frame = self.resolution.blank(0).pasted(
            self.full_view.frame_at(moment * self.speed), self.resolution.whole
        )
        if self.inset is not None:
            panel = self.inset_panel
            frame = frame.filled(panel.bloat(3, 3), self.theme.footage_text)
            frame = frame.pasted(self.inset_picture_at(moment), panel)
        # footage at its own pace carries no badge
        if self.speed == 1.0:
            return frame
        return FootageFrame(role_taker=frame, theme=self.theme).speed_badged(self.speed)


@dataclass
class TitleOverFootage(Video):
    """
    A title over footage, the footage shaded so the title is what is read; the title
    fades out at the end.
    """

    footage: Video
    """
    What plays under the title, for as long as it lasts.
    """

    title: str
    """
    The title.
    """

    line: str
    """
    What stands under the title, in smaller letters.
    """

    title_for: timedelta
    """
    How long the title stands, before it fades out.
    """

    fade: timedelta = timedelta(seconds=0.25)
    """
    How long the title takes to fade out.
    """

    shade: Shade = EvenShade(0.55)
    """
    What the footage is shaded with under the title.
    """

    theme: VideoTheme = field(default_factory=VideoTheme.neutral)
    """
    The colour the title is written in.
    """

    title_width_reduction: int = 120
    """
    Pixels narrower than the space inside the margins a title is wrapped to.
    """

    title_raise: int = 20
    """
    Pixels above the middle of the stage the title is centred.
    """

    line_drop: int = 44
    """
    Pixels below the title's last row the line is centred.
    """

    row_height_share: float = 0.7
    """
    The share of the title's letter height each of its rows takes when the line is
    placed under them.
    """

    @property
    def duration(self) -> timedelta:
        """
        How long the footage under the title plays.
        """
        return self.footage.duration

    def title_at(self, moment: timedelta) -> float:
        """
        :param moment: A moment from the start.
        :return: How far the title is up then, from zero to one.
        """
        return 1.0 - FrameFade(self.fade).weight_at(moment - self.title_for)

    def frame_at(self, moment: timedelta) -> Frame:
        """
        :param moment: A moment from the start.
        :return: The footage then, with the title over it while it stands.
        """
        frame = self.footage.frame_at(moment)
        weight = self.title_at(moment)
        if weight <= 0.0:
            return frame
        resolution = frame.resolution
        shaded = self.shade.over(frame, resolution.whole)
        heading = Typesetting(
            size=TextSize.CLAIM, face=FontFace.BOLD, color=self.theme.footage_text
        )
        text = heading.wrapped(
            self.title,
            resolution.width - 2 * resolution.margin - self.title_width_reduction,
        )
        rows = text.count("\n") + 1
        middle = resolution.stage_height / 2 - self.title_raise
        centre = resolution.width / 2
        written = heading.written(
            shaded, text, Point2(centre, middle), Anchor.CENTRE_MIDDLE
        )
        written = Typesetting(
            size=TextSize.BODY, color=self.theme.footage_text
        ).written(
            written,
            self.line,
            Point2(
                centre,
                middle + rows * TextSize.CLAIM * self.row_height_share + self.line_drop,
            ),
            Anchor.CENTRE_MIDDLE,
        )
        return frame.blended(written, weight)
