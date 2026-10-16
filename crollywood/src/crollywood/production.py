"""
A storyboard turned into a finished mp4: the slides grown to hold their lines, the
narration said and placed, the picture encoded to the size the file is allowed, and the
narration and the subtitles joined to it.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import timedelta
from enum import StrEnum
from pathlib import Path

from crollywood.captions import BandCaption, CaptionStyle, Subtitled
from crollywood.encoding import H264Encoder, VideoFile
from crollywood.multiplexer import MediaMultiplexer
from crollywood.narration import NarratedTimeline, Storyboard
from crollywood.speech import KokoroTextToSpeech, TextToSpeech
from crollywood.timeline import TimelineStyle, VideoTimeline
from krrood.units import BitRate, DataSize

logger = logging.getLogger(__name__)


class Subtitling(StrEnum):
    """
    How the subtitles go into the video.
    """

    BURNED_IN = "burned-in"
    """
    Drawn into the picture, in the band every scene keeps clear: part of the video.
    """

    SOFT = "soft"
    """
    A text track the viewer can switch off, with the same cues as a SubRip file next to
    the mp4 for players that take one.
    """


@dataclass
class VideoProduction:
    """
    Makes a narrated, subtitled mp4 from a storyboard.
    """

    storyboard: Storyboard
    """
    The scenes in order, each with the lines that start with it.
    """

    voice: TextToSpeech = field(default_factory=KokoroTextToSpeech)
    """
    What says the narration.
    """

    timeline_style: TimelineStyle = field(default_factory=TimelineStyle)
    """
    How the scenes are played: the frame rate, the dissolve between them and what is
    drawn over every frame, such as a :class:`~crollywood.overlays.ProgressBar`.
    """

    subtitling: Subtitling = Subtitling.BURNED_IN
    """
    How the subtitles go in.
    """

    captions: CaptionStyle = field(default_factory=BandCaption)
    """
    How burned-in subtitles are drawn.
    """

    encoder: H264Encoder = field(default_factory=H264Encoder)
    """
    What encodes the picture to the size it is allowed.
    """

    picture_bit_rate: BitRate = field(
        default_factory=lambda: BitRate(DataSize(2_000_000))
    )
    """
    How much data one second of picture may take: what keeps slides and footage at 720
    rows sharp.
    """

    multiplexer: MediaMultiplexer = field(default_factory=MediaMultiplexer)
    """
    What joins the picture, the narration and the subtitles.
    """

    def narrated_timeline(self) -> NarratedTimeline:
        """
        The timeline with its slides grown to hold their lines, and the narration placed
        on it, checked to run clear of itself and of the video's end.

        :return: The timeline with its narration.
        :raises NarrationOverrun: If a line runs into the next or past the end.
        """
        self.storyboard.fitted_to(self.voice)
        timeline = VideoTimeline(self.storyboard.scenes, self.timeline_style)
        narrated = NarratedTimeline(
            role_taker=timeline,
            narration=self.storyboard.narrated_by(self.voice, self.timeline_style),
        )
        narrated.check()
        return narrated

    def picture_of(self, narrated: NarratedTimeline) -> VideoTimeline:
        """
        :param narrated: The narrated timeline.
        :return: What is encoded as the picture: the timeline with its subtitles drawn
            in when they are burned in, else the timeline alone.
        """
        if self.subtitling is not Subtitling.BURNED_IN:
            return narrated.role_taker
        return Subtitled.over(narrated, captions=self.captions)

    def size_allowed_for(self, duration: timedelta) -> DataSize:
        """
        :param duration: How long the video plays.
        :return: The most the whole file may take: the picture bit rate over that long.
        """
        return self.picture_bit_rate.over(duration)

    def picture_budget_for(self, duration: timedelta) -> DataSize:
        """
        :param duration: How long the video plays.
        :return: What the picture may take: the size the file is allowed, leaving room
            for the sound.
        """
        return self.size_allowed_for(duration) - self.multiplexer.size_of_sound(
            duration
        )

    def written_to(self, path: Path) -> VideoFile:
        """
        Render and encode the video.

        :param path: Where the mp4 goes; the SubRip file of soft subtitles goes beside
            it.
        :return: The file written.
        """
        narrated = self.narrated_timeline()
        duration = narrated.duration
        logger.info(
            "%.1f s of video in %d scenes",
            duration.total_seconds(),
            len(narrated.scenes),
        )
        picture = self.picture_of(narrated)
        silent = self.encoder.encode(
            picture,
            path.with_name(path.stem + "_silent.mp4"),
            self.picture_budget_for(duration),
        )
        soundtrack = narrated.soundtrack().written_to(path.with_suffix(".wav"))
        subtitles = (
            narrated.narration.written_to(path.with_suffix(".srt"))
            if self.subtitling is Subtitling.SOFT
            else None
        )
        video = self.multiplexer.joined(silent, soundtrack, path, subtitles=subtitles)
        silent.path.unlink()
        soundtrack.unlink()
        logger.info(
            "%s: %.1f s, %d bytes",
            video.path,
            video.duration.total_seconds(),
            video.size.bytes,
        )
        return video
