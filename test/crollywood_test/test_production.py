"""
Tests for :mod:`crollywood.production`: a minimal storyboard made into a narrated mp4,
with its subtitles burned in or as a track, encoded to the size it is allowed.
"""

from __future__ import annotations


import subprocess
from dataclasses import dataclass, field
from datetime import timedelta
from enum import StrEnum
from pathlib import Path

import imageio_ffmpeg
import numpy as np
import pytest

from crollywood.captions import BoxedCaption, Subtitled
from crollywood.encoding import EncoderPreset, H264Encoder, VideoFile
from crollywood.frame import Frame, Resolution
from crollywood.narration import NarratedLine, NarratedScene, Storyboard
from crollywood.overlays import ProgressBar
from crollywood.production import Subtitling, VideoProduction
from crollywood.slides import (
    TabularDataRow,
    TabularDataSlide,
    SingleFrameSlide,
    TextCard,
)
from crollywood.speech import SoundLevel
from crollywood.theme import VideoTheme
from crollywood.timeline import TimelineStyle, Transition
from krrood.units import DataSize

from .voices import EvenlyPacedTextToSpeech

RESOLUTION = Resolution.standard()
FRAMES_PER_SECOND = 10
SLIDE_HOLD = timedelta(seconds=1)
DISSOLVE = timedelta(seconds=0.2)
DARK = 128
VOICE = EvenlyPacedTextToSpeech(
    time_per_word=timedelta(seconds=0.25),
    sample_rate=16_000,
    level=SoundLevel(decibels=-20.0),
)
"""
A voice at a rate an audio encoder takes, quiet enough not to clip.
"""


def gradient() -> Frame:
    """
    :return: A picture that changes across the frame, so there is something to encode.
    """
    pixels = RESOLUTION.blank(0).pixels.copy()
    pixels[..., 0] = np.linspace(0, 255, RESOLUTION.width, dtype=np.uint8)
    return Frame(pixels)


def minimal_storyboard() -> Storyboard:
    """
    A title card said over, a picture held, and a table of results said over.
    """
    return Storyboard(
        [
            NarratedScene(
                role_taker=TextCard(
                    "A minimal video", "made by a test", held_for=SLIDE_HOLD
                ),
                lines=(NarratedLine("This video was made by a test."),),
            ),
            NarratedScene(role_taker=SingleFrameSlide(gradient(), held_for=SLIDE_HOLD)),
            NarratedScene(
                role_taker=TabularDataSlide(
                    "Results",
                    (TabularDataRow("Trials", "2 of 2"),),
                    held_for=SLIDE_HOLD,
                ),
                lines=(NarratedLine("Both trials succeeded."),),
                transition=Transition.CUT,
            ),
        ]
    )


def production(
    subtitling: Subtitling = Subtitling.BURNED_IN,
) -> VideoProduction:
    """
    The minimal storyboard made quickly by the test voice.
    """
    return VideoProduction(
        minimal_storyboard(),
        voice=VOICE,
        timeline_style=TimelineStyle(
            frames_per_second=FRAMES_PER_SECOND, dissolve=DISSOLVE
        ),
        subtitling=subtitling,
        encoder=H264Encoder(preset=EncoderPreset.FAST),
    )


class StreamKind(StrEnum):
    """
    The kinds of stream a video file holds, as ffmpeg selects them.
    """

    SOUND = "a"
    SUBTITLES = "s"


def holds_stream(video: VideoFile, kind: StreamKind) -> bool:
    """
    :param video: A video file.
    :param kind: The kind of stream.
    :return: Whether the file holds a stream of that kind.
    """
    command = [
        imageio_ffmpeg.get_ffmpeg_exe(),
        "-v",
        "error",
        "-i",
        str(video.path),
        "-map",
        f"0:{kind}",
        "-c",
        "copy",
        "-f",
        "null",
        "-",
    ]
    return subprocess.run(command, capture_output=True).returncode == 0


@pytest.fixture(scope="module")
def burned_in(tmp_path_factory: pytest.TempPathFactory) -> VideoFile:
    """
    The minimal video with its subtitles in the picture, made once.
    """
    return production().written_to(tmp_path_factory.mktemp("burned_in") / "minimal.mp4")


# %% the video that comes out


def test_the_minimal_video_plays_its_scenes_at_their_size_and_rate(
    burned_in: VideoFile,
) -> None:
    narrated = production().narrated_timeline()
    assert burned_in.resolution == RESOLUTION
    assert burned_in.frames_per_second == pytest.approx(FRAMES_PER_SECOND)
    assert burned_in.duration.total_seconds() == pytest.approx(
        narrated.duration.total_seconds(), abs=1.5 / FRAMES_PER_SECOND
    )


def test_the_minimal_video_carries_its_narration_and_no_subtitle_track(
    burned_in: VideoFile,
) -> None:
    assert holds_stream(burned_in, StreamKind.SOUND)
    assert not holds_stream(burned_in, StreamKind.SUBTITLES)


def test_the_minimal_video_leaves_nothing_but_itself_beside_it(
    burned_in: VideoFile,
) -> None:
    assert [path.name for path in burned_in.path.parent.iterdir()] == [
        burned_in.path.name
    ]


def test_the_subtitles_are_burned_into_the_band_while_a_line_is_said(
    burned_in: VideoFile,
) -> None:
    band = slice(int(RESOLUTION.stage_height), RESOLUTION.height)
    lead = production().storyboard.lead
    said = burned_in.frame_at(lead + timedelta(seconds=0.5))
    assert said.pixels[band].min() < DARK


def test_the_video_is_encoded_to_its_picture_bit_rate(
    burned_in: VideoFile,
) -> None:
    allowed = production().picture_bit_rate.over(burned_in.duration)
    assert DataSize(0) < burned_in.size <= allowed


# %% the slides grow to hold their lines


def test_a_slide_is_held_until_its_line_has_been_said() -> None:
    made = production()
    narrated = made.narrated_timeline()
    said = VOICE.speaks(narrated.narration.lines[0].line.said).duration
    storyboard = made.storyboard
    assert narrated.scenes[0].duration == storyboard.lead + said + storyboard.tail
    assert narrated.scenes[1].duration == SLIDE_HOLD


# %% subtitles as a track


def test_soft_subtitles_go_in_as_a_track_and_a_subrip_file_beside_it(
    tmp_path: Path,
) -> None:
    video = production(Subtitling.SOFT).written_to(tmp_path / "minimal.mp4")
    assert holds_stream(video, StreamKind.SUBTITLES) and holds_stream(
        video, StreamKind.SOUND
    )
    subrip = video.path.with_suffix(".srt")
    assert "This video was made by a test." in subrip.read_text()


# %% a size the production is allowed


@dataclass
class FixedSizeProduction(VideoProduction):
    """
    A production allowed one size, however long the video plays.
    """

    largest: DataSize = field(kw_only=True)
    """
    The most the file may take.
    """

    def size_allowed_for(self, duration: timedelta) -> DataSize:
        return self.largest


def test_a_production_that_allows_another_size_budgets_its_picture_from_it() -> None:
    made = FixedSizeProduction(
        minimal_storyboard(), voice=VOICE, largest=DataSize.of_bytes(150_000)
    )
    duration = made.narrated_timeline().duration
    sound = made.multiplexer.size_of_sound(duration)
    assert made.picture_budget_for(duration) == made.largest - sound


# %% what is drawn over the scenes


def test_a_production_draws_its_overlays_and_its_captions_in_their_style() -> None:
    bar = ProgressBar()
    made = VideoProduction(
        minimal_storyboard(),
        voice=VOICE,
        timeline_style=TimelineStyle(
            frames_per_second=FRAMES_PER_SECOND, overlays=[bar]
        ),
        captions=BoxedCaption(),
    )
    picture = made.picture_of(made.narrated_timeline())
    assert isinstance(picture, Subtitled)
    assert isinstance(picture.captions, BoxedCaption)
    assert picture.style.overlays == [bar]
    top = picture.frame_at(timedelta(0)).pixels[: bar.thickness]
    assert (top == tuple(VideoTheme.neutral().panel.to_rgb_bytes())).all()


def test_soft_subtitles_leave_the_picture_its_overlays_and_no_captions() -> None:
    bar = ProgressBar()
    made = VideoProduction(
        minimal_storyboard(),
        voice=VOICE,
        timeline_style=TimelineStyle(
            frames_per_second=FRAMES_PER_SECOND, overlays=[bar]
        ),
        subtitling=Subtitling.SOFT,
    )
    picture = made.picture_of(made.narrated_timeline())
    assert not isinstance(picture, Subtitled)
    assert picture.style.overlays == [bar]


def test_the_timeline_of_a_production_plays_in_the_production_timeline_style() -> None:
    made = production()
    assert made.narrated_timeline().role_taker.style is made.timeline_style
