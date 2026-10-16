"""
Tests for :mod:`crollywood.multiplexer`: the tracks a video, its narration and its
subtitles are joined into.
"""

from __future__ import annotations

import subprocess
from datetime import timedelta
from pathlib import Path

import pytest

from crollywood.encoding import FfmpegEncoderProgram, VideoFile
from crollywood.multiplexer import (
    Language,
    MediaMultiplexer,
    NarrationTrack,
    PictureTrack,
    SubtitleTrack,
    Track,
)
from krrood.units import BitRate, DataSize

TRACK_FILE = Path("track")
INPUT_INDEX = 2
NARRATION_BIT_RATE = BitRate(DataSize(48_000))

# %% the tracks


@pytest.mark.parametrize(
    "track, stream_kind",
    [
        (PictureTrack(TRACK_FILE), "v"),
        (NarrationTrack(TRACK_FILE, bit_rate=NARRATION_BIT_RATE), "a"),
        (SubtitleTrack(TRACK_FILE), "s"),
    ],
)
def test_a_track_takes_its_stream_from_its_own_input(
    track: Track, stream_kind: str
) -> None:
    options = track.options(INPUT_INDEX)
    assert options[:2] == ["-map", f"{INPUT_INDEX}:{stream_kind}:0"]


def test_the_narration_is_encoded_at_its_bit_rate() -> None:
    options = NarrationTrack(TRACK_FILE, bit_rate=NARRATION_BIT_RATE).options(
        INPUT_INDEX
    )
    assert options[options.index("-b:a") + 1] == str(
        NARRATION_BIT_RATE.per_second.magnitude
    )


def test_a_track_is_labelled_with_its_language() -> None:
    options = SubtitleTrack(TRACK_FILE, language=Language.ENGLISH).options(INPUT_INDEX)
    assert f"language={Language.ENGLISH}" in options


def test_the_sound_takes_the_narration_over_the_video_and_the_subtitle_allowance() -> (
    None
):
    multiplexer = MediaMultiplexer(
        ffmpeg=FfmpegEncoderProgram(Path("ffmpeg")),
        audio_bit_rate=BitRate(DataSize(8_000)),
        subtitle_allowance=DataSize(100),
    )
    assert multiplexer.size_of_sound(timedelta(seconds=10)) == DataSize(80_100)


# %% the tracks joined


def test_the_subtitles_are_joined_only_when_given(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    commands = []
    monkeypatch.setattr(
        subprocess, "run", lambda command, check: commands.append(command)
    )
    multiplexer = MediaMultiplexer(ffmpeg=FfmpegEncoderProgram(Path("ffmpeg")))
    video, soundtrack, subtitles = (
        tmp_path / "video.mp4",
        tmp_path / "narration.wav",
        tmp_path / "subtitles.srt",
    )
    multiplexer.joined(VideoFile(video), soundtrack, tmp_path / "joined.mp4")
    multiplexer.joined(VideoFile(video), soundtrack, tmp_path / "joined.mp4", subtitles)
    without, with_subtitles = commands
    assert str(subtitles) not in without
    assert str(subtitles) in with_subtitles


def test_the_tracks_are_joined_by_the_multiplexer_program(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    commands = []
    monkeypatch.setattr(
        subprocess, "run", lambda command, check: commands.append(command)
    )
    multiplexer = MediaMultiplexer(ffmpeg=FfmpegEncoderProgram(Path("ffmpeg")))
    multiplexer.joined(
        VideoFile(tmp_path / "video.mp4"),
        tmp_path / "narration.wav",
        tmp_path / "joined.mp4",
    )
    (command,) = commands
    assert command[0] == str(multiplexer.ffmpeg.path)
