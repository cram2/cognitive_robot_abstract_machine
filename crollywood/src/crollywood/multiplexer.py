"""
The picture, the narration and the subtitles joined into one mp4.
"""

from __future__ import annotations

import subprocess
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import timedelta
from enum import StrEnum
from pathlib import Path

from crollywood.encoding import FfmpegEncoderProgram, VideoFile
from krrood.units import BitRate, DataSize

# %% the tracks of the mp4


class Language(StrEnum):
    """
    A language a track is in, as its ISO 639-2 code.
    """

    ENGLISH = "eng"


@dataclass(frozen=True)
class Track(ABC):
    """
    One stream of the joined mp4, read from a file of its own.
    """

    path: Path
    """
    The file the stream is read from.
    """

    language: Language = Language.ENGLISH
    """
    The language the track is in.
    """

    @abstractmethod
    def options(self, input_index: int) -> list[str]:
        """
        :param input_index: Where among the encoder's inputs the file is.
        :return: The encoder's options that put the stream into the mp4 and encode it.
        """


@dataclass(frozen=True)
class PictureTrack(Track):
    """
    The picture, copied as it was encoded.
    """

    def options(self, input_index: int) -> list[str]:
        return ["-map", f"{input_index}:v:0", "-c:v", "copy"]


@dataclass(frozen=True)
class NarrationTrack(Track):
    """
    The narration, encoded as AAC, the codec every player decodes, in one channel.
    """

    bit_rate: BitRate = field(kw_only=True)
    """
    How much data each second of narration is encoded into.
    """

    def options(self, input_index: int) -> list[str]:
        return [
            "-map",
            f"{input_index}:a:0",
            "-c:a",
            "aac",
            "-b:a",
            str(self.bit_rate.per_second.magnitude),
            "-ac",
            "1",
            "-metadata:s:a:0",
            f"language={self.language}",
        ]


@dataclass(frozen=True)
class SubtitleTrack(Track):
    """
    The subtitles, as a text track the viewer can switch off, on by default.
    """

    def options(self, input_index: int) -> list[str]:
        return [
            "-map",
            f"{input_index}:s:0",
            "-c:s",
            "mov_text",
            "-metadata:s:s:0",
            f"language={self.language}",
            "-disposition:s:0",
            "default",
        ]


# %% the tracks joined


@dataclass
class MediaMultiplexer:
    """
    Joins a silent video, its narration and its subtitles into one mp4.
    """

    ffmpeg: FfmpegEncoderProgram = field(default_factory=FfmpegEncoderProgram)
    """
    The program that joins the streams.
    """

    audio_bit_rate: BitRate = field(default_factory=lambda: BitRate(DataSize(64_000)))
    """
    How much data each second of narration is encoded into: one channel of speech needs
    no more.
    """

    subtitle_allowance: DataSize = field(
        default_factory=lambda: DataSize.of_bytes(40_000)
    )
    """
    What is set aside for the subtitle track and what joining the streams adds to the
    file.
    """

    def size_of_sound(self, duration: timedelta) -> DataSize:
        """
        :param duration: How long the video plays.
        :return: What the narration and the subtitles take over it, to keep out of the
            picture's budget.
        """
        return self.audio_bit_rate.over(duration) + self.subtitle_allowance

    def joined(
        self,
        video: VideoFile,
        soundtrack: Path,
        path: Path,
        subtitles: Path | None = None,
    ) -> VideoFile:
        """
        :param video: The silent video.
        :param soundtrack: The narration, as a wave file as long as the video.
        :param path: Where the joined mp4 goes.
        :param subtitles: The subtitles, as a SubRip file, or None where they are
            already in the picture.
        :return: The file written.
        """
        tracks: list[Track] = [
            PictureTrack(video.path),
            NarrationTrack(soundtrack, bit_rate=self.audio_bit_rate),
        ]
        if subtitles is not None:
            tracks.append(SubtitleTrack(subtitles))
        arguments: list[str] = []
        for track in tracks:
            arguments += ["-i", str(track.path)]
        for input_index, track in enumerate(tracks):
            arguments += track.options(input_index)
        arguments += ["-movflags", "+faststart", str(path)]
        subprocess.run(self.ffmpeg.command(arguments), check=True)
        return VideoFile(path=path)
