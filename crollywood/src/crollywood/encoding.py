"""
Frames written out as one H.264 video no larger than a size budget.

The encoder is told the size the file may not exceed and picks the bit rate from that and
the length, in two passes; the file that comes out can be read back frame by frame.
"""

from __future__ import annotations

import subprocess
from dataclasses import dataclass, field
from datetime import timedelta
from enum import StrEnum
from pathlib import Path
from tempfile import TemporaryDirectory

import cv2
import imageio_ffmpeg
import numpy as np
from typing_extensions import Sequence

from crollywood.exceptions import EncodingFailed, NothingToRead
from crollywood.frame import Frame, Resolution
from crollywood.timeline import VideoTimeline
from krrood.units import BitRate, DataSize
from crollywood.video import Video

# %% the encoder's names


class PixelFormat(StrEnum):
    """
    How the encoder lays out the pixels of a frame, as it names the layouts.
    """

    RGB24 = "rgb24"
    """
    A byte each of red, green and blue per pixel: what a :class:`~crollywood.frame.Frame`
    holds.
    """

    YUV420P = "yuv420p"
    """
    Brightness per pixel and colour per square of four: the layout every player decodes,
    which needs even frame sides.
    """


class EncoderPreset(StrEnum):
    """
    How long the encoder may take to make the file small, as it names its presets.
    """

    ULTRAFAST = "ultrafast"
    """
    As quick as it goes, for a file that is kept losslessly and read again.
    """

    FAST = "fast"
    """
    Quick, for a draft to look at.
    """

    SLOW = "slow"
    """
    Thorough, for the video handed on.
    """


# %% the file that comes out


@dataclass(frozen=True)
class VideoFile(Video):
    """
    A video on disk: a camera's recording, a simulation filmed as it ran, or a video
    written out, read for what a submission is checked on.
    """

    path: Path
    """
    Where it lies.
    """

    @property
    def size(self) -> DataSize:
        """
        How much it takes on disk.
        """
        return DataSize.of_bytes(self.path.stat().st_size)

    def _opened(self) -> cv2.VideoCapture:
        """
        :return: The file opened for reading.
        """
        return cv2.VideoCapture(str(self.path))

    @property
    def resolution(self) -> Resolution:
        """
        The size of its frames.
        """
        opened = self._opened()
        return Resolution(
            width=int(opened.get(cv2.CAP_PROP_FRAME_WIDTH)),
            height=int(opened.get(cv2.CAP_PROP_FRAME_HEIGHT)),
        )

    @property
    def frames_per_second(self) -> float:
        """
        How many frames one second of it holds.
        """
        return float(self._opened().get(cv2.CAP_PROP_FPS))

    @property
    def frame_count(self) -> int:
        """
        How many frames it holds.
        """
        return int(self._opened().get(cv2.CAP_PROP_FRAME_COUNT))

    @property
    def duration(self) -> timedelta:
        """
        How long it plays.
        """
        return timedelta(seconds=self.frame_count / self.frames_per_second)

    def frame_at(self, moment: timedelta) -> Frame:
        """
        :param moment: The moment, from the video's start; past its end counts as its
            last frame.
        :return: The frame there, as red, green and blue.
        :raises NothingToRead: If the file holds no frame.
        """
        opened = self._opened()
        index = round(moment.total_seconds() * self.frames_per_second)
        opened.set(cv2.CAP_PROP_POS_FRAMES, min(index, self.frame_count - 1))
        read, frame = opened.read()
        if not read:
            raise NothingToRead(path=self.path, moment=moment)
        return Frame(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))


# %% the encoder program


@dataclass(frozen=True)
class FfmpegEncoderProgram:
    """
    The ffmpeg program that encodes and joins the streams of a video.
    """

    path: Path = field(default_factory=lambda: Path(imageio_ffmpeg.get_ffmpeg_exe()))
    """
    Where the program is; by default the one :mod:`imageio_ffmpeg` ships.
    """

    def command(self, arguments: Sequence[str]) -> list[str]:
        """
        :param arguments: What the program is told to do.
        :return: The command that runs the program with those arguments, overwriting
            what it writes and reporting only errors.
        """
        return [str(self.path), "-y", "-loglevel", "error", *arguments]


# %% the encoder


@dataclass
class H264Encoder:
    """
    Writes a timeline out as an H.264 mp4 no larger than the size budget it is given.

    The frames are first kept losslessly, since two passes have to read them, and then
    encoded twice at the bit rate the budget and the length allow.
    """

    preset: EncoderPreset = EncoderPreset.SLOW
    """
    How long the encoder may take to make the file small.
    """

    ffmpeg: FfmpegEncoderProgram = field(default_factory=FfmpegEncoderProgram)
    """
    The program that encodes.
    """

    pixel_format: PixelFormat = PixelFormat.YUV420P
    """
    How the pixels of the file are laid out.
    """

    bit_rate_headroom: float = 0.90
    """
    How much of the size budget the video stream is aimed at, leaving the container and
    the encoder's own overshoot room under the limit.
    """

    peak_bit_rate_factor: float = 1.5
    """
    How far above the average bit rate the encoder may go for a moment.
    """

    rate_control_buffer: timedelta = timedelta(seconds=2)
    """
    How long a stretch the encoder holds to the average bit rate over.
    """

    def encode(
        self, timeline: VideoTimeline, path: Path, size_budget: DataSize
    ) -> VideoFile:
        """
        Write the timeline out.

        :param timeline: What to write.
        :param path: Where the mp4 goes.
        :param size_budget: The most the file may take on disk.
        :return: The file written.
        """
        path.parent.mkdir(parents=True, exist_ok=True)
        bit_rate = self.bit_rate_for(size_budget, timeline.duration)
        with TemporaryDirectory(prefix="video_master_") as scratch:
            master = Path(scratch) / "master.mkv"
            self._keep_losslessly(timeline, master)
            self._two_passes(master, path, bit_rate, Path(scratch) / "pass")
        return VideoFile(path=path)

    def bit_rate_for(self, size_budget: DataSize, duration: timedelta) -> BitRate:
        """
        :param size_budget: The most the file may take on disk.
        :param duration: How long the video plays.
        :return: The bit rate that fills the budget over that time, less headroom.
        """
        return BitRate.filling(size_budget, duration).scaled(self.bit_rate_headroom)

    def _keep_losslessly(self, timeline: VideoTimeline, master: Path) -> None:
        """
        Pipe every frame into a lossless file.

        :param timeline: What to keep.
        :param master: Where the lossless file goes.
        :raises EncodingFailed: If the encoder program exits with an error.
        """
        first = next(iter(timeline.frames()))
        resolution = first.resolution
        command = self.ffmpeg.command(
            [
                "-f",
                "rawvideo",
                "-pix_fmt",
                PixelFormat.RGB24,
                "-s",
                f"{resolution.width}x{resolution.height}",
                "-r",
                str(timeline.style.frames_per_second),
                "-i",
                "-",
                "-c:v",
                "libx264",
                "-qp",
                "0",
                "-preset",
                EncoderPreset.ULTRAFAST,
                str(master),
            ]
        )
        process = subprocess.Popen(command, stdin=subprocess.PIPE)
        for frame in timeline.frames():
            process.stdin.write(np.ascontiguousarray(frame.pixels).tobytes())
        process.stdin.close()
        if process.wait() != 0:
            raise EncodingFailed(path=master)

    def _two_passes(
        self, master: Path, path: Path, bit_rate: BitRate, log_prefix: Path
    ) -> None:
        """
        Encode the master twice at a bit rate, the second pass reading the first's log.

        :param master: The lossless file.
        :param path: Where the mp4 goes.
        :param bit_rate: The average bit rate the file is encoded at.
        :param log_prefix: Where the first pass keeps what the second reads.
        """
        shared = self.ffmpeg.command(
            [
                "-i",
                str(master),
                "-c:v",
                "libx264",
                "-preset",
                self.preset,
                "-profile:v",
                "high",
                "-pix_fmt",
                self.pixel_format,
                "-b:v",
                str(bit_rate.per_second.magnitude),
                "-maxrate",
                str(bit_rate.scaled(self.peak_bit_rate_factor).per_second.magnitude),
                "-bufsize",
                str(bit_rate.over(self.rate_control_buffer).magnitude),
                "-passlogfile",
                str(log_prefix),
                "-movflags",
                "+faststart",
            ]
        )
        subprocess.run(
            shared + ["-pass", "1", "-an", "-f", "mp4", "/dev/null"], check=True
        )
        subprocess.run(shared + ["-pass", "2", "-an", str(path)], check=True)
