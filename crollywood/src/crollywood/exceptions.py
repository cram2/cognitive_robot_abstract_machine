"""
What goes wrong when a video is made, each with a type of its own.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import timedelta
from pathlib import Path


from krrood.exceptions import DataclassException
from typing_extensions import TYPE_CHECKING

if TYPE_CHECKING:
    from crollywood.frame import Resolution
    from crollywood.narration import NarratedLine


@dataclass
class NarrationOverrun(DataclassException):
    """
    Raised when a line is still being said when the next starts, or when the video ends.
    """

    line: NarratedLine
    """
    The line cut off.
    """

    by: timedelta
    """
    By how long.
    """

    def error_message(self) -> str:
        """
        :return: What went wrong.
        """
        return (
            f'"{self.line.written[:40]}…" runs {self.by.total_seconds():.1f} s past '
            "what follows it"
        )

    def suggest_correction(self) -> str:
        """
        :return: How to put it right.
        """
        return "Shorten the line, or give the scenes it plays over more time."


@dataclass
class MomentOutsideScene(DataclassException):
    """
    Raised when a scene is asked for a moment before it starts or after it ends.
    """

    moment: timedelta
    """
    The moment asked for, from the scene's start.
    """

    duration: timedelta
    """
    How long the scene lasts.
    """

    def error_message(self) -> str:
        """
        :return: What went wrong.
        """
        return (
            f"{self.moment.total_seconds():.3f} s is outside a scene lasting "
            f"{self.duration.total_seconds():.3f} s"
        )

    def suggest_correction(self) -> str:
        """
        :return: How to put it right.
        """
        return "Ask for a moment from zero up to, but not including, the duration."


@dataclass
class FramesOfDifferentSizes(DataclassException):
    """
    Raised when two frames that have to share one video differ in size.
    """

    first: Resolution
    """
    The size of one frame.
    """

    second: Resolution
    """
    The size of the other.
    """

    def error_message(self) -> str:
        """
        :return: What went wrong.
        """
        return (
            f"frames of {self.first.width}x{self.first.height} and "
            f"{self.second.width}x{self.second.height} cannot share one video"
        )

    def suggest_correction(self) -> str:
        """
        :return: How to put it right.
        """
        return "Draw every scene at the resolution the video is made at."


@dataclass
class NothingToRead(DataclassException):
    """
    Raised when a video file holds no frame to read.
    """

    path: Path
    """
    The video file.
    """

    moment: timedelta
    """
    The moment asked for, from the file's start.
    """

    def error_message(self) -> str:
        """
        :return: What went wrong.
        """
        return f"nothing to read at {self.moment.total_seconds():.3f} s of {self.path}"

    def suggest_correction(self) -> str:
        """
        :return: How to put it right.
        """
        return "Check that the file exists and holds a video."


@dataclass
class EncodingFailed(DataclassException):
    """
    Raised when the encoder program exits with an error.
    """

    path: Path
    """
    The file it was writing.
    """

    def error_message(self) -> str:
        """
        :return: What went wrong.
        """
        return f"ffmpeg failed to write {self.path}"

    def suggest_correction(self) -> str:
        """
        :return: How to put it right.
        """
        return "Run it without the error level silenced to read what it reports."


@dataclass
class DarknessOutOfRange(DataclassException):
    """
    Raised when a shade is asked to be darker than black or lighter than no shade.
    """

    darkness: float
    """
    The darkness asked for.
    """

    def error_message(self) -> str:
        """
        :return: What went wrong.
        """
        return f"a shade of darkness {self.darkness} lies outside zero to one"

    def suggest_correction(self) -> str:
        """
        :return: How to put it right.
        """
        return "Give a darkness from zero, which leaves the picture, to one, black."
