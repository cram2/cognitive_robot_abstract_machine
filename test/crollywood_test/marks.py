"""
An overlay for tests: it stamps how far the video has played into one pixel, so a test
reads back which moment an overlay was drawn for.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import timedelta

from crollywood.frame import Frame
from crollywood.timeline import Overlay


@dataclass
class ProgressStamp(Overlay):
    """
    Writes the share of the video played, scaled to a byte, into the top left pixel.
    """

    def drawn_over(self, frame: Frame, moment: timedelta, runs_for: timedelta) -> Frame:
        stamped = frame.pixels.copy()
        stamped[0, 0] = stamped_value(moment, runs_for)
        return Frame(stamped)


def stamped_value(moment: timedelta, runs_for: timedelta) -> int:
    """
    :param moment: A moment of the video.
    :param runs_for: How long the video lasts.
    :return: The byte :class:`ProgressStamp` writes for that moment.
    """
    return round(255 * moment / runs_for)
