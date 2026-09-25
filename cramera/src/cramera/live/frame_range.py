"""
Selecting a contiguous stretch of a recorded run.

Kept apart from :mod:`cramera.live.recording` so that the pure-filesystem side of
recordings (:mod:`cramera.live.recording_storage`, which the always-on server uses
without any demo process) can express a trim without pulling in the bridge and the world
model behind it.
"""

from __future__ import annotations

from dataclasses import dataclass


class InvalidFrameRange(Exception):
    """
    Raised when a :class:`FrameRange` is malformed or reaches past the frames it is
    meant to select from.
    """


@dataclass(frozen=True)
class FrameRange:
    """
    A contiguous stretch of a recording, as the viewer's trim selects it.

    Both bounds are inclusive, so the range the viewer draws over a timeline is the
    range that gets kept.
    """

    first: int
    """
    Index of the first frame kept.
    """

    last: int
    """
    Index of the last frame kept.
    """

    def __post_init__(self) -> None:
        """
        Reject negative or reversed frame ranges.
        """
        if self.first < 0 or self.last < self.first:
            raise InvalidFrameRange(
                "a frame range must run forwards from zero, got %d..%d"
                % (self.first, self.last)
            )

    @classmethod
    def whole(cls, frame_count: int) -> FrameRange:
        """
        The range covering an entire recording.

        :param frame_count: How many frames the recording holds.
        :raises InvalidFrameRange: If the recording holds no frames.
        """
        if frame_count < 1:
            raise InvalidFrameRange("an empty recording has no frames to select")
        return cls(first=0, last=frame_count - 1)

    def length(self) -> int:
        """
        How many frames the range keeps.
        """
        return self.last - self.first + 1
