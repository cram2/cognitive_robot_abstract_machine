"""
When to replay a recorded demo around one answered moment.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime

from typing_extensions import Any, ClassVar, Dict


@dataclass(frozen=True)
class ReplayWindow:
    """
    The span of a recorded demo worth replaying around one moment.

    A moment by itself is too short to watch, so the window leads and trails it by fixed
    shifts, just far enough to show it happening.
    """

    LEAD_SECONDS: ClassVar[float] = 1.0
    """
    How long before its moment a replay begins.
    """

    TAIL_SECONDS: ClassVar[float] = 1.0
    """
    How long after its moment a replay ends.
    """

    start: float
    """
    When the replay begins, in seconds since the epoch.
    """

    end: float
    """
    When the replay ends, in seconds since the epoch.
    """

    @classmethod
    def around(cls, moment: datetime) -> ReplayWindow:
        """
        The window worth replaying around one moment.

        :param moment: When the thing worth watching happened.
        """
        at = moment.timestamp()
        return cls(start=at - cls.LEAD_SECONDS, end=at + cls.TAIL_SECONDS)

    def to_payload(self) -> Dict[str, Any]:
        """
        The JSON shape the viewer opens a replay from.
        """
        return {"start": self.start, "end": self.end}
