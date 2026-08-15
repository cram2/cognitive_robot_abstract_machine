"""
The plan steps and manipulations a live recording went through.

A bundle's ``segments`` are what the replay timeline is divided into and what it marks
its key moments from (see ``core/timeline-events.js``): where each step began, and the
frames an object was picked up and let go. An offline onboarded scene derives them from
the recorded action list (:meth:`cramera.onboard.demo.SceneBuilder.derive_segments`); a
live recording has no such list, so they are derived here from what each tick captured —
the action the plan reported performing, and where every loose object was.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from typing_extensions import (
    Any,
    ClassVar,
    Dict,
    List,
    Optional,
    Sequence,
    TYPE_CHECKING,
)

if TYPE_CHECKING:
    from cramera.live.recording import RecordedFrame


TRANSPORT_TOLERANCE = 0.02
"""
How far an object must end up from where it started, in metres, to count as
transported rather than nudged.
"""

MOTION_TOLERANCE = 0.005
"""
How far an object must move from a pose, in metres, to count as having left it.
"""


@dataclass(frozen=True)
class ObjectWindow:
    """
    The stretch of a recording over which one object was being carried.
    """

    object_key: str
    """
    The object's published mesh key.
    """

    attach: int
    """
    Index of the first frame the object had left where it started.
    """

    detach: int
    """
    Index of the first frame the object was resting where it ended up.
    """


@dataclass(frozen=True)
class RecordedSegment:
    """
    One stretch of a recording, as the replay timeline divides it.
    """

    UNLABELLED_STEP: ClassVar[str] = "run"
    """
    What a stretch is called when the plan reported no action for it.
    """

    step: str
    """
    What was happening: the action the plan reported, else a name for the manipulation.
    """

    start: int
    """
    Index of the frame the stretch begins at.
    """

    end: int
    """
    Index of the frame the stretch ends at.
    """

    window: Optional[ObjectWindow] = None
    """
    The object carried during this stretch, if one was.
    """

    def to_payload(self) -> Dict[str, Any]:
        """
        The segment in the ``scene.json`` shape a bundle carries and the viewer reads.
        """
        payload: Dict[str, Any] = {
            "step": self.step,
            "action": None,
            "arm": None,
            "start": self.start,
            "end": self.end,
        }
        if self.window is None:
            return payload
        payload.update(
            {
                "picks": self.window.object_key,
                "attach": self.window.attach,
                "detach": self.window.detach,
            }
        )
        return payload


def has_moved(one: Sequence[float], other: Sequence[float], tolerance: float) -> bool:
    """
    Whether two recorded poses are more than ``tolerance`` apart in position.

    :param one: The first pose, as ``[x, y, z, qx, qy, qz, qw]``.
    :param other: The pose to compare it against.
    :param tolerance: The distance, in metres, below which the two count as the same
        place.
    """
    return (
        math.dist(list(one[:3]), list(other[:3])) > tolerance
        if one and other
        else False
    )


def object_windows(frames: List[RecordedFrame]) -> List[ObjectWindow]:
    """
    The carry window of every object that ended the recording somewhere else.

    :param frames: The recording's ticks, in order.
    """
    if not frames:
        return []
    first, last = frames[0].objects, frames[-1].objects
    windows = []
    for key, spawn in first.items():
        final = last.get(key)
        if final is None or not has_moved(spawn, final, TRANSPORT_TOLERANCE):
            continue
        attach = _first_frame_away_from(frames, key, spawn)
        detach = _last_frame_away_from(frames, key, final) + 1
        # not ``<``: an object set down on the very frame it arrives is carried for a
        # single tick, and dropping that window would leave the move unmarked
        if attach <= detach:
            windows.append(ObjectWindow(object_key=key, attach=attach, detach=detach))
    windows.sort(key=lambda window: window.attach)
    return windows


def derive_segments(frames: List[RecordedFrame]) -> List[RecordedSegment]:
    """
    Divide a recording into the stretches its timeline is marked with.

    Each carried object gets a stretch of its own, named after whichever action the
    plan reported when it was picked up; whatever precedes the first pick is one
    further stretch. A recording in which nothing was carried is divided by the
    reported actions alone, so a run that only drove or looked around is still marked.

    :param frames: The recording's ticks, in order.
    """
    if not frames:
        return []
    windows = object_windows(frames)
    if not windows:
        return _step_segments(frames)
    last_frame = len(frames) - 1
    segments = []
    previous_end = 0
    for index, window in enumerate(windows):
        following = windows[index + 1].attach if index + 1 < len(windows) else None
        if index == 0 and window.attach > 0:
            segments.append(
                RecordedSegment(
                    step=_step_at(frames, 0) or RecordedSegment.UNLABELLED_STEP,
                    start=0,
                    end=window.attach,
                )
            )
            previous_end = window.attach
        segments.append(
            RecordedSegment(
                step=_step_at(frames, window.attach) or _carry_step(window),
                start=previous_end,
                end=following if following is not None else last_frame,
                window=window,
            )
        )
        previous_end = segments[-1].end
    if previous_end < last_frame:
        segments.append(
            RecordedSegment(
                step=_step_at(frames, previous_end) or RecordedSegment.UNLABELLED_STEP,
                start=previous_end,
                end=last_frame,
            )
        )
    return segments


def _carry_step(window: ObjectWindow) -> str:
    """
    What to call a manipulation the plan reported no action for.

    :param window: The carry window being named.
    """
    return "move_" + window.object_key.split(".")[0]


def _step_at(frames: List[RecordedFrame], index: int) -> Optional[str]:
    """
    The action reported at one frame, or the last one reported before it.

    A pick lands between two reported actions often enough — the plan is between motion
    nodes on exactly that tick — that reading the frame alone would leave the stretch
    unnamed.

    :param frames: The recording's ticks, in order.
    :param index: Index of the frame to read.
    """
    for frame in reversed(frames[: index + 1]):
        if frame.step:
            return frame.step
    return None


def _step_segments(frames: List[RecordedFrame]) -> List[RecordedSegment]:
    """
    One stretch per run of consecutive frames reporting the same action.

    :param frames: The recording's ticks, in order.
    """
    segments: List[RecordedSegment] = []
    start = 0
    for index in range(1, len(frames) + 1):
        ended = index == len(frames)
        if not ended and frames[index].step == frames[start].step:
            continue
        step = frames[start].step
        if step:
            segments.append(
                RecordedSegment(step=step, start=start, end=min(index, len(frames) - 1))
            )
        start = index
    if segments:
        return segments
    return [
        RecordedSegment(
            step=RecordedSegment.UNLABELLED_STEP, start=0, end=len(frames) - 1
        )
    ]


def _first_frame_away_from(
    frames: List[RecordedFrame], key: str, pose: Sequence[float]
) -> int:
    """
    Index of the first frame at which an object had left ``pose``.

    :param frames: The recording's ticks, in order.
    :param key: The object's published mesh key.
    :param pose: The pose the object is leaving.
    """
    for index, frame in enumerate(frames):
        current = frame.objects.get(key)
        if current is not None and has_moved(current, pose, MOTION_TOLERANCE):
            return index
    return len(frames) - 1


def _last_frame_away_from(
    frames: List[RecordedFrame], key: str, pose: Sequence[float]
) -> int:
    """
    Index of the last frame at which an object was not yet resting at ``pose``.

    :param frames: The recording's ticks, in order.
    :param key: The object's published mesh key.
    :param pose: The pose the object comes to rest at.
    """
    for index in range(len(frames) - 1, -1, -1):
        current = frames[index].objects.get(key)
        if current is not None and has_moved(current, pose, MOTION_TOLERANCE):
            return index
    return 0
