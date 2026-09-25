"""
Buffering one live run into a replayable trajectory.

:class:`Recording` only accumulates ticks and tracks its own lifecycle; it knows nothing
about the :class:`~cramera.live.bridge.Bridge` it is fed from or how a finalized
recording becomes a scene bundle on disk (see :mod:`cramera.live.recording_bundle`) —
the tick hook that appends to it lives in :mod:`cramera.live.visualization`, right
beside the hook that already publishes each tick to the bridge.
"""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass, field, replace
from enum import StrEnum

from typing_extensions import Dict, List, Optional

from cramera.live.bridge import ChartSnapshot, WorldStateSnapshot
from cramera.live.frame_range import FrameRange, InvalidFrameRange


class RecordingState(StrEnum):
    """
    Where one :class:`Recording` is in its lifecycle.
    """

    IDLE = "idle"
    """
    No run is being captured, and nothing is available to replay.
    """

    RECORDING = "recording"
    """
    A live run is being captured tick by tick.
    """

    FINALIZED = "finalized"
    """
    Capture has stopped; the buffered frames are ready to bundle or discard.
    """


FALLBACK_FRAME_RATE = 30.0
"""
Frame rate reported for a recording with too few ticks to measure a span from.
"""


class NoActiveRecording(Exception):
    """
    Raised by :meth:`Recording.stop` when no run is being captured.
    """


@dataclass(frozen=True)
class RecordedFrame:
    """
    One recorded tick, in the same shape
    :class:`~cramera.live.bridge.WorldStateSnapshot` reports it.
    """

    frames: Dict[str, float]
    """
    Movable connection position by prefixed name.
    """

    base: Optional[List[float]]
    """
    Robot base pose as ``[x, y, z, qx, qy, qz, qw]``, or None without a robot.
    """

    objects: Dict[str, List[float]]
    """
    Loose-object pose by mesh key, in the same 7-element form as :attr:`base`.
    """

    step: Optional[str] = None
    """
    Label of the action the plan was performing on this tick, or None between actions.

    What a replay's timeline names this stretch of the run (see
    :mod:`cramera.live.recording_segments`).
    """

    statechart: Optional[ChartSnapshot] = None
    """
    The motion statechart executing on this tick, or None while no motion runs.

    Held as the bridge published it, so a stretch of ticks that found the chart
    unchanged shares one snapshot (see :meth:`Recording.append`).
    """


@dataclass
class Recording:
    """
    One live run's buffered trajectory.

    The simulation thread appends ticks through :meth:`append`; HTTP threads read
    :meth:`status_payload` and call :meth:`stop`/:meth:`discard` — every mutable field
    is guarded by :attr:`_lock`, matching :class:`~cramera.live.bridge.Bridge`'s own
    sim-thread-writes/HTTP-thread-reads contract.
    """

    state: RecordingState = RecordingState.IDLE
    """
    The recording's current lifecycle state.
    """

    scene_name: Optional[str] = None
    """
    Name of the scene bundle the finalized recording was written to, filled in by
    whoever bundles it (see :mod:`cramera.live.recording_bundle`); None until then.
    """

    _frames: List[RecordedFrame] = field(default_factory=list)
    """
    Every tick captured so far, in recording order.
    """

    _tick_times: List[float] = field(default_factory=list)
    """
    Wall-clock time of each captured tick, used to estimate the frame rate.
    """

    _lock: threading.Lock = field(default_factory=threading.Lock)
    """
    Guards every field above.
    """

    def start(self) -> None:
        """
        Begin capturing, discarding whatever an earlier, unfinalized recording held.
        """
        with self._lock:
            self.state = RecordingState.RECORDING
            self.scene_name = None
            self._frames = []
            self._tick_times = []

    def append(
        self,
        snapshot: WorldStateSnapshot,
        step: Optional[str] = None,
        statechart: Optional[ChartSnapshot] = None,
    ) -> None:
        """
        Buffer one tick, or do nothing while not actively recording.

        :param snapshot: The tick's world state, as the bridge just published it.
        :param step: Label of the action being performed on this tick, if any.
        :param statechart: The motion statechart executing on this tick, if any.
        """
        with self._lock:
            if self.state is not RecordingState.RECORDING:
                return
            self._frames.append(
                RecordedFrame(
                    frames=dict(snapshot.frames),
                    base=list(snapshot.base) if snapshot.base is not None else None,
                    objects={
                        key: list(value) for key, value in snapshot.objects.items()
                    },
                    step=step,
                    statechart=self._held_statechart(statechart),
                )
            )
            self._tick_times.append(time.time())

    def _held_statechart(
        self, statechart: Optional[ChartSnapshot]
    ) -> Optional[ChartSnapshot]:
        """
        The snapshot to store for a tick: the one the previous tick already holds when
        the statechart has not changed since, so an unchanged stretch costs one object.

        :param statechart: The statechart the bridge published for this tick, if any.
        """
        if not self._frames:
            return statechart
        held = self._frames[-1].statechart
        return held if held == statechart else statechart

    def update_statechart(self, statechart: Optional[ChartSnapshot]) -> None:
        """Complete the latest world frame with a plan's final chart observation.

        A plan boundary can follow chart changes without another world update.
        Preserve the final observation on the latest pose and timestamp.

        :param statechart: The chart observed when the plan node completed.
        """
        with self._lock:
            if self.state is not RecordingState.RECORDING or not self._frames:
                return
            self._frames[-1] = replace(
                self._frames[-1], statechart=self._held_statechart(statechart)
            )

    def stop(self) -> List[RecordedFrame]:
        """
        Finalize the recording and return its buffered frames.

        Idempotent: calling this again while already :attr:`~RecordingState.FINALIZED`
        just returns the same frames, so a browser tab that already finalized a
        recording can ask again without side effects.

        :raises NoActiveRecording: If capture never started.
        """
        with self._lock:
            if self.state is RecordingState.IDLE:
                raise NoActiveRecording("no recording is in progress")
            self.state = RecordingState.FINALIZED
            return list(self._frames)

    def discard(self) -> None:
        """
        Drop the buffered frames and return to idle.
        """
        with self._lock:
            self.state = RecordingState.IDLE
            self.scene_name = None
            self._frames = []
            self._tick_times = []

    def frames_in(self, frame_range: FrameRange) -> List[RecordedFrame]:
        """
        The buffered frames a range selects, in recording order.

        :param frame_range: The stretch of the recording to keep.
        :raises InvalidFrameRange: If the range reaches past the buffered frames.
        """
        with self._lock:
            if frame_range.last >= len(self._frames):
                raise InvalidFrameRange(
                    "frame %d is past the recording's %d frames"
                    % (frame_range.last, len(self._frames))
                )
            return self._frames[frame_range.first : frame_range.last + 1]

    def frame_count(self) -> int:
        """
        How many ticks have been captured so far.
        """
        with self._lock:
            return len(self._frames)

    def frames_per_second(self, fallback: float = FALLBACK_FRAME_RATE) -> float:
        """
        The recording's frame rate, estimated from wall-clock time between ticks.

        :param fallback: Rate to report with fewer than two ticks to measure a span
            from.
        """
        with self._lock:
            return self._measured_frame_rate(fallback)

    def _measured_frame_rate(self, fallback: float) -> float:
        """
        The frame rate, with :attr:`_lock` already held by the caller.

        :param fallback: Rate to report with fewer than two ticks to measure a span
            from.
        """
        if len(self._tick_times) < 2:
            return fallback
        duration = self._tick_times[-1] - self._tick_times[0]
        if duration <= 0:
            return fallback
        return max(1.0, round(len(self._tick_times) / duration, 2))

    def status_payload(self) -> Dict[str, object]:
        """
        What the viewer polls to show recording/playback controls.
        """
        with self._lock:
            duration = (
                round(self._tick_times[-1] - self._tick_times[0], 2)
                if len(self._tick_times) >= 2
                else 0.0
            )
            return {
                "state": self.state.value,
                "frameCount": len(self._frames),
                "durationSeconds": duration,
                "framesPerSecond": self._measured_frame_rate(FALLBACK_FRAME_RATE),
                "sceneName": self.scene_name,
            }
