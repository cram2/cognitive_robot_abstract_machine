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
from dataclasses import dataclass, field
from enum import StrEnum

from typing_extensions import Dict, List, Optional

from cramera.live.bridge import WorldStateSnapshot


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
    :mod:`cramera.live.recording_segments`); an offline onboarded scene reads the same
    labels off the recorded action list instead.
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

    def append(self, snapshot: WorldStateSnapshot, step: Optional[str] = None) -> None:
        """
        Buffer one tick, or do nothing while not actively recording.

        :param snapshot: The tick's world state, as the bridge just published it.
        :param step: Label of the action being performed on this tick, if any.
        """
        with self._lock:
            if self.state is not RecordingState.RECORDING:
                return
            self._frames.append(
                RecordedFrame(
                    frames=dict(snapshot.frames),
                    base=snapshot.base,
                    objects={
                        key: list(value) for key, value in snapshot.objects.items()
                    },
                    step=step,
                )
            )
            self._tick_times.append(time.time())

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

    def frame_count(self) -> int:
        """
        How many ticks have been captured so far.
        """
        with self._lock:
            return len(self._frames)

    def frames_per_second(self, fallback: float = 30.0) -> float:
        """
        The recording's frame rate, estimated from wall-clock time between ticks.

        :param fallback: Rate to report with fewer than two ticks to measure a span
            from.
        """
        with self._lock:
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
                "sceneName": self.scene_name,
            }
