from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import imageio.v2 as imageio
import mujoco
import numpy as np
from scipy.spatial.transform import Rotation
from typing_extensions import List, Optional, Tuple

from semantic_digital_twin.adapters.multi_sim import MujocoCamera, MujocoSim
from semantic_digital_twin.callbacks.callback import StateChangeCallback
from semantic_digital_twin.exceptions import (
    EmptyVideoRecordingError,
    EmptyWorldVideoRecordingError,
    InvalidVideoRecordingRateError,
    VideoRecordingAlreadyStartedError,
    VideoRecordingNotStartedError,
)
from semantic_digital_twin.spatial_computations.raytracer import RayTracer
from semantic_digital_twin.world import World

_OVERVIEW_CAMERA_NAME = "cram_video_overview_camera"
"""Name given to the automatically attached default camera."""

_MINIMUM_OVERVIEW_DISTANCE = 1.0
"""
Floor (in meters) for the distance of the default overview camera to the scene center,
so a world whose geometry collapses to a point still gets a sensibly framed camera.
"""

_OVERVIEW_DISTANCE_FACTOR = 1.5
"""Multiplier applied to the scene's bounding diagonal to place the default overview camera."""


@dataclass
class VideoResolution:
    """
    Pixel dimensions of a rendered video frame.
    """

    width: int
    """The frame width in pixels."""

    height: int
    """The frame height in pixels."""


@dataclass
class RecordedVideo:
    """
    The raw result of a :class:`MujocoVideoRecorder` session, before encoding.
    """

    frames: List[np.ndarray]
    """The captured RGB frames, in playback order."""

    frame_timestamps: List[float]
    """
    Each frame's position, in seconds, on the *encoded video's own timeline*
    (``index / frames_per_second``) - not a real-world or simulated-world clock. Whatever
    drove :attr:`~MujocoVideoRecorder.world` (a plan, manual stepping, ...) is not
    necessarily paced to real time, so these are the only timestamps that stay meaningful
    once the frames are encoded into a video.
    """

    frames_per_second: int
    """The rate the frames are encoded at."""


def _look_at_quaternion(
    camera_position: np.ndarray, target_position: np.ndarray
) -> np.ndarray:
    """
    Computes the world-frame orientation of a camera at ``camera_position`` that looks at
    ``target_position``, following MuJoCo's camera convention (camera looks down its local
    -Z axis, +Y is up).

    :param camera_position: The world-frame position of the camera.
    :param target_position: The world-frame position the camera should look at.
    :return: A ``[w, x, y, z]`` quaternion.
    """
    forward = target_position - camera_position
    forward = forward / np.linalg.norm(forward)
    up_hint = np.array([0.0, 0.0, 1.0])
    if np.abs(np.dot(forward, up_hint)) > 0.99:
        up_hint = np.array([0.0, 1.0, 0.0])

    z_axis = -forward
    x_axis = np.cross(up_hint, z_axis)
    x_axis = x_axis / np.linalg.norm(x_axis)
    y_axis = np.cross(z_axis, x_axis)

    rotation_matrix = np.column_stack([x_axis, y_axis, z_axis])
    return Rotation.from_matrix(rotation_matrix).as_quat(scalar_first=True)


def overview_camera_pose(bounds: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Computes a fixed diagonal viewpoint that frames an axis-aligned bounding box.

    :param bounds: A ``(2, 3)`` array of the scene's ``[minimum, maximum]`` corners.
    :return: A ``(position, quaternion)`` tuple in world frame; the quaternion is ``[w, x, y, z]``.
    """
    minimum, maximum = bounds
    center = (minimum + maximum) / 2.0
    diagonal = float(np.linalg.norm(maximum - minimum))
    distance = max(diagonal, _MINIMUM_OVERVIEW_DISTANCE) * _OVERVIEW_DISTANCE_FACTOR
    direction = np.array([1.0, -1.0, 1.0])
    direction = direction / np.linalg.norm(direction)
    position = center + direction * distance
    quaternion = _look_at_quaternion(position, center)
    return position, quaternion


@dataclass(eq=False)
class _FrameCaptureCallback(StateChangeCallback):
    """
    Sibling callback owned by a :class:`MujocoVideoRecorder`. Notifies it every time
    :attr:`recorder`'s world changes, at whatever pace that happens.
    """

    recorder: MujocoVideoRecorder = field(kw_only=True)
    """The recorder to notify on every state change."""

    def on_state_change(self, **kwargs):
        self.recorder._on_world_state_change()


@dataclass
class MujocoVideoRecorder:
    """
    Records a video of a :class:`~semantic_digital_twin.world.World` while it is being
    mutated live (e.g. by a running plan), by mirroring it into a headless MuJoCo scene via
    :class:`~semantic_digital_twin.adapters.multi_sim.MujocoSim`.

    Frames are captured on world state changes rather than on a wall-clock timer: whatever
    drives the world (a Giskard-ticked plan, or manual stepping via
    :meth:`advance_simulation`) is not paced to real time, so a wall-clock timer would mostly
    just sample whatever state the world had already settled into by the time it fired.
    Since the changes are not evenly spaced in real (or simulated) time either, the resulting
    :class:`RecordedVideo` timestamps each kept frame by its position in the *encoded video*
    instead (see :attr:`RecordedVideo.frame_timestamps`).
    """

    world: World
    """The world to record. Must already contain at least one body with geometry."""

    frames_per_second: int = 30
    """The rate the encoded video plays back at."""

    capture_every_n_state_changes: int = 1
    """
    Only every Nth world state change is captured as a frame. Raise this to shorten the
    video for a world that changes state very often per unit of meaningful motion.
    """

    resolution: VideoResolution = field(
        default_factory=lambda: VideoResolution(width=640, height=480)
    )
    """The pixel resolution of the captured frames."""

    camera_name: Optional[str] = None
    """
    The name of an existing MuJoCo camera to record from. If ``None``, a fixed overview
    camera framing the world's bounding box is attached automatically when recording starts.
    """

    _multi_sim: Optional[MujocoSim] = field(init=False, default=None, repr=False)
    """The live MuJoCo mirror of :attr:`world`, present only while recording."""

    _frame_capture_callback: Optional[_FrameCaptureCallback] = field(
        init=False, default=None, repr=False
    )
    """Notifies on every world state change, present only while recording."""

    _state_change_count: int = field(init=False, default=0, repr=False)
    """Number of world state changes observed since :meth:`start`."""

    _frames: List[np.ndarray] = field(init=False, default_factory=list, repr=False)
    """Frames captured since the last :meth:`start`."""

    _auto_attached_camera: Optional[MujocoCamera] = field(
        init=False, default=None, repr=False
    )
    """
    The overview camera :meth:`start` attached to :attr:`world`, if :attr:`camera_name` was
    ``None``; removed again by :meth:`stop` so repeated recordings of the same world don't
    accumulate same-named cameras.
    """

    def __post_init__(self):
        if self.frames_per_second <= 0:
            raise InvalidVideoRecordingRateError(
                field_name="frames_per_second", value=self.frames_per_second
            )
        if self.capture_every_n_state_changes <= 0:
            raise InvalidVideoRecordingRateError(
                field_name="capture_every_n_state_changes",
                value=self.capture_every_n_state_changes,
            )

    def start(self) -> None:
        """
        Attaches a default camera if none was given, builds a headless MuJoCo mirror of
        :attr:`world`, and starts capturing a frame on every subsequent state change.

        The mirror's own physics stepping is not run in a background thread: for a world
        driven by a plan, MuJoCo only needs to mirror the poses Giskard already computed
        (see :meth:`_on_world_state_change`, which refreshes the derived kinematics itself);
        for a world with nothing else driving it, advance it explicitly with
        :meth:`advance_simulation`.
        """
        if self._multi_sim is not None:
            raise VideoRecordingAlreadyStartedError(world=self.world)

        if self.camera_name is None:
            self._auto_attached_camera = self._attach_overview_camera()
            self.camera_name = self._auto_attached_camera.name

        self._multi_sim = MujocoSim(world=self.world, headless=True)
        # The synchronizer throttles its own sim -> world sync (and thus notify_state_change)
        # to a wall-clock rate; advance_simulation() steps in a tight loop with no wall-clock
        # pacing of its own, so that throttle must be disabled or most steps would go unseen.
        self._multi_sim.synchronizer.sync_rate_hz = float("inf")
        self._multi_sim.simulator.start(simulate_in_thread=False, render_in_thread=False)

        self._state_change_count = 0
        self._frame_capture_callback = _FrameCaptureCallback(_world=self.world, recorder=self)
        self._on_world_state_change()

    def stop(self) -> RecordedVideo:
        """
        Stops capturing frames and tears down the MuJoCo mirror.

        :return: The frames captured since :meth:`start`.
        """
        if self._multi_sim is None:
            raise VideoRecordingNotStartedError(world=self.world)

        self._frame_capture_callback.stop()
        self._frame_capture_callback = None
        self._multi_sim.simulator.stop()
        self._multi_sim = None

        if self._auto_attached_camera is not None:
            # Building the MuJoCo mirror can reparent the world under a new synthetic root
            # (see MultiSimBuilder.build_world), so world.root may no longer be the body the
            # camera was attached to; remove it from the camera's own recorded body instead.
            self._auto_attached_camera.body.simulator_additional_properties.remove(
                self._auto_attached_camera
            )
            self._auto_attached_camera = None
            self.camera_name = None

        frame_timestamps = [i / self.frames_per_second for i in range(len(self._frames))]
        recorded_video = RecordedVideo(
            frames=self._frames,
            frame_timestamps=frame_timestamps,
            frames_per_second=self.frames_per_second,
        )
        self._frames = []
        return recorded_video

    def advance_simulation(self, duration: float) -> None:
        """
        Steps the MuJoCo mirror's own physics (gravity, contacts, ...) forward by
        ``duration`` simulated seconds, capturing frames along the way.

        Use this when nothing else (e.g. a coraplex plan) is already driving :attr:`world`;
        it single-steps the mirror synchronously so capturing stays deterministic regardless
        of how fast the host machine can step MuJoCo. A physics step is far finer-grained
        than a usable video frame (often 1 ms), so :attr:`capture_every_n_state_changes` is
        temporarily raised to match :attr:`frames_per_second` for the duration of this call -
        otherwise every single step would trigger an expensive render.

        :param duration: How many simulated seconds to advance.
        """
        if self._multi_sim is None:
            raise VideoRecordingNotStartedError(world=self.world)

        step_size = self._multi_sim.simulator.step_size
        steps_per_frame = max(1, round((1.0 / self.frames_per_second) / step_size))
        previous_decimation = self.capture_every_n_state_changes
        self.capture_every_n_state_changes = steps_per_frame
        # Restart the decimation period cleanly so the first frame of this call lands
        # exactly steps_per_frame steps in, rather than wherever the previous decimation
        # period's phase happened to leave off.
        self._state_change_count = 0
        try:
            for _ in range(max(1, round(duration / step_size))):
                self._multi_sim.simulator.step()
        finally:
            self.capture_every_n_state_changes = previous_decimation

    @property
    def captured_frame_count(self) -> int:
        """
        The number of frames captured (and kept) since :meth:`start`.
        """
        return len(self._frames)

    def _on_world_state_change(self) -> None:
        """
        Captures a frame if this is the Nth world state change since the last capture (see
        :attr:`capture_every_n_state_changes`).
        """
        keep_frame = self._state_change_count % self.capture_every_n_state_changes == 0
        self._state_change_count += 1
        if not keep_frame:
            return

        # Nothing but MuJoCo's own stepping normally recomputes body poses (xpos/xmat) from
        # qpos; since the mirror's physics is not stepped in the background (see start()),
        # a plan/Giskard-driven qpos write needs an explicit forward pass before rendering,
        # or the frame would show the previous pose. Held across both calls (the lock is
        # reentrant) so nothing can step the model in between the forward pass and the render
        # that depends on it.
        simulator = self._multi_sim.simulator
        with simulator._model_lock:
            mujoco.mj_forward(simulator._mj_model, simulator._mj_data)
            capture_result = simulator.capture_rgb(
                camera_name=self.camera_name,
                height=self.resolution.height,
                width=self.resolution.width,
            )
        self._frames.append(capture_result.result)

    def _attach_overview_camera(self) -> MujocoCamera:
        """
        Computes and attaches a fixed camera framing :attr:`world`'s bounding box to its
        root body.

        :return: The newly attached camera.
        """
        bounds = RayTracer(self.world).scene.bounds
        if bounds is None:
            raise EmptyWorldVideoRecordingError(world=self.world)

        position, quaternion = overview_camera_pose(np.asarray(bounds))
        camera = MujocoCamera(
            name=_OVERVIEW_CAMERA_NAME,
            body=self.world.root,
            position=position.tolist(),
            quaternion=quaternion.tolist(),
            resolution=[float(self.resolution.width), float(self.resolution.height)],
        )
        self.world.root.simulator_additional_properties.append(camera)
        return camera


def write_video(recorded_video: RecordedVideo, output_path: Path) -> Path:
    """
    Encodes a :class:`RecordedVideo`'s frames into a video file.

    :param recorded_video: The frames (and frame rate) to encode.
    :param output_path: The file path the video is written to.
    :return: ``output_path``.
    """
    if len(recorded_video.frames) == 0:
        raise EmptyVideoRecordingError(output_path=output_path)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with imageio.get_writer(
        str(output_path), fps=recorded_video.frames_per_second
    ) as writer:
        for frame in recorded_video.frames:
            writer.append_data(frame)
    return output_path
