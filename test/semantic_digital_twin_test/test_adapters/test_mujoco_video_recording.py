"""
Tests for :mod:`semantic_digital_twin.adapters.mujoco_video_recording`.

The pure camera-pose math and the video encoding are exercised as fast, always-on unit
tests. Starting an actual :class:`MujocoVideoRecorder` spins up a headless MuJoCo
simulation (via :class:`~semantic_digital_twin.adapters.multi_sim.MujocoSim`), so those
tests follow the same CI-only gating as ``test_multi_sim.py``.
"""

import os

import numpy as np
import pytest

from semantic_digital_twin.adapters.multi_sim import MujocoCamera
from semantic_digital_twin.adapters.mujoco_video_recording import (
    MujocoVideoRecorder,
    RecordedVideo,
    VideoResolution,
)
from semantic_digital_twin.exceptions import (
    EmptyVideoRecordingError,
    EmptyWorldVideoRecordingError,
    InvalidVideoRecordingRateError,
    VideoRecordingAlreadyStartedError,
    VideoRecordingNotStartedError,
)
from semantic_digital_twin.testing import ray_test_world
from semantic_digital_twin.world import World

only_run_test_in_CI = os.environ.get("CI", "false").lower() == "false"
requires_mujoco_ci = pytest.mark.skipif(
    only_run_test_in_CI,
    reason="Only run MuJoCo-backed recording tests in CI.",
)


# %% MujocoCamera.overview_pose - pure math, no simulator needed


def test_overview_camera_pose_applies_distance_floor_for_a_degenerate_box():
    point_bounds = np.array([[1.0, 1.0, 1.0], [1.0, 1.0, 1.0]])

    position = MujocoCamera.overview_pose(point_bounds).to_position().to_np()[:3]

    assert np.linalg.norm(position - np.array([1.0, 1.0, 1.0])) == pytest.approx(1.5)


def test_overview_camera_pose_scales_with_the_bounding_diagonal():
    small_bounds = np.array([[-1.0, -1.0, -1.0], [1.0, 1.0, 1.0]])
    large_bounds = np.array([[-2.0, -2.0, -2.0], [2.0, 2.0, 2.0]])

    small_position = MujocoCamera.overview_pose(small_bounds).to_position().to_np()[:3]
    large_position = MujocoCamera.overview_pose(large_bounds).to_position().to_np()[:3]

    small_distance = np.linalg.norm(small_position)
    large_distance = np.linalg.norm(large_position)
    assert large_distance == pytest.approx(2 * small_distance)


def test_overview_camera_pose_looks_at_the_bounding_box_center():
    bounds = np.array([[-1.0, -1.0, -1.0], [1.0, 1.0, 1.0]])

    pose = MujocoCamera.overview_pose(bounds)
    position = pose.to_position().to_np()[:3]

    from scipy.spatial.transform import Rotation

    rotation_matrix = Rotation.from_quat(pose.to_quaternion().to_np()).as_matrix()
    view_direction = rotation_matrix @ np.array([0.0, 0.0, -1.0])
    expected_direction = -position / np.linalg.norm(position)
    assert np.allclose(view_direction, expected_direction, atol=1e-6)


# %% RecordedVideo - encoding and derived timestamps, no simulator needed


def test_recorded_video_write_round_trips_frame_count_and_shape(tmp_path):
    frames = [(np.random.rand(48, 64, 3) * 255).astype(np.uint8) for _ in range(5)]
    recorded_video = RecordedVideo(frames=frames, frames_per_second=30)
    output_path = tmp_path / "video.mp4"

    result_path = recorded_video.write(output_path)

    assert result_path == output_path
    assert output_path.exists()
    assert output_path.stat().st_size > 0


def test_recorded_video_write_raises_on_empty_recording(tmp_path):
    recorded_video = RecordedVideo(frames=[], frames_per_second=30)

    with pytest.raises(EmptyVideoRecordingError):
        recorded_video.write(tmp_path / "video.mp4")


def test_recorded_video_frame_timestamps_are_derived_from_frames_per_second():
    frames = [(np.random.rand(4, 4, 3) * 255).astype(np.uint8) for _ in range(4)]
    recorded_video = RecordedVideo(frames=frames, frames_per_second=8)

    assert recorded_video.frame_timestamps == pytest.approx([0.0, 0.125, 0.25, 0.375])


# %% MujocoVideoRecorder guard clauses - cheap, no simulator started


def test_stop_without_start_raises(ray_test_world):
    world, *_ = ray_test_world
    recorder = MujocoVideoRecorder(world=world)

    with pytest.raises(VideoRecordingNotStartedError):
        recorder.stop()


def test_default_camera_is_resolved_at_construction_not_at_start(ray_test_world):
    """
    The default overview camera is attached as soon as the recorder is constructed, not
    lazily on start(): a caller that never starts the recorder still gets a fully resolved
    camera, and a world without any geometry to frame fails fast at construction instead of
    only once start() is later called.
    """
    world, *_ = ray_test_world
    recorder = MujocoVideoRecorder(world=world)

    assert recorder.camera is not None


def test_constructing_a_recorder_for_an_empty_world_raises():
    with pytest.raises(EmptyWorldVideoRecordingError):
        MujocoVideoRecorder(world=World())


def test_non_positive_frames_per_second_raises(ray_test_world):
    world, *_ = ray_test_world

    with pytest.raises(InvalidVideoRecordingRateError):
        MujocoVideoRecorder(world=world, frames_per_second=0)


def test_non_positive_capture_decimation_raises(ray_test_world):
    world, *_ = ray_test_world

    with pytest.raises(InvalidVideoRecordingRateError):
        MujocoVideoRecorder(world=world, capture_every_n_state_changes=0)


# %% MujocoVideoRecorder lifecycle - spins up a real headless MuJoCo simulation


@requires_mujoco_ci
def test_recorder_captures_frames_at_the_configured_resolution(ray_test_world):
    world, *_ = ray_test_world
    recorder = MujocoVideoRecorder(
        world=world,
        frames_per_second=10,
        resolution=VideoResolution(width=32, height=24),
    )

    recorder.start()
    try:
        recorder.advance_simulation(duration=0.5)
    finally:
        recorded_video = recorder.stop()

    assert len(recorded_video.frames) > 0
    assert recorded_video.frames[0].shape == (24, 32, 3)


@requires_mujoco_ci
def test_frame_timestamps_are_positions_on_the_encoded_videos_own_timeline(
    ray_test_world,
):
    world, *_ = ray_test_world
    recorder = MujocoVideoRecorder(
        world=world, frames_per_second=10, capture_every_n_state_changes=50
    )

    recorder.start()
    try:
        recorder.advance_simulation(duration=0.5)
    finally:
        recorded_video = recorder.stop()

    assert recorded_video.frame_timestamps == pytest.approx(
        [i / 10 for i in range(len(recorded_video.frames))]
    )


@requires_mujoco_ci
def test_captured_frame_count_tracks_frames_kept_so_far(ray_test_world):
    world, *_ = ray_test_world
    recorder = MujocoVideoRecorder(world=world, capture_every_n_state_changes=100)

    recorder.start()
    try:
        assert (
            recorder.captured_frame_count == 1
        )  # the initial frame captured by start()
        recorder.advance_simulation(duration=0.5)
        assert recorder.captured_frame_count >= 1
        frame_count_before_stop = recorder.captured_frame_count
    finally:
        recorded_video = recorder.stop()
    assert len(recorded_video.frames) == frame_count_before_stop


@requires_mujoco_ci
def test_start_twice_raises(ray_test_world):
    world, *_ = ray_test_world
    recorder = MujocoVideoRecorder(world=world)

    recorder.start()
    try:
        with pytest.raises(VideoRecordingAlreadyStartedError):
            recorder.start()
    finally:
        recorder.stop()


@requires_mujoco_ci
def test_recording_the_same_world_twice_does_not_accumulate_cameras(ray_test_world):
    world, *_ = ray_test_world

    first_recorder = MujocoVideoRecorder(world=world)
    first_recorder.start()
    first_recorder.stop()

    second_recorder = MujocoVideoRecorder(world=world)
    second_recorder.start()
    try:
        assert second_recorder.captured_frame_count == 1
    finally:
        second_recorder.stop()
