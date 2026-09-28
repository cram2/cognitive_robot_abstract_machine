"""
Managing an already-finalized live-recording bundle on disk.

Deliberately free of :mod:`cramera.live.bridge`/``semantic_digital_twin`` — once a
recording has been written to disk (by :func:`cramera.live.recording_bundle.
finalize_recording`, whether from an explicit ``/recording/stop`` or its exit-time
safety net), discarding or saving it is a pure filesystem operation that works whether
or not the demo process that produced it is still running. This is what lets
:mod:`cramera.server` (the always-on viewer process, on a different port than the live
bridge) offer the same actions as a fallback once that process is gone.
"""

from __future__ import annotations

import json
import shutil
from contextlib import ExitStack
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path
from tempfile import TemporaryDirectory

from typing_extensions import Optional

from cramera import paths
from cramera.recording_fields import SceneField, TrajectoryField
from cramera.generated_json import write_json_atomically
from cramera.live.frame_range import FrameRange, InvalidFrameRange
from cramera.live.recording_segments import clip_segment_payloads
from cramera.onboard.scene_index import validate_scene_name, write_scene_index


class NoSavedRecording(Exception):
    """
    Raised by :func:`save_recording_bundle` when no finalized ``__recording__`` bundle
    exists on disk to save.
    """


class SceneNameTaken(Exception):
    """
    Raised by :func:`save_recording_bundle` when the requested name already names a
    scene in a shared or local scenes root.
    """


def has_saveable_recording() -> bool:
    """
    Whether a finalized ``__recording__`` bundle currently exists on disk.
    """
    return (
        paths.local_scenes_directory() / paths.RECORDING_SCENE_NAME / "scene.json"
    ).is_file()


def discard_recording_bundle() -> None:
    """
    Delete the unsaved ``__recording__`` bundle from disk, if one exists.
    """
    shutil.rmtree(
        paths.local_scenes_directory() / paths.RECORDING_SCENE_NAME, ignore_errors=True
    )


class SharedScenesUnavailable(Exception):
    """
    Raised when a scene is to be shared but no shared scenes root is configured, so
    sharing it would quietly be an ordinary local save.
    """


class SceneDestination(StrEnum):
    """
    Which scenes root a saved recording is written to.
    """

    LOCAL = "local"
    """
    The user's own data directory: the recording stays on this machine.
    """

    SHARED = "shared"
    """
    The optional scene archive shared with other viewers.
    """

    def directory(self) -> Path:
        """
        The scenes root this destination writes into.
        """
        if self is SceneDestination.LOCAL:
            return paths.local_scenes_directory()
        return paths.scenes_directory()


def save_recording_bundle(
    name: str,
    destination: SceneDestination = SceneDestination.LOCAL,
    robot: Optional[str] = None,
    environment: Optional[str] = None,
    task: Optional[str] = None,
    frame_range: Optional[FrameRange] = None,
) -> str:
    """
    Promote the finalized ``__recording__`` bundle to a permanent, saved scene.

    Sharing only moves files: the scene appears in the shared checkout's working tree,
    and committing it there stays a deliberate act.

    :param name: Name to save the recording under.
    :param destination: Which scenes root to save it into.
    :param robot: What the person saving it calls the robot, or None to leave the
        recording's own answer standing.
    :param environment: What they call the environment, likewise.
    :param task: What the run was doing, which only they can say.
    :param frame_range: Optional inclusive range to keep in the saved copy.
    :raises cramera.onboard.scene_index.InvalidSceneName: If ``name`` is unsafe or
        reserved.
    :raises NoSavedRecording: If no finalized ``__recording__`` bundle exists on disk.
    :raises SceneNameTaken: If ``name`` already names a scene in any scenes root.
    :raises SharedScenesUnavailable: If sharing is asked for without a shared root
        distinct from the local one.
    """
    validate_scene_name(name)
    root = destination.directory()
    if (
        destination is SceneDestination.SHARED
        and root == paths.local_scenes_directory()
    ):
        raise SharedScenesUnavailable("no shared scenes root: set " "CRAMERA_SCENES")
    source = paths.local_scenes_directory() / paths.RECORDING_SCENE_NAME
    if not (source / "scene.json").is_file():
        raise NoSavedRecording("no finalized recording to save")
    if any((existing / name).is_dir() for existing in paths.scene_roots()):
        raise SceneNameTaken("a scene named '%s' already exists" % name)
    RecordingBundle(source).save(root / name, frame_range, robot, environment, task)
    return name


# %% cutting a finalized bundle down before it is saved


def trim_recording_bundle(frame_range: FrameRange) -> None:
    """
    Cut the unsaved ``__recording__`` bundle down to the frames a range keeps.

    Everything a replay reads is derived from frame indices, so the trim rewrites all
    three together: the trajectory keeps only the selected ticks, the timeline segments
    are rebased on them (see
    :func:`cramera.live.recording_segments.clip_segment_payloads`), and each object
    spawns where the kept stretch starts rather than where the cut-away run did.

    :param frame_range: The stretch of the run to keep.
    :raises NoSavedRecording: If no finalized bundle exists on disk.
    :raises cramera.live.recording.InvalidFrameRange: If the range reaches past the
        bundled run.
    """
    RecordingBundle(paths.local_scenes_directory() / paths.RECORDING_SCENE_NAME).trim(
        frame_range
    )


# %% preparing and publishing recording copies


@dataclass(frozen=True)
class RecordingBundle:
    """
    A finalized recording whose edits can be prepared separately from its source.
    """

    directory: Path
    """
    Directory containing the scene, trajectory, and supporting assets.
    """

    def save(
        self,
        destination: Path,
        frame_range: Optional[FrameRange],
        robot: Optional[str],
        environment: Optional[str],
        task: Optional[str],
    ) -> None:
        """
        Publish a prepared copy, retaining the original until indexing succeeds.

        If restoration fails, the original remains at the recovery path named in an
        exception note.

        :param destination: Validated unoccupied permanent scene directory.
        :param frame_range: Optional inclusive range to retain in the saved copy.
        :param robot: Optional display name for the recorded robot.
        :param environment: Optional display name for the recorded environment.
        :param task: Optional description of the recorded task.
        """
        destination.parent.mkdir(parents=True, exist_ok=True)
        with TemporaryDirectory(
            dir=destination.parent, ignore_cleanup_errors=True
        ) as staging_directory:
            prepared = RecordingBundle(Path(staging_directory) / destination.name)
            shutil.copytree(self.directory, prepared.directory)
            if frame_range is not None:
                prepared.trim(frame_range)
            prepared.rename(destination.name, robot, environment, task)
            backup = TemporaryDirectory(
                dir=self.directory.parent, ignore_cleanup_errors=True, delete=False
            )
            original = Path(backup.name) / self.directory.name
            try:
                with ExitStack() as rollback:
                    prepared.directory.rename(destination)
                    rollback.callback(shutil.rmtree, destination)
                    self.directory.rename(original)
                    rollback.callback(original.rename, self.directory)
                    write_scene_index(
                        destination.parent / "index.json", destination.name
                    )
                    rollback.pop_all()
            except BaseException as error:
                if original.exists():
                    error.add_note(f"Original recording preserved at {original}")
                else:
                    backup.cleanup()
                raise
            else:
                backup.cleanup()

    def rename(
        self,
        name: str,
        robot: Optional[str],
        environment: Optional[str],
        task: Optional[str],
    ) -> None:
        """
        Update the saved scene's identity without changing its recorded data.

        :param name: Permanent name of the saved scene.
        :param robot: Optional robot display name.
        :param environment: Optional environment display name.
        :param task: Optional description of the run.
        """
        scene_path = self.directory / "scene.json"
        scene = json.loads(scene_path.read_text(encoding="utf-8"))
        scene["name"] = name
        scene.update(
            {
                field.value: given
                for field, given in [
                    (SceneField.ROBOT_NAME, robot),
                    (SceneField.ENVIRONMENT_NAME, environment),
                    (SceneField.TASK, task),
                ]
                if given
            }
        )
        write_json_atomically(scene_path, scene, indent=1)

    def trim(self, frame_range: FrameRange) -> None:
        """
        Keep a frame range with aligned trajectory, chart, and timeline data.

        :param frame_range: Inclusive recording range to retain.
        :raises NoSavedRecording: If the scene or trajectory is missing.
        :raises InvalidFrameRange: If the range extends past the recording.
        """
        scene_path, trajectory_path = (
            self.directory / "scene.json",
            self.directory / "trajectory.json",
        )
        if not scene_path.is_file() or not trajectory_path.is_file():
            raise NoSavedRecording("no finalized recording to trim")
        trajectory = json.loads(trajectory_path.read_text(encoding="utf-8"))
        if frame_range.last >= len(trajectory["frames"]):
            raise InvalidFrameRange(
                "frame %d is past the recording's %d frames"
                % (frame_range.last, len(trajectory["frames"]))
            )
        kept = slice(frame_range.first, frame_range.last + 1)
        for track in ("frames", "base", "objects"):
            trajectory[track] = trajectory[track][kept]
        if TrajectoryField.FRAME_TIMES in trajectory:
            trajectory[TrajectoryField.FRAME_TIMES] = trajectory[
                TrajectoryField.FRAME_TIMES
            ][kept]
        scene = json.loads(scene_path.read_text(encoding="utf-8"))
        statecharts_path = self.directory / scene.get("statecharts", "statecharts.json")
        if statecharts_path.is_file():
            statecharts = json.loads(statecharts_path.read_text(encoding="utf-8"))
            statecharts["frames"] = statecharts["frames"][kept]
            write_json_atomically(statecharts_path, statecharts)
        scene["segments"] = clip_segment_payloads(scene["segments"], frame_range)
        for entry in scene["objects"]:
            spawn = trajectory["objects"][0].get(entry["key"])
            if spawn is not None:
                entry["spawn"] = spawn
        write_json_atomically(trajectory_path, trajectory)
        write_json_atomically(scene_path, scene, indent=1)
