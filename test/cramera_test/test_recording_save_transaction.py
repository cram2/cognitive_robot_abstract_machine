"""
Saving a selected recording range must leave the original recoverable on failure.
"""

from __future__ import annotations

from collections.abc import Iterator
import json
from pathlib import Path

import pytest

from cramera import paths
from cramera.live import recording_storage
from cramera.live.bridge import Bridge, WorldStateSnapshot
from cramera.live.http import serve
from cramera.live.recording import Recording, RecordingState
from cramera.live.frame_range import FrameRange
from cramera.live.recording_bundle import finalize_recording
from cramera.live.recording_storage import SceneDestination, save_recording_bundle
from cramera.recording_fields import SceneField

from .test_live_bundle import attached_bridge
from .test_live_recording import statechart
from .test_server import post, server as viewer_server


# %% shared recording and endpoint fixtures


def directory_contents(directory: Path) -> dict[Path, bytes]:
    """
    Capture every file byte under a directory, including scene indices.

    :param directory: Root whose saved and unsaved data must survive.
    """
    return {
        path.relative_to(directory): path.read_bytes()
        for path in directory.rglob("*")
        if path.is_file()
    }


@pytest.fixture()
def recorded_bridge(fixture_scene: Path) -> Bridge:
    """
    Finalize a moving recording alongside an already indexed scene.

    :param fixture_scene: Isolated existing scene and index.
    """
    bridge = attached_bridge()
    bridge.recording = Recording()
    bridge.recording.start()
    for frame in range(6):
        bridge.recording.append(
            WorldStateSnapshot(
                frames={},
                base=None,
                objects={"milk.stl": [frame, 0, 0, 0, 0, 0, 1]},
            ),
            statechart=statechart("RUNNING"),
        )
    finalize_recording(bridge, bridge.recording)
    return bridge


@pytest.fixture(params=("viewer", "live"))
def save_endpoint(
    request: pytest.FixtureRequest, recorded_bridge: Bridge
) -> Iterator[str]:
    """
    Exercise each real save endpoint against the same finalized recording.

    :param request: Select the always-on viewer or the live bridge.
    :param recorded_bridge: Bridge owning the finalized recording.
    """
    if request.param == "viewer":
        yield request.getfixturevalue("viewer_server") + "/api/recording/save"
        return
    server = serve(recorded_bridge, 0)
    try:
        yield "http://localhost:%d/recording/save" % server.server_address[1]
    finally:
        server.shutdown()
        server.server_close()


# %% rejected HTTP saves preserve their recording


@pytest.mark.parametrize(
    "name,destination,status",
    (
        ("../escape", SceneDestination.LOCAL, 400),
        ("fixture", SceneDestination.LOCAL, 409),
        ("saved", SceneDestination.SHARED, 400),
    ),
)
def test_rejected_trimmed_save_preserves_recording_and_index(
    save_endpoint: str,
    fixture_scene: Path,
    recorded_bridge: Bridge,
    name: str,
    destination: SceneDestination,
    status: int,
) -> None:
    """
    Rejecting a name or destination must preserve every source frame and index byte.

    :param save_endpoint: One of the two save routes.
    :param fixture_scene: Isolated scene storage.
    :param recorded_bridge: Owner whose finalized recording must remain available.
    :param name: Invalid, occupied, or otherwise valid scene name.
    :param destination: Unavailable shared root or local storage.
    :param status: Expected HTTP rejection status.
    """
    before = directory_contents(fixture_scene / "scenes")

    response_status, body = post(
        save_endpoint,
        {
            "name": name,
            "destination": destination,
            "firstFrame": 1,
            "lastFrame": 3,
        },
    )

    assert response_status == status
    assert body["ok"] is False
    assert directory_contents(fixture_scene / "scenes") == before
    assert recorded_bridge.recording.state is RecordingState.FINALIZED


@pytest.mark.parametrize(
    "payload",
    (
        {"name": "saved", "destination": "elsewhere"},
        {"name": "saved", "destination": []},
        {"name": "saved", "firstFrame": "invalid", "lastFrame": 3},
        {"name": "saved", "firstFrame": 1, "lastFrame": None},
        {"name": "saved", "firstFrame": 1.5, "lastFrame": 3},
        {"name": "saved", "firstFrame": True, "lastFrame": 3},
        {"name": "saved", "lastFrame": 3},
        {"name": ["saved"]},
        {"name": "saved", "robot": ["robot"]},
        ["saved"],
    ),
)
def test_malformed_save_is_rejected_without_changing_recording(
    save_endpoint: str, fixture_scene: Path, payload: object
) -> None:
    """
    Invalid destinations and frame values receive a usable JSON rejection.

    :param save_endpoint: One of the two save routes.
    :param fixture_scene: Isolated scene storage.
    :param payload: Malformed JSON save request.
    """
    before = directory_contents(fixture_scene / "scenes")

    status, body = post(save_endpoint, payload)

    assert status == 400
    assert body["ok"] is False
    assert directory_contents(fixture_scene / "scenes") == before


def test_trimmed_save_keeps_selected_frames_and_chart_indices(
    save_endpoint: str, fixture_scene: Path
) -> None:
    """
    Successful saving publishes aligned data and removes the unsaved recording.

    :param save_endpoint: One of the two save routes.
    :param fixture_scene: Isolated scene storage.
    """
    source = paths.local_scenes_directory() / paths.RECORDING_SCENE_NAME
    trajectory = json.loads((source / "trajectory.json").read_text())
    scene = json.loads((source / "scene.json").read_text())
    charts = json.loads((source / scene["statecharts"]).read_text())
    first, last = 1, 3
    task = "carry the milk"

    status, body = post(
        save_endpoint,
        {"name": "saved", "firstFrame": first, "lastFrame": last, "task": task},
    )

    assert status == 200
    assert body == {"ok": True, "scene": "saved"}
    saved = fixture_scene / "scenes" / body["scene"]
    saved_trajectory = json.loads((saved / "trajectory.json").read_text())
    for track in ("frames", "base", "objects"):
        assert saved_trajectory[track] == trajectory[track][first : last + 1]
    saved_charts = json.loads((saved / scene["statecharts"]).read_text())
    assert saved_charts["frames"] == charts["frames"][first : last + 1]
    assert json.loads((saved / "scene.json").read_text())[SceneField.TASK] == task
    assert not source.exists()


# %% filesystem failures preserve their recording


@pytest.mark.parametrize("operation", ("write_json_atomically", "write_scene_index"))
def test_failed_save_preserves_recording_and_index(
    fixture_scene: Path,
    recorded_bridge: Bridge,
    monkeypatch: pytest.MonkeyPatch,
    operation: str,
) -> None:
    """
    Metadata and index write failures must not consume the only recording copy.

    :param fixture_scene: Isolated scene storage.
    :param recorded_bridge: Fixture finalizing a real recording.
    :param monkeypatch: Replace one filesystem write with an injected failure.
    :param operation: Storage write to fail.
    """
    before = directory_contents(fixture_scene / "scenes")

    def fail_write(*arguments: object, **keywords: object) -> None:
        """
        Report a failed disk write without modifying the destination.

        :param arguments: Positional write arguments.
        :param keywords: Named write arguments.
        """
        raise OSError("recording storage unavailable")

    monkeypatch.setattr(recording_storage, operation, fail_write)

    with pytest.raises(OSError):
        save_recording_bundle("saved")

    assert directory_contents(fixture_scene / "scenes") == before
    assert (paths.local_scenes_directory() / paths.RECORDING_SCENE_NAME).is_dir()


@pytest.mark.parametrize("destination", tuple(SceneDestination))
def test_trimmed_save_rolls_back_an_index_failure(
    fixture_scene: Path,
    recorded_bridge: Bridge,
    monkeypatch: pytest.MonkeyPatch,
    destination: SceneDestination,
) -> None:
    """
    Indexing failure restores source frames, charts, and both destination indices.

    :param fixture_scene: Isolated local scene storage.
    :param recorded_bridge: Fixture finalizing a real recording.
    :param monkeypatch: Configure shared storage and inject a failed index write.
    :param destination: Local or shared root to publish into.
    """
    shared = fixture_scene / "shared"
    shared.mkdir()
    monkeypatch.setenv("CRAMERA_SCENES", str(shared))
    before = directory_contents(fixture_scene)

    def fail_index(path: Path, name: str) -> None:
        """
        Refuse the index after the prepared scene has been published.

        :param path: Destination index path.
        :param name: Published scene name.
        """
        assert (path.parent / name / "scene.json").is_file()
        raise OSError("scene index unavailable")

    monkeypatch.setattr(recording_storage, "write_scene_index", fail_index)

    with pytest.raises(OSError):
        save_recording_bundle(
            "saved", destination, frame_range=FrameRange(first=1, last=3)
        )

    assert directory_contents(fixture_scene) == before


def test_shared_save_creates_the_configured_destination(
    fixture_scene: Path, recorded_bridge: Bridge, monkeypatch: pytest.MonkeyPatch
) -> None:
    """
    Sharing can initialize a configured scene directory that does not exist yet.

    :param fixture_scene: Isolated local scene storage.
    :param recorded_bridge: Fixture finalizing a real recording.
    :param monkeypatch: Configure a new shared storage directory.
    """
    shared = fixture_scene / "shared"
    monkeypatch.setenv("CRAMERA_SCENES", str(shared))

    name = save_recording_bundle("saved", SceneDestination.SHARED)

    assert (shared / name / "scene.json").is_file()


@pytest.mark.parametrize("destination", tuple(SceneDestination))
def test_failed_restore_retains_original_recording_and_reports_location(
    fixture_scene: Path,
    recorded_bridge: Bridge,
    monkeypatch: pytest.MonkeyPatch,
    destination: SceneDestination,
) -> None:
    """
    A failed rollback leaves the complete original recoverable at its reported path.

    :param fixture_scene: Isolated local scene storage.
    :param recorded_bridge: Fixture finalizing a real recording.
    :param monkeypatch: Inject index and source-restoration failures.
    :param destination: Local or shared root selected for the failed save.
    """
    shared = fixture_scene / "shared"
    monkeypatch.setenv("CRAMERA_SCENES", str(shared))
    source = paths.local_scenes_directory() / paths.RECORDING_SCENE_NAME
    before = directory_contents(source)
    restored_from: list[Path] = []
    restore_failure = PermissionError("recording directory cannot be restored")
    rename = Path.rename

    def fail_index(path: Path, name: str) -> None:
        """
        Reject publication after the original has moved into the backup.

        :param path: Destination index path.
        :param name: Published scene name.
        """
        raise OSError("scene index unavailable")

    def fail_restore(path: Path, target: Path) -> Path:
        """
        Fail only the attempt to move the original back into its initial directory.

        :param path: Directory being renamed.
        :param target: Requested directory after the rename.
        """
        if target == source:
            restored_from.append(path)
            raise restore_failure
        return rename(path, target)

    monkeypatch.setattr(recording_storage, "write_scene_index", fail_index)
    monkeypatch.setattr(Path, "rename", fail_restore)

    with pytest.raises(type(restore_failure)) as caught:
        save_recording_bundle(
            "saved", destination, frame_range=FrameRange(first=1, last=3)
        )

    assert caught.value is restore_failure
    [preserved] = restored_from
    assert directory_contents(preserved) == before
    assert any(str(preserved) in note for note in caught.value.__notes__)
    assert not (destination.directory() / "saved").exists()
