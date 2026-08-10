import subprocess
import sys
import tempfile
from dataclasses import dataclass, field
from pathlib import Path

import pytest
import trimesh
from typing_extensions import Set

from semantic_digital_twin.world_description.geometry import Mesh
from semantic_digital_twin.world_description.mesh_file_storage import (
    MeshFileStorage,
    ProcessLiveness,
)

from .dataset import export_mesh_and_print_session_root


@dataclass
class DeclaredProcessLiveness(ProcessLiveness):
    """
    Reports exactly the processes it was told are running, so a test can describe the
    machine it wants instead of hunting for a process id that is really dead.
    """

    live_process_ids: Set[int] = field(default_factory=set)
    """
    The process ids to report as running.
    """

    def is_alive(self, process_id: int) -> bool:
        return process_id in self.live_process_ids


@pytest.fixture(autouse=True)
def unclaimed_storage():
    """
    Let every test here build the storage the way it needs to, rather than inheriting
    whatever an earlier test happened to construct first.
    """
    MeshFileStorage.clear_instance()
    yield
    MeshFileStorage.clear_instance()


# %% where exported mesh files live


def test_exported_meshes_share_one_session_root(mesh_file_storage):
    """
    Meshes exported without an explicit directory collect under a single root owned by
    this process, rather than being scattered directly across the system temporary
    directory where nothing can ever find them again.
    """
    first = Mesh.from_trimesh(mesh=trimesh.creation.box(extents=(1.0, 1.0, 1.0)))
    second = Mesh.from_trimesh(mesh=trimesh.creation.box(extents=(2.0, 2.0, 2.0)))

    assert Path(first.filename).parent.parent == mesh_file_storage.root
    assert Path(second.filename).parent.parent == mesh_file_storage.root
    assert mesh_file_storage.root.parent == Path(tempfile.gettempdir())


# %% the lifetime of exported mesh files


def test_session_root_is_removed_on_process_exit():
    """
    A process that exports meshes leaves nothing behind once it exits normally.

    The export is run in a subprocess because the cleanup runs at interpreter shutdown,
    which the test process itself never reaches while the test is running.
    """
    script_path = Path(export_mesh_and_print_session_root.__file__)

    result = subprocess.run(
        [sys.executable, str(script_path)],
        capture_output=True,
        text=True,
        timeout=300,
    )

    assert result.returncode == 0, result.stderr
    session_root = Path(result.stdout.strip())
    assert not session_root.exists()


def test_remove_deletes_the_session_root():
    storage = MeshFileStorage()
    Mesh.from_trimesh(mesh=trimesh.creation.box(extents=(1.0, 1.0, 1.0)))

    storage.remove()
    MeshFileStorage.clear_instance()

    assert not storage.root.exists()


def test_file_supplied_mesh_is_not_removed(mesh_file_storage, tmp_path):
    """
    Cleanup reclaims only what the export wrote; a mesh whose path the caller supplied
    stays where the caller put it.
    """
    caller_owned_path = tmp_path / "caller_owned.stl"
    trimesh.creation.box(extents=(1.0, 1.0, 1.0)).export(caller_owned_path)
    mesh = Mesh.from_file(file_path=str(caller_owned_path))

    mesh_file_storage.remove()

    assert Path(mesh.filename).exists()


# %% reclaiming what a killed process could not clean up


def stale_root(temporary_directory: Path, process_id: int) -> Path:
    """
    Create a root that looks as though the given process had left it behind.

    :param temporary_directory: The directory to create the root in.
    :param process_id: The process to name the root after.
    :return: The path of the created root.
    """
    root = temporary_directory / f"{MeshFileStorage.root_prefix}{process_id}_abcdef"
    root.mkdir()
    (root / "leftover.obj").touch()
    return root


def test_root_of_dead_process_is_removed(tmp_path):
    """
    A process killed before it could clean up leaves its meshes behind; the next process
    to export a mesh reclaims them.
    """
    abandoned = stale_root(tmp_path, process_id=424242)

    MeshFileStorage(
        temporary_directory=tmp_path,
        process_liveness=DeclaredProcessLiveness(live_process_ids=set()),
    )

    assert not abandoned.exists()


def test_root_of_live_process_is_kept(tmp_path):
    """
    A root belonging to a process that is still running is left alone, so two processes
    exporting meshes at the same time cannot delete each other's files.
    """
    in_use = stale_root(tmp_path, process_id=424242)

    MeshFileStorage(
        temporary_directory=tmp_path,
        process_liveness=DeclaredProcessLiveness(live_process_ids={424242}),
    )

    assert (in_use / "leftover.obj").exists()


def test_own_root_survives_the_sweep(tmp_path):
    """
    The sweep runs while the new root already exists, and must not mistake it for
    abandoned just because the sweeping process reports as dead.
    """
    storage = MeshFileStorage(
        temporary_directory=tmp_path,
        process_liveness=DeclaredProcessLiveness(live_process_ids=set()),
    )

    assert storage.root.exists()


def test_unrelated_directory_is_kept(tmp_path):
    """
    Only directories this package named are candidates, so nothing else sharing the
    temporary directory is at risk.
    """
    stranger = tmp_path / "someone_elses_data"
    stranger.mkdir()

    MeshFileStorage(
        temporary_directory=tmp_path,
        process_liveness=DeclaredProcessLiveness(live_process_ids=set()),
    )

    assert stranger.exists()
