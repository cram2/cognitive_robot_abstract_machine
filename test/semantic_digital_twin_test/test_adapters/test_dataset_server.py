from __future__ import annotations

import os
import shutil
from http import HTTPStatus
from importlib.resources import files
from pathlib import Path, PurePosixPath

import pytest
import trimesh
from PIL import Image

from semantic_digital_twin.adapters.dataset_server import (
    DatasetServer,
    DatasetServerVariable,
    ListedItemKind,
)
from semantic_digital_twin.adapters.package_resolver import (
    CompositePathResolver,
    FileUriResolver,
)
from semantic_digital_twin.exceptions import DatasetServerError, PathResolutionError
from semantic_digital_twin.utils import create_cache_dir
from semantic_digital_twin.world_description.geometry import Mesh
from semantic_digital_twin.world_description.mesh_file_storage import MeshFileSources

from .mock_dataset_file_server import MockDatasetFileServer

# %% the dataset the tests serve

PLY_RESOURCES = Path(files("semantic_digital_twin")).parent.parent / "resources" / "ply"
"""
The directory holding the chair the served entry is made from.
"""

ENTRY = PurePosixPath("mesh_store/ab/ab5f1c3d9e2b4a6c8d0f1e2a3b4c5d6e7f809a1b")
"""
The one entry the served dataset holds, relative to the dataset root.
"""

MESH_NAME = "chair.obj"
"""
The mesh file inside that entry, which refers to the other files by name.
"""

DATASET_ROOT = PurePosixPath("/raid/users/tom_sch/datasets")
"""
Where the dataset sits on the machine holding it, which is what the references recorded
against it are written relative to.
"""

REFERENCE = str(DATASET_ROOT / ENTRY / MESH_NAME)
"""
The reference a mesh records for the served mesh file.
"""


@pytest.fixture(scope="module")
def served_dataset(tmp_path_factory):
    """
    A dataset holding one entry, laid out the way the real store is.

    The entry is the repository's chair exported as a mesh file with the material and
    texture it names beside it, which is the several-files-per-mesh shape the copying is
    about.
    """
    exported = Mesh.from_ply_file(
        ply_file_path=str(PLY_RESOURCES / "chair.ply"),
        texture_file_path=str(PLY_RESOURCES / "chair_texture.png"),
    )
    root = tmp_path_factory.mktemp("dataset")
    entry = root / ENTRY
    entry.parent.mkdir(parents=True)
    shutil.copytree(Path(exported.filename).parent, entry)
    (entry / Path(exported.filename).name).rename(entry / MESH_NAME)
    return root


@pytest.fixture
def file_server(served_dataset):
    server = MockDatasetFileServer(root=served_dataset)
    yield server
    server.stop()


@pytest.fixture
def dataset_server(file_server, tmp_path):
    return DatasetServer(
        base_url=file_server.base_url,
        dataset_root=DATASET_ROOT,
        cache=tmp_path / "cache",
    )


@pytest.fixture(autouse=True)
def forget_registered_sources():
    """
    Keep the process-wide sources of one test out of the next.
    """
    MeshFileSources.clear_instance()
    yield
    MeshFileSources.clear_instance()


# %% copying an entry


class TestEntryIsCopiedWhole:
    """
    A reference names one file, but the files beside it are what the mesh refers to, so
    the whole directory is what has to arrive.
    """

    def test_every_file_of_the_entry_is_copied(self, dataset_server, served_dataset):
        answered = Path(dataset_server.resolve(REFERENCE))

        copied = {
            item.name
            for item in answered.parent.iterdir()
            if item.name != dataset_server.completion_marker
        }
        assert copied == {item.name for item in (served_dataset / ENTRY).iterdir()}

    def test_answer_is_the_copy_and_not_the_reference(self, dataset_server):
        answered = Path(dataset_server.resolve(REFERENCE))

        assert answered == dataset_server.cache / ENTRY / MESH_NAME
        assert answered.is_file()

    def test_directories_in_a_listing_are_not_copied(self, dataset_server):
        listing = dataset_server.list_directory(ENTRY)

        assert {item.kind for item in listing} == {ListedItemKind.FILE}


# %% reading the copied mesh


class TestCopiedMeshLoads:
    """
    The point of copying the entry whole is that the material and texture the mesh names
    resolve against the copy.
    """

    def test_texture_resolves_from_the_copied_entry(self, dataset_server):
        mesh = trimesh.load_mesh(dataset_server.resolve(REFERENCE), process=False)

        with Image.open(PLY_RESOURCES / "chair_texture.png") as expected:
            assert mesh.visual.material.image.size == expected.size

    def test_reading_through_the_server_matches_reading_the_file(
        self, dataset_server, served_dataset
    ):
        MeshFileSources().use(dataset_server)

        through_server = Mesh(filename=REFERENCE).mesh
        directly = Mesh(filename=str(served_dataset / ENTRY / MESH_NAME)).mesh

        assert through_server.vertices.tolist() == directly.vertices.tolist()
        assert through_server.faces.tolist() == directly.faces.tolist()


# %% not asking twice


class TestCachedEntryIsNotFetchedAgain:
    """
    Entries of a store addressed by content hash never change, so a copy is kept for
    good.
    """

    def test_second_resolution_makes_no_request(self, dataset_server, file_server):
        dataset_server.resolve(REFERENCE)
        after_first = list(file_server.requested_paths)

        dataset_server.resolve(REFERENCE)

        assert file_server.requested_paths == after_first

    def test_copy_without_its_marker_is_made_again(self, dataset_server, file_server):
        answered = Path(dataset_server.resolve(REFERENCE))
        (answered.parent / dataset_server.completion_marker).unlink()
        after_first = len(file_server.requested_paths)

        dataset_server.resolve(REFERENCE)

        assert len(file_server.requested_paths) > after_first


# %% which references are claimed


class TestOnlyReferencesBelowTheRootAreClaimed:
    """
    A process reads from the dataset and from its own disk, so a resolver that claimed
    everything would take files it has no copy of.
    """

    def test_path_outside_the_root_is_not_claimed(self, dataset_server):
        assert not dataset_server.supports("/home/somebody/local.obj")

    def test_resolving_an_unclaimed_path_is_refused(self, dataset_server):
        with pytest.raises(PathResolutionError):
            dataset_server.resolve("/home/somebody/local.obj")

    def test_local_path_falls_through_to_the_file_resolver(
        self, dataset_server, tmp_path
    ):
        composite = CompositePathResolver(resolvers=[dataset_server, FileUriResolver()])
        local_file = str(tmp_path / "local.obj")

        assert composite.resolve(local_file) == os.path.abspath(local_file)


# %% when the server has nothing to answer with


class TestMissingEntryIsReported:
    """
    A reference to something the server does not hold is a broken dataset, not a
    transport failure, and says so.
    """

    def test_missing_entry_raises_with_the_status(self, dataset_server):
        missing = str(DATASET_ROOT / "mesh_store/zz/absent/thing.obj")

        with pytest.raises(DatasetServerError) as raised:
            dataset_server.resolve(missing)

        assert raised.value.status_code == HTTPStatus.NOT_FOUND


# %% the sources a process reads through


class TestMeshReadsThroughRegisteredSources:
    """
    A mesh records where it came from; the registered sources decide what that means on
    the machine reading it.
    """

    def test_reference_is_answered_with_itself_when_nothing_claims_it(self):
        assert MeshFileSources().resolve(REFERENCE) == Path(REFERENCE)

    def test_registered_source_answers_the_reference(self, dataset_server):
        MeshFileSources().use(dataset_server)

        assert Mesh(filename=REFERENCE).local_file == (
            dataset_server.cache / ENTRY / MESH_NAME
        )


# %% describing the server through the environment


class TestServerIsDescribedByTheEnvironment:
    """
    A user of the dataset has no account on the machine holding it, so the address is
    all the configuration there is.
    """

    def test_no_server_without_an_address(self, monkeypatch):
        monkeypatch.delenv(DatasetServerVariable.BASE_URL, raising=False)

        assert DatasetServer.from_environment() is None

    def test_address_and_cache_are_read_from_the_environment(
        self, monkeypatch, tmp_path
    ):
        monkeypatch.setenv(DatasetServerVariable.BASE_URL, "http://host:18080/datasets")
        monkeypatch.setenv(DatasetServerVariable.DATASET_ROOT, str(DATASET_ROOT))
        monkeypatch.setenv(DatasetServerVariable.CACHE_DIRECTORY, str(tmp_path))

        server = DatasetServer.from_environment()

        assert server.base_url == "http://host:18080/datasets"
        assert server.dataset_root == DATASET_ROOT
        assert server.cache == tmp_path

    def test_cache_defaults_to_the_directory_the_package_downloads_into(
        self, monkeypatch
    ):
        monkeypatch.setenv(DatasetServerVariable.BASE_URL, "http://host:18080/datasets")
        monkeypatch.delenv(DatasetServerVariable.CACHE_DIRECTORY, raising=False)

        server = DatasetServer.from_environment()

        assert server.cache == create_cache_dir("dataset_server")
