import pytest

from semantic_digital_twin.world_description.mesh_file_storage import MeshFileStorage


@pytest.fixture
def mesh_file_storage() -> MeshFileStorage:
    """
    The session root, removed again once the test that asked for it is done.

    Not autouse: tests that merely export a mesh are meant to share one root, and tearing
    it down between them would hide exactly the sharing this module is about.
    """
    storage = MeshFileStorage()
    yield storage
    storage.remove()
    MeshFileStorage.clear_instance()
