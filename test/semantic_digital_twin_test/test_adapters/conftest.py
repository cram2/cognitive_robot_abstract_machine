import os

import pytest


@pytest.fixture(scope="session", autouse=True)
def unique_mujoco_scene_file(tmp_path_factory):
    """
    Give each xdist worker its own MuJoCo scene file to prevent /tmp/scene.xml race
    conditions.
    """
    try:
        from semantic_digital_twin.adapters.multi_sim import MujocoSim

        worker_id = os.environ.get("PYTEST_XDIST_WORKER", "main")
        tmp_path = tmp_path_factory.mktemp(f"mujoco_{worker_id}")
        MujocoSim.default_file_path = str(tmp_path / "scene.xml")
    except ImportError:
        pass
