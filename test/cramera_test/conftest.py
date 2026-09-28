"""
Fixtures for the cramera tests.

``fixture_scene`` builds a tiny but complete scene bundle (scene.json + trajectory.json
+ a minimal URDF) in a tmp directory and points cramera at it via the CRAMERA_*
environment variables, so the knowledge base and the server tests run against deterministic data
instead of a real (huge, generated) bundle.
"""

import importlib.util
import json
from pathlib import Path

import pytest
import rclpy

from cramera import paths

DATASET_DIR = Path(__file__).parent / "dataset"
"""
Real files instead of embedded strings/dicts, so they stay valid URDF/JSON and Python.
"""

ROBOT_URDF_PATH = DATASET_DIR / "fixture_robot.urdf"
"""
The scene's robot description.
"""

ARCHITECTURE_DIR = DATASET_DIR / "architecture"
"""
A miniature CRAM repository, so the knowledge base's scan is fast and deterministic.
"""

SCENE = json.loads((DATASET_DIR / "scene.json").read_text())
"""
The fixture scene bundle's scene.json and trajectory.json content.
"""

TRAJECTORY = json.loads((DATASET_DIR / "trajectory.json").read_text())


def reset_knowledge_base_cache() -> None:
    """
    Drop the cached knowledge base so the next build picks up the current environment.

    Does nothing without krrood: :mod:`cramera.knowledge` is then not importable, and
    the tests that need it skip themselves.
    """
    if importlib.util.find_spec("krrood") is None:
        return
    from cramera.knowledge.knowledge_base import EpisodeKnowledgeBase

    EpisodeKnowledgeBase.reset()


@pytest.fixture(autouse=True)
def unchecked_out_scenes_submodule(tmp_path, monkeypatch):
    """
    Point the scenes submodule at a directory that does not exist.

    A developer machine with the submodule checked out would otherwise let every test
    that leaves ``CRAMERA_SCENES`` unset fall back to the real ``cramera/scenes``
    checkout, read the scenes in it and write its own bundles into it. Tests that want a
    shared root override this with their own directory.
    """
    monkeypatch.setattr(paths, "SCENES_SUBMODULE", tmp_path / "no-submodule")


@pytest.fixture()
def fixture_scene(tmp_path, monkeypatch):
    """
    A complete miniature scene bundle + architecture; returns the data dir.
    """
    scenes = tmp_path / "scenes" / "fixture"
    scenes.mkdir(parents=True)
    (scenes / "scene.json").write_text(json.dumps(SCENE))
    (scenes / "trajectory.json").write_text(json.dumps(TRAJECTORY))
    (scenes / "robot.urdf").write_text(ROBOT_URDF_PATH.read_text())
    (scenes / "milk.stl").write_bytes(b"solid milk\nendsolid milk\n")
    (tmp_path / "scenes" / "index.json").write_text(
        json.dumps(
            {
                "default": "fixture",
                "scenes": [{"name": "fixture", "robot": "pr2", "environment": None}],
            }
        )
    )

    monkeypatch.setenv("CRAMERA_DATA", str(tmp_path))
    monkeypatch.setenv("CRAMERA_SCENES", str(tmp_path / "scenes"))
    monkeypatch.setenv("CRAMERA_SCENE", "fixture")
    monkeypatch.setenv("CRAMERA_ARCHITECTURE", str(ARCHITECTURE_DIR))

    # a fresh knowledge base per test: it is cached, and the environment just changed
    reset_knowledge_base_cache()
    yield tmp_path
    reset_knowledge_base_cache()


@pytest.fixture(autouse=True)
def leave_no_ros_context():
    """
    Shut a ROS context down again that a cramera test started.

    Serving markers or a live bridge initializes rclpy, and a context left running is
    somebody else's context for every test that follows -- the ones asserting who owns
    the context then read these leftovers instead of their own doing.
    """
    yield
    if rclpy.ok():
        rclpy.shutdown()
