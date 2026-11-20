"""
Tests that every package of the workspace has the version of the repository.
"""

from __future__ import annotations

import importlib.metadata
from pathlib import Path

PACKAGES = [
    "cognitive_robot_abstract_machine",
    "coraplex",
    "cramph",
    "experiments",
    "giskardpy",
    "krrood",
    "krrood_compute_resources",
    "krrood_spatial_math",
    "krrood_synchronization",
    "physics_simulators",
    "probabilistic_model",
    "random_events",
    "robokudo",
    "segmind",
    "semantic_digital_twin",
]

VERSION_FILE = Path(__file__).parents[2] / "VERSION"


def test_all_package_versions_match_root_version() -> None:
    """
    Every workspace package reports the version written in the repository's ``VERSION``
    file.
    """
    expected = VERSION_FILE.read_text().strip()

    for package_name in PACKAGES:
        version = (
            importlib.metadata.version(package_name)
            if package_name == "krrood"
            else __import__(package_name).__version__
        )
        assert version == expected, (
            f"{package_name} version {version} does not match root VERSION {expected}"
        )
