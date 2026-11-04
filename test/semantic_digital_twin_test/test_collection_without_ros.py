"""
Tests that the semantic digital twin tests load on a machine without ROS.
"""

import os
import subprocess
import sys
from pathlib import Path

import pytest

from ..orm_interface_build import OrmBuild
from ..pytest_environment import PytestEnvironmentVariable

# %% collection without ROS


def test_every_test_module_loads_without_ros():
    """
    Every test module either loads or is skipped when ROS cannot be imported, so the
    tests that need no ROS run on a machine without it.
    """
    result = subprocess.run(
        [sys.executable, str(Path(__file__).parent / "collection_without_ros.py")],
        capture_output=True,
        text=True,
        timeout=600,
        env={**os.environ, PytestEnvironmentVariable.ORM_BUILD: OrmBuild.NEVER},
    )

    assert result.returncode == pytest.ExitCode.OK, result.stdout + result.stderr
