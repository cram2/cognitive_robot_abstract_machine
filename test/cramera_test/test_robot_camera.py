"""
Robot camera selection and pose tracking use the viewer's actual Three.js transforms.
"""

from pathlib import Path
import shutil
import subprocess

import pytest


# %% robot perspective
@pytest.mark.skipif(shutil.which("node") is None, reason="Node.js is not installed")
def test_robot_camera() -> None:
    """
    Exercise camera sources and articulated robot viewpoints with vendored Three.js.
    """
    result = subprocess.run(
        [
            "node",
            "--test",
            str(Path(__file__).parent / "js" / "test_robot_camera.js"),
            str(Path(__file__).parent / "js" / "test_robot_camera_panel.js"),
        ],
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert result.returncode == 0, result.stdout + result.stderr
