"""
Tests of the world synchronization messages.
"""

import subprocess
import sys
from pathlib import Path

# %% importing without ROS


def test_messages_load_no_ros_module():
    """
    The world synchronization messages import without loading ROS, so code that stores
    or reads world snapshots runs without it.
    """
    result = subprocess.run(
        [sys.executable, str(Path(__file__).parent / "messages_import_ros_modules.py")],
        capture_output=True,
        text=True,
        timeout=300,
    )

    assert result.returncode == 0, result.stdout + result.stderr
    assert result.stdout.split() == []
