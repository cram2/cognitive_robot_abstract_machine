"""
Leaves the ROS adapter tests uncollected on a machine without ROS, where their modules
cannot be imported.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from semantic_digital_twin.utils import rclpy_installed

# %% collection without ROS


def pytest_ignore_collect(collection_path: Path, config: pytest.Config) -> bool | None:
    """
    :return: ``True`` without ROS, leaving every path of this directory uncollected;
        otherwise nothing, so pytest decides as it would.
    """
    if rclpy_installed():
        return None
    return True
