"""
A test that passes only where :attr:`LibraryEnvironmentSetting.HEADLESS_MUJOCO` is set.

Named so the suite does not collect it.
"""

from __future__ import annotations

import os

from cognitive_robot_abstract_machine.continuous_integration_jobs import (
    LibraryEnvironmentSetting,
)


def test_headless_mujoco_is_set() -> None:
    """
    MuJoCo's rendering backend is the one its library's tests need.
    """
    setting = LibraryEnvironmentSetting.HEADLESS_MUJOCO
    assert os.environ.get(setting.variable_name) == setting.variable_value
