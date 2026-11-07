"""
Tests of the DAiSy robot description.
"""

import logging
import subprocess
import sys
from pathlib import Path

import pytest

from semantic_digital_twin.datastructures.definitions import GripperState
from semantic_digital_twin.robots import griplink_gripper
from semantic_digital_twin.robots.daisy import DAiSy, DAiSyLeftGripper
from semantic_digital_twin.robots.exceptions import GriplinkInterfacesNotInstalledError
from semantic_digital_twin.robots.griplink_gripper import (
    GriplinkAction,
    GriplinkEndpoint,
    GriplinkFlexConfiguration,
    GriplinkPresetConfiguration,
)

# %% griplink configurations


def test_daisy_grippers_build_griplink_configurations(daisy_world):
    """
    Both DAiSy end effectors build griplink-specific configurations from their declared
    states, so generic actions produce robot-appropriate configurations without naming
    the robot.
    """
    daisy = daisy_world.get_semantic_annotations_by_type(DAiSy)[0]

    for end_effector in daisy.all_end_effectors:
        open_configuration = end_effector.default_configuration(GripperState.OPEN)
        close_configuration = end_effector.default_configuration(GripperState.CLOSE)
        flex_close_configuration = end_effector.default_configuration(
            GripperState.FLEXCLOSE
        )

        assert isinstance(open_configuration, GriplinkPresetConfiguration)
        assert isinstance(close_configuration, GriplinkPresetConfiguration)
        assert isinstance(flex_close_configuration, GriplinkFlexConfiguration)


# %% importing without ROS


def import_daisy_without_griplink_interfaces() -> list[str]:
    """
    :return: The lines the DAiSy import printed in a process without the ROS griplink
        action package: one per logged record, then the robot class name.
    """
    result = subprocess.run(
        [
            sys.executable,
            str(Path(__file__).parent / "daisy_import_without_griplink_interfaces.py"),
        ],
        capture_output=True,
        text=True,
        timeout=300,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    return result.stdout.strip().splitlines()


def test_daisy_is_importable_without_griplink_interfaces():
    """
    The DAiSy robot description loads in a process without the ROS griplink action
    package, so code and tests that never command the real grippers run without ROS.
    """
    assert import_daisy_without_griplink_interfaces()[-1] == DAiSy.__name__


def test_griplink_gripper_import_warns_without_griplink_interfaces():
    """
    Loading the griplink gripper module without the ROS griplink action package logs one
    warning from it.
    """
    logged = [
        line
        for line in import_daisy_without_griplink_interfaces()
        if line.startswith(f"{GriplinkEndpoint.__module__} ")
    ]

    assert logged == [
        f"{GriplinkEndpoint.__module__} {logging.getLevelName(logging.WARNING)}"
    ]


# %% griplink endpoints


@pytest.fixture
def without_griplink_action_types(monkeypatch: pytest.MonkeyPatch) -> None:
    """
    Leaves the griplink gripper module without its ROS action types, as when
    ``griplink_interfaces`` is not installed.
    """
    for action in GriplinkAction:
        monkeypatch.setattr(griplink_gripper, action.value, None)


def test_daisy_griplink_endpoints_are_built_without_griplink_action_types(
    without_griplink_action_types: None,
):
    """
    The DAiSy grippers' endpoints are built without the ROS griplink action types, so
    reading the class property does not need ROS.
    """
    endpoint = DAiSyLeftGripper.griplink_endpoints[GripperState.OPEN]

    assert endpoint.action == GriplinkAction.RELEASE


def test_griplink_endpoint_message_type_without_griplink_action_types_is_rejected(
    without_griplink_action_types: None,
):
    """
    Asking an endpoint for its ROS action type without the ROS griplink action types
    fails instead of answering with no type.
    """
    endpoint = DAiSyLeftGripper.griplink_endpoints[GripperState.CLOSE]

    with pytest.raises(GriplinkInterfacesNotInstalledError):
        endpoint.message_type
