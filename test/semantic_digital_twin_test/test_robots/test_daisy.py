"""
Tests of the DAiSy robot description.
"""

import subprocess
import sys
from pathlib import Path

import pytest

from semantic_digital_twin.datastructures.definitions import GripperState
from semantic_digital_twin.robots.daisy import DAiSy
from semantic_digital_twin.robots.griplink_gripper import (
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


# %% action topics


@pytest.mark.parametrize(
    "state", [GripperState.OPEN, GripperState.CLOSE, GripperState.FLEXCLOSE]
)
def test_daisy_grippers_serve_each_state_under_the_topic_of_its_griplink_endpoint(
    daisy_world, state
):
    daisy = daisy_world.get_semantic_annotations_by_type(DAiSy)[0]

    for end_effector in daisy.all_end_effectors:
        assert (
            end_effector.action_topic_serving(state)
            == end_effector.griplink_endpoint(state).action_topic
        )


def test_daisy_grippers_serve_no_state_without_a_griplink_endpoint(daisy_world):
    daisy = daisy_world.get_semantic_annotations_by_type(DAiSy)[0]

    for end_effector in daisy.all_end_effectors:
        assert end_effector.action_topic_serving(None) is None


# %% importing without ROS


def test_daisy_is_importable_without_griplink_interfaces():
    """
    The DAiSy robot description loads in a process without the ROS griplink action
    package, so code and tests that never command the real grippers run without ROS.
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
    assert result.stdout.strip() == DAiSy.__name__
