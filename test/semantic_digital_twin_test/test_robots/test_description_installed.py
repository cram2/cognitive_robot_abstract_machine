"""
Tests for
:meth:`~semantic_digital_twin.robots.robot_parts.AbstractRobot.is_description_installed`.
"""

from __future__ import annotations

import os

from semantic_digital_twin.robots.minimal_robot import MinimalRobot

# %% mimics


class RobotDescribedByAFileOnThisMachine(MinimalRobot):
    """
    A robot whose description is a file that exists.
    """

    @classmethod
    def get_ros_file_path(cls) -> str:
        return os.path.abspath(__file__)


class RobotDescribedByAMissingFile(MinimalRobot):
    """
    A robot whose description is a file that does not exist.
    """

    @classmethod
    def get_ros_file_path(cls) -> str:
        return os.path.join(os.path.dirname(__file__), "no_such_robot.urdf")


class RobotDescribedInAMissingPackage(MinimalRobot):
    """
    A robot whose description is in a package that is not installed.
    """

    @classmethod
    def get_ros_file_path(cls) -> str:
        return "package://no_such_robot_description/urdf/robot.urdf"


# %% whether the description is installed


def test_a_description_on_this_machine_is_installed():
    assert RobotDescribedByAFileOnThisMachine.is_description_installed() is True


def test_a_missing_description_file_is_not_installed():
    assert RobotDescribedByAMissingFile.is_description_installed() is False


def test_a_description_in_a_missing_package_is_not_installed():
    assert RobotDescribedInAMissingPackage.is_description_installed() is False
