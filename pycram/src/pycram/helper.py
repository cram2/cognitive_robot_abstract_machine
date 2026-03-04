"""Implementation of helper functions and classes for internal usage only.

Classes:
Singleton -- implementation of singleton metaclass
"""

import os
from typing_extensions import Dict, Optional, Tuple
import xml.etree.ElementTree as ET

import logging

logger = logging.getLogger(__name__)


class Singleton(type):
    """
    Metaclass for singletons
    """

    _instances = {}
    """
    Dictionary of singleton child classes inheriting from this metaclass, keyed by child class objects.
    """

    def __call__(cls, *args, **kwargs):
        if cls not in cls._instances:
            cls._instances[cls] = super(Singleton, cls).__call__(*args, **kwargs)
        return cls._instances[cls]


def get_robot_urdf_and_mjcf_file_paths(
    robot_name: str, robot_relative_dir: str, multiverse_resources: Optional[str] = None
) -> Tuple[Optional[str], Optional[str]]:
    """
    Get the paths to the MJCF and URDF files of a robot from the Multiverse resources directory.

    :param robot_name: The name of the robot.
    :param robot_relative_dir: The relative directory of the robot in the Multiverse resources/robots directory.
    :param multiverse_resources: The path to the Multiverse resources directory.
    """
    multiverse_resources = (
        find_multiverse_resources_path()
        if multiverse_resources is None
        else multiverse_resources
    )

    urdf_filename: Optional[str] = None
    mjcf_filename: Optional[str] = None
    if multiverse_resources is not None:
        urdf_filename = get_robot_description_path(
            robot_relative_dir,
            robot_name,
            description_type=URDFObject,
            resources_dir=multiverse_resources,
        )
        mjcf_filename = get_robot_description_path(
            robot_relative_dir, robot_name, resources_dir=multiverse_resources
        )
    return urdf_filename, mjcf_filename


def perform(action_instance):
    """
    Executes the perform logic for a given action instance.

    :param action_instance: An instance of an action class.
    """
    return action_instance.perform()


def an(designator):
    """
    Resolve the first available action from the designator.

    :param designator: The designator description instance.
    :return: The first resolved action instance.
    """
    return designator.resolve()
