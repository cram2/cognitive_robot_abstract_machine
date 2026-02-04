from __future__ import annotations

import importlib
import os
import weakref
from copy import deepcopy
from functools import lru_cache, wraps
from typing import List

try:
    from ament_index_python import PackageNotFoundError
except ModuleNotFoundError:
    PackageNotFoundError = None
from xml.etree import ElementTree as ET

from typing_extensions import Any, Tuple


class IDGenerator:
    """
    A class that generates incrementing, unique IDs and caches them for every object this is called on.
    """

    _counter = 0
    """
    The counter of the unique IDs.
    """

    def __init__(self):
        self._counter = 0
        self._by_obj = weakref.WeakKeyDictionary()  # type: ignore[var-annotated]

    def __call__(self, obj: Any) -> int:
        """
        Creates a unique ID and caches it for every object this is called on.

        :param obj: The object to generate a unique ID for, must be hashable.
        :return: The unique ID.
        """
        try:
            return self._by_obj[obj]
        except KeyError:
            self._counter += 1
            self._by_obj[obj] = self._counter
            return self._counter


class suppress_stdout_stderr(object):
    """
    A context manager for doing a "deep suppression" of stdout and stderr in
    Python, i.e. will suppress all prints, even if the print originates in a
    compiled C/Fortran sub-function.

    This will not suppress raised exceptions, since exceptions are printed
    to stderr just before a script exits, and after the context manager has
    exited (at least, I think that is why it lets exceptions through).
    Copied from https://stackoverflow.com/questions/11130156/suppress-stdout-stderr-print-from-python-functions
    """

    def __init__(self):
        # Open a pair of null files
        self.null_fds = [os.open(os.devnull, os.O_RDWR) for _ in range(2)]
        # Save the actual stdout (1) and stderr (2) file descriptors.
        self.save_fds = [os.dup(1), os.dup(2)]

    def __enter__(self):
        # Assign the null pointers to stdout and stderr.
        # This one is not needed for URDF parsing output
        # os.dup2(self.null_fds[0], 1)
        os.dup2(self.null_fds[1], 2)

    def __exit__(self, *_):
        # Re-assign the real stdout/stderr back to (1) and (2)
        # This one is not needed for URDF parsing output
        # os.dup2(self.save_fds[0], 1)
        os.dup2(self.save_fds[1], 2)
        # Close all file descriptors
        for fd in self.null_fds + self.save_fds:
            os.close(fd)


def hacky_urdf_parser_fix(
    urdf: str, blacklist: Tuple[str] = ("transmission", "gazebo")
) -> str:
    # Parse input string
    root = ET.fromstring(urdf)

    # Iterate through each section in the blacklist
    for section_name in blacklist:
        # Find all sections with the given name and remove them
        for elem in root.findall(f".//{section_name}"):
            parent = root.find(f".//{section_name}/..")
            if parent is not None:
                parent.remove(elem)

    # Turn back to string
    return ET.tostring(root, encoding="unicode")


def robot_name_from_urdf_string(urdf_string: str) -> str:
    """
    Returns the name defined in the robot tag, e.g., 'pr2' from <robot name="pr2"> ... </robot>.
    :param urdf_string: URDF string
    :return: Extracted name
    """
    return urdf_string.split('robot name="')[1].split('"')[0]


def copy_lru_cache(maxsize=None, typed=False):
    def decorator(func):
        cached_func = lru_cache(maxsize=maxsize, typed=typed)(func)

        @wraps(func)
        def wrapper(*args, **kwargs):
            result = cached_func(*args, **kwargs)
            return deepcopy(result)

        # Preserve lru_cache methods
        wrapper.cache_info = cached_func.cache_info
        wrapper.cache_clear = cached_func.cache_clear

        return wrapper

    return decorator


def bpy_installed() -> bool:
    try:
        import bpy

        return True
    except ImportError:
        return False


def rclpy_installed() -> bool:
    try:
        import rclpy

        return True
    except ImportError:
        return False


def tracy_installed() -> bool:
    try:
        from ament_index_python.packages import get_package_share_directory

        pkg_name = "iai_tracy_description"

        if get_package_share_directory(pkg_name):
            return True
        return False
    except (ImportError, PackageNotFoundError, ValueError):
        return False


def hsrb_installed() -> bool:
    try:
        from ament_index_python.packages import get_package_share_directory

        pkg_name = "hsr_description"

        if get_package_share_directory(pkg_name):
            return True
        return False
    except (ImportError, PackageNotFoundError, ValueError):
        return False


def type_string_to_type(type_string: str) -> type:
    """
    Convert a string representation of a type to the actual type.

    :param type_string: The string representation of the type, e.g., 'module.submodule.ClassName'.
    :return: The actual type.
    """
    module_path, class_name = type_string.rsplit(".", 1)
    module = importlib.import_module(module_path)
    return getattr(module, class_name)


def camel_case_split(word: str) -> List[str]:
    """
    :param word: The word to split
    :return: A set of strings where each string is a camel case split of the original word
    """
    result = []
    start = 0
    for i, c in enumerate(word[1:], 1):
        if c.isupper():
            result.append(word[start:i])
            start = i
    result.append(word[start:])
    return result
