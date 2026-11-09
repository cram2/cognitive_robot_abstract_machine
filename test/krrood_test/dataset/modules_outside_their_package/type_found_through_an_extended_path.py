"""
A submodule of :mod:`package_extending_its_path` that lies outside that package's
directory.
"""

from dataclasses import dataclass


@dataclass
class TypeFoundThroughAnExtendedPath:
    """
    A type importable only through the path its package adds to itself.
    """
