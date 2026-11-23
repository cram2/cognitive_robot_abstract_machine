"""
Tests that the repository files code refers to exist.
"""

from __future__ import annotations

from cognitive_robot_abstract_machine.orm_interfaces import REPOSITORY_ROOT
from cognitive_robot_abstract_machine.repository_files import (
    RepositoryDirectory,
    RepositoryFile,
)

# %% existence


def test_every_repository_file_exists():
    missing = [
        file for file in RepositoryFile if not (REPOSITORY_ROOT / file).is_file()
    ]
    assert missing == []


def test_every_repository_directory_exists():
    """
    Every directory code refers to is there.
    """
    missing = [
        directory
        for directory in RepositoryDirectory
        if not (REPOSITORY_ROOT / directory).is_dir()
    ]
    assert missing == []
