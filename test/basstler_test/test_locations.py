"""
The paths the package names, and how a member stands in for its path.
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest

import basstler
from basstler.locations import PackageLocation, ProjectLocation

# %% a member stands in for its path


@pytest.mark.parametrize(
    "location", list(ProjectLocation), ids=lambda member: member.name
)
def test_a_member_formats_as_its_path(location: ProjectLocation):
    """
    Members are written into ``git`` references, ``bash -c`` lines and reports as text,
    where an enumeration's own formatting would print its name.
    """
    assert f"{location}" == str(location.value)


def test_a_member_is_accepted_where_a_path_is(tmp_path: Path):
    """
    Joining onto a root, joining beneath a member and handing one to ``os`` all see the
    path.
    """
    assert tmp_path / ProjectLocation.HOOKS == tmp_path / ProjectLocation.HOOKS.value
    assert ProjectLocation.HOOKS / "a.sh" == ProjectLocation.HOOKS.value / "a.sh"
    assert os.fspath(ProjectLocation.HOOKS) == str(ProjectLocation.HOOKS.value)


# %% the package's own files


def test_the_package_directory_is_where_the_package_is_imported_from():
    """
    Every location shipped in the package is found from this one.
    """
    assert PackageLocation.DIRECTORY.value == Path(basstler.__file__).parent


@pytest.mark.parametrize(
    "location",
    [location for location in PackageLocation if location is not PackageLocation.BOARD],
    ids=lambda member: member.name,
)
def test_every_shipped_location_exists(location: PackageLocation):
    """
    A misspelled location would only fail at the first read of it.

    The board is left out, as scratch state a maintenance pass writes.
    """
    assert location.value.exists()


def test_the_package_directory_is_named_the_same_from_the_project_root():
    """
    The project-relative and the absolute spelling of the package name one directory.
    """
    assert (
        PackageLocation.REPOSITORY_ROOT / ProjectLocation.PACKAGE
        == PackageLocation.DIRECTORY.value
    )
