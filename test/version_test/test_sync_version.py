"""Which packages ``scripts/sync_version.py`` writes a version into, and where.

The script is a standalone file rather than part of an importable
package, so it is loaded from its path.
"""

from __future__ import annotations

import importlib.util
import shutil
import sys
import tomllib
from pathlib import Path
from types import ModuleType

import pytest
from packaging.requirements import Requirement
from packaging.specifiers import SpecifierSet
from packaging.utils import canonicalize_name

# %% the script under test

SYNC_VERSION_SCRIPT = Path(__file__).parents[2] / "scripts" / "sync_version.py"
"""The script under test."""


def load_sync_version() -> ModuleType:
    """Registered under its name before it runs, as an import would be, since a
    dataclass looks its own module up while it is defined.

    :return: The script, imported from its file.
    """
    specification = importlib.util.spec_from_file_location(
        SYNC_VERSION_SCRIPT.stem, SYNC_VERSION_SCRIPT
    )
    module = importlib.util.module_from_spec(specification)
    sys.modules[specification.name] = module
    specification.loader.exec_module(module)
    return module


sync_version = load_sync_version()

repository = sync_version.Repository.of_this_script()

RELEASE_WORKSPACE = Path(__file__).parent / "dataset" / "release_workspace"
"""A miniature workspace: a root and members, one of them nested and one
without a version module of its own."""


@pytest.fixture
def release_workspace(tmp_path: Path) -> sync_version.Repository:
    """
    :return: A copy of :data:`RELEASE_WORKSPACE`, which a test may write to.
    """
    copy = tmp_path / RELEASE_WORKSPACE.name
    shutil.copytree(RELEASE_WORKSPACE, copy)
    return sync_version.Repository(copy)


def dependencies_of(pyproject_file: Path) -> dict[str, Requirement]:
    """
    :return: The ``[project]`` dependencies of a ``pyproject.toml``, by canonical name.
    """
    project = tomllib.loads(pyproject_file.read_text())["project"]
    requirements = (Requirement(text) for text in project["dependencies"])
    return {canonicalize_name(r.name): r for r in requirements}


# %% finding the packages
def test_every_version_module_in_the_repository_is_found():
    """
    A package is found from its own ``pyproject.toml``, so adding one needs no edit to
    the script - and one the script missed would keep an old version.
    """
    on_disk = {
        path
        for pattern in (
            "*/_version.py",
            "*/src/*/_version.py",
            "*/*/src/*/_version.py",
        )
        for path in repository.root.glob(pattern)
    }

    assert {
        package.version_file for package in repository.versioned_packages()
    } == on_disk


def test_members_are_read_from_the_workspace_declaration(release_workspace):
    """A member below the first directory level, such as ``nested/beta``, is a
    package too."""
    directories = {
        package.directory.relative_to(release_workspace.root)
        for package in release_workspace.versioned_packages()
    }

    assert directories == {Path("."), Path("alpha"), Path("nested/beta")}


# %% writing the version modules
def test_every_version_module_is_written_with_the_repository_version(
    release_workspace,
):
    release_workspace.synchronize()

    for package in release_workspace.versioned_packages():
        assert package.version_file.read_text() == package.version_module_text(
            release_workspace.version
        )


# %% bounding the dependencies on other packages
@pytest.fixture
def synchronized_dependencies(release_workspace) -> dict[str, dict[str, Requirement]]:
    """
    :return: The dependencies of each package, by canonical name, after synchronizing.
    """
    release_workspace.synchronize()
    return {
        package.distribution_name: dependencies_of(package.pyproject_file)
        for package in release_workspace.versioned_packages()
    }


def test_a_dependency_on_a_versioned_package_requires_the_repository_version(
    release_workspace, synchronized_dependencies
):
    expected = SpecifierSet(f">={release_workspace.version}")

    assert synchronized_dependencies["alpha"]["beta"].specifier == expected
    assert (
        synchronized_dependencies["alpha"]["alpha-plugin"].specifier == SpecifierSet()
    )
    assert synchronized_dependencies["beta"]["alpha"].specifier == expected
    assert synchronized_dependencies["release-root"]["alpha"].specifier == expected


def test_extras_survive_bounding(synchronized_dependencies):
    beta = synchronized_dependencies["alpha"]["beta"]

    assert beta.extras == {"extra", "another"}


def test_markers_survive_bounding(synchronized_dependencies):
    alpha = synchronized_dependencies["beta"]["alpha"]

    assert alpha.marker == Requirement("alpha ; sys_platform == 'linux'").marker


def test_a_package_does_not_bound_itself(synchronized_dependencies):
    assert synchronized_dependencies["beta"]["beta"].specifier == SpecifierSet()


def test_a_dependency_on_a_package_without_a_version_module_is_left_alone(
    synchronized_dependencies,
):
    assert synchronized_dependencies["alpha"]["unversioned"].specifier == SpecifierSet()


def test_a_dependency_on_a_third_party_package_is_left_alone(
    synchronized_dependencies,
):
    third_party = synchronized_dependencies["alpha"]["third-party"]

    assert third_party.specifier == SpecifierSet(">=2.0")


def test_only_the_dependencies_array_is_rewritten(release_workspace):
    original = (RELEASE_WORKSPACE / "alpha" / "pyproject.toml").read_text()
    alpha = release_workspace.root / "alpha" / "pyproject.toml"

    release_workspace.synchronize()

    changed = [
        line
        for line, before in zip(alpha.read_text().splitlines(), original.splitlines())
        if line != before
    ]
    assert len(changed) == 1
    assert alpha.read_text().count('"beta",') == 1


# %% staying synchronized
def test_synchronizing_twice_writes_nothing_the_second_time(release_workspace):
    written = release_workspace.synchronize()

    assert written
    assert release_workspace.synchronize() == []
    assert release_workspace.outdated_files() == []


def test_an_unsynchronized_workspace_reports_exactly_what_would_be_written(
    release_workspace,
):
    outdated = {file.path for file in release_workspace.outdated_files()}
    written = {file.path for file in release_workspace.synchronize()}

    assert outdated == written


def test_check_writes_nothing_and_fails_while_outdated(release_workspace, monkeypatch):
    before = {
        path: path.read_text()
        for path in release_workspace.root.rglob("*")
        if path.is_file()
    }
    monkeypatch.setattr(
        sync_version.Repository, "of_this_script", lambda: release_workspace
    )
    monkeypatch.setattr(
        sys, "argv", ["sync_version.py", sync_version.CommandFlag.CHECK]
    )

    assert sync_version.main() == 1
    assert before == {
        path: path.read_text()
        for path in release_workspace.root.rglob("*")
        if path.is_file()
    }

    release_workspace.synchronize()

    assert sync_version.main() == 0


# %% this repository
def test_this_repository_is_synchronized():
    """Fails when a version or a bound on another package was edited by hand or
    a package was added without running ``scripts/sync_version.py``."""
    assert repository.outdated_files() == []


def test_every_dependency_on_another_package_of_this_repository_requires_its_version():
    """Read back from the files with a TOML parser, independent of how the
    script writes them."""
    packages = repository.versioned_packages()
    siblings = {package.distribution_name for package in packages}
    expected = SpecifierSet(f">={repository.version}")

    for package in packages:
        for name, requirement in dependencies_of(package.pyproject_file).items():
            if name in siblings and name != package.distribution_name:
                assert requirement.specifier == expected, (package.directory, name)
