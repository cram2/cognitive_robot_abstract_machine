"""
Which packages ``scripts/sync_version.py`` writes a version into, and where.

The script is a standalone file rather than part of an importable package, so it is
loaded from its path.
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
"""
The script under test.
"""


def load_sync_version() -> ModuleType:
    """
    Registered under its name before it runs, as an import would be, since a dataclass
    looks its own module up while it is defined.

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

repository = sync_version.Repository.from_this_checkout()

RELEASE_WORKSPACE = Path(__file__).parent / "dataset" / "release_workspace"
"""
A miniature workspace: a root and members, one of them nested and one without a version
module of its own.
"""


@pytest.fixture
def release_workspace(tmp_path: Path) -> sync_version.Repository:
    """
    :return: A copy of :data:`RELEASE_WORKSPACE`, which a test may write to.
    """
    copy = tmp_path / RELEASE_WORKSPACE.name
    shutil.copytree(RELEASE_WORKSPACE, copy)
    return sync_version.Repository(root=copy)


def dependencies_of(pyproject_file: Path) -> dict[str, Requirement]:
    """
    :return: The ``[project]`` dependencies of a ``pyproject.toml``, by canonical name.
    """
    project = tomllib.loads(pyproject_file.read_text())["project"]
    requirements = (Requirement(text) for text in project["dependencies"])
    return {canonicalize_name(r.name): r for r in requirements}


def optional_dependencies_of(pyproject_file: Path) -> dict[str, Requirement]:
    """
    :return: The ``[project.optional-dependencies]`` of a ``pyproject.toml`` across every
        extra, by canonical name.
    """
    project = tomllib.loads(pyproject_file.read_text())["project"]
    requirements = (
        Requirement(text)
        for specifiers in project.get("optional-dependencies", {}).values()
        for text in specifiers
    )
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
            "*/*/src/*/*/_version.py",
        )
        for path in repository.root.glob(pattern)
    }

    assert {
        package.version_file
        for package in repository.versioned_packages()
        if package.has_version_module
    } == on_disk


def test_members_are_read_from_the_workspace_declaration(release_workspace):
    """
    A member below the first directory level, such as ``nested/beta``, is a package too.
    """
    directories = {
        package.directory.relative_to(release_workspace.root)
        for package in release_workspace.versioned_packages()
    }

    assert directories == {Path("."), Path("alpha"), Path("nested/beta")}


# %% writing the version modules
def test_every_version_module_is_written_with_the_repository_version(
    release_workspace,
):
    """
    A synchronized package's version module states the repository's version.
    """
    release_workspace.synchronize()

    for package in release_workspace.versioned_packages():
        assert package.version_file.read_text() == package.version_module_text(
            version=release_workspace.version
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
    """
    ``--check`` reports an outdated workspace without changing it.
    """
    before = {
        path: path.read_text()
        for path in release_workspace.root.rglob("*")
        if path.is_file()
    }
    monkeypatch.setattr(
        sync_version.Repository, "from_this_checkout", lambda: release_workspace
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


# %% packages released together with the package whose directory holds them
FAMILY_WORKSPACE = Path(__file__).parent / "dataset" / "family_workspace"
"""
A miniature workspace whose ``family`` package holds two members in its own directory,
released together with it, and a ``loner`` released on its own.
"""


@pytest.fixture
def family_workspace(tmp_path: Path) -> sync_version.Repository:
    """
    :return: A copy of :data:`FAMILY_WORKSPACE`, which a test may write to.
    """
    copy = tmp_path / FAMILY_WORKSPACE.name
    shutil.copytree(FAMILY_WORKSPACE, copy)
    return sync_version.Repository(root=copy)


@pytest.fixture
def family_dependencies(family_workspace) -> dict[str, dict[str, Requirement]]:
    """
    :return: The dependencies of each package of the family workspace, by canonical
        name, after synchronizing.
    """
    family_workspace.synchronize()
    return {
        package.distribution_name: dependencies_of(package.pyproject_file)
        for package in family_workspace.versioned_packages()
    }


def test_a_package_released_with_another_requires_exactly_its_version(
    family_workspace, family_dependencies
):
    """
    A member of a family, or the package holding it, never installs next to another
    release of a member.
    """
    expected = SpecifierSet(f"=={family_workspace.version}")

    assert family_dependencies["family-extra"]["family-core"].specifier == expected
    assert family_dependencies["family"]["family-core"].specifier == expected
    assert family_dependencies["family"]["family-extra"].specifier == expected


def test_a_dependency_across_families_requires_at_least_the_version(
    family_workspace, family_dependencies
):
    """
    Packages released on their own accept any later release of each other.
    """
    expected = SpecifierSet(f">={family_workspace.version}")

    assert family_dependencies["family-extra"]["loner"].specifier == expected
    assert family_dependencies["loner"]["family-core"].specifier == expected
    assert family_dependencies["family-root"]["family"].specifier == expected


def test_a_package_holding_others_declares_the_repository_version(family_workspace):
    """
    A package that ships no module of its own carries the version in its declaration.
    """
    family_workspace.synchronize()

    family = tomllib.loads(
        (family_workspace.root / "family" / "pyproject.toml").read_text()
    )
    assert family["project"]["version"] == family_workspace.version


def test_a_member_of_a_namespace_package_has_its_version_module_written(
    family_workspace,
):
    """
    A version module below a namespace package is found through the dotted attribute.
    """
    family_workspace.synchronize()

    extra_version = (
        family_workspace.root
        / "family"
        / "family-extra"
        / "src"
        / "family"
        / "extra"
        / "_version.py"
    )
    assert extra_version.read_text() == f'__version__ = "{family_workspace.version}"\n'


def test_a_static_version_of_a_package_holding_no_other_is_left_alone(
    release_workspace,
):
    """
    A package that holds no other package keeps the version it states.
    """
    unversioned = release_workspace.root / "unversioned" / "pyproject.toml"
    before = unversioned.read_text()

    release_workspace.synchronize()

    assert unversioned.read_text() == before


# %% bounding the optional dependencies on other packages
@pytest.fixture
def family_core_optional_dependencies(family_workspace) -> dict[str, Requirement]:
    """
    :return: The optional dependencies of ``family-core``, by canonical name, after
        synchronizing.
    """
    family_workspace.synchronize()
    return optional_dependencies_of(
        family_workspace.root / "family" / "family-core" / "pyproject.toml"
    )


def test_an_optional_dependency_released_together_requires_exactly_its_version(
    family_workspace, family_core_optional_dependencies
):
    """
    Installing an extra never pulls in another release of the family.
    """
    expected = SpecifierSet(f"=={family_workspace.version}")

    assert family_core_optional_dependencies["family-extra"].specifier == expected


def test_an_optional_dependency_across_families_requires_at_least_the_version(
    family_workspace, family_core_optional_dependencies
):
    loner = family_core_optional_dependencies["loner"]

    assert loner.specifier == SpecifierSet(f">={family_workspace.version}")
    assert loner.extras == {"cli"}


def test_an_optional_dependency_on_a_third_party_package_is_left_alone(
    family_core_optional_dependencies,
):
    third_party = family_core_optional_dependencies["third-party"]

    assert third_party.specifier == SpecifierSet(">=2.0")


def test_an_optional_dependency_in_an_inline_array_keeps_its_marker(
    family_workspace,
):
    family_workspace.synchronize()

    project = tomllib.loads(
        (
            family_workspace.root / "family" / "family-core" / "pyproject.toml"
        ).read_text()
    )["project"]
    (inline,) = (
        Requirement(text) for text in project["optional-dependencies"]["inline"]
    )
    assert inline.specifier == SpecifierSet(f"=={family_workspace.version}")
    assert inline.marker == Requirement("family-extra ; sys_platform == 'linux'").marker


def test_a_dependency_group_naming_a_sibling_is_left_alone(family_workspace):
    family_workspace.synchronize()

    document = tomllib.loads(
        (
            family_workspace.root / "family" / "family-core" / "pyproject.toml"
        ).read_text()
    )
    assert document["dependency-groups"]["workspace"] == ["family-extra"]


def test_synchronizing_a_family_twice_writes_nothing_the_second_time(
    family_workspace,
):
    """
    Exact bounds and declared versions are stable once written.
    """
    assert family_workspace.synchronize()
    assert family_workspace.synchronize() == []


# %% this repository
def test_this_repository_is_synchronized():
    """
    Fails when a version or a bound on another package was edited by hand or a package
    was added without running ``scripts/sync_version.py``.
    """
    assert repository.outdated_files() == []


@pytest.mark.parametrize(
    "declared_dependencies", [dependencies_of, optional_dependencies_of]
)
def test_every_dependency_on_another_package_of_this_repository_requires_its_version(
    declared_dependencies,
):
    """
    Read back from the files with a TOML parser, independent of how the script writes
    them.
    """
    packages = {
        package.distribution_name: package
        for package in repository.versioned_packages()
    }

    for package in packages.values():
        for name, requirement in declared_dependencies(package.pyproject_file).items():
            if name not in packages or name == package.distribution_name:
                continue
            released_together = repository.release_directory(
                package=packages[name]
            ) == repository.release_directory(package=package)
            operator = (
                sync_version.VersionOperator.EXACTLY
                if released_together
                else sync_version.VersionOperator.AT_LEAST
            )
            expected = SpecifierSet(f"{operator}{repository.version}")
            assert requirement.specifier == expected, (package.directory, name)
