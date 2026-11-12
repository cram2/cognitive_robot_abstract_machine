"""
Tests for building the ORM interfaces a checkout needs before it can persist objects.
"""

from __future__ import annotations

import os
import shutil
import subprocess
from dataclasses import dataclass, field
from enum import StrEnum
from pathlib import Path

import pytest
from typing_extensions import List, Optional

from cognitive_robot_abstract_machine import orm_interfaces
from cognitive_robot_abstract_machine.exceptions import (
    MissingORMGeneratorError,
    OrmGenerationFailedError,
    UnknownOrmPackageError,
    UnorderableOrmInterfacesError,
)
from cognitive_robot_abstract_machine.orm_interfaces import (
    INTERFACE_FILE_NAME,
    OrmInterface,
    REPOSITORY_ROOT,
    WORKSPACE_ORM_INTERFACES,
    WorkspaceOrmInterfaces,
)

from . import dataset
from .dataset import failing_generate_orm, generate_orm, mapped_module

# %% a checkout of packages that generate an interface

PACKAGE_NAMES: tuple[str, ...] = ("upstream", "downstream")
"""
Packages of the checkout under test, in dependency order.
"""

STALE_INTERFACE_CONTENT = "# interface of a previous run\n"
"""
Content the interfaces of the checkout hold before it is regenerated.
"""

SOURCE_MODULE_NAME = Path(mapped_module.__file__).name
"""
Name of the module of a package whose classes its interface maps.
"""


@pytest.fixture
def checkout(tmp_path: Path) -> Path:
    """
    A git checkout of two packages whose interfaces hold content of a previous run,
    beside the mapping engine every one of their generators reads.
    """
    for package_name in PACKAGE_NAMES:
        package_root = tmp_path / package_name
        (package_root / "scripts").mkdir(parents=True)
        shutil.copy(
            Path(generate_orm.__file__),
            package_root / "scripts" / "generate_orm.py",
        )
        interface = generate_orm.interface_of(package_root)
        interface.parent.mkdir(parents=True)
        interface.write_text(STALE_INTERFACE_CONTENT, encoding="utf-8")
        shutil.copy(
            Path(mapped_module.__file__), interface.parent.parent / SOURCE_MODULE_NAME
        )

    for source_folder in orm_interfaces.MAPPING_ENGINE_SOURCE_FOLDERS:
        mapping_engine = tmp_path / source_folder
        mapping_engine.mkdir(parents=True)
        shutil.copy(Path(mapped_module.__file__), mapping_engine / SOURCE_MODULE_NAME)

    subprocess.run(["git", "init"], cwd=tmp_path, check=True, capture_output=True)
    subprocess.run(
        ["git", "add", "--all"], cwd=tmp_path, check=True, capture_output=True
    )
    return tmp_path


@pytest.fixture
def workspace(checkout: Path) -> WorkspaceOrmInterfaces:
    """
    The ORM interfaces of the checkout under test.
    """
    return WorkspaceOrmInterfaces(
        tuple(OrmInterface(package_name, checkout) for package_name in PACKAGE_NAMES)
    )


def tracked_interfaces(repository_root: Path) -> set[str]:
    """
    Read which ORM interfaces of a checkout git tracks.

    :param repository_root: Root of the checkout.
    :return: The repository-relative paths of the tracked interfaces.
    """
    listing = subprocess.run(
        ["git", "ls-files", "--", f"*/{INTERFACE_FILE_NAME}"],
        cwd=repository_root,
        check=True,
        capture_output=True,
        text=True,
    )
    return set(listing.stdout.splitlines())


def git_ignores(repository_root: Path, path: Path) -> bool:
    """
    Ask git whether a checkout's ignore rules cover a path.

    :param repository_root: Root of the checkout.
    :param path: Path to ask about.
    :return: Whether git ignores it.
    """
    return (
        subprocess.run(
            ["git", "check-ignore", "--quiet", str(path)],
            cwd=repository_root,
        ).returncode
        == 0
    )


# %% regeneration


def test_regeneration_runs_the_generators_in_dependency_order(
    workspace: WorkspaceOrmInterfaces, checkout: Path
):
    workspace.regenerate()

    records = generate_orm.read_generation_log(checkout)
    assert [record.package_name for record in records] == list(PACKAGE_NAMES)


def test_regeneration_clears_every_interface_before_generating_any(
    workspace: WorkspaceOrmInterfaces, checkout: Path
):
    workspace.regenerate()

    records = generate_orm.read_generation_log(checkout)
    assert records[0].generated_packages == []
    assert records[1].generated_packages == [PACKAGE_NAMES[0]]


def test_regeneration_fills_every_interface(workspace: WorkspaceOrmInterfaces):
    workspace.regenerate()

    for interface in workspace.interfaces:
        assert interface.path.read_text(
            encoding="utf-8"
        ) == generate_orm.interface_content(interface.package_name)


# %% one interpreter for every generator


def test_the_generators_share_one_interpreter(
    workspace: WorkspaceOrmInterfaces, checkout: Path
):
    """
    Every generator runs in the same interpreter, so what one of them imports is
    already imported for the ones after it.
    """
    workspace.regenerate()

    records = generate_orm.read_generation_log(checkout)
    assert len({record.process_id for record in records}) == 1


def test_the_build_stays_out_of_the_calling_interpreter(
    workspace: WorkspaceOrmInterfaces, checkout: Path
):
    """
    The interpreter the generators share is not the one that asked for the build, which
    is what keeps the packages they import out of it.
    """
    workspace.regenerate()

    records = generate_orm.read_generation_log(checkout)
    assert records[0].process_id != os.getpid()


def test_a_generator_starts_from_the_logging_the_build_was_launched_with(
    workspace: WorkspaceOrmInterfaces, checkout: Path
):
    """
    A generator configures logging for its own run, so what it leaves behind does not
    reach the generators after it.
    """
    workspace.regenerate()

    records = generate_orm.read_generation_log(checkout)
    assert [record.root_logger_handlers for record in records] == [1] * len(
        PACKAGE_NAMES
    )


# %% incomplete checkouts


def test_missing_generator_names_its_package(workspace: WorkspaceOrmInterfaces):
    incomplete = workspace.interfaces[-1]
    incomplete.generator.unlink()

    with pytest.raises(MissingORMGeneratorError) as error:
        workspace.regenerate()

    assert error.value.package_name == incomplete.package_name
    assert error.value.path == incomplete.generator


# %% this repository


def test_every_workspace_package_has_a_generator():
    without_generator = [
        interface.package_name
        for interface in WORKSPACE_ORM_INTERFACES.interfaces
        if not interface.generator.exists()
    ]
    assert without_generator == []


def test_this_repository_tracks_no_generated_interface():
    assert tracked_interfaces(REPOSITORY_ROOT) == set()


def test_this_repository_ignores_every_generated_interface():
    not_ignored = [
        interface.package_name
        for interface in WORKSPACE_ORM_INTERFACES.interfaces
        if not git_ignores(REPOSITORY_ROOT, interface.path)
    ]
    assert not_ignored == []


def test_this_repository_ignores_a_generated_interface_outside_a_workspace_package():
    krrood_test_dataset_interface = (
        REPOSITORY_ROOT / "test" / "krrood_test" / "dataset" / INTERFACE_FILE_NAME
    )

    assert git_ignores(REPOSITORY_ROOT, krrood_test_dataset_interface)


# %% the interfaces a package needs


class BranchingPackage(StrEnum):
    """
    Packages of a workspace whose interfaces build on one another along two branches.
    """

    BASE = "base"
    """
    A package whose interface builds on no other.
    """

    MIDDLE = "middle"
    """
    A package whose interface builds on :attr:`BASE`.
    """

    TOP = "top"
    """
    A package whose interface builds on :attr:`MIDDLE`.
    """

    SIDE = "side"
    """
    A package whose interface builds on :attr:`BASE`, beside :attr:`MIDDLE`.
    """


def branching_interfaces(*package_names: BranchingPackage) -> WorkspaceOrmInterfaces:
    """
    :return: The interfaces of the packages, in the order given, whose generators are
        in the test dataset.
    """
    workspace_root = Path(dataset.__file__).parent / "branching_workspace"
    return WorkspaceOrmInterfaces(
        tuple(
            OrmInterface(package_name, workspace_root) for package_name in package_names
        )
    )


@pytest.fixture
def branching_workspace() -> WorkspaceOrmInterfaces:
    """
    The interfaces of every :class:`BranchingPackage`.
    """
    return branching_interfaces(*BranchingPackage)


def test_interfaces_are_built_after_the_interfaces_they_build_on():
    """
    The build order follows the dependencies, whatever order the interfaces are given
    in.
    """
    workspace = branching_interfaces(
        BranchingPackage.TOP, BranchingPackage.MIDDLE, BranchingPackage.BASE
    )

    assert package_names_of_workspace(workspace) == [
        BranchingPackage.BASE,
        BranchingPackage.MIDDLE,
        BranchingPackage.TOP,
    ]


def test_interfaces_building_on_an_interface_outside_the_workspace_cannot_be_ordered():
    """
    An interface whose generator imports an interface the workspace lacks has no place
    in the build order.
    """
    workspace = branching_interfaces(BranchingPackage.TOP, BranchingPackage.MIDDLE)

    with pytest.raises(UnorderableOrmInterfacesError):
        workspace.interfaces_in_build_order


def test_an_interface_builds_on_the_interfaces_its_generator_imports(
    branching_workspace: WorkspaceOrmInterfaces,
):
    """
    The packages an interface builds on are those whose generated interface its
    generator imports.
    """
    dependencies_by_package = {
        interface.package_name: interface.dependencies
        for interface in branching_workspace.interfaces
    }

    assert dependencies_by_package == {
        BranchingPackage.BASE: (),
        BranchingPackage.MIDDLE: (BranchingPackage.BASE,),
        BranchingPackage.SIDE: (BranchingPackage.BASE,),
        BranchingPackage.TOP: (BranchingPackage.MIDDLE,),
    }


def package_names_of_workspace(workspace: WorkspaceOrmInterfaces) -> list[str]:
    """
    :return: The names of the packages whose interfaces the workspace holds, in build
        order.
    """
    return [interface.package_name for interface in workspace.interfaces_in_build_order]


def test_a_package_needs_its_interface_and_every_one_it_builds_on(
    branching_workspace: WorkspaceOrmInterfaces,
):
    """
    The interfaces a package builds on come along, through every level, in build order.
    """
    needed = branching_workspace.interfaces_needed_by_package(BranchingPackage.TOP)

    assert package_names_of_workspace(needed) == [
        BranchingPackage.BASE,
        BranchingPackage.MIDDLE,
        BranchingPackage.TOP,
    ]


def test_a_package_does_not_need_the_interfaces_of_another_branch(
    branching_workspace: WorkspaceOrmInterfaces,
):
    """
    An interface building on the same base, but not on the package, stays out.
    """
    needed = branching_workspace.interfaces_needed_by_package(BranchingPackage.SIDE)

    assert package_names_of_workspace(needed) == [
        BranchingPackage.BASE,
        BranchingPackage.SIDE,
    ]


def test_a_package_building_on_nothing_needs_only_its_own_interface(
    branching_workspace: WorkspaceOrmInterfaces,
):
    """
    The interfaces built on a package are not needed to read it.
    """
    needed = branching_workspace.interfaces_needed_by_package(BranchingPackage.BASE)

    assert package_names_of_workspace(needed) == [BranchingPackage.BASE]


def test_a_package_without_an_interface_is_rejected(
    branching_workspace: WorkspaceOrmInterfaces,
):
    """
    Asking for a package the workspace holds no interface of is an error, not an
    empty build.
    """
    with pytest.raises(UnknownOrmPackageError):
        branching_workspace.interfaces_needed_by_package("package_without_an_interface")


# %% what a build lets through to the terminal


@pytest.fixture
def failing_workspace(checkout: Path) -> WorkspaceOrmInterfaces:
    """
    The interfaces of a checkout whose first generator fails.
    """
    shutil.copy(
        Path(failing_generate_orm.__file__),
        checkout / PACKAGE_NAMES[0] / "scripts" / "generate_orm.py",
    )
    return WorkspaceOrmInterfaces(
        tuple(OrmInterface(package_name, checkout) for package_name in PACKAGE_NAMES)
    )


def test_a_quiet_build_keeps_the_generator_output_off_the_terminal(
    workspace: WorkspaceOrmInterfaces, capfd
):
    workspace.regenerate()

    assert generate_orm.PROGRESS_LINE not in capfd.readouterr().out


def test_a_build_showing_generator_output_lets_it_through(
    workspace: WorkspaceOrmInterfaces, capfd
):
    workspace.regenerate(show_generator_output=True)

    assert capfd.readouterr().out.count(generate_orm.PROGRESS_LINE) == len(
        PACKAGE_NAMES
    )


def test_a_failing_generator_reports_what_it_wrote(
    failing_workspace: WorkspaceOrmInterfaces,
):
    with pytest.raises(OrmGenerationFailedError) as failure:
        failing_workspace.regenerate()

    assert failing_generate_orm.DIAGNOSTIC in str(failure.value)
    assert failure.value.package_name == PACKAGE_NAMES[0]


def test_a_failing_generator_is_named_when_it_had_the_terminal(checkout: Path):
    """
    A build writing to the terminal reports nothing back, so the package that failed is
    read off the interface it left unwritten.
    """
    shutil.copy(
        Path(failing_generate_orm.__file__),
        checkout / PACKAGE_NAMES[-1] / "scripts" / "generate_orm.py",
    )
    workspace = WorkspaceOrmInterfaces(
        tuple(OrmInterface(package_name, checkout) for package_name in PACKAGE_NAMES)
    )

    with pytest.raises(OrmGenerationFailedError) as failure:
        workspace.regenerate(show_generator_output=True)

    assert failure.value.package_name == PACKAGE_NAMES[-1]


def test_the_bar_counts_every_class_of_every_interface(
    workspace: WorkspaceOrmInterfaces, monkeypatch
):
    advanced = []
    monkeypatch.setattr(
        orm_interfaces.BuildProgress,
        "advance",
        lambda self, report: advanced.append(report.class_name),
    )

    workspace.regenerate()

    assert advanced == list(generate_orm.MAPPED_CLASS_NAMES) * len(PACKAGE_NAMES)


def test_the_bar_learns_how_many_classes_an_interface_holds(
    workspace: WorkspaceOrmInterfaces,
):
    progress = orm_interfaces.BuildProgress(len(PACKAGE_NAMES), False)
    with progress:
        workspace.run_reporting_to(progress)

        assert progress.bar.total == len(generate_orm.MAPPED_CLASS_NAMES)
        assert progress.bar.n == len(generate_orm.MAPPED_CLASS_NAMES)


def test_the_interfaces_done_are_counted_as_the_build_goes(
    workspace: WorkspaceOrmInterfaces,
):
    progress = orm_interfaces.BuildProgress(len(PACKAGE_NAMES), False)
    with progress:
        workspace.run_reporting_to(progress)

    assert progress.completed_interfaces == len(PACKAGE_NAMES)


def test_a_build_showing_generator_output_keeps_no_bar(
    workspace: WorkspaceOrmInterfaces,
):
    progress = orm_interfaces.BuildProgress(len(PACKAGE_NAMES), True)
    with progress:
        assert progress.bar is None


# %% interfaces their sources have outrun


def change_after_the_build(path: Path, interface: OrmInterface) -> None:
    """
    Give a file a modification time later than the interface's, as an edit made after
    the build would.

    :param path: The file to mark as changed.
    :param interface: The interface the change comes after.
    """
    changed_at = interface.path.stat().st_mtime + 1
    os.utime(path, (changed_at, changed_at))


def test_a_freshly_built_checkout_is_current(workspace: WorkspaceOrmInterfaces):
    workspace.regenerate()

    assert workspace.is_outdated is False


def test_a_missing_interface_is_outdated(workspace: WorkspaceOrmInterfaces):
    workspace.regenerate()
    workspace.interfaces[-1].remove()

    assert workspace.interfaces[-1].is_outdated is True
    assert workspace.is_outdated is True


def test_a_changed_module_outdates_the_interface_of_its_package(
    workspace: WorkspaceOrmInterfaces,
):
    workspace.regenerate()
    interface = workspace.interfaces[0]

    change_after_the_build(interface.sources / SOURCE_MODULE_NAME, interface)

    assert interface.is_outdated is True


def test_a_changed_module_leaves_the_interface_of_another_package_alone(
    workspace: WorkspaceOrmInterfaces,
):
    workspace.regenerate()
    interface = workspace.interfaces[0]

    change_after_the_build(interface.sources / SOURCE_MODULE_NAME, interface)

    assert workspace.interfaces[-1].is_outdated is False


def test_a_changed_mapping_engine_module_outdates_every_interface(
    workspace: WorkspaceOrmInterfaces,
):
    """
    Every generator reads the library that maps the classes, so a change to it leaves no
    interface of the checkout current.
    """
    workspace.regenerate()
    interface = workspace.interfaces[0]

    for source_folder in interface.mapping_engine_sources:
        change_after_the_build(source_folder / SOURCE_MODULE_NAME, interface)

    assert all(member.is_outdated for member in workspace.interfaces)
    assert workspace.is_outdated is True


def test_a_changed_generator_outdates_the_interface_it_writes(
    workspace: WorkspaceOrmInterfaces,
):
    workspace.regenerate()
    interface = workspace.interfaces[0]

    change_after_the_build(interface.generator, interface)

    assert interface.is_outdated is True


def test_a_missing_generator_leaves_its_interface_outdated(
    workspace: WorkspaceOrmInterfaces,
):
    """
    A checkout that cannot build an interface is never current, so the build runs and
    reports the missing generator rather than being skipped.
    """
    workspace.regenerate()
    interface = workspace.interfaces[0]
    interface.generator.unlink()

    assert interface.is_outdated is True
