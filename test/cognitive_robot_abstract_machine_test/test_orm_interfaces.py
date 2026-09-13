"""
Tests for building the ORM interfaces a checkout needs before it can persist objects.
"""

from __future__ import annotations

import logging
import os
import shutil
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path

import pytest
from typing_extensions import List, Optional, Set, Tuple

from cognitive_robot_abstract_machine import orm_interfaces
from cognitive_robot_abstract_machine.exceptions import (
    MissingORMGeneratorError,
    OrmGenerationFailedError,
    OrmImportFailedError,
)
from cognitive_robot_abstract_machine.orm_interfaces import (
    InterfaceLocation,
    OrmInterface,
    REPOSITORY_ROOT,
    WORKSPACE_ORM_INTERFACES,
    WorkspaceOrmInterfaces,
)

from .dataset import failing_generate_orm, generate_orm, mapped_module

# %% a checkout of packages that generate an interface

PACKAGE_NAMES: Tuple[str, ...] = ("upstream", "downstream")
"""
Packages of the checkout under test, in dependency order.
"""

PREVIOUS_INTERFACE_CONTENT = "# interface of a previous run\n"
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
    A git checkout of two packages whose interfaces hold content of a previous run.
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
        interface.write_text(PREVIOUS_INTERFACE_CONTENT, encoding="utf-8")
        shutil.copy(
            Path(mapped_module.__file__), interface.parent.parent / SOURCE_MODULE_NAME
        )

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


def tracked_interfaces(repository_root: Path) -> Set[str]:
    """
    Read which ORM interfaces of a checkout git tracks.

    :param repository_root: Root of the checkout.
    :return: The repository-relative paths of the tracked interfaces.
    """
    listing = subprocess.run(
        ["git", "ls-files", "--", f"*/{InterfaceLocation.FILE_NAME}"],
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
        REPOSITORY_ROOT
        / "test"
        / "krrood_test"
        / "dataset"
        / InterfaceLocation.FILE_NAME
    )

    assert git_ignores(REPOSITORY_ROOT, krrood_test_dataset_interface)


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


# %% interfaces that no longer match their classes

DATASET = Path(mapped_module.__file__).parent
"""
Folder holding the modules a checkout under test is built from.
"""

REMOVED_CLASS_INTERFACE = DATASET / "interface_of_a_removed_class.py"
"""
An interface of a package that no longer holds a class it maps.
"""

FAILING_INTERFACE = DATASET / "failing_interface.py"
"""
An interface that fails to import for a reason other than a stale mapping.
"""


def leave_behind(stand_in: Path, interface: OrmInterface) -> None:
    """
    Put an interface of a previous build in a package's place.

    :param stand_in: The module standing in for what a build wrote.
    :param interface: The interface of the package it is left in.
    """
    shutil.copy(stand_in, interface.path)


def test_a_freshly_built_checkout_imports(workspace: WorkspaceOrmInterfaces):
    workspace.regenerate()

    assert workspace.stale_interface() is None


def test_a_missing_interface_is_stale(workspace: WorkspaceOrmInterfaces):
    workspace.regenerate()
    missing = workspace.interfaces[-1]
    missing.remove()

    stale = workspace.stale_interface()

    assert stale.module_name == missing.module_name
    assert stale.error_type == ModuleNotFoundError.__name__


def test_an_interface_of_a_removed_class_is_stale(workspace: WorkspaceOrmInterfaces):
    """
    A generated interface reaches every class it maps as an attribute of its module, so
    one that has been renamed or removed since the build fails the import as a missing
    attribute rather than as a missing module.
    """
    workspace.regenerate()
    outrun = workspace.interfaces[0]
    leave_behind(REMOVED_CLASS_INTERFACE, outrun)

    stale = workspace.stale_interface()

    assert stale.module_name == outrun.module_name
    assert stale.error_type == AttributeError.__name__


def test_the_first_interface_that_does_not_import_is_the_one_reported(
    workspace: WorkspaceOrmInterfaces,
):
    workspace.regenerate()
    for interface in workspace.interfaces:
        leave_behind(REMOVED_CLASS_INTERFACE, interface)

    stale = workspace.stale_interface()

    assert stale.module_name == workspace.interfaces[0].module_name


def test_an_interface_failing_for_another_reason_reports_what_it_wrote(
    workspace: WorkspaceOrmInterfaces,
):
    """
    An interface failing for anything but a stale mapping is a broken checkout rather
    than one to rebuild, so the attempt says what happened instead of answering with a
    build.
    """
    workspace.regenerate()
    failing = workspace.interfaces[0]
    leave_behind(FAILING_INTERFACE, failing)

    with pytest.raises(OrmImportFailedError) as failure:
        workspace.stale_interface()

    assert ValueError.__name__ in failure.value.output
    assert str(failing.path) in failure.value.output


def test_the_import_attempt_stays_out_of_the_calling_interpreter(
    workspace: WorkspaceOrmInterfaces,
):
    """
    The interfaces are imported in an interpreter of their own, so a build following a
    failed attempt cannot leave a stale interface behind in the one that asked for it.
    """
    workspace.regenerate()

    workspace.stale_interface()

    assert [
        interface.module_name
        for interface in workspace.interfaces
        if interface.module_name in sys.modules
    ] == []


@pytest.fixture
def reported_staleness(caplog, monkeypatch) -> logging.Handler:
    """
    What a build writes about the interfaces it found stale.

    ..note:: Whether the logger under test propagates to the root logger depends on the
        logger class the ROS overlay installs, and the capturing handler sits on the root
        logger already. Attaching it to the logger under test and turning propagation off
        records every report exactly once, with or without the overlay.

    :return: The handler holding the records, empty until a build writes one.
    """
    monkeypatch.setattr(orm_interfaces.logger, "propagate", False)
    orm_interfaces.logger.addHandler(caplog.handler)
    yield caplog.handler
    orm_interfaces.logger.removeHandler(caplog.handler)


def test_the_stale_interface_is_reported_through_logging(
    workspace: WorkspaceOrmInterfaces, reported_staleness: logging.Handler
):
    """
    The report explains a build the run did not ask for, so it goes to logging rather
    than to whatever the caller has its output pointed at.
    """
    workspace.regenerate()
    leave_behind(REMOVED_CLASS_INTERFACE, workspace.interfaces[0])

    stale = workspace.stale_interface()

    assert [record.getMessage() for record in reported_staleness.records] == [
        stale.report()
    ]


def test_a_checkout_that_imports_is_reported_on_at_all(
    workspace: WorkspaceOrmInterfaces, reported_staleness: logging.Handler
):
    workspace.regenerate()

    workspace.stale_interface()

    assert reported_staleness.records == []
