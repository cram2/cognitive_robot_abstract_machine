"""
Tests that CI runs the tests a change can affect, and only those.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import pytest

from cognitive_robot_abstract_machine.ci_jobs import (
    CIJob,
    CIJobLayout,
    CIJobMatrix,
    DocumentationPath,
    LibraryTests,
    MatrixKey,
    Workspace,
    WorkspaceMember,
)
from cognitive_robot_abstract_machine.exceptions import UnknownSharedJobLibraryError
from cognitive_robot_abstract_machine.orm_interfaces import REPOSITORY_ROOT
from cognitive_robot_abstract_machine.repository_files import (
    RepositoryDirectory,
    RepositoryFile,
)

from . import dataset

# %% fixtures


@pytest.fixture
def dependent_workspace_root() -> Path:
    """
    The root of a workspace whose members ``base``, ``middle`` and ``top`` depend on
    each other in a chain, next to ``side``, which depends on none.
    """
    return Path(dataset.__file__).parent / "dependent_workspace"


@pytest.fixture
def dependent_workspace(dependent_workspace_root: Path) -> Workspace:
    """
    The workspace of :func:`dependent_workspace_root`.
    """
    return Workspace.from_repository_root(repository_root=dependent_workspace_root)


@pytest.fixture
def repository_workspace() -> Workspace:
    """
    The workspace of this repository.
    """
    return Workspace.from_repository_root(repository_root=REPOSITORY_ROOT)


def library_names(tests: tuple[LibraryTests, ...]) -> list[str]:
    """
    :return: The names of the libraries the tests belong to, in their order.
    """
    return [library_tests.library_name for library_tests in tests]


# %% reading the workspace


def test_repository_tests_are_those_of_no_member(
    dependent_workspace: Workspace,
) -> None:
    """
    A test directory named after no member holds tests of the repository itself.
    """
    assert library_names(dependent_workspace.repository_tests) == ["repository"]


def test_member_dependencies_are_read_by_canonical_name(
    dependent_workspace: Workspace,
) -> None:
    """
    A dependency is known by its distribution name, without its extras and version.
    """
    top = next(
        member
        for member in dependent_workspace.members
        if member.distribution_name == "top"
    )
    assert top.dependency_names == frozenset({"middle"})


# %% affected tests


def test_member_change_affects_the_members_depending_on_it(
    dependent_workspace: Workspace,
) -> None:
    """
    A change to a member affects its tests, those of the members depending on it and
    those of the repository, but not those of the members it depends on.
    """
    affected = dependent_workspace.affected_tests(
        changed_paths=[Path("middle") / "src" / "middle" / "module.py"]
    )
    assert library_names(affected) == ["middle", "top", "repository"]


def test_change_to_member_tests_affects_the_member(
    dependent_workspace: Workspace,
) -> None:
    """
    A change to a member's tests affects the member as a change to its code does.
    """
    affected = dependent_workspace.affected_tests(
        changed_paths=[RepositoryDirectory.TESTS / "side_test" / "test_side.py"]
    )
    assert library_names(affected) == ["side", "repository"]


def test_documentation_change_affects_only_the_repository_tests(
    dependent_workspace: Workspace,
) -> None:
    """
    A change to the repository's documentation affects no member.
    """
    affected = dependent_workspace.affected_tests(
        changed_paths=[DocumentationPath.DOCUMENTATION / "index.md"]
    )
    assert affected == dependent_workspace.repository_tests


def test_change_to_repository_tests_affects_only_the_repository_tests(
    dependent_workspace: Workspace,
) -> None:
    """
    A change to the repository's own tests affects no member.
    """
    affected = dependent_workspace.affected_tests(
        changed_paths=[
            dependent_workspace.repository_tests[0].directory / "test_repository.py"
        ]
    )
    assert affected == dependent_workspace.repository_tests


def test_change_outside_every_member_affects_all_tests(
    dependent_workspace: Workspace,
) -> None:
    """
    A change to a file no member owns can affect every test.
    """
    affected = dependent_workspace.affected_tests(
        changed_paths=[RepositoryFile.PYPROJECT]
    )
    assert affected == dependent_workspace.all_tests


def test_members_depending_on_each_other_in_a_cycle_are_all_affected() -> None:
    """
    Members depending on each other in a cycle are found without looping forever.
    """
    first = WorkspaceMember(
        directory=Path("first"),
        distribution_name="first",
        dependency_names=frozenset({"second"}),
    )
    second = WorkspaceMember(
        directory=Path("second"),
        distribution_name="second",
        dependency_names=frozenset({"first"}),
    )
    workspace = Workspace(members=(first, second), repository_tests=())
    assert workspace.members_depending_on(members=[first]) == {first, second}


def test_semantic_digital_twin_change_does_not_affect_krrood_tests(
    repository_workspace: Workspace,
) -> None:
    """
    :mod:`krrood` does not depend on :mod:`semantic_digital_twin`, so its tests stay
    unaffected by a change there.
    """
    affected = repository_workspace.affected_tests(
        changed_paths=[Path("semantic_digital_twin") / "src" / "module.py"]
    )
    assert "krrood" not in library_names(affected)


def test_member_tests_import_only_tests_of_members_they_depend_on(
    repository_workspace: Workspace,
) -> None:
    """
    A change to a member's tests affects only that member and the members depending on
    it, so tests importing the tests of any other member would miss that change.
    """
    members_by_library_name = {
        member.tests.library_name: member for member in repository_workspace.members
    }
    test_import = re.compile(
        r"^\s*(?:from|import)\s+(?:test\.|\.\.+)(\w+)"
        + re.escape(LibraryTests.directory_suffix)
        + r"\b",
        re.MULTILINE,
    )
    unaffected_imports = set()
    for member in repository_workspace.members:
        for test_file in (REPOSITORY_ROOT / member.tests.directory).rglob("*.py"):
            for imported in test_import.findall(test_file.read_text(errors="ignore")):
                imported_member = members_by_library_name.get(imported)
                if imported_member is None:
                    continue
                if member not in repository_workspace.members_depending_on(
                    members=[imported_member]
                ):
                    unaffected_imports.add((member.tests.library_name, imported))
    assert unaffected_imports == set()


# %% jobs


def test_shared_job_runs_only_its_affected_libraries(
    dependent_workspace_root: Path, dependent_workspace: Workspace
) -> None:
    """
    A shared job runs the tests of its affected libraries only, and every affected
    library it does not share gets a job of its own.
    """
    layout = CIJobLayout.from_toml(
        text=(dependent_workspace_root / RepositoryFile.CI_JOBS.name).read_text()
    )
    jobs = layout.jobs_running(
        affected_tests=dependent_workspace.affected_tests(
            changed_paths=[Path("middle") / "module.py"]
        ),
        workspace=dependent_workspace,
    )
    assert jobs == (
        CIJob(name="top", library_names=("top",)),
        CIJob(name="chain", library_names=("middle",)),
        CIJob(name="rules", library_names=("repository",), check_rules=True),
    )


def test_shared_job_checking_rules_runs_without_affected_libraries(
    dependent_workspace_root: Path, dependent_workspace: Workspace
) -> None:
    """
    The job checking the rules runs even when the change affects none of its libraries.
    """
    layout = CIJobLayout.from_toml(
        text=(dependent_workspace_root / RepositoryFile.CI_JOBS.name).read_text()
    )
    jobs = layout.jobs_running(affected_tests=(), workspace=dependent_workspace)
    assert jobs == (CIJob(name="rules", library_names=(), check_rules=True),)


def test_shared_job_naming_a_library_without_tests_is_rejected(
    dependent_workspace_root: Path, dependent_workspace: Workspace
) -> None:
    """
    A shared job naming a library without tests is a mistake in the layout.
    """
    layout = CIJobLayout.from_toml(
        text=(dependent_workspace_root / "ci_jobs_unknown_library.toml").read_text()
    )
    with pytest.raises(UnknownSharedJobLibraryError):
        layout.jobs_running(
            affected_tests=dependent_workspace.all_tests,
            workspace=dependent_workspace,
        )


def test_repository_jobs_run_every_library_once(
    repository_workspace: Workspace,
) -> None:
    """
    The repository's job layout runs the tests of every library in exactly one job.
    """
    layout = CIJobLayout.from_toml(
        text=(REPOSITORY_ROOT / RepositoryFile.CI_JOBS).read_text()
    )
    jobs = layout.jobs_running(
        affected_tests=repository_workspace.all_tests, workspace=repository_workspace
    )
    run_library_names = [name for job in jobs for name in job.library_names]
    assert sorted(run_library_names) == sorted(
        library_names(repository_workspace.all_tests)
    )


def test_matrix_lists_the_libraries_of_a_job_as_json() -> None:
    """
    The libraries of a job are a JSON list, as the reusable test workflow reads them.
    """
    job = CIJob(name="chain", library_names=("base", "middle"))
    matrix = json.loads(CIJobMatrix(jobs=(job,)).to_json())
    [entry] = matrix[MatrixKey.INCLUDE]
    assert json.loads(entry[MatrixKey.LIBRARIES]) == list(job.library_names)
