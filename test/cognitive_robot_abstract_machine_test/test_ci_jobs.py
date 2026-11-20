"""
Tests that CI runs the tests a change can affect, and only those.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from cognitive_robot_abstract_machine.ci_jobs import (
    CIJob,
    CIJobLayout,
    CIJobMatrix,
    DocumentationPath,
    LibraryTests,
    MatrixKey,
    RepositoryTestSuite,
    Workspace,
    WorkspaceMember,
)
from cognitive_robot_abstract_machine.exceptions import UnknownCILibraryError
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
    each other in a chain, next to ``side``, which depends on none, and ``family``,
    whose members ``family-core`` and ``family-extra`` nest in its directory.
    """
    return Path(dataset.__file__).parent / "dependent_workspace"


@pytest.fixture
def dependent_layout(dependent_workspace_root: Path) -> CIJobLayout:
    """
    The CI job layout of :func:`dependent_workspace_root`.
    """
    return CIJobLayout.from_toml(
        text=(dependent_workspace_root / RepositoryFile.CI_JOBS.name).read_text()
    )


@pytest.fixture
def dependent_tests(
    dependent_workspace_root: Path, dependent_layout: CIJobLayout
) -> RepositoryTestSuite:
    """
    The tests of :func:`dependent_workspace_root`.
    """
    return RepositoryTestSuite(
        repository_root=dependent_workspace_root,
        workspace=Workspace.from_repository_root(
            repository_root=dependent_workspace_root
        ),
        assigned_tests=dependent_layout.assigned_tests,
    )


@pytest.fixture
def repository_layout() -> CIJobLayout:
    """
    The CI job layout of this repository.
    """
    return CIJobLayout.from_toml(
        text=(REPOSITORY_ROOT / RepositoryFile.CI_JOBS).read_text()
    )


@pytest.fixture
def repository_tests(repository_layout: CIJobLayout) -> RepositoryTestSuite:
    """
    The tests of this repository.
    """
    return RepositoryTestSuite(
        repository_root=REPOSITORY_ROOT,
        workspace=Workspace.from_repository_root(repository_root=REPOSITORY_ROOT),
        assigned_tests=repository_layout.assigned_tests,
    )


def paths_by_library(tests: tuple[LibraryTests, ...]) -> dict[str, list[str]]:
    """
    :return: The paths each library's tests run, by library, as text.
    """
    return {
        library_tests.library_name: [str(path) for path in library_tests.test_paths]
        for library_tests in tests
    }


def library_test_directory(library_name: str) -> str:
    """
    :return: The test directory named after the library, as text.
    """
    return str(
        RepositoryDirectory.TESTS / (library_name + LibraryTests.directory_suffix)
    )


def family_test(file_name: str) -> str:
    """
    :return: The path of a test of the family members, as text.
    """
    return str(Path(library_test_directory("family")) / file_name)


# %% reading the workspace


def test_member_dependencies_are_read_by_canonical_name(
    dependent_tests: RepositoryTestSuite,
) -> None:
    """
    A dependency is known by its distribution name, without its extras and version.
    """
    top = next(
        member
        for member in dependent_tests.workspace.members
        if member.distribution_name == "top"
    )
    assert top.dependency_names == frozenset({"middle"})


def test_path_belongs_to_the_innermost_member(
    dependent_tests: RepositoryTestSuite,
) -> None:
    """
    A path in a member nested in another one's directory belongs to the nested member.
    """
    owner = dependent_tests.workspace.owner(
        path=Path("family") / "family-core" / "pyproject.toml"
    )
    assert owner.distribution_name == "family-core"


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
    workspace = Workspace(members=(first, second))
    assert workspace.members_depending_on(members=[first]) == {first, second}


# %% libraries of the tests


def test_test_of_nested_members_belongs_to_the_least_dependent_covering_member(
    dependent_tests: RepositoryTestSuite,
) -> None:
    """
    A test in a test directory shared by nested members belongs to the nested member
    depending on the fewest members while depending on all the test imports, unless it
    is assigned by hand.
    """
    assert dependent_tests.test_libraries == {
        "test.base_test.test_chain_base": "base",
        "test.middle_test.test_chain_middle": "middle",
        "test.top_test.test_chain_top": "top",
        "test.side_test.test_chain_side": "side",
        "test.family_test.test_family_core": "family-core",
        "test.family_test.test_family_extra": "family-extra",
        "test.family_test.test_family_both": "family",
        "test.family_test.test_family_layout": "family",
        "test.repository_test.test_chain_repository": "repository",
    }


# %% affected tests


def test_member_change_affects_the_tests_of_the_members_depending_on_it(
    dependent_tests: RepositoryTestSuite,
) -> None:
    """
    A change to a member affects its tests, those of the members depending on it and
    those of the repository, but not those of the members it depends on.
    """
    affected = dependent_tests.affected_tests(
        changed_paths=[Path("middle") / "src" / "middle" / "__init__.py"]
    )
    assert paths_by_library(affected) == {
        "middle": [library_test_directory("middle")],
        "top": [library_test_directory("top")],
        "repository": [library_test_directory("repository")],
    }


def test_nested_member_change_affects_only_the_tests_depending_on_it(
    dependent_tests: RepositoryTestSuite,
) -> None:
    """
    A change to a nested member affects the tests in the shared test directory whose
    library depends on it, not the tests of the member it depends on.
    """
    affected = dependent_tests.affected_tests(
        changed_paths=[Path("family") / "family-extra" / "src" / "family" / "extra.py"]
    )
    assert paths_by_library(affected) == {
        "family": [
            family_test("test_family_both.py"),
            family_test("test_family_layout.py"),
        ],
        "family-extra": [family_test("test_family_extra.py")],
        "repository": [library_test_directory("repository")],
    }


def test_change_to_a_test_helper_affects_the_tests_loading_it(
    dependent_tests: RepositoryTestSuite,
) -> None:
    """
    A change to a test module affects the tests loading it, in other test directories
    too, and no other test.
    """
    affected = dependent_tests.affected_tests(
        changed_paths=[RepositoryDirectory.TESTS / "base_test" / "chain_helpers.py"]
    )
    assert paths_by_library(affected) == {
        "side": [library_test_directory("side")],
        "repository": [library_test_directory("repository")],
    }


def test_change_to_test_data_affects_every_test_of_its_directory(
    dependent_tests: RepositoryTestSuite,
) -> None:
    """
    A changed file of a test directory that is not a module can be read by any test in
    it.
    """
    affected = dependent_tests.affected_tests(
        changed_paths=[Path(library_test_directory("family")) / "data" / "sample.json"]
    )
    assert paths_by_library(affected) == {
        "family": [
            family_test("test_family_both.py"),
            family_test("test_family_layout.py"),
        ],
        "family-core": [family_test("test_family_core.py")],
        "family-extra": [family_test("test_family_extra.py")],
        "repository": [library_test_directory("repository")],
    }


def test_change_to_a_test_configuration_affects_every_test_below_it(
    dependent_tests: RepositoryTestSuite,
) -> None:
    """
    A changed ``conftest.py`` affects every test below its directory.
    """
    affected = dependent_tests.affected_tests(
        changed_paths=[Path(family_test("conftest.py"))]
    )
    assert set(paths_by_library(affected)) == {
        "family",
        "family-core",
        "family-extra",
        "repository",
    }


def test_documentation_change_affects_only_the_repository_tests(
    dependent_tests: RepositoryTestSuite,
) -> None:
    """
    A change to the repository's documentation affects no member's tests.
    """
    affected = dependent_tests.affected_tests(
        changed_paths=[DocumentationPath.DOCUMENTATION / "index.md"]
    )
    assert paths_by_library(affected) == {
        "repository": [library_test_directory("repository")]
    }


def test_change_outside_every_member_affects_all_tests(
    dependent_tests: RepositoryTestSuite,
) -> None:
    """
    A change to a file no member owns can affect every test.
    """
    affected = dependent_tests.affected_tests(changed_paths=[RepositoryFile.PYPROJECT])
    assert affected == dependent_tests.all_tests()


def test_semantic_digital_twin_change_does_not_affect_krrood_core_tests(
    repository_tests: RepositoryTestSuite,
) -> None:
    """
    No krrood distribution depends on :mod:`semantic_digital_twin`, so the tests of
    krrood-core stay unaffected by a change there.
    """
    affected = repository_tests.affected_tests(
        changed_paths=[Path("semantic_digital_twin") / "src" / "module.py"]
    )
    assert "krrood-core" not in paths_by_library(affected)


def test_every_assigned_test_path_exists(repository_layout: CIJobLayout) -> None:
    """
    Every test assigned to a library by hand is there.
    """
    missing = [
        test_path
        for assigned in repository_layout.assigned_tests
        for test_path in assigned.test_paths
        if not (REPOSITORY_ROOT / test_path).exists()
    ]
    assert missing == []


# %% jobs


def test_shared_job_runs_only_its_affected_libraries(
    dependent_tests: RepositoryTestSuite, dependent_layout: CIJobLayout
) -> None:
    """
    A shared job runs the tests of its affected libraries only, and every affected
    library it does not share gets a job of its own.
    """
    affected = dependent_tests.affected_tests(
        changed_paths=[Path("middle") / "module.py"]
    )
    affected_by_name = {tests.library_name: tests for tests in affected}
    jobs = dependent_layout.jobs_running(
        affected_tests=affected,
        known_library_names=dependent_tests.library_names,
    )
    assert jobs == (
        CIJob(name="top", libraries=(affected_by_name["top"],)),
        CIJob(name="chain", libraries=(affected_by_name["middle"],)),
        CIJob(
            name="rules",
            libraries=(affected_by_name["repository"],),
            check_rules=True,
        ),
    )


def test_shared_job_checking_rules_runs_without_affected_libraries(
    dependent_tests: RepositoryTestSuite, dependent_layout: CIJobLayout
) -> None:
    """
    The job checking the rules runs even when the change affects none of its libraries.
    """
    jobs = dependent_layout.jobs_running(
        affected_tests=(), known_library_names=dependent_tests.library_names
    )
    assert jobs == (CIJob(name="rules", libraries=(), check_rules=True),)


@pytest.mark.parametrize(
    "layout_file_name",
    ["ci_jobs_unknown_library.toml", "ci_jobs_unknown_assigned_library.toml"],
)
def test_layout_naming_a_library_without_tests_is_rejected(
    layout_file_name: str,
    dependent_workspace_root: Path,
    dependent_tests: RepositoryTestSuite,
) -> None:
    """
    A shared job or an assignment naming a library without tests is a mistake in the
    layout.
    """
    layout = CIJobLayout.from_toml(
        text=(dependent_workspace_root / layout_file_name).read_text()
    )
    with pytest.raises(UnknownCILibraryError):
        layout.jobs_running(
            affected_tests=dependent_tests.all_tests(),
            known_library_names=dependent_tests.library_names,
        )


def test_repository_jobs_run_every_library_once(
    repository_tests: RepositoryTestSuite, repository_layout: CIJobLayout
) -> None:
    """
    The repository's job layout runs the tests of every library in exactly one job.
    """
    all_tests = repository_tests.all_tests()
    jobs = repository_layout.jobs_running(
        affected_tests=all_tests,
        known_library_names=repository_tests.library_names,
    )
    run_library_names = [
        library.library_name for job in jobs for library in job.libraries
    ]
    assert sorted(run_library_names) == sorted(
        library.library_name for library in all_tests
    )


def test_matrix_lists_the_tests_of_each_library_of_a_job_as_json() -> None:
    """
    The tests of each library of a job are JSON, as the reusable test workflow reads
    them.
    """
    library = LibraryTests(
        library_name="base", test_paths=(RepositoryDirectory.TESTS / "base_test",)
    )
    job = CIJob(name="chain", libraries=(library,))
    [entry] = json.loads(CIJobMatrix(jobs=(job,)).to_json())[MatrixKey.INCLUDE]
    assert json.loads(entry[MatrixKey.TEST_PATHS]) == {
        library.library_name: [str(path) for path in library.test_paths]
    }
