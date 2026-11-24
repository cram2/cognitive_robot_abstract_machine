"""
Tests that CI runs the tests a change can affect, and only those.
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import pytest

from krrood.development_tools.continuous_integration_jobs import (
    ContinuousIntegrationJob,
    ContinuousIntegrationJobLayout,
    ContinuousIntegrationJobMatrix,
    GitHubActionsName,
    LibraryTests,
    MatrixKey,
    RepositoryTestSuite,
    Workspace,
    WorkspaceMember,
)
from krrood.development_tools.exceptions import (
    UnknownContinuousIntegrationLibraryError,
)
from krrood.development_tools.pyproject import Pyproject

from .dataset_files import DatasetFile

# %% fixtures


@pytest.fixture
def dependent_workspace_root() -> Path:
    """
    The root of :attr:`DatasetFile.DEPENDENT_WORKSPACE`.
    """
    return DatasetFile.DEPENDENT_WORKSPACE


@pytest.fixture
def dependent_layout(dependent_workspace_root: Path) -> ContinuousIntegrationJobLayout:
    """
    The CI job layout of :func:`dependent_workspace_root`.
    """
    return ContinuousIntegrationJobLayout.from_toml(
        text=(dependent_workspace_root / "continuous_integration_jobs.toml").read_text()
    )


@pytest.fixture
def dependent_tests(
    dependent_workspace_root: Path, dependent_layout: ContinuousIntegrationJobLayout
) -> RepositoryTestSuite:
    """
    The tests of :func:`dependent_workspace_root`.
    """
    return RepositoryTestSuite(
        repository_root=dependent_workspace_root,
        workspace=Workspace.from_repository_root(
            repository_root=dependent_workspace_root
        ),
        layout=dependent_layout,
    )


def paths_by_library(tests: tuple[LibraryTests, ...]) -> dict[str, list[str]]:
    """
    :return: The paths each library's tests run, by library, as text.
    """
    return {
        library_tests.library_name: [str(path) for path in library_tests.test_paths]
        for library_tests in tests
    }


def library_test_directory(
    layout: ContinuousIntegrationJobLayout, library_name: str
) -> str:
    """
    :return: The test directory named after the library, as text.
    """
    return str(layout.tests_directory / (library_name + LibraryTests.directory_suffix))


def family_test(layout: ContinuousIntegrationJobLayout, file_name: str) -> str:
    """
    :return: The path of a test of the family members, as text.
    """
    return str(Path(library_test_directory(layout, "family")) / file_name)


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
    dependent_layout: ContinuousIntegrationJobLayout,
) -> None:
    """
    A change to a member affects its tests, those of the members depending on it and
    those of the repository, but not those of the members it depends on.
    """
    affected = dependent_tests.affected_tests(
        changed_paths=[Path("middle") / "src" / "middle" / "__init__.py"]
    )
    assert paths_by_library(affected) == {
        "middle": [library_test_directory(dependent_layout, "middle")],
        "top": [library_test_directory(dependent_layout, "top")],
        "repository": [library_test_directory(dependent_layout, "repository")],
    }


def test_nested_member_change_affects_only_the_tests_depending_on_it(
    dependent_tests: RepositoryTestSuite,
    dependent_layout: ContinuousIntegrationJobLayout,
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
            family_test(dependent_layout, "test_family_both.py"),
            family_test(dependent_layout, "test_family_layout.py"),
        ],
        "family-extra": [family_test(dependent_layout, "test_family_extra.py")],
        "repository": [library_test_directory(dependent_layout, "repository")],
    }


def test_change_to_a_test_helper_affects_the_tests_loading_it(
    dependent_tests: RepositoryTestSuite,
    dependent_layout: ContinuousIntegrationJobLayout,
) -> None:
    """
    A change to a test module affects the tests loading it, in other test directories
    too, and no other test.
    """
    affected = dependent_tests.affected_tests(
        changed_paths=[
            dependent_layout.tests_directory / "base_test" / "chain_helpers.py"
        ]
    )
    assert paths_by_library(affected) == {
        "side": [library_test_directory(dependent_layout, "side")],
        "repository": [library_test_directory(dependent_layout, "repository")],
    }


def test_change_to_test_data_affects_every_test_of_its_directory(
    dependent_tests: RepositoryTestSuite,
    dependent_layout: ContinuousIntegrationJobLayout,
) -> None:
    """
    A changed file of a test directory that is not a module can be read by any test in
    it.
    """
    affected = dependent_tests.affected_tests(
        changed_paths=[
            Path(library_test_directory(dependent_layout, "family"))
            / "data"
            / "sample.json"
        ]
    )
    assert paths_by_library(affected) == {
        "family": [
            family_test(dependent_layout, "test_family_both.py"),
            family_test(dependent_layout, "test_family_layout.py"),
        ],
        "family-core": [family_test(dependent_layout, "test_family_core.py")],
        "family-extra": [family_test(dependent_layout, "test_family_extra.py")],
        "repository": [library_test_directory(dependent_layout, "repository")],
    }


def test_change_to_a_test_configuration_affects_every_test_below_it(
    dependent_tests: RepositoryTestSuite,
    dependent_layout: ContinuousIntegrationJobLayout,
) -> None:
    """
    A changed ``conftest.py`` affects every test below its directory.
    """
    affected = dependent_tests.affected_tests(
        changed_paths=[Path(family_test(dependent_layout, "conftest.py"))]
    )
    assert set(paths_by_library(affected)) == {
        "family",
        "family-core",
        "family-extra",
        "repository",
    }


def test_documentation_change_affects_only_the_repository_tests(
    dependent_tests: RepositoryTestSuite,
    dependent_layout: ContinuousIntegrationJobLayout,
) -> None:
    """
    A change to the repository's documentation affects no member's tests.
    """
    affected = dependent_tests.affected_tests(
        changed_paths=[dependent_layout.documentation_paths[0] / "index.md"]
    )
    assert paths_by_library(affected) == {
        "repository": [library_test_directory(dependent_layout, "repository")]
    }


def test_change_outside_every_member_affects_all_tests(
    dependent_tests: RepositoryTestSuite,
) -> None:
    """
    A change to a file no member owns can affect every test.
    """
    affected = dependent_tests.affected_tests(changed_paths=[Path(Pyproject.file_name)])
    assert affected == dependent_tests.all_tests()


# %% jobs


def test_shared_job_runs_only_its_affected_libraries(
    dependent_tests: RepositoryTestSuite,
    dependent_layout: ContinuousIntegrationJobLayout,
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
        ContinuousIntegrationJob(name="top", libraries=(affected_by_name["top"],)),
        ContinuousIntegrationJob(name="chain", libraries=(affected_by_name["middle"],)),
        ContinuousIntegrationJob(
            name="rules",
            libraries=(affected_by_name["repository"],),
            check_rules=True,
        ),
    )


def test_shared_job_checking_rules_runs_without_affected_libraries(
    dependent_tests: RepositoryTestSuite,
    dependent_layout: ContinuousIntegrationJobLayout,
) -> None:
    """
    The job checking the rules runs even when the change affects none of its libraries.
    """
    jobs = dependent_layout.jobs_running(
        affected_tests=(), known_library_names=dependent_tests.library_names
    )
    assert jobs == (
        ContinuousIntegrationJob(name="rules", libraries=(), check_rules=True),
    )


@pytest.mark.parametrize(
    "layout_file_name",
    [
        "continuous_integration_jobs_unknown_library.toml",
        "continuous_integration_jobs_unknown_assigned_library.toml",
        "continuous_integration_jobs_unknown_environment_library.toml",
    ],
)
def test_layout_naming_a_library_without_tests_is_rejected(
    layout_file_name: str,
    dependent_workspace_root: Path,
    dependent_tests: RepositoryTestSuite,
) -> None:
    """
    A shared job, an assignment or a test environment naming a library without tests is
    a mistake in the layout.
    """
    layout = ContinuousIntegrationJobLayout.from_toml(
        text=(dependent_workspace_root / layout_file_name).read_text()
    )
    with pytest.raises(UnknownContinuousIntegrationLibraryError):
        layout.jobs_running(
            affected_tests=dependent_tests.all_tests(),
            known_library_names=dependent_tests.library_names,
        )


def test_matrix_entry_gives_back_the_tests_of_each_library() -> None:
    """
    The tests of each library a matrix entry lists are those of its job.
    """
    library = LibraryTests(
        library_name="base", test_paths=(Path("test") / "base_test",)
    )
    job = ContinuousIntegrationJob(name="chain", libraries=(library,))
    [entry] = json.loads(ContinuousIntegrationJobMatrix(jobs=(job,)).to_json())[
        MatrixKey.INCLUDE
    ]
    assert (
        LibraryTests.from_test_paths_json(text=entry[MatrixKey.TEST_PATHS])
        == job.libraries
    )


def test_matrix_is_appended_to_the_step_output(tmp_path: Path) -> None:
    """
    The matrix is added to the outputs a step already wrote, under its output name.
    """
    output_file = tmp_path / "output"
    earlier_output = "earlier=output\n"
    output_file.write_text(earlier_output)
    matrix = ContinuousIntegrationJobMatrix(
        jobs=(ContinuousIntegrationJob(name="rules", libraries=(), check_rules=True),)
    )
    matrix.write_github_output(output_file=output_file)
    assert output_file.read_text() == (
        f"{earlier_output}{GitHubActionsName.MATRIX_OUTPUT}={matrix.to_json()}\n"
    )


# %% running the tests of a library


@pytest.fixture
def library_test_runs(tmp_path: Path) -> Path:
    """
    A copy of the tests run in a pytest process of their own, outside the repository so
    that process loads none of its ``conftest.py`` files.
    """
    shutil.copytree(DatasetFile.LIBRARY_TEST_RUNS, tmp_path, dirs_exist_ok=True)
    shutil.copy(
        DatasetFile.DEPENDENT_WORKSPACE / "continuous_integration_jobs.toml", tmp_path
    )
    return tmp_path


def test_library_tests_pass_with_their_environment(
    library_test_runs: Path, dependent_layout: ContinuousIntegrationJobLayout
) -> None:
    """
    A library's tests run with the environment variables its layout declares.
    """
    library_tests = LibraryTests(
        library_name="base", test_paths=(Path("environment_check.py"),)
    )
    assert library_tests.run(
        repository_root=library_test_runs,
        environment=dependent_layout.test_environments[library_tests.library_name],
    )


def test_failing_library_tests_report_a_failure(library_test_runs: Path) -> None:
    """
    A run with a failing test reports that not every test passed.
    """
    library_tests = LibraryTests(
        library_name="base", test_paths=(Path("failing_check.py"),)
    )
    assert not library_tests.run(repository_root=library_test_runs, environment={})
