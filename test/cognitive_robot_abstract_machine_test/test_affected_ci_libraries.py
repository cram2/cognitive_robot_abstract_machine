"""
Tests for the script that selects the libraries a pull request's CI tests.
"""

from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
from pathlib import Path, PurePosixPath

import pytest

from cognitive_robot_abstract_machine.orm_interfaces import REPOSITORY_ROOT

SCRIPT = REPOSITORY_ROOT / "scripts" / "affected_ci_libraries.py"
_specification = importlib.util.spec_from_file_location("affected_ci_libraries", SCRIPT)
affected_ci_libraries = importlib.util.module_from_spec(_specification)
sys.modules[_specification.name] = affected_ci_libraries
_specification.loader.exec_module(affected_ci_libraries)

AlwaysTestedLibrary = affected_ci_libraries.AlwaysTestedLibrary
PythonSource = affected_ci_libraries.PythonSource
RepositoryPath = affected_ci_libraries.RepositoryPath
RequirementName = affected_ci_libraries.RequirementName
Workspace = affected_ci_libraries.Workspace


@pytest.fixture(scope="module")
def workspace() -> Workspace:
    return Workspace(REPOSITORY_ROOT)


@pytest.fixture(scope="module")
def always_tested() -> set[str]:
    return {library.value for library in AlwaysTestedLibrary}


def selected_for(workspace: Workspace, *paths: str) -> set[str]:
    return set(workspace.libraries_to_test([PurePosixPath(path) for path in paths]))


def a_file_of_member(workspace: Workspace, member: str) -> str:
    return str(workspace.members[member].folder / "src" / member / "changed.py")


# %% the libraries


def test_the_libraries_are_the_test_folders(workspace: Workspace):
    test_folders = sorted(
        folder.name.removesuffix("_test")
        for folder in (REPOSITORY_ROOT / "test").glob("*_test")
        if folder.is_dir()
    )
    assert workspace.libraries == test_folders


def test_all_prints_every_library(workspace: Workspace):
    printed = subprocess.run(
        [sys.executable, str(SCRIPT), "--all"],
        capture_output=True,
        text=True,
        check=True,
    ).stdout
    assert json.loads(printed) == workspace.libraries


def test_the_changed_paths_are_read_from_standard_input(
    workspace: Workspace, always_tested: set[str]
):
    printed = subprocess.run(
        [sys.executable, str(SCRIPT)],
        input="README.md\n\nAGENTS.md\n",
        capture_output=True,
        text=True,
        check=True,
    ).stdout
    assert json.loads(printed) == sorted(
        always_tested | {workspace.agent_rules_library}
    )


def test_the_script_imports_only_the_standard_library():
    """
    The selection runs on the bare runner before anything is installed.
    """
    imported = PythonSource(SCRIPT).absolute_imports
    assert imported - sys.stdlib_module_names - {"__future__"} == set()


# %% what a test job reads


def test_every_library_reads_what_the_harness_imports(workspace: Workspace):
    assert "semantic_digital_twin" in workspace.harness_members
    for library in workspace.libraries:
        assert workspace.harness_members <= workspace.members_read_by(library)


def test_the_harness_includes_the_modules_the_conftest_imports_relatively(
    workspace: Workspace,
):
    conftest = PythonSource(REPOSITORY_ROOT / RepositoryPath.TEST_HARNESS.value)
    imported = conftest.relative_imports
    reached = {source.path for source in PythonSource.closure([conftest])}
    assert imported and imported <= reached
    assert (
        workspace.members_imported_by([PythonSource(path) for path in imported])
        <= workspace.harness_members
    )


def test_a_library_building_the_orm_interfaces_reads_every_orm_member(
    workspace: Workspace,
):
    assert workspace.orm_members <= workspace.members_read_by("semantic_digital_twin")


def test_a_library_reads_what_its_members_require(workspace: Workspace):
    for library in workspace.libraries:
        if library not in workspace.members:
            continue
        required = workspace.dependency_closure({library})
        assert required <= workspace.members_read_by(library)


# %% changed paths


def test_a_harness_member_change_tests_every_library(workspace: Workspace):
    for member in workspace.harness_members:
        assert selected_for(workspace, a_file_of_member(workspace, member)) == set(
            workspace.libraries
        )


def test_a_member_change_tests_the_libraries_reading_it(
    workspace: Workspace, always_tested: set[str]
):
    for member in workspace.members:
        reading = {
            library
            for library in workspace.libraries
            if member in workspace.members_read_by(library)
        }
        assert selected_for(workspace, a_file_of_member(workspace, member)) == (
            always_tested | reading
        )


def test_a_member_no_other_library_reads_tests_only_itself(
    workspace: Workspace, always_tested: set[str]
):
    assert selected_for(workspace, a_file_of_member(workspace, "robokudo")) == (
        always_tested | {"robokudo"}
    )


def test_a_test_folder_change_tests_the_libraries_loading_it(
    workspace: Workspace, always_tested: set[str]
):
    assert selected_for(workspace, "test/robokudo_test/test_changed.py") == (
        always_tested | {"robokudo"}
    )


def test_a_test_helper_another_library_imports_tests_that_library_too(
    workspace: Workspace,
):
    helper = REPOSITORY_ROOT / "test" / "cramph_test" / "test_statechart"
    importing = {
        library
        for library, sources in workspace.test_sources.items()
        if library != "cramph"
        and any(source.path.is_relative_to(helper) for source in sources)
    }
    assert importing
    assert importing <= selected_for(
        workspace, "test/cramph_test/test_statechart/changed.py"
    )


def test_the_agent_rules_test_the_library_checking_them(
    workspace: Workspace, always_tested: set[str]
):
    assert selected_for(workspace, str(RepositoryPath.AGENT_RULES.value)) == (
        always_tested | {workspace.agent_rules_library}
    )


@pytest.mark.parametrize("path", ["README.md", "doc/intro.md", "doc/conf.py"])
def test_documentation_tests_only_the_libraries_always_tested(
    workspace: Workspace, always_tested: set[str], path: str
):
    assert selected_for(workspace, path) == always_tested


@pytest.mark.parametrize(
    "path",
    [
        "pyproject.toml",
        ".github/workflows/ci.yml",
        "scripts/regenerate_all_orm.py",
        "test/conftest.py",
        "test/dataset/robots_on_a_drive.py",
        "a/path/no/rule/knows",
    ],
)
def test_any_other_path_tests_every_library(workspace: Workspace, path: str):
    assert selected_for(workspace, path) == set(workspace.libraries)


def test_no_change_tests_the_libraries_always_tested(
    workspace: Workspace, always_tested: set[str]
):
    assert selected_for(workspace) == always_tested


# %% reading names and imports


def test_requirement_names_are_normalized():
    assert RequirementName.of("random-events>=26.10.0") == RequirementName.of(
        "random_events"
    )
    assert RequirementName.of("Krrood_Spatial.Math[casadi]") == RequirementName.of(
        "krrood-spatial-math"
    )


def test_a_dependency_spelled_with_dashes_is_a_workspace_edge(workspace: Workspace):
    assert "random_events" in workspace.members["probabilistic_model"].dependencies


def test_imports_are_read_from_a_file_that_does_not_parse():
    source = PythonSource(
        Path(__file__).parent
        / "dataset"
        / "affected_ci_libraries"
        / "unparsable_imports.py"
    )
    assert source.absolute_imports == {"krrood", "segmind", "semantic_digital_twin"}
