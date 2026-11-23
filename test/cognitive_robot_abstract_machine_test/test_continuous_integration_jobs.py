"""
Tests that this repository's CI job layout fits its workspace and tests.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from krrood.development_tools.continuous_integration_jobs import (
    ContinuousIntegrationJobLayout,
    RepositoryTestSuite,
    Workspace,
)

from cognitive_robot_abstract_machine.orm_interfaces import REPOSITORY_ROOT
from cognitive_robot_abstract_machine.repository_files import RepositoryFile

# %% fixtures


@pytest.fixture
def repository_layout() -> ContinuousIntegrationJobLayout:
    """
    The CI job layout of this repository.
    """
    return ContinuousIntegrationJobLayout.from_toml(
        text=(REPOSITORY_ROOT / RepositoryFile.CONTINUOUS_INTEGRATION_JOBS).read_text()
    )


@pytest.fixture
def repository_tests(
    repository_layout: ContinuousIntegrationJobLayout,
) -> RepositoryTestSuite:
    """
    The tests of this repository.
    """
    return RepositoryTestSuite(
        repository_root=REPOSITORY_ROOT,
        workspace=Workspace.from_repository_root(repository_root=REPOSITORY_ROOT),
        layout=repository_layout,
    )


# %% the repository's layout


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
    assert "krrood-core" not in {tests.library_name for tests in affected}


def test_every_path_the_layout_names_exists(
    repository_layout: ContinuousIntegrationJobLayout,
) -> None:
    """
    The tests directory, the documentation paths and every test assigned to a library by
    hand are there.
    """
    named_paths = [
        repository_layout.tests_directory,
        *repository_layout.documentation_paths,
        *(
            test_path
            for assigned in repository_layout.assigned_tests
            for test_path in assigned.test_paths
        ),
    ]
    missing = [path for path in named_paths if not (REPOSITORY_ROOT / path).exists()]
    assert missing == []


def test_repository_jobs_run_every_library_once(
    repository_tests: RepositoryTestSuite,
    repository_layout: ContinuousIntegrationJobLayout,
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
