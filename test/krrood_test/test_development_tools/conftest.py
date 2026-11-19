"""
Fixtures shared by the development tools tests.
"""

from __future__ import annotations

import shutil
import sys
from pathlib import Path

import pytest
from typing_extensions import Callable, Iterator

from krrood.development_tools.issue_forms.issue_form_files import (
    RepositoryIssueFormsModule,
)


from .dataset_files import DatasetFile, ProseSnippet
from .git_repository import git


@pytest.fixture
def copy_into_temporary_directory(
    tmp_path: Path,
) -> Callable[[DatasetFile, str], Path]:
    """
    :return: A function copying a dataset file or directory into ``tmp_path`` under a
        given name.
    """

    def copy(dataset_file: DatasetFile, file_name: str) -> Path:
        destination = tmp_path / file_name
        if dataset_file.is_dir():
            shutil.copytree(dataset_file, destination)
        else:
            shutil.copyfile(dataset_file, destination)
        return destination

    return copy


@pytest.fixture
def repository_root(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """
    :return: An empty directory, made the working directory, standing in for a
        repository root.
    """
    monkeypatch.chdir(tmp_path)
    return tmp_path


@pytest.fixture
def repository(tmp_path: Path) -> Path:
    """
    :return: A git repository whose one commit holds an overlong module docstring.
    """
    git(tmp_path, "init", "--quiet")
    git(tmp_path, "config", "user.name", "Tester")
    git(tmp_path, "config", "user.email", "tester@example.com")
    shutil.copy(ProseSnippet.OVERLONG_MODULE_DOCSTRING, tmp_path / "module.py")
    git(tmp_path, "add", "module.py")
    git(tmp_path, "commit", "--quiet", "-m", "Add")
    return tmp_path


@pytest.fixture
def forget_project_issue_forms() -> Iterator[None]:
    """
    Remove the repository's issue forms module from the imported modules after the test,
    so that the next test imports its own.
    """
    yield
    for module_name in list(sys.modules):
        if module_name.split(".")[0] == RepositoryIssueFormsModule.ISSUE_FORMS:
            del sys.modules[module_name]


@pytest.fixture
def project_issue_forms_repository(
    repository_root: Path,
    copy_into_temporary_directory: Callable[[DatasetFile, str], Path],
    forget_project_issue_forms: None,
) -> Path:
    """
    :return: :func:`repository_root`, holding the repository's issue forms module.
    """
    copy_into_temporary_directory(
        DatasetFile.PROJECT_ISSUE_FORMS_MODULE,
        f"{RepositoryIssueFormsModule.ISSUE_FORMS}.py",
    )
    return repository_root
