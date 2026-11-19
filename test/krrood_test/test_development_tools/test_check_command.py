"""
Tests of the ``krrood check`` command.
"""

from __future__ import annotations

import shutil
from pathlib import Path

import pytest
from typing_extensions import Callable

from krrood.development_tools.command_line import (
    CommandLine,
    ExitStatus,
    Option,
)
from krrood.development_tools.subcommand import Subcommand

from .dataset_files import DatasetFile, DocstringSnippet, ProseSnippet
from .git_repository import git

# %% check command


def test_checking_fails_when_code_breaks_a_rule(repository_root: Path) -> None:
    """
    A file with an overlong comment block makes ``krrood check`` exit with a failure.
    """
    exit_status = CommandLine.main(
        command_line=[
            Subcommand.CHECK,
            Option.ALL_LINES,
            str(ProseSnippet.COMMENT_BLOCKS),
        ]
    )

    assert exit_status == ExitStatus.FAILURE


def test_checking_passes_when_nothing_breaks_a_rule(repository_root: Path) -> None:
    """
    A fully documented module makes ``krrood check`` exit successfully.
    """
    exit_status = CommandLine.main(
        command_line=[
            Subcommand.CHECK,
            Option.ALL_LINES,
            str(DocstringSnippet.DOCUMENTED_MODULE),
        ]
    )

    assert exit_status == ExitStatus.SUCCESS


def test_checking_without_files_passes(repository_root: Path) -> None:
    """
    ``krrood check`` without files has nothing to report.
    """
    assert CommandLine.main(command_line=[Subcommand.CHECK]) == ExitStatus.SUCCESS


# %% checking the change since a revision


def test_checking_since_a_revision_finds_the_changed_files(
    repository: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """
    Without files named, the Python files changed since the revision are checked: an
    added module with an overlong comment block fails the check.
    """
    monkeypatch.chdir(repository)
    shutil.copy(ProseSnippet.COMMENT_BLOCKS, repository / "added.py")
    git(repository, "add", "added.py")

    exit_status = CommandLine.main(
        command_line=[Subcommand.CHECK, Option.SINCE, "HEAD"]
    )

    assert exit_status == ExitStatus.FAILURE


def test_checking_since_a_revision_ignores_unchanged_code(
    repository: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """
    The committed module breaks a rule, but nothing changed since the revision.
    """
    monkeypatch.chdir(repository)

    exit_status = CommandLine.main(
        command_line=[Subcommand.CHECK, Option.SINCE, "HEAD"]
    )

    assert exit_status == ExitStatus.SUCCESS


# %% files a repository excludes


def test_a_file_the_repository_excludes_is_not_checked(
    repository_root: Path,
    copy_into_temporary_directory: Callable[[DatasetFile, str], Path],
) -> None:
    """
    An overlong comment block under ``excluded/`` passes, since the repository's
    ``pyproject.toml`` keeps that directory out of the check.
    """
    copy_into_temporary_directory(DatasetFile.EXCLUDING_PYPROJECT, "pyproject.toml")
    (repository_root / "excluded").mkdir()
    path = shutil.copy(
        ProseSnippet.COMMENT_BLOCKS, repository_root / "excluded" / "module.py"
    )

    exit_status = CommandLine.main(
        command_line=[Subcommand.CHECK, Option.ALL_LINES, str(path)]
    )

    assert exit_status == ExitStatus.SUCCESS


def test_a_file_outside_the_exclusions_is_checked(
    repository_root: Path,
    copy_into_temporary_directory: Callable[[DatasetFile, str], Path],
) -> None:
    """
    The same module outside ``excluded/`` fails the check.
    """
    copy_into_temporary_directory(DatasetFile.EXCLUDING_PYPROJECT, "pyproject.toml")
    path = shutil.copy(ProseSnippet.COMMENT_BLOCKS, repository_root / "module.py")

    exit_status = CommandLine.main(
        command_line=[Subcommand.CHECK, Option.ALL_LINES, str(path)]
    )

    assert exit_status == ExitStatus.FAILURE
