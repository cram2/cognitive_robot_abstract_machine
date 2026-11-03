from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import pytest

from krrood.development_tools.agents.checker import (
    CodeRulesCheck,
    GitArgument,
    StagedLines,
)
from krrood.development_tools.agents.prose_length import ProseLengthChecker
from krrood.development_tools.exceptions import GitCommandFailed

from .dataset_files import ProseSnippet

# %% the staged change


def git(repository: Path, *arguments: str) -> None:
    """
    Run a git command in *repository*, which must succeed.
    """
    subprocess.run([GitArgument.GIT, *arguments], cwd=repository, check=True)


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


def test_only_code_the_staged_change_touches_is_reported(repository: Path) -> None:
    path = repository / "module.py"
    committed_lines = len(path.read_text().splitlines())
    with path.open("a") as source:
        source.write(f"\n{ProseSnippet.OVERLONG_ATTRIBUTE_DOCSTRING.read_text()}")
    git(repository, "add", "module.py")
    check = CodeRulesCheck((ProseLengthChecker(),), StagedLines(repository))

    violations = check.violations([path])

    assert [violation.first_line > committed_lines for violation in violations] == [
        True
    ]


def test_every_violation_is_reported_without_a_staged_change(
    repository: Path,
) -> None:
    check = CodeRulesCheck((ProseLengthChecker(),))

    assert len(check.violations([repository / "module.py"])) == 1


def test_listing_the_staged_change_outside_a_repository_fails(tmp_path: Path) -> None:
    with pytest.raises(GitCommandFailed):
        StagedLines(tmp_path).find_changed_lines(tmp_path / "module.py")
