"""
Tests of running checkers over files, alone or restricted to a change.
"""

from __future__ import annotations

import shutil
from pathlib import Path

import pytest

from krrood.development_tools.agents.checker import (
    CodeRulesCheck,
    LinesChangedSince,
    StagedLines,
)
from krrood.development_tools.agents.prose_length import ProseLengthChecker
from krrood.development_tools.exceptions import GitCommandFailed

from .dataset_files import ProseSnippet
from .git_repository import git

# %% the staged change


def test_only_code_the_staged_change_touches_is_reported(repository: Path) -> None:
    """
    A violation in committed code is left out; one in appended, staged code is reported.
    """
    path = repository / "module.py"
    committed_lines = len(path.read_text().splitlines())
    with path.open("a") as source:
        source.write(f"\n{ProseSnippet.OVERLONG_ATTRIBUTE_DOCSTRING.read_text()}")
    git(repository, "add", "module.py")
    check = CodeRulesCheck(
        checkers=(ProseLengthChecker(),),
        changed_lines=StagedLines(repository=repository),
    )

    violations = check.violations(paths=[path])

    assert [violation.first_line > committed_lines for violation in violations] == [
        True
    ]


def test_every_violation_is_reported_without_a_staged_change(
    repository: Path,
) -> None:
    """
    Without a staged change, the violations of committed code are reported too.
    """
    check = CodeRulesCheck(checkers=(ProseLengthChecker(),))

    assert len(check.violations(paths=[repository / "module.py"])) == 1


def test_listing_the_staged_change_outside_a_repository_fails(tmp_path: Path) -> None:
    """
    Listing a staged change where git finds no repository raises
    :class:`GitCommandFailed`.
    """
    with pytest.raises(GitCommandFailed):
        StagedLines(repository=tmp_path).find_changed_lines(path=tmp_path / "module.py")


# %% the change since a revision


def test_only_code_changed_since_the_revision_is_reported(repository: Path) -> None:
    """
    Code appended without staging it is compared with the revision, and so reported.
    """
    path = repository / "module.py"
    committed_lines = len(path.read_text().splitlines())
    with path.open("a") as source:
        source.write(f"\n{ProseSnippet.OVERLONG_ATTRIBUTE_DOCSTRING.read_text()}")
    check = CodeRulesCheck(
        checkers=(ProseLengthChecker(),),
        changed_lines=LinesChangedSince(repository=repository, revision="HEAD"),
    )

    violations = check.violations(paths=[path])

    assert [violation.first_line > committed_lines for violation in violations] == [
        True
    ]


def test_the_changed_files_are_the_python_files_modified_or_added(
    repository: Path,
) -> None:
    """
    Deleted, untracked and non-Python files are left out.
    """
    shutil.copy(ProseSnippet.COMMENT_BLOCKS, repository / "deleted.py")
    git(repository, "add", "deleted.py")
    git(repository, "commit", "--quiet", "-m", "Add a module to delete")
    (repository / "deleted.py").unlink()
    with (repository / "module.py").open("a") as source:
        source.write("\n")
    shutil.copy(ProseSnippet.COMMENT_BLOCKS, repository / "added.py")
    git(repository, "add", "added.py")
    shutil.copy(ProseSnippet.COMMENT_BLOCKS, repository / "untracked.py")
    (repository / "notes.txt").write_text("not Python\n")
    git(repository, "add", "notes.txt")

    changed_files = LinesChangedSince(
        repository=repository, revision="HEAD"
    ).find_changed_files()

    assert sorted(changed_files) == [repository / "added.py", repository / "module.py"]


def test_comparing_with_an_unknown_revision_fails(repository: Path) -> None:
    """
    A revision git cannot resolve raises :class:`GitCommandFailed`.
    """
    with pytest.raises(GitCommandFailed):
        LinesChangedSince(
            repository=repository, revision="no-such-revision"
        ).find_changed_files()
