"""
Covers the two mechanical questions ``check_scope_overlap.py`` answers: whether the
work's paths already exist on the base, and which unlanded branches already touch them.
"""

from __future__ import annotations

import json
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

import pytest

import check_scope_overlap
from check_scope_overlap import (
    Candidate,
    ReportKey,
    UnknownBranchError,
    build_scope_report,
)
from scratch_repository import ScratchRepository

BASE_BRANCH = "main"
"""
The branch the scratch repository's candidates are compared against.
"""

NEW_SKILL_PATH = ".claude/skills/new-thing/SKILL.md"
"""
A path no branch but ``overlapping`` introduces - the fold test's positive case.
"""

EXISTING_PATH = "already-on-base.md"
"""
A path that exists on the base already - the fold test's negative case.
"""


@dataclass
class CandidateScratchRepository:
    """
    A scratch repository with a base branch and two candidate branches: one touching
    the paths under test, one touching something else entirely.
    """

    repository: ScratchRepository

    @classmethod
    def create(cls, parent_directory: Path) -> CandidateScratchRepository:
        """
        Build the repository, its base branch, and both candidate branches.

        :param parent_directory: Where to put it, typically pytest's temporary directory.
        :return: The prepared repository.
        """
        repository = ScratchRepository.create(parent_directory)
        repository.run_git("checkout", "--quiet", "-b", BASE_BRANCH)
        repository.write(EXISTING_PATH, "already here\n")
        repository.commit_everything("base")

        repository.run_git("checkout", "--quiet", "-b", "overlapping")
        repository.write(NEW_SKILL_PATH, "the new skill\n")
        repository.write("shared/helper.py", "helper\n")
        repository.commit_everything("introduce the new skill")

        repository.run_git("checkout", "--quiet", BASE_BRANCH)
        repository.run_git("checkout", "--quiet", "-b", "unrelated")
        repository.write("docs/something-else.md", "unrelated\n")
        repository.commit_everything("unrelated work")

        repository.run_git("checkout", "--quiet", BASE_BRANCH)
        return cls(repository)

    @property
    def project_root(self) -> Path:
        """
        The repository the script runs against.
        """
        return self.repository.project_root

    def commit_to_base(self, relative_path: str) -> None:
        """
        Add a file to the base branch after the candidates already branched off it.

        :param relative_path: The file to add, relative to the project root.
        """
        self.repository.run_git("checkout", "--quiet", BASE_BRANCH)
        self.repository.write(relative_path, "landed later\n")
        self.repository.commit_everything("later work on the base")


@pytest.fixture
def candidate_repository(tmp_path: Path) -> CandidateScratchRepository:
    """
    A scratch repository with a base branch and both candidate branches.
    """
    return CandidateScratchRepository.create(tmp_path)


# %% the fold test - does the base already have these paths?


def test_path_missing_from_base_is_reported_and_present_one_is_not(
    candidate_repository: CandidateScratchRepository,
) -> None:
    report = build_scope_report(
        candidate_repository.project_root,
        BASE_BRANCH,
        [NEW_SKILL_PATH, EXISTING_PATH],
        [],
    )

    assert report.paths_absent_from_base == [NEW_SKILL_PATH]


# %% candidate overlap - which unlanded branches already touch them?


def test_candidate_touching_a_requested_path_reports_it_as_shared(
    candidate_repository: CandidateScratchRepository,
) -> None:
    report = build_scope_report(
        candidate_repository.project_root,
        BASE_BRANCH,
        [NEW_SKILL_PATH],
        [Candidate("the parent", "overlapping")],
    )

    assert [overlap.shared_paths for overlap in report.candidates] == [[NEW_SKILL_PATH]]


def test_candidate_touching_nothing_requested_still_reports_its_changed_paths(
    candidate_repository: CandidateScratchRepository,
) -> None:
    report = build_scope_report(
        candidate_repository.project_root,
        BASE_BRANCH,
        [NEW_SKILL_PATH],
        [Candidate("elsewhere", "unrelated")],
    )
    overlap = report.candidates[0]

    assert overlap.shared_paths == []
    assert overlap.changed_paths == ["docs/something-else.md"]


def test_changed_paths_exclude_commits_added_to_the_base_after_branching(
    candidate_repository: CandidateScratchRepository,
) -> None:
    candidate_repository.commit_to_base("landed-after-branching.md")

    report = build_scope_report(
        candidate_repository.project_root,
        BASE_BRANCH,
        [NEW_SKILL_PATH],
        [Candidate("the parent", "overlapping")],
    )

    assert report.candidates[0].changed_paths == [NEW_SKILL_PATH, "shared/helper.py"]


# %% a branch that does not resolve must never read as "no overlap"


def test_unresolvable_base_branch_raises(
    candidate_repository: CandidateScratchRepository,
) -> None:
    with pytest.raises(UnknownBranchError):
        build_scope_report(
            candidate_repository.project_root,
            "no-such-base",
            [NEW_SKILL_PATH],
            [],
        )


def test_unresolvable_candidate_branch_raises(
    candidate_repository: CandidateScratchRepository,
) -> None:
    with pytest.raises(UnknownBranchError):
        build_scope_report(
            candidate_repository.project_root,
            BASE_BRANCH,
            [NEW_SKILL_PATH],
            [Candidate("missing", "no-such-candidate")],
        )


# %% the wire format itself


def test_the_report_key_names_are_the_documented_wire_format() -> None:
    """
    Pin the key names as literals, since every other assertion now reads them from
    :class:`ReportKey` and so would follow a rename rather than catch it. This is the one
    place that owns what a reader of the JSON actually sees.
    """
    assert {key.name: str(key) for key in ReportKey} == {
        "PATHS_ABSENT_FROM_BASE": "paths_absent_from_base",
        "CANDIDATES": "candidates",
        "LABEL": "label",
        "BRANCH": "branch",
        "SHARED_PATHS": "shared_paths",
        "CHANGED_PATHS": "changed_paths",
    }


# %% command line


def test_command_line_prints_the_report_as_json(
    candidate_repository: CandidateScratchRepository,
) -> None:
    result = subprocess.run(
        [
            sys.executable,
            check_scope_overlap.__file__,
            "--repository",
            str(candidate_repository.project_root),
            "--base",
            BASE_BRANCH,
            "--path",
            NEW_SKILL_PATH,
            "--path",
            EXISTING_PATH,
            "--candidate",
            "the parent=overlapping",
        ],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr

    assert json.loads(result.stdout) == {
        ReportKey.PATHS_ABSENT_FROM_BASE: [NEW_SKILL_PATH],
        ReportKey.CANDIDATES: [
            {
                ReportKey.LABEL: "the parent",
                ReportKey.BRANCH: "overlapping",
                ReportKey.SHARED_PATHS: [NEW_SKILL_PATH],
                ReportKey.CHANGED_PATHS: [NEW_SKILL_PATH, "shared/helper.py"],
            }
        ],
    }


def test_command_line_reports_an_unresolvable_branch_as_a_failure(
    candidate_repository: CandidateScratchRepository,
) -> None:
    result = subprocess.run(
        [
            sys.executable,
            check_scope_overlap.__file__,
            "--repository",
            str(candidate_repository.project_root),
            "--base",
            "no-such-base",
            "--path",
            NEW_SKILL_PATH,
        ],
        capture_output=True,
        text=True,
    )

    assert result.returncode != 0
    assert "no-such-base" in result.stderr
