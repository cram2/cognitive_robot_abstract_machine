"""
Integration tests for check-setup.sh's per-check reporting and exit code.

Run against a scratch project root with a local bare repository standing in for the
personal-notes remote - no network access or real personal-notes branch involved.
"""

from __future__ import annotations

import os
import subprocess
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path

import pytest

from scratch_repository import (
    NOTES_BRANCH,
    ScratchRepository,
    initialize_bare_repository,
)

# The files check-setup.sh's `tooling_files` check requires, relative to the
# project root. Kept as literals rather than sourced from
# resolve-personal-notes-config.sh so a rename that breaks the check has to be
# made deliberately in both places, instead of the test silently following
# along and asserting nothing.
TOOLING_FILES = (
    ".claude/skills/plan-dashboard/build_dashboard.py",
    ".claude/skills/plan-dashboard/refresh_dashboard.sh",
    ".claude/skills/plan-dashboard/requirements.txt",
    ".claude/skills/plan-dashboard/plan-schema.md",
)

REQUIREMENTS_FILE = ".claude/skills/plan-dashboard/requirements.txt"

NOTES_PATH = ".claude/personal/cram-notes.md"


# %% what a report is made of


class SetupCheck(StrEnum):
    """
    The checks check-setup.sh reports on, in the order it prints them.
    """

    TOOLING_FILES = "tooling_files"
    SESSION_START_HOOK = "session_start_hook"
    CLAUDE_LOCAL_MD_IGNORED = "claude_local_md_ignored"
    NOTES_REMOTE = "notes_remote"
    NOTES_REMOTE_URL = "notes_remote_url"
    NOTES_BRANCH_NAME = "notes_branch_name"
    NOTES_PATH = "notes_path"
    NOTES_BRANCH = "notes_branch"
    NOTES_FILE = "notes_file"
    DASHBOARD_DEPENDENCIES = "dashboard_dependencies"
    CLAUDE_LOCAL_MD = "claude_local_md"


class CheckStatus(StrEnum):
    """
    The status check-setup.sh reports for a single check.
    """

    OK = "ok"
    NEEDS_SETUP = "needs-setup"
    INFORMATIONAL = "info"


@dataclass
class CheckResult:
    """
    What check-setup.sh reported for one check.
    """

    status: CheckStatus
    """
    Whether the check passed, needs setup, or is context rather than a verdict.
    """

    detail: str
    """
    The human-readable explanation printed alongside the status.
    """


@dataclass
class SetupReport:
    """
    One parsed run of check-setup.sh: what it reported, and how it exited.
    """

    exit_code: int
    """
    The script's exit code: 0 when nothing needs setup, 1 otherwise.
    """

    results: dict[SetupCheck, CheckResult]
    """
    Every reported check, keyed by the check it reports on.
    """

    @classmethod
    def from_completed_process(
        cls, process: subprocess.CompletedProcess[str]
    ) -> SetupReport:
        """
        Parse a finished check-setup.sh run.

        Raises if a row names a check this test module doesn't know about, so a new
        check has to be declared here rather than silently going unasserted.

        :param process: The finished check-setup.sh subprocess.
        :return: The parsed report.
        """
        results = {}
        for line in process.stdout.splitlines():
            check, status, detail = line.split("\t")
            results[SetupCheck(check)] = CheckResult(CheckStatus(status), detail)
        return cls(process.returncode, results)


# %% the scratch layout


@pytest.fixture
def check_setup_repository(scratch_repository: ScratchRepository) -> ScratchRepository:
    """
    A scratch repository set up so every check-setup.sh check passes: the real check-
    setup.sh and resolve-personal-notes-config.sh, placeholder tooling files, a
    registered SessionStart hook, a gitignored CLAUDE.local.md, and a notes branch
    carrying a notes file.

    Individual tests break exactly one of those conditions to assert the matching check
    reports it.

    :param scratch_repository: The initialized scratch repository and notes remote.
    :return: The same repository, fully set up.
    """
    scratch_repository.install_hook_scripts(
        "resolve-personal-notes-config.sh", "check-setup.sh"
    )

    for tooling_file in TOOLING_FILES:
        scratch_repository.write(tooling_file, "placeholder\n")
    # The dependency check reads this file rather than a hardcoded list, so a
    # requirement that is certainly installed keeps the fixture's baseline green.
    scratch_repository.write(REQUIREMENTS_FILE, "pytest>=1\n")

    scratch_repository.write(
        ".claude/settings.json",
        '{"hooks": {"SessionStart": [{"hooks": [{"type": "command",'
        ' "command": "$CLAUDE_PROJECT_DIR/.claude/hooks/session-start.sh"}]}]}}\n',
    )
    scratch_repository.write(".gitignore", "CLAUDE.local.md\n")
    scratch_repository.write("CLAUDE.local.md", "notes\n")

    scratch_repository.commit_everything("initial commit")
    scratch_repository.publish_notes_branch({NOTES_PATH: "my notes\n"})
    scratch_repository.resolve_notes_remote_to()
    return scratch_repository


def run_check_setup(
    repository: ScratchRepository, **environment_overrides: str
) -> SetupReport:
    """
    Run the scratch layout's check-setup.sh and parse its report.

    Every ``CLAUDE_PERSONAL_NOTES_*`` variable is stripped from the inherited
    environment first, so a value that happens to be set in whoever's shell is running
    the tests can never change what they assert.

    :param repository: A fixture-built scratch repository.
    :param environment_overrides: Personal-notes environment variables to set for this
        run, for the tests that exercise resolution from the environment.
    :return: The parsed report.
    """
    environment = {
        name: value
        for name, value in os.environ.items()
        if not name.startswith("CLAUDE_PERSONAL_NOTES_")
    }
    environment.update(environment_overrides)
    return SetupReport.from_completed_process(
        subprocess.run(
            [
                "bash",
                str(repository.project_root / ".claude" / "hooks" / "check-setup.sh"),
            ],
            cwd=repository.project_root,
            capture_output=True,
            text=True,
            env=environment,
        )
    )


# %% the already-set-up fast path


def test_reports_no_work_needed_when_everything_is_in_place(
    check_setup_repository: ScratchRepository,
):
    report = run_check_setup(check_setup_repository)
    assert report.exit_code == 0
    needing_setup = [
        check
        for check, result in report.results.items()
        if result.status == CheckStatus.NEEDS_SETUP
    ]
    assert needing_setup == []


def test_reports_every_check_it_documents(check_setup_repository: ScratchRepository):
    report = run_check_setup(check_setup_repository)
    assert set(report.results) == set(SetupCheck)


# %% the personal-notes branch


def test_reports_a_missing_notes_branch_and_the_remotes_it_tried(
    check_setup_repository: ScratchRepository, tmp_path: Path
):
    empty_remote = initialize_bare_repository(tmp_path / "empty-remote.git")
    check_setup_repository.resolve_notes_remote_to(empty_remote)

    report = run_check_setup(check_setup_repository)
    assert report.exit_code == 1
    assert report.results[SetupCheck.NOTES_BRANCH].status == CheckStatus.NEEDS_SETUP
    assert str(empty_remote) in report.results[SetupCheck.NOTES_BRANCH].detail


def test_does_not_check_for_the_notes_file_when_its_branch_is_missing(
    check_setup_repository: ScratchRepository, tmp_path: Path
):
    check_setup_repository.resolve_notes_remote_to(
        initialize_bare_repository(tmp_path / "empty-remote.git")
    )

    report = run_check_setup(check_setup_repository)
    assert report.results[SetupCheck.NOTES_FILE].status == CheckStatus.NEEDS_SETUP
    assert report.results[SetupCheck.NOTES_FILE].detail == (
        "not checked - the branch that would hold it doesn't exist yet"
    )


def test_reports_a_notes_branch_that_exists_but_holds_no_notes_file(
    check_setup_repository: ScratchRepository,
):
    check_setup_repository.run_git(
        "config", "claude.personalNotesPath", ".claude/personal/some-other-notes.md"
    )

    report = run_check_setup(check_setup_repository)
    assert report.exit_code == 1
    assert report.results[SetupCheck.NOTES_BRANCH].status == CheckStatus.OK
    assert report.results[SetupCheck.NOTES_FILE].status == CheckStatus.NEEDS_SETUP
    assert (
        ".claude/personal/some-other-notes.md"
        in report.results[SetupCheck.NOTES_FILE].detail
    )


# %% how each setting was resolved


def test_reports_which_source_each_resolved_setting_came_from(
    check_setup_repository: ScratchRepository,
):
    report = run_check_setup(check_setup_repository)
    assert (
        "from git config claude.personalNotesRemote"
        in report.results[SetupCheck.NOTES_REMOTE].detail
    )
    assert report.results[SetupCheck.NOTES_BRANCH_NAME].detail == (
        f"{NOTES_BRANCH} (from built-in default)"
    )
    assert report.results[SetupCheck.NOTES_PATH].detail == (
        f"{NOTES_PATH} (from built-in default)"
    )


def test_reports_a_setting_resolved_from_the_environment(
    check_setup_repository: ScratchRepository,
):
    report = run_check_setup(
        check_setup_repository,
        CLAUDE_PERSONAL_NOTES_PATH=".claude/personal/from-the-environment.md",
    )
    assert report.results[SetupCheck.NOTES_PATH].detail == (
        ".claude/personal/from-the-environment.md"
        " (from environment variable CLAUDE_PERSONAL_NOTES_PATH)"
    )


# %% the tooling this checkout is expected to carry


def test_reports_which_tooling_files_this_checkout_is_missing(
    check_setup_repository: ScratchRepository,
):
    (
        check_setup_repository.project_root
        / ".claude/skills/plan-dashboard/plan-schema.md"
    ).unlink()

    report = run_check_setup(check_setup_repository)
    assert report.exit_code == 1
    assert report.results[SetupCheck.TOOLING_FILES].status == CheckStatus.NEEDS_SETUP
    assert (
        ".claude/skills/plan-dashboard/plan-schema.md"
        in report.results[SetupCheck.TOOLING_FILES].detail
    )
    assert (
        ".claude/skills/plan-dashboard/build_dashboard.py"
        not in report.results[SetupCheck.TOOLING_FILES].detail
    )


def test_reports_a_session_start_hook_that_is_not_registered(
    check_setup_repository: ScratchRepository,
):
    check_setup_repository.write(".claude/settings.json", "{}\n")

    report = run_check_setup(check_setup_repository)
    assert report.exit_code == 1
    assert (
        report.results[SetupCheck.SESSION_START_HOOK].status == CheckStatus.NEEDS_SETUP
    )


def test_reports_a_claude_local_md_that_is_not_gitignored(
    check_setup_repository: ScratchRepository,
):
    check_setup_repository.write(".gitignore", "something-else\n")

    report = run_check_setup(check_setup_repository)
    assert report.exit_code == 1
    assert (
        report.results[SetupCheck.CLAUDE_LOCAL_MD_IGNORED].status
        == CheckStatus.NEEDS_SETUP
    )


# %% plan-dashboard dependencies


def test_reports_dashboard_requirements_that_are_not_installed(
    check_setup_repository: ScratchRepository,
):
    check_setup_repository.write(
        REQUIREMENTS_FILE, "pytest>=1\nno-such-distribution-exists>=2  # a comment\n"
    )

    report = run_check_setup(check_setup_repository)
    assert report.exit_code == 1
    assert (
        report.results[SetupCheck.DASHBOARD_DEPENDENCIES].status
        == CheckStatus.NEEDS_SETUP
    )
    assert (
        "no-such-distribution-exists"
        in report.results[SetupCheck.DASHBOARD_DEPENDENCIES].detail
    )
    assert "pytest" not in report.results[SetupCheck.DASHBOARD_DEPENDENCIES].detail


# %% the outcome of it all working


def test_reports_a_claude_local_md_that_was_never_written(
    check_setup_repository: ScratchRepository,
):
    (check_setup_repository.project_root / "CLAUDE.local.md").unlink()

    report = run_check_setup(check_setup_repository)
    assert report.exit_code == 1
    assert report.results[SetupCheck.CLAUDE_LOCAL_MD].status == CheckStatus.NEEDS_SETUP
