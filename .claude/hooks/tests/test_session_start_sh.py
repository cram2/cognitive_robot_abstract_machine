"""
Integration tests for session-start.sh's summary report.

Cover the two guards the report exists for: naming which situation a branch with no plan
item is actually in, and surfacing check-setup.sh's verdict rather than leaving it to be
remembered. Both stay invisible to anyone who uses neither plans nor personal notes.

Run against a scratch project root with a local bare repository standing in for the
personal-notes remote - no network access or real personal-notes branch involved.
"""

from __future__ import annotations

import subprocess
from collections.abc import Mapping
from pathlib import Path

import pytest

from scratch_repository import (
    NOTES_BRANCH,
    PERSONAL_GIT_IDENTITY_PATH,
    SCRATCH_IDENTITY,
    WORK_BRANCH,
    ScratchRepository,
)
from session_start_summary import SummaryMessage, summary_message, summary_value

FIXTURES_DIRECTORY = Path(__file__).parent / "fixtures"

PLAN_MANIFEST = (FIXTURES_DIRECTORY / "plan.yaml").read_text()

PLAN_MANIFEST_WITH_TRACKING_ISSUE = (
    FIXTURES_DIRECTORY / "plan-with-tracking-issue.yaml"
).read_text()

TRACKING_ISSUE = "55"

PLAN_IDENTIFIER = "test-plan"

NOTES_PATH = ".claude/personal/cram-notes.md"

BRANCH_INDEX_PATH = ".claude/personal/plans/_generated/branch-index.tsv"

MANIFEST_PATH = f".claude/personal/plans/{PLAN_IDENTIFIER}/plan.yaml"

CLAUDE_LOCAL_MD = "CLAUDE.local.md"


def branch_index(plan_identifier_by_branch: Mapping[str, str]) -> str:
    """
    Build a branch index mapping each branch to the plan that tracks it.

    :param plan_identifier_by_branch: Plan ids, keyed by the branch each one tracks.
    :return: The index's tab-separated content.
    """
    return "".join(
        f"{branch}\t{plan_identifier}\n"
        for branch, plan_identifier in plan_identifier_by_branch.items()
    )


# %% the scratch layout


@pytest.fixture
def session_start_repository(
    scratch_repository: ScratchRepository,
) -> ScratchRepository:
    """
    A scratch repository carrying the real session-start.sh and everything else a set up
    clone has, with nothing published to the notes remote yet.

    :param scratch_repository: The initialized scratch repository and notes remote.
    :return: The same repository, ready to publish a notes branch and run the hook.
    """
    scratch_repository.install_hook_scripts(
        "resolve-personal-notes-config.sh",
        "session-start-messages.sh",
        "session-start.sh",
        "check-setup.sh",
    )
    scratch_repository.write_setup_prerequisites()
    scratch_repository.commit_everything("initial commit")
    scratch_repository.resolve_notes_remote_to()
    return scratch_repository


def run_session_start(
    repository: ScratchRepository,
) -> subprocess.CompletedProcess[str]:
    """
    Run the scratch layout's session-start.sh.

    :param repository: A fixture-built scratch repository.
    :return: The finished subprocess.
    """
    return repository.run_hook_script("session-start.sh")


def publish_and_run(
    repository: ScratchRepository, notes_branch_files: Mapping[str, str] | None = None
) -> subprocess.CompletedProcess[str]:
    """
    Publish everything a set up notes branch carries, plus *notes_branch_files*, then
    run session-start.sh against it.

    The recorded git identity is the one this repository already commits with, so the
    baseline these tests assert against is a clone with nothing left to set up - what
    the git identity round trip itself does is
    ``test_git_identity_sync.py``'s subject, not this module's.

    :param repository: The fixture-built scratch repository.
    :param notes_branch_files: Extra file contents, keyed by path relative to the
        project root.
    :return: The finished session-start.sh process.
    """
    repository.publish_notes_branch(
        {
            NOTES_PATH: "personal notes\n",
            PERSONAL_GIT_IDENTITY_PATH: SCRATCH_IDENTITY.as_git_config_file(),
            **(notes_branch_files or {}),
        }
    )
    return run_session_start(repository)


# %% someone who uses neither plans nor personal notes


def test_reports_nothing_when_no_notes_branch_exists(
    session_start_repository: ScratchRepository,
):
    session_start_repository.run_git("checkout", "--quiet", "-b", WORK_BRANCH)

    result = run_session_start(session_start_repository)

    assert result.returncode == 0, result.stderr
    assert result.stdout == ""
    assert not (session_start_repository.project_root / CLAUDE_LOCAL_MD).exists()


# %% the plan line


def test_reports_no_plans_when_none_are_tracked(
    session_start_repository: ScratchRepository,
):
    result = publish_and_run(session_start_repository)

    assert result.returncode == 0, result.stderr
    assert summary_value(result.stdout, "plan") == summary_message(
        SummaryMessage.NO_PLANS_TRACKED, NOTES_BRANCH
    )


def test_names_the_missing_item_when_other_plans_are_tracked(
    session_start_repository: ScratchRepository,
):
    result = publish_and_run(
        session_start_repository,
        {
            BRANCH_INDEX_PATH: branch_index(
                {
                    "some-other-branch": PLAN_IDENTIFIER,
                    "a-third-branch": "another-plan",
                }
            ),
            MANIFEST_PATH: PLAN_MANIFEST,
        },
    )

    assert result.returncode == 0, result.stderr
    assert summary_value(result.stdout, "plan") == summary_message(
        SummaryMessage.NO_PLAN_ITEM_TRACKS_BRANCH, WORK_BRANCH, "2"
    )


def test_reports_the_plan_that_tracks_this_branch(
    session_start_repository: ScratchRepository,
):
    result = publish_and_run(
        session_start_repository,
        {
            BRANCH_INDEX_PATH: branch_index({WORK_BRANCH: PLAN_IDENTIFIER}),
            MANIFEST_PATH: PLAN_MANIFEST_WITH_TRACKING_ISSUE,
        },
    )

    assert result.returncode == 0, result.stderr
    assert summary_value(result.stdout, "plan") == summary_message(
        SummaryMessage.BRANCH_TRACKED_IN_PLAN, PLAN_IDENTIFIER, TRACKING_ISSUE
    )


def test_reports_a_tracked_plan_that_has_no_tracking_issue(
    session_start_repository: ScratchRepository,
):
    result = publish_and_run(
        session_start_repository,
        {
            BRANCH_INDEX_PATH: branch_index({WORK_BRANCH: PLAN_IDENTIFIER}),
            MANIFEST_PATH: PLAN_MANIFEST,
        },
    )

    assert result.returncode == 0, result.stderr
    assert summary_value(result.stdout, "plan") == summary_message(
        SummaryMessage.BRANCH_TRACKED_IN_PLAN, PLAN_IDENTIFIER, "none"
    )


def test_reports_a_tracked_branch_whose_manifest_is_missing(
    session_start_repository: ScratchRepository,
):
    result = publish_and_run(
        session_start_repository,
        {BRANCH_INDEX_PATH: branch_index({WORK_BRANCH: PLAN_IDENTIFIER})},
    )

    assert result.returncode == 0, result.stderr
    assert summary_value(result.stdout, "plan") == summary_message(
        SummaryMessage.PLAN_MANIFEST_MISSING,
        PLAN_IDENTIFIER,
        MANIFEST_PATH,
        NOTES_BRANCH,
    )


# %% branches no plan item can ever track


def test_reports_plan_as_not_applicable_on_the_default_branch(
    session_start_repository: ScratchRepository,
):
    session_start_repository.publish_notes_branch(
        {
            NOTES_PATH: "personal notes\n",
            BRANCH_INDEX_PATH: branch_index({WORK_BRANCH: PLAN_IDENTIFIER}),
            MANIFEST_PATH: PLAN_MANIFEST,
        }
    )
    session_start_repository.run_git("checkout", "--quiet", "-b", "main")

    result = run_session_start(session_start_repository)

    assert result.returncode == 0, result.stderr
    assert summary_value(result.stdout, "plan") == summary_message(
        SummaryMessage.PLAN_NOT_APPLICABLE
    )


def test_reports_plan_as_not_applicable_on_the_notes_branch(
    session_start_repository: ScratchRepository,
):
    session_start_repository.publish_notes_branch(
        {
            NOTES_PATH: "personal notes\n",
            BRANCH_INDEX_PATH: branch_index({WORK_BRANCH: PLAN_IDENTIFIER}),
            MANIFEST_PATH: PLAN_MANIFEST,
        }
    )
    session_start_repository.run_git("checkout", "--quiet", NOTES_BRANCH)

    result = run_session_start(session_start_repository)

    assert result.returncode == 0, result.stderr
    assert summary_value(result.stdout, "plan") == summary_message(
        SummaryMessage.PLAN_NOT_APPLICABLE
    )


# %% the setup line


def test_reports_setup_as_ok_when_every_check_passes(
    session_start_repository: ScratchRepository,
):
    result = publish_and_run(session_start_repository)

    assert result.returncode == 0, result.stderr
    assert summary_value(result.stdout, "setup") == summary_message(
        SummaryMessage.SETUP_OK
    )


def test_names_every_check_that_needs_setup(
    session_start_repository: ScratchRepository,
):
    (session_start_repository.project_root / ".claude" / "settings.json").unlink()
    session_start_repository.commit_everything("unregister the SessionStart hook")

    result = publish_and_run(session_start_repository)

    failing_check = "session_start_hook"
    detail = next(
        row.split("\t")[2]
        for row in session_start_repository.run_hook_script(
            "check-setup.sh"
        ).stdout.splitlines()
        if row.split("\t")[0] == failing_check
    )
    assert summary_value(result.stdout, "setup") == summary_message(
        SummaryMessage.CHECKS_NEED_SETUP, "1"
    )
    assert f"    {failing_check}: {detail}" in result.stdout


def test_a_failing_setup_check_does_not_fail_the_hook(
    session_start_repository: ScratchRepository,
):
    (session_start_repository.project_root / ".claude" / "settings.json").unlink()
    session_start_repository.commit_everything("unregister the SessionStart hook")

    result = publish_and_run(session_start_repository)

    assert result.returncode == 0, result.stderr
    assert (session_start_repository.project_root / CLAUDE_LOCAL_MD).exists()


# %% every message renders


def test_every_summary_message_renders_something():
    """
    Check that every message this module can name resolves to a shell function that
    prints something.

    Deliberately not an assertion about the wording: every other assertion here renders
    its expectation from session-start-messages.sh, so a reword moves both sides at once
    and none of them fail. What is still worth catching is a member naming a function
    that does not exist, or one that prints nothing - neither of which survives here.

    Three placeholder arguments cover the widest message; a shell function ignores the
    ones it does not read.
    """
    for message in SummaryMessage:
        assert summary_message(message, "first", "second", "third").strip() != ""
