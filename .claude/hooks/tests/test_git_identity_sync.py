"""
Integration tests for the git identity round trip between a clone and the personal-notes
branch.

Cover both halves: session-start.sh writing a recorded identity into a clone that has
none, and save-git-identity.sh recording one in the first place. A clone that already
has its own identity is left alone, so the sync can only ever fill a gap.

Run against a scratch project root with a local bare repository standing in for the
personal-notes remote - no network access or real personal-notes branch involved.
"""

from __future__ import annotations

import subprocess

import pytest

from scratch_repository import (
    NOTES_BRANCH,
    PERSONAL_GIT_IDENTITY_PATH,
    SCRATCH_IDENTITY,
    GitIdentity,
    ScratchRepository,
)
from session_start_summary import SummaryMessage, summary_message, summary_value

NOTES_PATH = ".claude/personal/cram-notes.md"

RECORDED_IDENTITY = GitIdentity("Ada Lovelace", "ada@example.com")
"""
The identity the notes branch carries in these tests, deliberately different from
:data:`SCRATCH_IDENTITY` so the two can never be confused for one another.
"""

IDENTITY_FILE_WITHOUT_EMAIL = f"[user]\n\tname = {RECORDED_IDENTITY.name}\n"
"""
A recording that names the author but cannot authorize a commit - the half-written state
a hand-edited file can land in.
"""

SUMMARY_LABEL = "git identity"


# %% the scratch layout


@pytest.fixture
def git_identity_repository(scratch_repository: ScratchRepository) -> ScratchRepository:
    """
    A scratch repository carrying the real scripts of both halves of the round trip,
    with nothing published to the notes remote yet.

    :param scratch_repository: The initialized scratch repository and notes remote.
    :return: The same repository, ready to publish a notes branch and run the hooks.
    """
    scratch_repository.install_hook_scripts(
        "resolve-personal-notes-config.sh",
        "session-start-messages.sh",
        "session-start.sh",
        "save-git-identity.sh",
        "write-personal-notes-file.sh",
    )
    scratch_repository.write_setup_prerequisites()
    scratch_repository.commit_everything("initial commit")
    scratch_repository.resolve_notes_remote_to()
    return scratch_repository


def publish_notes_branch(
    repository: ScratchRepository, identity_file: str | None = None
) -> None:
    """
    Publish the notes file, and optionally a recorded identity, to the notes branch.

    :param repository: The fixture-built scratch repository.
    :param identity_file: The git-identity file's contents, or ``None`` to publish a
        notes branch that records no identity at all.
    """
    files = {NOTES_PATH: "personal notes\n"}
    if identity_file is not None:
        files[PERSONAL_GIT_IDENTITY_PATH] = identity_file
    repository.publish_notes_branch(files)


def run_session_start(
    repository: ScratchRepository,
) -> subprocess.CompletedProcess[str]:
    """
    Run the scratch layout's session-start.sh.

    :param repository: A fixture-built scratch repository.
    :return: The finished subprocess.
    """
    return repository.run_hook_script("session-start.sh")


# %% filling a gap in a clone that has no identity of its own


def test_sets_the_repository_local_identity_from_the_notes_branch(
    git_identity_repository: ScratchRepository,
):
    publish_notes_branch(
        git_identity_repository, RECORDED_IDENTITY.as_git_config_file()
    )
    git_identity_repository.clear_local_git_identity()

    result = run_session_start(git_identity_repository)

    assert result.returncode == 0, result.stderr
    assert git_identity_repository.local_git_identity() == RECORDED_IDENTITY


def test_reports_the_identity_it_set(git_identity_repository: ScratchRepository):
    publish_notes_branch(
        git_identity_repository, RECORDED_IDENTITY.as_git_config_file()
    )
    git_identity_repository.clear_local_git_identity()

    result = run_session_start(git_identity_repository)

    assert summary_value(result.stdout, SUMMARY_LABEL) == summary_message(
        SummaryMessage.GIT_IDENTITY_WRITTEN,
        NOTES_BRANCH,
        PERSONAL_GIT_IDENTITY_PATH,
        f"{RECORDED_IDENTITY.name} <{RECORDED_IDENTITY.email}>",
    )


# %% an identity the clone already has


def test_leaves_an_identity_the_clone_already_has_untouched(
    git_identity_repository: ScratchRepository,
):
    publish_notes_branch(
        git_identity_repository, RECORDED_IDENTITY.as_git_config_file()
    )

    result = run_session_start(git_identity_repository)

    assert result.returncode == 0, result.stderr
    assert git_identity_repository.local_git_identity() == SCRATCH_IDENTITY


def test_reports_the_identity_it_left_alone(
    git_identity_repository: ScratchRepository,
):
    publish_notes_branch(
        git_identity_repository, RECORDED_IDENTITY.as_git_config_file()
    )

    result = run_session_start(git_identity_repository)

    assert summary_value(result.stdout, SUMMARY_LABEL) == summary_message(
        SummaryMessage.CLONE_HAS_ITS_OWN_GIT_IDENTITY,
        f"{SCRATCH_IDENTITY.name} <{SCRATCH_IDENTITY.email}>",
    )


# %% a notes branch that records no identity


def test_writes_no_identity_when_none_is_recorded(
    git_identity_repository: ScratchRepository,
):
    publish_notes_branch(git_identity_repository)
    git_identity_repository.clear_local_git_identity()

    result = run_session_start(git_identity_repository)

    assert result.returncode == 0, result.stderr
    assert git_identity_repository.local_git_identity() is None


def test_reports_that_no_identity_is_recorded(
    git_identity_repository: ScratchRepository,
):
    publish_notes_branch(git_identity_repository)

    result = run_session_start(git_identity_repository)

    assert summary_value(result.stdout, SUMMARY_LABEL) == summary_message(
        SummaryMessage.NO_GIT_IDENTITY_RECORDED,
        NOTES_BRANCH,
        PERSONAL_GIT_IDENTITY_PATH,
    )


# %% a recording that is only half there


def test_writes_neither_half_of_an_identity_that_has_no_email(
    git_identity_repository: ScratchRepository,
):
    publish_notes_branch(git_identity_repository, IDENTITY_FILE_WITHOUT_EMAIL)
    git_identity_repository.clear_local_git_identity()

    result = run_session_start(git_identity_repository)

    assert result.returncode == 0, result.stderr
    assert git_identity_repository.local_git_identity() is None


def test_names_what_an_incomplete_recording_is_missing(
    git_identity_repository: ScratchRepository,
):
    publish_notes_branch(git_identity_repository, IDENTITY_FILE_WITHOUT_EMAIL)

    result = run_session_start(git_identity_repository)

    assert summary_value(result.stdout, SUMMARY_LABEL) == summary_message(
        SummaryMessage.GIT_IDENTITY_INCOMPLETE,
        PERSONAL_GIT_IDENTITY_PATH,
        NOTES_BRANCH,
    )


# %% recording an identity in the first place


def test_records_the_given_identity_on_the_notes_branch(
    git_identity_repository: ScratchRepository, tmp_path
):
    publish_notes_branch(git_identity_repository)

    result = git_identity_repository.run_hook_script(
        "save-git-identity.sh",
        "--name",
        RECORDED_IDENTITY.name,
        "--email",
        RECORDED_IDENTITY.email,
    )

    assert result.returncode == 0, result.stderr
    checkout = git_identity_repository.clone_notes_branch(tmp_path / "notes-checkout")
    assert (
        GitIdentity.from_git_config_file(checkout / PERSONAL_GIT_IDENTITY_PATH)
        == RECORDED_IDENTITY
    )


def test_recording_the_same_identity_again_pushes_nothing(
    git_identity_repository: ScratchRepository,
):
    publish_notes_branch(git_identity_repository)
    arguments = (
        "--name",
        RECORDED_IDENTITY.name,
        "--email",
        RECORDED_IDENTITY.email,
    )
    git_identity_repository.run_hook_script("save-git-identity.sh", *arguments)

    result = git_identity_repository.run_hook_script("save-git-identity.sh", *arguments)

    assert result.returncode == 0, result.stderr
    assert "already up to date" in result.stdout


def test_refuses_to_record_an_identity_with_no_name(
    git_identity_repository: ScratchRepository,
):
    publish_notes_branch(git_identity_repository)

    result = git_identity_repository.run_hook_script(
        "save-git-identity.sh", "--email", RECORDED_IDENTITY.email
    )

    assert result.returncode == 1
    assert "--name" in result.stderr


def test_refuses_to_record_an_identity_with_no_email(
    git_identity_repository: ScratchRepository,
):
    publish_notes_branch(git_identity_repository)

    result = git_identity_repository.run_hook_script(
        "save-git-identity.sh", "--name", RECORDED_IDENTITY.name
    )

    assert result.returncode == 1
    assert "--email" in result.stderr
