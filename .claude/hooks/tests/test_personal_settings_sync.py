"""
Integration tests for the personal Claude Code settings round trip: session-start.sh
writing `.claude/settings.local.json` from the personal-notes branch, and save-personal-
settings.sh pushing local edits back to it.

Runs the real scripts against a local `git init --bare` fixture instead of a real
remote - no network access or real personal-notes branch involved.
"""

import os
import subprocess
from pathlib import Path

import pytest

from scratch_repository import ScratchRepository

FIXTURES_DIRECTORY = Path(__file__).parent / "fixtures"

SETTINGS_PATH_ON_NOTES_BRANCH = ".claude/personal/settings.local.json"
LOCAL_SETTINGS_PATH = ".claude/settings.local.json"

PERSONAL_SETTINGS = (FIXTURES_DIRECTORY / "personal-settings.json").read_text()
UPDATED_PERSONAL_SETTINGS = (
    FIXTURES_DIRECTORY / "personal-settings-updated.json"
).read_text()
LOCALLY_EDITED_SETTINGS = (
    FIXTURES_DIRECTORY / "personal-settings-locally-edited.json"
).read_text()


@pytest.fixture
def settings_repository(scratch_repository: ScratchRepository) -> ScratchRepository:
    """
    A scratch repository carrying the real session-start.sh, save-personal-settings.sh
    and the scripts they source, with a notes branch already published to its notes
    remote but carrying no settings yet.

    :param scratch_repository: The initialized scratch repository and notes remote.
    :return: The same repository, ready to run the settings scripts against.
    """
    scratch_repository.install_hook_scripts(
        "resolve-personal-notes-config.sh",
        "session-start-messages.sh",
        "session-start.sh",
        "save-personal-settings.sh",
        "write-personal-notes-file.sh",
    )
    scratch_repository.write("README.md", "scratch repo\n")
    scratch_repository.commit_everything("initial commit")
    scratch_repository.publish_notes_branch(
        {".claude/personal/placeholder.md": "notes\n"}
    )
    scratch_repository.resolve_notes_remote_to()
    return scratch_repository


def run_hook(
    repository: ScratchRepository, script_name: str
) -> subprocess.CompletedProcess[str]:
    """
    Run one of the scratch layout's hook scripts.

    Every ``CLAUDE_PERSONAL_NOTES_*`` variable is stripped from the inherited
    environment first, so a value that happens to be set in whoever's shell is running
    the tests can never change what they assert.

    :param repository: A fixture-built scratch repository.
    :param script_name: File name of the script under ``.claude/hooks``.
    :return: The finished subprocess, whether it succeeded or not.
    """
    environment = {
        name: value
        for name, value in os.environ.items()
        if not name.startswith("CLAUDE_PERSONAL_NOTES_")
    }
    return subprocess.run(
        ["bash", str(repository.project_root / ".claude" / "hooks" / script_name)],
        cwd=repository.project_root,
        capture_output=True,
        text=True,
        env=environment,
    )


def local_settings_of(repository: ScratchRepository) -> str:
    """
    Return the project's current `.claude/settings.local.json` content.

    :param repository: The scratch repository to read from.
    :return: The file's content.
    """
    return (repository.project_root / LOCAL_SETTINGS_PATH).read_text()


def settings_on_notes_branch(repository: ScratchRepository) -> str:
    """
    Return the settings the notes branch actually carries, read from a fresh checkout of
    the notes remote rather than from what a script reported.

    :param repository: The scratch repository whose notes remote to read.
    :return: The committed content.
    """
    checkout = repository.clone_notes_branch(
        repository.project_root.parent / "settings-verify-checkout"
    )
    return (checkout / SETTINGS_PATH_ON_NOTES_BRANCH).read_text()


# %% syncing settings out of the personal-notes branch


def test_writes_the_branch_settings_when_the_project_has_none(
    settings_repository: ScratchRepository,
):
    settings_repository.update_notes_branch_file(
        SETTINGS_PATH_ON_NOTES_BRANCH, PERSONAL_SETTINGS
    )

    result = run_hook(settings_repository, "session-start.sh")

    assert result.returncode == 0, result.stderr
    assert local_settings_of(settings_repository) == PERSONAL_SETTINGS
    assert f"local settings:  synced to {LOCAL_SETTINGS_PATH}" in result.stdout


def test_writes_no_settings_when_the_branch_has_none(
    settings_repository: ScratchRepository,
):
    result = run_hook(settings_repository, "session-start.sh")

    assert result.returncode == 0, result.stderr
    assert not (settings_repository.project_root / LOCAL_SETTINGS_PATH).exists()
    assert (
        f"local settings:  none on 'claude/personal-notes' "
        f"({SETTINGS_PATH_ON_NOTES_BRANCH})" in result.stdout
    )


def test_updates_settings_untouched_since_the_last_sync(
    settings_repository: ScratchRepository,
):
    settings_repository.update_notes_branch_file(
        SETTINGS_PATH_ON_NOTES_BRANCH, PERSONAL_SETTINGS
    )
    run_hook(settings_repository, "session-start.sh")
    settings_repository.update_notes_branch_file(
        SETTINGS_PATH_ON_NOTES_BRANCH, UPDATED_PERSONAL_SETTINGS
    )

    result = run_hook(settings_repository, "session-start.sh")

    assert result.returncode == 0, result.stderr
    assert local_settings_of(settings_repository) == UPDATED_PERSONAL_SETTINGS


def test_keeps_settings_edited_since_the_last_sync(
    settings_repository: ScratchRepository,
):
    settings_repository.update_notes_branch_file(
        SETTINGS_PATH_ON_NOTES_BRANCH, PERSONAL_SETTINGS
    )
    run_hook(settings_repository, "session-start.sh")
    settings_repository.write(LOCAL_SETTINGS_PATH, LOCALLY_EDITED_SETTINGS)
    settings_repository.update_notes_branch_file(
        SETTINGS_PATH_ON_NOTES_BRANCH, UPDATED_PERSONAL_SETTINGS
    )

    result = run_hook(settings_repository, "session-start.sh")

    assert result.returncode == 0, result.stderr
    assert local_settings_of(settings_repository) == LOCALLY_EDITED_SETTINGS
    assert (
        f"local settings:  kept local edits to {LOCAL_SETTINGS_PATH} - run "
        "save-personal-settings.sh to push them" in result.stdout
    )


def test_keeps_settings_that_were_never_synced(
    settings_repository: ScratchRepository,
):
    settings_repository.write(LOCAL_SETTINGS_PATH, LOCALLY_EDITED_SETTINGS)
    settings_repository.update_notes_branch_file(
        SETTINGS_PATH_ON_NOTES_BRANCH, PERSONAL_SETTINGS
    )

    result = run_hook(settings_repository, "session-start.sh")

    assert result.returncode == 0, result.stderr
    assert local_settings_of(settings_repository) == LOCALLY_EDITED_SETTINGS


# %% saving local settings back to the personal-notes branch


def test_saves_local_settings_to_the_branch(settings_repository: ScratchRepository):
    settings_repository.write(LOCAL_SETTINGS_PATH, PERSONAL_SETTINGS)

    result = run_hook(settings_repository, "save-personal-settings.sh")

    assert result.returncode == 0, result.stderr
    assert settings_on_notes_branch(settings_repository) == PERSONAL_SETTINGS


def test_saved_settings_are_no_longer_treated_as_local_edits(
    settings_repository: ScratchRepository,
):
    settings_repository.write(LOCAL_SETTINGS_PATH, LOCALLY_EDITED_SETTINGS)
    run_hook(settings_repository, "save-personal-settings.sh")
    settings_repository.update_notes_branch_file(
        SETTINGS_PATH_ON_NOTES_BRANCH, UPDATED_PERSONAL_SETTINGS
    )

    result = run_hook(settings_repository, "session-start.sh")

    assert result.returncode == 0, result.stderr
    assert local_settings_of(settings_repository) == UPDATED_PERSONAL_SETTINGS


def test_saving_without_local_settings_fails_with_a_clear_message(
    settings_repository: ScratchRepository,
):
    result = run_hook(settings_repository, "save-personal-settings.sh")

    assert result.returncode == 1
    assert result.stderr.startswith(
        f"No {LOCAL_SETTINGS_PATH} at the project root "
        f"({settings_repository.project_root / LOCAL_SETTINGS_PATH})"
        " - nothing to save.\n"
    )
