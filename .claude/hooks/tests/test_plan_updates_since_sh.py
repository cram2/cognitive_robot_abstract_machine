"""
Integration tests for plan-updates-since.sh: the recheck-deltas helper that diffs a
plan's directory since a baseline commit and prints tracking-issue comments newer than
that commit's timestamp.

Run against the shared ScratchRepository fixture (see conftest.py/scratch_repository.py)
and stubbed `gh`/`curl` executables standing in for the two GitHub backends - no network
access, no real personal-notes branch, no real GitHub credentials.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
from pathlib import Path

import pytest

from plan_updates_since_support import (
    NO_CHANGES_MESSAGE,
    NO_TRACKING_ISSUE_MESSAGE,
    IssueComment,
    PlanUpdatesSinceOption,
    no_default_repository_message,
)
from scratch_repository import ScratchRepository

FIXTURES_DIRECTORY = Path(__file__).parent / "fixtures"
STUBS_DIRECTORY = Path(__file__).parent / "stubs"

PLAN_MANIFEST_NOT_STARTED = (FIXTURES_DIRECTORY / "plan.yaml").read_text()
PLAN_MANIFEST_IN_PROGRESS = (FIXTURES_DIRECTORY / "plan-in-progress.yaml").read_text()
PLAN_MANIFEST_WITH_TRACKING_ISSUE = (
    FIXTURES_DIRECTORY / "plan-with-tracking-issue.yaml"
).read_text()
PLAN_MANIFEST_TRACKING_ISSUE_NO_REPOSITORY = (
    FIXTURES_DIRECTORY / "plan-tracking-issue-no-repository.yaml"
).read_text()
PLAN_ROADMAP = (FIXTURES_DIRECTORY / "roadmap.md").read_text()

PLAN_ID = "test-plan"
STAMP_RELATIVE_PATH = ".claude/.plan-state-sync-sha"

TRACKING_ISSUE_REPOSITORY = "octo-org/octo-repo"
"""
The default_repository plan-with-tracking-issue.yaml sets.
"""

TRACKING_ISSUE_NUMBER = "55"
"""
The tracking_issue plan-with-tracking-issue.yaml sets.
"""

CREDENTIAL_VARIABLE_NAMES = ("GH_TOKEN", "GITHUB_TOKEN", "GH_HOST")
"""
GitHub credential variables stripped from every test subprocess's environment.

Whoever runs this suite may well have real ones set (a Claude Code session's own git-
proxy credentials do), and a test that reached GitHub with them would be neither
reproducible nor safe.
"""

PERSONAL_NOTES_VARIABLE_PREFIX = "CLAUDE_PERSONAL_NOTES_"
"""
Prefix of the settings resolve-personal-notes-config.sh reads from the environment,
stripped for the same reason: a value set in the environment actually running these
tests must never change what a test asserts.
"""


def _run_git(*arguments: str, cwd: Path) -> subprocess.CompletedProcess[str]:
    result = subprocess.run(
        ["git", *arguments], cwd=cwd, capture_output=True, text=True
    )
    assert result.returncode == 0, result.stderr
    return result


def install_plan_updates_since(repository: ScratchRepository) -> None:
    """
    Copy plan-updates-since.sh and its dependencies into *repository*, then commit an
    initial commit - the common setup every test below builds on, whether or not it
    goes on to bootstrap a personal-notes branch too.

    :param repository: A fixture-built scratch repository.
    """
    repository.install_hook_scripts(
        "resolve-personal-notes-config.sh",
        "plan-updates-since.sh",
        "plan_updates_since_support.py",
    )
    repository.write("README.md", "scratch repo\n")
    repository.commit_everything("initial commit")


@pytest.fixture
def scratch_repo(scratch_repository: ScratchRepository) -> ScratchRepository:
    """
    scratch_repository with plan-updates-since.sh installed and a personal-notes branch
    bootstrapped with a placeholder file, checked out on a throwaway work branch.

    :param scratch_repository: The shared scratch repository fixture.
    :return: The prepared repository.
    """
    install_plan_updates_since(scratch_repository)
    scratch_repository.publish_notes_branch(
        {".claude/personal/placeholder.md": "notes\n"}
    )
    scratch_repository.resolve_notes_remote_to()
    return scratch_repository


@pytest.fixture
def stub_bin(tmp_path: Path) -> Path:
    """
    An empty directory meant to be placed first on a test subprocess's PATH, into which
    a test installs whichever of the stubbed `gh`/`curl` it needs via
    :func:`install_stub`.

    :param tmp_path: pytest's per-test temporary directory.
    :return: The stub directory.
    """
    directory = tmp_path / "stub-bin"
    directory.mkdir()
    return directory


def install_stub(stub_bin: Path, executable_name: str) -> None:
    """
    Copy the stub backing *executable_name* (from ``stubs/<executable_name>.sh``) into
    *stub_bin*, executable.

    :param stub_bin: The directory built by the stub_bin fixture.
    :param executable_name: ``"gh"`` or ``"curl"``.
    """
    source = STUBS_DIRECTORY / f"{executable_name}.sh"
    destination = stub_bin / executable_name
    shutil.copy(source, destination)
    destination.chmod(0o755)


def path_hiding_executable(executable_name: str, mirror_parent: Path) -> str:
    """
    Build a ``PATH`` string equivalent to the current one but with *executable_name*
    unfindable through it.

    Mirrors (via symlinks) any directory that provides *executable_name* into a copy
    missing just that one file, rather than dropping the whole directory from `PATH` -
    the directory providing it (typically ``/usr/bin``) also provides ``bash``, ``git``
    and ``python3``, which the script under test still needs to run at all.

    :param executable_name: The executable to hide, e.g. ``"gh"``.
    :param mirror_parent: Where to create mirror directories.
    :return: The adjusted ``PATH`` string.
    """
    entries = []
    mirror_index = 0
    for entry in os.environ.get("PATH", "").split(os.pathsep):
        directory = Path(entry)
        if not directory.is_dir() or not (directory / executable_name).exists():
            entries.append(entry)
            continue
        mirror = mirror_parent / f"hide-{executable_name}-{mirror_index}"
        mirror_index += 1
        mirror.mkdir()
        for item in directory.iterdir():
            if item.name == executable_name:
                continue
            try:
                (mirror / item.name).symlink_to(item)
            except OSError:
                continue
        entries.append(str(mirror))
    return os.pathsep.join(entries)


def clean_environment() -> dict[str, str]:
    """
    Return ``os.environ`` with every GitHub credential and personal-notes override
    variable stripped, so a test's assertions can't be changed by whatever happens to be
    set in the environment actually running the suite.

    :return: The filtered environment mapping.
    """
    return {
        name: value
        for name, value in os.environ.items()
        if name not in CREDENTIAL_VARIABLE_NAMES
        and not name.startswith(PERSONAL_NOTES_VARIABLE_PREFIX)
    }


def write_plan_commit(
    repository: ScratchRepository,
    plan_id: str,
    manifest: str,
    roadmap: str,
    message: str,
) -> str:
    """
    Commit *manifest*/*roadmap* at ``.claude/personal/plans/<plan_id>/`` onto the
    personal-notes branch, via a throwaway clone so the repository's own checked-out
    branch and working tree are untouched.

    :param repository: A fixture-built scratch repository whose notes branch already
        exists (see ScratchRepository.publish_notes_branch).
    :param plan_id: The plan id to write under.
    :param manifest: The plan.yaml content to commit.
    :param roadmap: The roadmap.md content to commit.
    :param message: The commit message.
    :return: The new commit's SHA.
    """
    checkout = (
        repository.project_root.parent / f"seed-checkout-{message.replace(' ', '-')}"
    )
    shutil.rmtree(checkout, ignore_errors=True)
    _run_git(
        "clone",
        "--quiet",
        "--branch",
        "claude/personal-notes",
        str(repository.notes_remote_path),
        str(checkout),
        cwd=repository.project_root.parent,
    )
    _run_git("config", "user.name", "Scratch Repo", cwd=checkout)
    _run_git("config", "user.email", "scratch-repo@example.com", cwd=checkout)
    plan_directory = checkout / ".claude" / "personal" / "plans" / plan_id
    plan_directory.mkdir(parents=True, exist_ok=True)
    (plan_directory / "plan.yaml").write_text(manifest)
    (plan_directory / "roadmap.md").write_text(roadmap)
    _run_git("add", f".claude/personal/plans/{plan_id}", cwd=checkout)
    _run_git("commit", "--quiet", "-m", message, cwd=checkout)
    _run_git("push", "--quiet", "origin", "claude/personal-notes", cwd=checkout)
    sha = _run_git("rev-parse", "HEAD", cwd=checkout).stdout.strip()
    shutil.rmtree(checkout)
    return sha


def run_plan_updates_since(
    repository: ScratchRepository, *arguments: str, env: dict[str, str] | None = None
) -> subprocess.CompletedProcess[str]:
    """
    Run the scratch repository's plan-updates-since.sh with *arguments*.

    :param repository: A fixture-built scratch repository.
    :param arguments: CLI arguments to pass to plan-updates-since.sh.
    :param env: Environment overrides applied on top of a cleaned copy of this process's
        own environment (see clean_environment).
    :return: The finished subprocess.
    """
    full_environment = {**clean_environment(), **(env or {})}
    return subprocess.run(
        [
            "bash",
            str(
                repository.project_root / ".claude" / "hooks" / "plan-updates-since.sh"
            ),
            *arguments,
        ],
        cwd=repository.project_root,
        capture_output=True,
        text=True,
        env=full_environment,
    )


# %% argument validation


def test_missing_plan_id_fails_with_a_clear_message(scratch_repo: ScratchRepository):
    result = run_plan_updates_since(scratch_repo)
    assert result.returncode == 1
    assert result.stderr.startswith("Usage:")


def test_unreachable_personal_notes_branch_fails_clearly(
    scratch_repository: ScratchRepository,
):
    install_plan_updates_since(scratch_repository)
    scratch_repository.resolve_notes_remote_to()

    result = run_plan_updates_since(scratch_repository, PLAN_ID)

    assert result.returncode == 1
    assert "doesn't exist yet" in result.stderr


def test_unknown_plan_id_fails_clearly(scratch_repo: ScratchRepository):
    result = run_plan_updates_since(scratch_repo, "no-such-plan")

    assert result.returncode == 1
    assert "No such plan 'no-such-plan'" in result.stderr


def test_no_since_and_no_stamp_fails_with_a_clear_message(
    scratch_repo: ScratchRepository,
):
    write_plan_commit(
        scratch_repo, PLAN_ID, PLAN_MANIFEST_NOT_STARTED, PLAN_ROADMAP, "seed"
    )

    result = run_plan_updates_since(scratch_repo, PLAN_ID)

    assert result.returncode == 1
    assert "No baseline SHA known" in result.stderr


# %% diffing the plan directory


def test_explicit_since_diffs_the_plan_directory(scratch_repo: ScratchRepository):
    sha_before = write_plan_commit(
        scratch_repo, PLAN_ID, PLAN_MANIFEST_NOT_STARTED, PLAN_ROADMAP, "v1"
    )
    write_plan_commit(
        scratch_repo, PLAN_ID, PLAN_MANIFEST_IN_PROGRESS, PLAN_ROADMAP, "v2"
    )

    result = run_plan_updates_since(
        scratch_repo, PLAN_ID, PlanUpdatesSinceOption.SINCE, sha_before
    )

    assert result.returncode == 0, result.stderr
    assert "-    status: not_started" in result.stdout
    assert "+    status: in_progress" in result.stdout


def test_no_changes_since_the_baseline_prints_a_clear_message(
    scratch_repo: ScratchRepository,
):
    sha = write_plan_commit(
        scratch_repo, PLAN_ID, PLAN_MANIFEST_NOT_STARTED, PLAN_ROADMAP, "v1"
    )

    result = run_plan_updates_since(
        scratch_repo, PLAN_ID, PlanUpdatesSinceOption.SINCE, sha
    )

    assert result.returncode == 0, result.stderr
    assert NO_CHANGES_MESSAGE in result.stdout


# %% the recheck stamp


def test_uses_the_recorded_stamp_when_since_is_omitted(scratch_repo: ScratchRepository):
    sha_before = write_plan_commit(
        scratch_repo, PLAN_ID, PLAN_MANIFEST_NOT_STARTED, PLAN_ROADMAP, "v1"
    )
    (scratch_repo.project_root / STAMP_RELATIVE_PATH).write_text(f"{sha_before}\n")
    write_plan_commit(
        scratch_repo, PLAN_ID, PLAN_MANIFEST_IN_PROGRESS, PLAN_ROADMAP, "v2"
    )

    result = run_plan_updates_since(scratch_repo, PLAN_ID)

    assert result.returncode == 0, result.stderr
    assert "+    status: in_progress" in result.stdout


def test_stamp_is_updated_after_running(scratch_repo: ScratchRepository):
    sha = write_plan_commit(
        scratch_repo, PLAN_ID, PLAN_MANIFEST_NOT_STARTED, PLAN_ROADMAP, "v1"
    )

    result = run_plan_updates_since(
        scratch_repo, PLAN_ID, PlanUpdatesSinceOption.SINCE, sha
    )
    assert result.returncode == 0, result.stderr
    assert (scratch_repo.project_root / STAMP_RELATIVE_PATH).read_text().strip() == sha

    second_result = run_plan_updates_since(scratch_repo, PLAN_ID)
    assert second_result.returncode == 0, second_result.stderr
    assert NO_CHANGES_MESSAGE in second_result.stdout


# %% tracking-issue comments


def test_no_tracking_issue_skips_the_comments_step(scratch_repo: ScratchRepository):
    sha = write_plan_commit(
        scratch_repo, PLAN_ID, PLAN_MANIFEST_NOT_STARTED, PLAN_ROADMAP, "v1"
    )

    result = run_plan_updates_since(
        scratch_repo, PLAN_ID, PlanUpdatesSinceOption.SINCE, sha
    )

    assert result.returncode == 0, result.stderr
    assert NO_TRACKING_ISSUE_MESSAGE in result.stdout


def test_tracking_issue_without_default_repository_fails_clearly(
    scratch_repo: ScratchRepository,
):
    sha = write_plan_commit(
        scratch_repo,
        PLAN_ID,
        PLAN_MANIFEST_TRACKING_ISSUE_NO_REPOSITORY,
        PLAN_ROADMAP,
        "v1",
    )

    result = run_plan_updates_since(
        scratch_repo, PLAN_ID, PlanUpdatesSinceOption.SINCE, sha
    )

    assert result.returncode == 1
    assert no_default_repository_message(PLAN_ID, TRACKING_ISSUE_NUMBER) in (
        result.stderr
    )


def test_prints_tracking_issue_comments_via_the_gh_backend(
    scratch_repo: ScratchRepository, stub_bin: Path, tmp_path: Path
):
    install_stub(stub_bin, "gh")
    sha = write_plan_commit(
        scratch_repo, PLAN_ID, PLAN_MANIFEST_WITH_TRACKING_ISSUE, PLAN_ROADMAP, "v1"
    )
    call_log = tmp_path / "gh-calls.txt"
    comment = IssueComment(
        author_login="octocat", created_at="2026-08-01T00:00:00Z", body="Looks good"
    )
    comments_json = json.dumps([comment.to_api_response()])

    result = run_plan_updates_since(
        scratch_repo,
        PLAN_ID,
        PlanUpdatesSinceOption.SINCE,
        sha,
        env={
            "PATH": f"{stub_bin}{os.pathsep}{os.environ.get('PATH', '')}",
            "STUB_GH_ISSUE_COMMENTS_JSON": comments_json,
            "STUB_GH_CALL_LOG": str(call_log),
        },
    )

    assert result.returncode == 0, result.stderr
    assert comment.formatted() in result.stdout
    logged_call = call_log.read_text()
    assert (
        f"repos/{TRACKING_ISSUE_REPOSITORY}/issues/{TRACKING_ISSUE_NUMBER}/comments"
        in logged_call
    )
    assert "since=" in logged_call


def test_prints_tracking_issue_comments_via_the_curl_fallback(
    scratch_repo: ScratchRepository, stub_bin: Path, tmp_path: Path
):
    install_stub(stub_bin, "curl")
    sha = write_plan_commit(
        scratch_repo, PLAN_ID, PLAN_MANIFEST_WITH_TRACKING_ISSUE, PLAN_ROADMAP, "v1"
    )
    call_log = tmp_path / "curl-calls.txt"
    comment = IssueComment(
        author_login="hubot", created_at="2026-08-01T01:00:00Z", body="Ship it"
    )
    comments_json = json.dumps([comment.to_api_response()])
    hidden_gh_path = path_hiding_executable("gh", tmp_path)

    result = run_plan_updates_since(
        scratch_repo,
        PLAN_ID,
        PlanUpdatesSinceOption.SINCE,
        sha,
        env={
            "PATH": f"{stub_bin}{os.pathsep}{hidden_gh_path}",
            "GH_TOKEN": "a-token",
            "STUB_CURL_ISSUE_COMMENTS_JSON": comments_json,
            "STUB_CURL_CALL_LOG": str(call_log),
        },
    )

    assert result.returncode == 0, result.stderr
    assert comment.formatted() in result.stdout
    logged_call = call_log.read_text()
    assert (
        f"repos/{TRACKING_ISSUE_REPOSITORY}/issues/{TRACKING_ISSUE_NUMBER}/comments"
        in logged_call
    )


def test_fails_when_neither_gh_nor_a_token_is_available(
    scratch_repo: ScratchRepository, tmp_path: Path
):
    sha = write_plan_commit(
        scratch_repo, PLAN_ID, PLAN_MANIFEST_WITH_TRACKING_ISSUE, PLAN_ROADMAP, "v1"
    )
    hidden_gh_path = path_hiding_executable("gh", tmp_path)

    result = run_plan_updates_since(
        scratch_repo,
        PLAN_ID,
        PlanUpdatesSinceOption.SINCE,
        sha,
        env={"PATH": hidden_gh_path},
    )

    assert result.returncode != 0
    assert "No GitHub credentials available" in result.stderr
