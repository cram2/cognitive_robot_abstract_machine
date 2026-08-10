"""
Integration tests for save-plan.sh's argument validation and CLAUDE.local.md.

marker-block extraction, run against a local `git init --bare` fixture instead of a
real remote - no network access or real personal-notes branch involved.
"""

import subprocess
from pathlib import Path

import pytest

from scratch_repository import NOTES_BRANCH, ScratchRepository

FIXTURES_DIRECTORY = Path(__file__).parent / "fixtures"

PLAN_MANIFEST = (FIXTURES_DIRECTORY / "plan.yaml").read_text()
PLAN_ROADMAP = (FIXTURES_DIRECTORY / "roadmap.md").read_text()


@pytest.fixture
def save_plan_repository(scratch_repository: ScratchRepository) -> ScratchRepository:
    """
    A scratch repository carrying the real save-plan.sh, resolve-personal-notes-
    config.sh and plan_manifest_tools.py, with a notes branch already published to its
    notes remote.

    :param scratch_repository: The initialized scratch repository and notes remote.
    :return: The same repository, ready to run save-plan.sh against.
    """
    scratch_repository.install_hook_scripts(
        "resolve-personal-notes-config.sh",
        "save-plan.sh",
        "plan_manifest_tools.py",
    )
    scratch_repository.write("README.md", "scratch repo\n")
    scratch_repository.commit_everything("initial commit")
    scratch_repository.publish_notes_branch(
        {".claude/personal/placeholder.md": "notes\n"}
    )
    scratch_repository.resolve_notes_remote_to()
    return scratch_repository


def run_save_plan(
    repository: ScratchRepository, *arguments: str
) -> subprocess.CompletedProcess[str]:
    """
    Run the scratch layout's save-plan.sh with *arguments*.

    :param repository: A fixture-built scratch repository.
    :param arguments: CLI arguments to pass to save-plan.sh.
    :return: The finished subprocess.
    """
    return subprocess.run(
        [
            "bash",
            str(repository.project_root / ".claude" / "hooks" / "save-plan.sh"),
            *arguments,
        ],
        cwd=repository.project_root,
        capture_output=True,
        text=True,
    )


# %% --manifest/--roadmap pairing


def test_manifest_without_roadmap_fails(save_plan_repository: ScratchRepository):
    manifest_path = save_plan_repository.write("plan.yaml", PLAN_MANIFEST)
    result = run_save_plan(
        save_plan_repository, "test-plan", "--manifest", str(manifest_path)
    )
    assert result.returncode == 1
    assert result.stderr == (
        "--manifest was given without --roadmap - they must be passed together.\n"
    )


def test_roadmap_without_manifest_fails(save_plan_repository: ScratchRepository):
    roadmap_path = save_plan_repository.write("roadmap.md", PLAN_ROADMAP)
    result = run_save_plan(
        save_plan_repository, "test-plan", "--roadmap", str(roadmap_path)
    )
    assert result.returncode == 1
    assert result.stderr == (
        "--roadmap was given without --manifest - they must be passed together.\n"
    )


# %% CLAUDE.local.md marker-block extraction


def test_saves_the_manifest_and_roadmap_extracted_from_claude_local_md_markers(
    save_plan_repository: ScratchRepository,
):
    save_plan_repository.write(
        "CLAUDE.local.md",
        "<!-- BEGIN-PLAN-MANIFEST: test-plan -->\n"
        f"{PLAN_MANIFEST}"
        "<!-- END-PLAN-MANIFEST -->\n"
        "<!-- BEGIN-PLAN-ROADMAP: test-plan -->\n"
        f"{PLAN_ROADMAP}"
        "<!-- END-PLAN-ROADMAP -->\n",
    )

    result = run_save_plan(save_plan_repository, "test-plan")
    assert result.returncode == 0, result.stderr
    assert "Saved plan 'test-plan'" in result.stdout
    assert str(save_plan_repository.notes_remote_path) in result.stdout

    verify_checkout = save_plan_repository.clone_notes_branch(
        save_plan_repository.project_root.parent / "verify-checkout"
    )
    saved_manifest = (
        verify_checkout / ".claude" / "personal" / "plans" / "test-plan" / "plan.yaml"
    ).read_text()
    saved_roadmap = (
        verify_checkout / ".claude" / "personal" / "plans" / "test-plan" / "roadmap.md"
    ).read_text()
    assert saved_manifest == PLAN_MANIFEST
    assert saved_roadmap == PLAN_ROADMAP

    branch_index = (
        verify_checkout
        / ".claude"
        / "personal"
        / "plans"
        / "_generated"
        / "branch-index.tsv"
    ).read_text()
    assert branch_index == "item-a-branch\ttest-plan\n"


def test_missing_marker_pair_fails_with_a_clear_message(
    save_plan_repository: ScratchRepository,
):
    save_plan_repository.write("CLAUDE.local.md", "no markers here\n")
    result = run_save_plan(save_plan_repository, "test-plan")
    assert result.returncode == 1
    assert result.stderr.startswith(
        "CLAUDE.local.md has no plan-manifest/plan-roadmap section to extract.\n"
    )
