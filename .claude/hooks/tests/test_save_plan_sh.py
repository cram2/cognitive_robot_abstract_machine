"""
Integration tests for save-plan.sh's argument validation and CLAUDE.local.md.

marker-block extraction, run against a local `git init --bare` fixture instead of a
real remote - no network access or real personal-notes branch involved.
"""

import shutil
import subprocess
from pathlib import Path

import pytest

import plan_manifest_tools

HOOKS_SOURCE_DIRECTORY = Path(plan_manifest_tools.__file__).parent
FIXTURES_DIRECTORY = Path(__file__).parent / "fixtures"

PLAN_MANIFEST = (FIXTURES_DIRECTORY / "plan.yaml").read_text()
PLAN_ROADMAP = (FIXTURES_DIRECTORY / "roadmap.md").read_text()


def _run_git(*arguments: str, cwd: Path) -> subprocess.CompletedProcess[str]:
    result = subprocess.run(
        ["git", *arguments], cwd=cwd, capture_output=True, text=True
    )
    assert result.returncode == 0, result.stderr
    return result


@pytest.fixture
def scratch_repo(tmp_path: Path) -> Path:
    """
    Build a scratch project root with the real save-plan.sh/ resolve-personal-notes-
    config.sh/plan_manifest_tools.py, a local `git init --bare` fixture standing in for
    the personal-notes remote (already carrying an empty `claude/personal-notes`
    branch), and the notes remote pointed at that fixture via local git config.

    :param tmp_path: pytest's per-test temporary directory.
    :return: The scratch project root, checked out on a throwaway branch.
    """
    project_root = tmp_path / "project"
    hooks_directory = project_root / ".claude" / "hooks"
    hooks_directory.mkdir(parents=True)
    for script in (
        "resolve-personal-notes-config.sh",
        "save-plan.sh",
        "plan_manifest_tools.py",
    ):
        shutil.copy(HOOKS_SOURCE_DIRECTORY / script, hooks_directory / script)

    _run_git("init", "--quiet", cwd=project_root)
    # A CI runner has no ambient git identity configured - set one locally so
    # the commits below don't depend on the environment already having one.
    _run_git("config", "user.name", "Scratch Repo", cwd=project_root)
    _run_git("config", "user.email", "scratch-repo@example.com", cwd=project_root)
    (project_root / "README.md").write_text("scratch repo\n")
    _run_git("add", ".", cwd=project_root)
    _run_git("commit", "--quiet", "-m", "initial commit", cwd=project_root)

    bare_repository_path = tmp_path / "personal-notes.git"
    _run_git("init", "--quiet", "--bare", str(bare_repository_path), cwd=tmp_path)

    _run_git("checkout", "--quiet", "-b", "claude/personal-notes", cwd=project_root)
    (project_root / ".claude" / "personal").mkdir(parents=True)
    (project_root / ".claude" / "personal" / "placeholder.md").write_text("notes\n")
    _run_git("add", ".claude/personal/placeholder.md", cwd=project_root)
    _run_git("commit", "--quiet", "-m", "bootstrap personal-notes", cwd=project_root)
    _run_git(
        "push", str(bare_repository_path), "claude/personal-notes", cwd=project_root
    )
    _run_git("checkout", "--quiet", "-b", "some-work-branch", cwd=project_root)

    _run_git(
        "config",
        "claude.personalNotesRemote",
        str(bare_repository_path),
        cwd=project_root,
    )
    return project_root


def run_save_plan(
    scratch_repo: Path, *arguments: str
) -> subprocess.CompletedProcess[str]:
    """
    Run the scratch layout's save-plan.sh with *arguments*.

    :param scratch_repo: A fixture-built scratch project root.
    :param arguments: CLI arguments to pass to save-plan.sh.
    :return: The finished subprocess.
    """
    return subprocess.run(
        ["bash", str(scratch_repo / ".claude" / "hooks" / "save-plan.sh"), *arguments],
        cwd=scratch_repo,
        capture_output=True,
        text=True,
    )


# %% --manifest/--roadmap pairing


def test_manifest_without_roadmap_fails(scratch_repo: Path):
    manifest_path = scratch_repo / "plan.yaml"
    manifest_path.write_text(PLAN_MANIFEST)
    result = run_save_plan(scratch_repo, "test-plan", "--manifest", str(manifest_path))
    assert result.returncode == 1
    assert result.stderr == (
        "--manifest was given without --roadmap - they must be passed together.\n"
    )


def test_roadmap_without_manifest_fails(scratch_repo: Path):
    roadmap_path = scratch_repo / "roadmap.md"
    roadmap_path.write_text(PLAN_ROADMAP)
    result = run_save_plan(scratch_repo, "test-plan", "--roadmap", str(roadmap_path))
    assert result.returncode == 1
    assert result.stderr == (
        "--roadmap was given without --manifest - they must be passed together.\n"
    )


# %% CLAUDE.local.md marker-block extraction


def test_saves_the_manifest_and_roadmap_extracted_from_claude_local_md_markers(
    scratch_repo: Path,
):
    claude_local_md = scratch_repo / "CLAUDE.local.md"
    claude_local_md.write_text(
        "<!-- BEGIN-PLAN-MANIFEST: test-plan -->\n"
        f"{PLAN_MANIFEST}"
        "<!-- END-PLAN-MANIFEST -->\n"
        "<!-- BEGIN-PLAN-ROADMAP: test-plan -->\n"
        f"{PLAN_ROADMAP}"
        "<!-- END-PLAN-ROADMAP -->\n"
    )

    result = run_save_plan(scratch_repo, "test-plan")
    assert result.returncode == 0, result.stderr
    bare_repository_path = scratch_repo.parent / "personal-notes.git"
    assert "Saved plan 'test-plan'" in result.stdout
    assert str(bare_repository_path) in result.stdout

    verify_checkout = scratch_repo.parent / "verify-checkout"
    _run_git(
        "clone",
        "--quiet",
        "--branch",
        "claude/personal-notes",
        str(scratch_repo.parent / "personal-notes.git"),
        str(verify_checkout),
        cwd=scratch_repo.parent,
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


def test_missing_marker_pair_fails_with_a_clear_message(scratch_repo: Path):
    (scratch_repo / "CLAUDE.local.md").write_text("no markers here\n")
    result = run_save_plan(scratch_repo, "test-plan")
    assert result.returncode == 1
    assert result.stderr.startswith(
        "CLAUDE.local.md has no plan-manifest/plan-roadmap section to extract.\n"
    )
