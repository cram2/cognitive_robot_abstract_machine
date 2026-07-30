"""
Tests for refresh_dashboard.sh's own orchestration logic: argument parsing, the
correction-triggers-a-push gate, and the optional --tracking-url passthrough.

The two Python scripts it calls (sync_manifest_status.py, build_dashboard.py) and
the personal-notes push (write-personal-notes-file.sh) are replaced with stubs in a
scratch project-root layout, so these tests exercise only refresh_dashboard.sh's own
shell logic - no real git remote, network access, or GitHub data is involved.
refresh_dashboard_support.py has no such dependencies, so the real script is reused
unchanged.
"""

import json
import shutil
import subprocess
from pathlib import Path

import pytest

import refresh_dashboard_support

PLAN_DASHBOARD_DIRECTORY = Path(refresh_dashboard_support.__file__).parent


def _repository_root() -> Path:
    """
    Walk upward from *PLAN_DASHBOARD_DIRECTORY* until a directory containing
    pyproject.toml is found.

    :return: The repository root.
    """
    for candidate in (PLAN_DASHBOARD_DIRECTORY, *PLAN_DASHBOARD_DIRECTORY.parents):
        if (candidate / "pyproject.toml").is_file():
            return candidate
    raise RuntimeError(f"No pyproject.toml found above {PLAN_DASHBOARD_DIRECTORY}.")


REPOSITORY_ROOT = _repository_root()
STUBS_DIRECTORY = Path(__file__).parent / "fixtures" / "stubs"


@pytest.fixture
def scratch_project_root(tmp_path: Path) -> Path:
    """
    Build a scratch project-root layout refresh_dashboard.sh can run
    against unmodified: the real script plus resolve-personal-notes-config.sh,
    and stand-ins for every script/hook it calls out to.

    :param tmp_path: pytest's per-test temporary directory.
    :return: The scratch project root.
    """
    plan_dashboard_directory = tmp_path / ".claude" / "skills" / "plan-dashboard"
    hooks_directory = tmp_path / ".claude" / "hooks"
    plan_dashboard_directory.mkdir(parents=True)
    hooks_directory.mkdir(parents=True)

    shutil.copy(
        PLAN_DASHBOARD_DIRECTORY / "refresh_dashboard.sh",
        plan_dashboard_directory / "refresh_dashboard.sh",
    )
    shutil.copy(
        PLAN_DASHBOARD_DIRECTORY / "refresh_dashboard_support.py",
        plan_dashboard_directory / "refresh_dashboard_support.py",
    )
    shutil.copy(
        REPOSITORY_ROOT / ".claude" / "hooks" / "resolve-personal-notes-config.sh",
        hooks_directory / "resolve-personal-notes-config.sh",
    )

    shutil.copy(
        STUBS_DIRECTORY / "sync_manifest_status_stub.py",
        plan_dashboard_directory / "sync_manifest_status.py",
    )
    shutil.copy(
        STUBS_DIRECTORY / "build_dashboard_stub.py",
        plan_dashboard_directory / "build_dashboard.py",
    )
    shutil.copy(
        STUBS_DIRECTORY / "write_personal_notes_file_stub.sh",
        hooks_directory / "write-personal-notes-file.sh",
    )
    for stub in ("sync_manifest_status.py", "build_dashboard.py"):
        (plan_dashboard_directory / stub).chmod(0o755)
    (hooks_directory / "write-personal-notes-file.sh").chmod(0o755)

    return tmp_path


def _refresh_dashboard_script_path(scratch_project_root: Path) -> Path:
    """
    The scratch layout's copy of refresh_dashboard.sh.

    :param scratch_project_root: A fixture-built scratch project root.
    """
    return (
        scratch_project_root
        / ".claude"
        / "skills"
        / "plan-dashboard"
        / "refresh_dashboard.sh"
    )


def run_refresh_dashboard(
    scratch_project_root: Path, corrected_ids: list[str], extra_arguments: list[str]
) -> subprocess.CompletedProcess[str]:
    """
    Run the scratch layout's refresh_dashboard.sh, working directory set to the scratch
    root so any relative file it writes lands there.

    :param scratch_project_root: A fixture-built scratch project root.
    :param corrected_ids: Item ids the sync_manifest_status.py stub should report as
        corrected. Written into --pr-data's file, which the stub echoes back verbatim.
    :param extra_arguments: Additional CLI arguments to pass through.
    :return: The finished subprocess.
    """
    plan_path = scratch_project_root / "plan.yaml"
    plan_path.write_text("schema_version: 1\n")
    pull_request_data_path = scratch_project_root / "pr_data.json"
    pull_request_data_path.write_text(
        json.dumps([{"id": identifier} for identifier in corrected_ids])
    )
    output_path = scratch_project_root / "dashboard.html"

    return subprocess.run(
        [
            "bash",
            str(_refresh_dashboard_script_path(scratch_project_root)),
            "--plan-id",
            "test-plan",
            "--plan",
            str(plan_path),
            "--roadmap",
            str(scratch_project_root / "roadmap.md"),
            "--pr-data",
            str(pull_request_data_path),
            "--output",
            str(output_path),
            *extra_arguments,
        ],
        cwd=scratch_project_root,
        capture_output=True,
        text=True,
    )


# %% argument parsing


def test_missing_required_argument_fails_with_usage(scratch_project_root: Path):
    (scratch_project_root / "roadmap.md").write_text("")
    script_path = _refresh_dashboard_script_path(scratch_project_root)
    result = subprocess.run(
        ["bash", str(script_path), "--plan-id", "test-plan"],
        cwd=scratch_project_root,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 1
    assert result.stderr.startswith(f"Usage: {script_path} ")


def test_unrecognized_argument_fails(scratch_project_root: Path):
    script_path = _refresh_dashboard_script_path(scratch_project_root)
    result = subprocess.run(
        ["bash", str(script_path), "--not-a-real-flag", "value"],
        cwd=scratch_project_root,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 1
    assert result.stderr == "Unrecognized argument: --not-a-real-flag\n"


# %% correction-triggers-a-push gate


def test_zero_corrections_does_not_push_to_personal_notes(scratch_project_root: Path):
    (scratch_project_root / "roadmap.md").write_text("")
    result = run_refresh_dashboard(
        scratch_project_root, corrected_ids=[], extra_arguments=[]
    )
    assert result.returncode == 0, result.stderr
    assert not (
        scratch_project_root / "write_personal_notes_file_invocation.txt"
    ).exists()


def test_a_correction_pushes_to_personal_notes(scratch_project_root: Path):
    (scratch_project_root / "roadmap.md").write_text("")
    result = run_refresh_dashboard(
        scratch_project_root, corrected_ids=["a"], extra_arguments=[]
    )
    assert result.returncode == 0, result.stderr
    invocation = (
        scratch_project_root / "write_personal_notes_file_invocation.txt"
    ).read_text()
    plan_path = scratch_project_root / "plan.yaml"
    assert f"--source\n{plan_path}\n" in invocation
    assert "--destination\n.claude/personal/plans/test-plan/plan.yaml\n" in invocation
    assert "1 item(s) to done" in invocation


# %% --tracking-url passthrough


def test_tracking_url_is_not_forwarded_when_omitted(scratch_project_root: Path):
    (scratch_project_root / "roadmap.md").write_text("")
    run_refresh_dashboard(scratch_project_root, corrected_ids=[], extra_arguments=[])
    build_invocation = json.loads(
        (scratch_project_root / "build_dashboard_invocation.json").read_text()
    )
    assert "--tracking-url" not in build_invocation


def test_tracking_url_is_forwarded_when_given(scratch_project_root: Path):
    (scratch_project_root / "roadmap.md").write_text("")
    run_refresh_dashboard(
        scratch_project_root,
        corrected_ids=[],
        extra_arguments=["--tracking-url", "https://github.com/owner/repo/issues/1"],
    )
    build_invocation = json.loads(
        (scratch_project_root / "build_dashboard_invocation.json").read_text()
    )
    tracking_url_index = build_invocation.index("--tracking-url")
    assert build_invocation[tracking_url_index + 1] == (
        "https://github.com/owner/repo/issues/1"
    )


# %% final merged summary


def test_prints_the_merged_sync_and_build_summary(scratch_project_root: Path):
    (scratch_project_root / "roadmap.md").write_text("")
    result = run_refresh_dashboard(
        scratch_project_root, corrected_ids=["a"], extra_arguments=[]
    )
    assert result.returncode == 0, result.stderr
    merged = json.loads(result.stdout)
    assert merged == {"corrected": [{"id": "a"}], "drift_count": 0}
