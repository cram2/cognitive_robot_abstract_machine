"""
Tests for refresh_dashboard.sh's own orchestration logic: argument parsing, the
correction-triggers-a-push gate, and the optional --tracking-url passthrough.

The two modules it calls (basstler.sync_manifest_status, basstler.build_dashboard) and
the personal-notes push (write-personal-notes-file.sh) are replaced with stubs in a
scratch project-root layout, so these tests exercise only refresh_dashboard.sh's own
shell logic - no real git remote, network access, or GitHub data is involved.
basstler.refresh_dashboard_support has no such dependencies, so the real module is reused
unchanged.
"""

import json
import shutil
import subprocess
from pathlib import Path

import pytest

from .constants import PACKAGE_DIRECTORY, STUBS_DIRECTORY, ToolingDirectory
from .scratch_repository import install_hook_scripts_into

PLAN_DASHBOARD_DIRECTORY = ToolingDirectory.PLAN_DASHBOARD_SKILL.path
"""
Where refresh_dashboard.sh itself lives - the skill directory, not the package. Claude
Code discovers a skill by path, so its shell entry point stays with it.
"""


@pytest.fixture
def scratch_project_root(tmp_path: Path) -> Path:
    """
    Build a scratch project-root layout refresh_dashboard.sh can run
    against unmodified: the real script plus resolve-personal-notes-config.sh,
    and stand-ins for every script/hook it calls out to.

    :param tmp_path: pytest's per-test temporary directory.
    :return: The scratch project root.
    """
    plan_dashboard_directory = tmp_path / ToolingDirectory.PLAN_DASHBOARD_SKILL
    hooks_directory = tmp_path / ToolingDirectory.HOOKS
    package_directory = tmp_path / PACKAGE_DIRECTORY.name
    plan_dashboard_directory.mkdir(parents=True)
    package_directory.mkdir()

    shutil.copy(
        PLAN_DASHBOARD_DIRECTORY / "refresh_dashboard.sh",
        plan_dashboard_directory / "refresh_dashboard.sh",
    )
    install_hook_scripts_into(tmp_path, "resolve-personal-notes-config.sh")

    # The real support module, whose own dependencies are the standard library only, in a
    # scratch package the script's `python3 -m basstler.<module>` calls resolve against.
    (package_directory / "__init__.py").touch()
    shutil.copy(
        PACKAGE_DIRECTORY / "refresh_dashboard_support.py",
        package_directory / "refresh_dashboard_support.py",
    )
    shutil.copy(
        STUBS_DIRECTORY / "sync_manifest_status_stub.py",
        package_directory / "sync_manifest_status.py",
    )
    shutil.copy(
        STUBS_DIRECTORY / "build_dashboard_stub.py",
        package_directory / "build_dashboard.py",
    )
    shutil.copy(
        STUBS_DIRECTORY / "write_personal_notes_file_stub.sh",
        hooks_directory / "write-personal-notes-file.sh",
    )
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
