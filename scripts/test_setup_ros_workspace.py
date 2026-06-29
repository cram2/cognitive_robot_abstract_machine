from __future__ import annotations

import os
import stat
import subprocess
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class ShellEnvironment:
    """Filesystem paths used to run the setup wrapper in isolation."""

    repository_root: Path
    """Absolute path to the repository under test."""

    temporary_home: Path
    """Temporary home directory used while running the wrapper."""

    python_log: Path
    """File containing the setup tool path passed to the fake interpreter."""

    bin_path: Path
    """Directory containing executable stubs used by the test."""


def create_shell_environment(temporary_path: Path) -> ShellEnvironment:
    """Create an isolated shell environment for the wrapper test."""
    repository_root = Path(__file__).resolve().parents[1]
    bin_path = temporary_path / "bin"
    temporary_home = temporary_path / "home"
    python_log = temporary_path / "python_invocation.txt"

    bin_path.mkdir()
    temporary_home.mkdir()
    fake_python_path = bin_path / "python3"
    fake_python_path.write_text(
        "#!/bin/sh\n"
        'if [ ! -f "$1" ]; then\n'
        '  echo "missing setup file: $1" >&2\n'
        "  exit 64\n"
        "fi\n"
        'printf "%s\\n" "$1" > "$FAKE_PYTHON_LOG"\n'
        "exit 0\n"
    )
    fake_python_path.chmod(stat.S_IRWXU)

    return ShellEnvironment(
        repository_root=repository_root,
        temporary_home=temporary_home,
        python_log=python_log,
        bin_path=bin_path,
    )


def test_setup_ros_workspace_delegates_to_existing_setup_tool(
    tmp_path: Path,
) -> None:
    """The ROS workspace wrapper delegates to a setup implementation that exists."""
    shell_environment = create_shell_environment(tmp_path)
    environment = os.environ.copy()
    environment["FAKE_PYTHON_LOG"] = str(shell_environment.python_log)
    environment["HOME"] = str(shell_environment.temporary_home)
    environment["OVERLAY_WS"] = str(tmp_path / "ros_ws")
    environment["PATH"] = (
        f"{shell_environment.bin_path}{os.pathsep}{environment['PATH']}"
    )

    result = subprocess.run(
        [str(shell_environment.repository_root / "scripts/setup_ros_workspace.sh")],
        cwd=shell_environment.repository_root,
        env=environment,
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    invoked_setup_tool = Path(shell_environment.python_log.read_text().strip())
    assert invoked_setup_tool.is_file()
    assert invoked_setup_tool.name == "setup_workspace.py"
