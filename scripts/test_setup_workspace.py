from __future__ import annotations

import importlib.util
import subprocess
from dataclasses import dataclass
from pathlib import Path
from types import ModuleType

import pytest


@dataclass(frozen=True)
class SetupWorkspaceModule:
    """Loaded setup workspace module used by focused script tests."""

    module: ModuleType
    """Python module loaded from the setup workspace script."""


def load_setup_workspace_module() -> SetupWorkspaceModule:
    """Load the setup workspace script as a Python module."""
    repository_root = Path(__file__).resolve().parents[1]
    setup_workspace_path = repository_root / ".github/docker/setup_workspace.py"
    module_specification = importlib.util.spec_from_file_location(
        "setup_workspace", setup_workspace_path
    )
    assert module_specification is not None
    assert module_specification.loader is not None

    module = importlib.util.module_from_spec(module_specification)
    module_specification.loader.exec_module(module)
    return SetupWorkspaceModule(module=module)


def test_dependency_manager_skips_apt_when_packages_are_installed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Installed system dependencies do not trigger apt commands."""
    setup_workspace_module = load_setup_workspace_module()
    recorded_commands: list[list[str]] = []

    def record_installed_package_command(
        command: list[str],
        check: bool,
        stdout: int | None = None,
        stderr: int | None = None,
        text: bool = False,
    ) -> subprocess.CompletedProcess[list[str]]:
        recorded_commands.append(command)
        return subprocess.CompletedProcess(command, 0, stdout="install ok installed")

    monkeypatch.setattr(
        setup_workspace_module.module.subprocess,
        "run",
        record_installed_package_command,
    )

    dependency_manager = setup_workspace_module.module.SystemDependencyManager(
        is_root=False
    )
    dependency_manager.install_packages(["git", "python3-pip"])

    assert recorded_commands == [
        ["dpkg-query", "-W", "-f=${Status}", "git"],
        ["dpkg-query", "-W", "-f=${Status}", "python3-pip"],
    ]


def test_dependency_manager_installs_only_missing_packages_non_interactively(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Missing system dependencies are installed with non-interactive sudo."""
    setup_workspace_module = load_setup_workspace_module()
    recorded_commands: list[list[str]] = []

    def record_package_command(
        command: list[str],
        check: bool,
        stdout: int | None = None,
        stderr: int | None = None,
        text: bool = False,
    ) -> subprocess.CompletedProcess[list[str]]:
        recorded_commands.append(command)
        if command == ["dpkg-query", "-W", "-f=${Status}", "missing-package"]:
            return subprocess.CompletedProcess(command, 1, stdout="")
        if command[0] == "dpkg-query":
            return subprocess.CompletedProcess(
                command, 0, stdout="install ok installed"
            )
        return subprocess.CompletedProcess(command, 0)

    monkeypatch.setattr(
        setup_workspace_module.module.subprocess,
        "run",
        record_package_command,
    )

    dependency_manager = setup_workspace_module.module.SystemDependencyManager(
        is_root=False
    )
    dependency_manager.install_packages(["installed-package", "missing-package"])

    assert recorded_commands == [
        ["dpkg-query", "-W", "-f=${Status}", "installed-package"],
        ["dpkg-query", "-W", "-f=${Status}", "missing-package"],
        ["sudo", "-n", "apt", "update"],
        ["sudo", "-n", "apt", "install", "-y", "missing-package"],
    ]


def test_dependency_manager_treats_not_installed_status_as_missing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Known apt packages are missing unless their status is installed."""
    setup_workspace_module = load_setup_workspace_module()
    recorded_commands: list[list[str]] = []

    def record_package_status_command(
        command: list[str],
        check: bool,
        stdout: int | None = None,
        stderr: int | None = None,
        text: bool = False,
    ) -> subprocess.CompletedProcess[list[str]]:
        recorded_commands.append(command)
        if command[0] == "dpkg-query":
            return subprocess.CompletedProcess(
                command, 0, stdout="unknown ok not-installed"
            )
        return subprocess.CompletedProcess(command, 0)

    monkeypatch.setattr(
        setup_workspace_module.module.subprocess,
        "run",
        record_package_status_command,
    )

    dependency_manager = setup_workspace_module.module.SystemDependencyManager(
        is_root=False
    )
    dependency_manager.install_packages(["known-package"])

    assert recorded_commands == [
        ["dpkg-query", "-W", "-f=${Status}", "known-package"],
        ["sudo", "-n", "apt", "update"],
        ["sudo", "-n", "apt", "install", "-y", "known-package"],
    ]


def test_weiss_gripper_repository_keeps_griplink_dependency() -> None:
    """The Weiss gripper workspace keeps packages needed by its description."""
    setup_workspace_module = load_setup_workspace_module()

    gripper_repository = next(
        repository
        for repository in setup_workspace_module.module.create_repositories()
        if repository.name == "iai_weiss_wpg_300-120-gripper"
    )

    assert "griplink" not in gripper_repository.cleanup_dirs
