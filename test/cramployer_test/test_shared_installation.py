"""Exercise shared dependency installation with local command recorders."""

from __future__ import annotations

# %% Imports
import json
import os
import shutil
import subprocess
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path

import pytest

from scripts.cramployer.vocabulary import (
    CommandArgument,
    DependencySource,
    FileName,
    OptionalPackage,
    Program,
)
from scripts.cramployer.launcher_vocabulary import EnvironmentVariable, ExitStatus

from .dataset.paths import FixturePath
from .dataset.install_command import InstallationRecorder, RecorderSetting


# %% Installation inputs
class InstallValue(StrEnum):
    """External command names and settings exercised by installation tests."""

    SYSTEM_INSTALLER = "apt-get"
    """System dependency installer replaced during tests."""
    PROCESSOR_INDEX = "https://download.pytorch.org/whl/cpu"
    """PyTorch index serving CPU wheels."""
    INDEX_ARGUMENT = "--extra-index-url"
    """Argument adding the CPU package index."""
    DRAKE = "drake"
    """Optional Drake distribution."""
    VISION_PROJECT = "robokudo/pyproject.toml"
    """Dependency metadata for optional vision requirements."""
    PIP = "pip"
    """uv's pip-compatible command group."""
    INSTALL = "install"
    """Install dependency distributions."""
    REQUIREMENTS = "-r"
    """Read dependency metadata from a project."""
    DEVELOPMENT_EXTRA = "dev"
    """Development dependencies selected by the installer."""
    NO_SOURCES = "--no-sources"
    """Preserve editable workspace installations."""
    TREE = "tree"
    """Display resolved dependencies."""
    DEPTH = "--depth"
    """Limit dependency tree display depth."""
    DIRECT_DEPENDENCIES = "1"
    """Display only direct dependency relationships."""


@dataclass
class DependencyInstallation:
    """Run the shared script with external installation commands replaced."""

    repository: Path
    """Checkout containing the shared installation script."""
    command_log: Path
    """Recorded installation requests."""
    process_environment: dict[str, str]
    """Environment selecting local command recorders."""

    def run(self, mode: str, packages: str) -> subprocess.CompletedProcess[str]:
        """Execute dependency installation without modifying the machine."""
        return subprocess.run(
            [
                Program.BASH,
                str(self.repository / FileName.DEPENDENCY_INSTALLER),
                mode,
                packages,
            ],
            cwd=self.repository.parent,
            env=self.process_environment,
            text=True,
            capture_output=True,
            check=False,
        )

    def read_commands(self) -> list[list[str]]:
        """Return installation requests in execution order."""
        return [json.loads(line) for line in self.command_log.read_text().splitlines()]


# %% Installation fixture
@pytest.fixture
def dependency_installation(tmp_path: Path) -> DependencyInstallation:
    """Provide a checkout with mocked package and system installation commands."""
    checkout = tmp_path / FixturePath.CHECKOUT
    installer = checkout / FileName.DEPENDENCY_INSTALLER
    installer.parent.mkdir(parents=True)
    shutil.copyfile(FixturePath.REPOSITORY / FileName.DEPENDENCY_INSTALLER, installer)
    executable_directory = tmp_path / FixturePath.EXECUTABLES
    executable_directory.mkdir()
    for executable in (
        Program.UV,
        InstallValue.SYSTEM_INSTALLER,
        RecorderSetting.PRIVILEGE_COMMAND,
    ):
        destination = executable_directory / executable
        shutil.copyfile(FixturePath.INSTALL_RECORDER, destination)
        destination.chmod(0o755)
    command_log = tmp_path / FixturePath.COMMAND_LOG
    command_log.touch()
    return DependencyInstallation(
        checkout,
        command_log,
        {
            **os.environ,
            RecorderSetting.COMMAND_LOG: str(command_log),
            EnvironmentVariable.EXECUTABLE_PATH: os.pathsep.join(
                [
                    str(executable_directory),
                    os.environ[EnvironmentVariable.EXECUTABLE_PATH],
                ]
            ),
        },
    )


# %% PyTorch selection
@pytest.mark.parametrize("mode", list(DependencySource))
def test_shared_installer_selects_torch_index_explicitly(
    dependency_installation: DependencyInstallation, mode: DependencySource
) -> None:
    """Request CPU wheels only when CPU mode is selected."""
    dependency_installation.run(mode, OptionalPackage.ROBOKUDO).check_returncode()
    commands = dependency_installation.read_commands()
    vision = next(
        command for command in commands if InstallValue.VISION_PROJECT in command
    )
    expected = [
        Program.UV,
        InstallValue.PIP,
        InstallValue.INSTALL,
        InstallValue.REQUIREMENTS,
        InstallValue.VISION_PROJECT,
        CommandArgument.EXTRA,
        InstallValue.DEVELOPMENT_EXTRA,
        InstallValue.NO_SOURCES,
    ]
    if mode == DependencySource.PROCESSOR_ONLY:
        expected.extend([InstallValue.INDEX_ARGUMENT, InstallValue.PROCESSOR_INDEX])
    assert vision == expected


@pytest.mark.parametrize("mode", list(DependencySource))
def test_shared_installer_allows_common_dependencies_without_vision(
    dependency_installation: DependencyInstallation, mode: DependencySource
) -> None:
    """Install the workspace without enabling optional packages."""
    dependency_installation.run(mode, "").check_returncode()
    assert [
        command
        for command in dependency_installation.read_commands()
        if command[0] == Program.UV
    ] == [
        [
            Program.UV,
            CommandArgument.SYNC,
            CommandArgument.EXTRA,
            InstallValue.DEVELOPMENT_EXTRA,
            CommandArgument.ACTIVE,
        ],
        [
            Program.UV,
            InstallValue.TREE,
            InstallValue.DEPTH,
            InstallValue.DIRECT_DEPENDENCIES,
        ],
    ]


def test_shared_installer_selects_drake(
    dependency_installation: DependencyInstallation,
) -> None:
    """Install Drake when the semantic digital twin dependencies are selected."""
    dependency_installation.run(
        DependencySource.DEFAULT, OptionalPackage.SEMANTIC_DIGITAL_TWIN
    ).check_returncode()
    assert dependency_installation.read_commands()[-1] == [
        Program.UV,
        InstallValue.PIP,
        InstallValue.INSTALL,
        InstallValue.DRAKE,
    ]


@pytest.mark.parametrize(
    "mode, packages", [("unsupported", ""), (DependencySource.DEFAULT, "unsupported")]
)
def test_shared_installer_rejects_invalid_selection_before_installing(
    dependency_installation: DependencyInstallation, mode: str, packages: str
) -> None:
    """Leave the environment untouched when an installation option is invalid."""
    assert (
        dependency_installation.run(mode, packages).returncode
        == ExitStatus.INVALID_SELECTION
    )
    assert dependency_installation.read_commands() == []


def test_shared_installer_stops_on_system_installation_failure(
    dependency_installation: DependencyInstallation,
) -> None:
    """Stop before Python installation when apt fails."""
    dependency_installation.process_environment[RecorderSetting.FAILURE_COMMAND] = (
        InstallValue.SYSTEM_INSTALLER
    )
    assert (
        dependency_installation.run(DependencySource.DEFAULT, "").returncode
        == InstallationRecorder.failure_status
    )
    assert [command[0] for command in dependency_installation.read_commands()] == [
        InstallValue.SYSTEM_INSTALLER
    ]
