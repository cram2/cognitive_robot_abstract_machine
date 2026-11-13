"""
Preserve command failures across the host launcher and setup stages.
"""

from __future__ import annotations

# %% Imports
import subprocess

import pytest
from typing_extensions import TYPE_CHECKING

from scripts.cramployer.commands import EnvironmentSetup
from scripts.cramployer.launcher_vocabulary import (
    CommandName,
    DeploymentName,
    DockerArgument,
    Executable,
    ExitStatus,
)
from scripts.cramployer.settings import DeploymentSettings

from .dataset.docker import FixtureEnvironment, RecordedExitStatus
from .dataset.paths import FixturePath

if TYPE_CHECKING:
    from pathlib import Path

    from .dataset.deployment import DevelopmentEnvironment


# %% Command exit status
@pytest.mark.parametrize(
    "command_name",
    [
        CommandName.SHELL,
        CommandName.ROOT_SHELL,
        CommandName.CONFIGURATION,
        CommandName.BUILD_ROBOT_WORKSPACE,
        CommandName.GENERATE_INTERFACES,
    ],
)
@pytest.mark.parametrize(
    "failure_status", [ExitStatus.INVALID_SELECTION, RecordedExitStatus.FAILURE]
)
def test_commands_preserve_failure_status(
    deployment: DevelopmentEnvironment,
    monkeypatch: pytest.MonkeyPatch,
    command_name: CommandName,
    failure_status: int,
) -> None:
    """
    Return the selected process status without a Python traceback.
    """
    failure_arguments = {
        CommandName.SHELL: DockerArgument.RUN,
        CommandName.ROOT_SHELL: DockerArgument.EXECUTE,
        CommandName.CONFIGURATION: DockerArgument.CONFIGURATION,
        CommandName.BUILD_ROBOT_WORKSPACE: DeploymentName.ROBOT_WORKSPACE_SCRIPT,
        CommandName.GENERATE_INTERFACES: DeploymentName.INTERFACE_GENERATOR,
    }
    monkeypatch.setenv(FixtureEnvironment.RUNNING_CONTAINERS, DeploymentName.SERVICE)
    monkeypatch.setenv(
        FixtureEnvironment.FAILURE_ARGUMENT, failure_arguments[command_name]
    )
    monkeypatch.setenv(FixtureEnvironment.FAILURE_STATUS, str(failure_status))
    result = deployment.run(command_name)
    assert (result.returncode, result.stderr) == (failure_status, "")


def test_shell_preserves_application_failure_status(
    deployment: DevelopmentEnvironment, monkeypatch: pytest.MonkeyPatch
) -> None:
    """
    Pass a supplied application's failing status through the shell command.
    """
    monkeypatch.setenv(FixtureEnvironment.FAILURE_ARGUMENT, DockerArgument.RUN)
    result = deployment.run(CommandName.SHELL, Executable.PYTHON)
    assert (result.returncode, result.stderr) == (RecordedExitStatus.FAILURE, "")


def test_root_shell_preserves_container_listing_failure_status(
    deployment: DevelopmentEnvironment, monkeypatch: pytest.MonkeyPatch
) -> None:
    """
    Stop before container selection when Docker cannot list containers.
    """
    monkeypatch.setenv(FixtureEnvironment.FAILURE_ARGUMENT, DockerArgument.LIST)
    result = deployment.run(CommandName.ROOT_SHELL)
    assert (result.returncode, result.stderr) == (RecordedExitStatus.FAILURE, "")
    assert len(deployment.read_commands()) == 1


# %% Setup failure handling
@pytest.mark.parametrize(
    "failure_argument",
    [DeploymentName.ROBOT_WORKSPACE_SCRIPT, DeploymentName.INTERFACE_GENERATOR],
)
def test_setup_stops_at_failed_container_stage(
    deployment: DevelopmentEnvironment,
    monkeypatch: pytest.MonkeyPatch,
    failure_argument: str,
) -> None:
    """
    Stop setup at the failing build stage and return its status.
    """
    monkeypatch.setenv(FixtureEnvironment.FAILURE_ARGUMENT, failure_argument)
    result = deployment.run(CommandName.SETUP)
    assert result.returncode == RecordedExitStatus.FAILURE
    assert deployment.read_commands()[-1][-1] == failure_argument


def test_setup_image_build_still_raises_on_failure(
    deployment: DevelopmentEnvironment,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """
    Keep image builds checked so setup cannot continue after a failed build.
    """
    monkeypatch.setenv(FixtureEnvironment.FAILURE_ARGUMENT, DockerArgument.BUILD)
    settings = DeploymentSettings(
        deployment.deployment_directory,
        tmp_path / FixturePath.CHECKOUT,
        tmp_path / FixturePath.ROBOT_WORKSPACE_SOURCES,
    )
    with pytest.raises(subprocess.CalledProcessError) as failure:
        EnvironmentSetup(settings).build_base_image()
    assert failure.value.returncode == RecordedExitStatus.FAILURE
