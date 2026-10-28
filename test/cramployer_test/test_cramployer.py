"""Check deployment commands without building images or running applications."""

from __future__ import annotations

# %% Imports
import os
import shutil
import subprocess
from enum import StrEnum

import pytest
from typing_extensions import TYPE_CHECKING

from .dataset.docker import (
    EnvironmentVariable,
)
from scripts.cramployer.launcher_vocabulary import (
    CommandName,
    ContainerArgument,
    DeploymentName,
    DockerArgument,
    Executable,
    ExitStatus,
)

from .dataset.paths import FixturePath

if TYPE_CHECKING:
    from pathlib import Path
    from .dataset.deployment import DevelopmentEnvironment


# %% Container selections
class ContainerName(StrEnum):
    """Running and unrelated container selections supplied by command fixtures."""

    FIRST = "first"
    """First available development container."""
    SECOND = "second"
    """Second available development container."""
    UNRELATED = "other"
    """Container outside the development project."""


# %% General development commands
def test_shell_starts_an_independent_interactive_container(
    deployment: DevelopmentEnvironment,
) -> None:
    """Open Bash in a disposable container."""
    deployment.run(CommandName.SHELL).check_returncode()
    assert deployment.read_commands()[-1][-5:] == [
        DockerArgument.RUN,
        DockerArgument.REMOVE_ON_EXIT,
        DockerArgument.NO_DEPENDENCIES,
        DeploymentName.SERVICE,
        Executable.BASH,
    ]


def test_setup_does_not_start_an_application(
    deployment: DevelopmentEnvironment,
) -> None:
    """Finish setup after native checks, ROS compilation, and ORM generation."""
    deployment.run(CommandName.SETUP).check_returncode()
    assert deployment.read_commands()[-1][-4:] == [
        DockerArgument.NO_DEPENDENCIES,
        DeploymentName.SERVICE,
        Executable.PYTHON,
        DeploymentName.INTERFACE_GENERATOR,
    ]


def test_interface_generation_is_available_without_rebuilding_images(
    deployment: DevelopmentEnvironment,
) -> None:
    """Regenerate interfaces in a one-off development container."""
    deployment.run(CommandName.GENERATE_INTERFACES).check_returncode()
    assert deployment.read_commands()[-1][-3:] == [
        DeploymentName.SERVICE,
        Executable.PYTHON,
        DeploymentName.INTERFACE_GENERATOR,
    ]


# %% Root access to running containers
def test_root_shell_attaches_with_the_development_environment(
    deployment: DevelopmentEnvironment, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Open root Bash in the sole running container with ROS and Python sourced."""
    monkeypatch.setenv(EnvironmentVariable.RUNNING_CONTAINERS, ContainerName.FIRST)
    deployment.run(CommandName.ROOT_SHELL).check_returncode()
    assert deployment.read_commands()[-1] == [
        DockerArgument.EXECUTE,
        DockerArgument.INTERACTIVE,
        DockerArgument.TERMINAL,
        DockerArgument.USER,
        DeploymentName.ROOT,
        ContainerName.FIRST,
        Executable.BASH,
        ContainerArgument.STARTUP_FILE,
        DeploymentName.ENVIRONMENT,
        ContainerArgument.INTERACTIVE,
    ]


def test_root_shell_requires_selection_with_multiple_containers(
    deployment: DevelopmentEnvironment, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Require a selection when several development containers are running."""
    monkeypatch.setenv(
        EnvironmentVariable.RUNNING_CONTAINERS,
        "\n".join([ContainerName.FIRST, ContainerName.SECOND]),
    )
    assert (
        deployment.run(CommandName.ROOT_SHELL).returncode
        == ExitStatus.INVALID_SELECTION
    )


def test_root_shell_accepts_an_explicit_running_container(
    deployment: DevelopmentEnvironment, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Attach to the selected container when several shells are open."""
    monkeypatch.setenv(
        EnvironmentVariable.RUNNING_CONTAINERS,
        "\n".join([ContainerName.FIRST, ContainerName.SECOND]),
    )
    deployment.run(CommandName.ROOT_SHELL, ContainerName.SECOND).check_returncode()
    command = deployment.read_commands()[-1]
    assert command[command.index(DeploymentName.ROOT) + 1] == ContainerName.SECOND


def test_root_shell_requires_a_running_development_container(
    deployment: DevelopmentEnvironment,
) -> None:
    """Reject attachment when no development container is running."""
    assert deployment.run(CommandName.ROOT_SHELL).returncode == ExitStatus.UNAVAILABLE


def test_root_shell_rejects_an_unrelated_container(
    deployment: DevelopmentEnvironment, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Restrict privileged attachment to the project's running development containers."""
    monkeypatch.setenv(EnvironmentVariable.RUNNING_CONTAINERS, ContainerName.FIRST)
    assert (
        deployment.run(CommandName.ROOT_SHELL, ContainerName.UNRELATED).returncode
        == ExitStatus.INVALID_SELECTION
    )


# %% Checkout defaults
def test_example_configuration_uses_enclosing_checkout(
    deployment: DevelopmentEnvironment,
) -> None:
    """Resolve the checkout from the launcher's enclosing repository."""
    directory = deployment.deployment_directory
    shutil.copyfile(
        directory / FixturePath.EXAMPLE, directory / FixturePath.CONFIGURATION
    )
    repository = directory.parents[1]
    result = subprocess.run(
        [Executable.BASH, str(FixturePath.READ_SOURCE)],
        cwd=directory,
        env={**os.environ, EnvironmentVariable.REPOSITORY: str(repository)},
        capture_output=True,
        text=True,
        check=True,
    )
    assert result.stdout == str(repository)


def test_launcher_finds_checkout_from_another_working_directory(
    deployment: DevelopmentEnvironment, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Use the enclosing checkout when the launcher runs outside its directory."""
    directory = deployment.deployment_directory
    shutil.copyfile(
        directory / FixturePath.EXAMPLE, directory / FixturePath.CONFIGURATION
    )
    shutil.copyfile(FixturePath.SOURCE_PROBE, deployment.docker_executable)
    monkeypatch.setenv(
        EnvironmentVariable.REPOSITORY, str(tmp_path / ContainerName.UNRELATED)
    )
    result = deployment.run(CommandName.CONFIGURATION, working_directory=tmp_path)
    result.check_returncode()
    assert result.stdout.strip() == str(directory.parents[1])
