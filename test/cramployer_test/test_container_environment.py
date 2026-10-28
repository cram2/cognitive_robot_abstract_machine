"""
Validate optional container environment files using Compose configuration.
"""

from __future__ import annotations

# %% Imports
import json
import shutil
import subprocess
from dataclasses import dataclass

import pytest
from typing_extensions import TYPE_CHECKING

from scripts.cramployer.vocabulary import PayloadKey
from scripts.cramployer.launcher_vocabulary import (
    ContainerArgument,
    DockerArgument,
    Executable,
)

from .dataset.docker import EnvironmentVariable
from .dataset.paths import FixturePath

if TYPE_CHECKING:
    from pathlib import Path
    from typing_extensions import Any, Self


# %% Compose configuration
@dataclass
class ServiceConfiguration:
    """
    Resolved environment of the development service.
    """

    environment: dict[str, str]
    """Environment variables passed into development containers."""

    @classmethod
    def from_json(class_type, payload: dict[str, Any]) -> Self:
        """
        Parse the development service from Compose's configuration response.
        """
        return class_type(
            payload[PayloadKey.SERVICES][PayloadKey.CRAM][PayloadKey.ENVIRONMENT]
        )


@dataclass
class ContainerConfiguration:
    """
    An isolated Compose project whose runtime environment can be inspected.
    """

    project_directory: Path
    """Directory containing Compose configuration and its optional environment file."""
    docker_executable: str
    """
    Docker executable providing the Compose configuration command.
    """

    def resolve_environment(self) -> dict[str, str]:
        """
        Resolve the development service environment without creating containers.
        """
        result = subprocess.run(
            [
                self.docker_executable,
                DockerArgument.COMPOSE,
                DockerArgument.ENVIRONMENT_FILE,
                ContainerArgument.EMPTY_CONFIGURATION,
                DockerArgument.FILE,
                str(self.project_directory / FixturePath.COMPOSE),
                DockerArgument.CONFIGURATION,
                DockerArgument.FORMAT,
                DockerArgument.JSON,
            ],
            check=True,
            capture_output=True,
            text=True,
        )
        return ServiceConfiguration.from_json(json.loads(result.stdout)).environment


# %% Configuration fixture
@pytest.fixture
def configuration(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> ContainerConfiguration:
    """
    Prepare a Compose configuration with no optional environment file.
    """
    docker_executable = shutil.which(Executable.DOCKER)
    if docker_executable is None:
        pytest.skip("Docker Compose is required for configuration validation")
    availability = subprocess.run(
        [docker_executable, DockerArgument.COMPOSE, DockerArgument.VERSION],
        check=False,
        capture_output=True,
    )
    if availability.returncode:
        pytest.skip("Docker Compose is required for configuration validation")
    shutil.copyfile(
        FixturePath.DEPLOYMENT / FixturePath.COMPOSE, tmp_path / FixturePath.COMPOSE
    )
    monkeypatch.setenv(EnvironmentVariable.CRAM_SOURCE, str(tmp_path))
    monkeypatch.setenv(EnvironmentVariable.ROBOT_WORKSPACE_SOURCES, str(tmp_path))
    monkeypatch.delenv(EnvironmentVariable.MIDDLEWARE, raising=False)
    return ContainerConfiguration(tmp_path, docker_executable)


# %% Runtime environment forwarding
def test_arbitrary_variables_are_forwarded_from_environment_file(
    configuration: ContainerConfiguration,
) -> None:
    """
    Forward middleware selection and unrelated environment values.
    """
    source = FixturePath.DATASET / FixturePath.CONTAINER_ENVIRONMENT
    shutil.copyfile(
        source, configuration.project_directory / FixturePath.CONTAINER_ENVIRONMENT
    )
    expected = dict(line.split("=", 1) for line in source.read_text().splitlines())
    environment = configuration.resolve_environment()
    assert {key: environment[key] for key in expected} == expected


def test_missing_environment_file_keeps_middleware_unset(
    configuration: ContainerConfiguration,
) -> None:
    """
    Allow a fresh checkout to use the default middleware without a local env file.
    """
    assert EnvironmentVariable.MIDDLEWARE not in configuration.resolve_environment()
