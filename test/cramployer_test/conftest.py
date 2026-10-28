"""
Shared source and command fixtures for deployment tooling.
"""

from __future__ import annotations

# %% Imports
import os
import shutil
import sys
from pathlib import Path


import pytest

from .dataset.deployment import DevelopmentEnvironment
from .dataset.docker import EnvironmentVariable
from .dataset.paths import FixturePath
from scripts.cramployer.vocabulary import FileName


# %% Checkout fixtures
@pytest.fixture
def repository() -> Path:
    """
    Provide the repository containing the shared installation script.
    """
    return FixturePath.REPOSITORY


# %% Launcher fixture
@pytest.fixture
def deployment(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> DevelopmentEnvironment:
    """
    Prepare local configuration and a Docker command recorder.
    """
    directory = tmp_path / FixturePath.ISOLATED_DEPLOYMENT
    shutil.copytree(
        FixturePath.DEPLOYMENT,
        directory,
        ignore=shutil.ignore_patterns(".git", "__pycache__"),
    )
    shutil.copyfile(
        FixturePath.DEPLOYMENT_ENVIRONMENT, directory / FixturePath.CONFIGURATION
    )
    executable_directory = tmp_path / FixturePath.EXECUTABLES
    executable_directory.mkdir()
    docker_executable = executable_directory / FixturePath.DOCKER
    shutil.copyfile(FixturePath.RECORDER, docker_executable)
    docker_executable.chmod(0o755)
    monkeypatch.setenv(
        EnvironmentVariable.PATH,
        os.pathsep.join(
            [
                str(executable_directory),
                str(Path(sys.executable).parent),
                os.environ[EnvironmentVariable.PATH],
            ]
        ),
    )
    monkeypatch.setenv(EnvironmentVariable.TEST_ROOT, str(tmp_path))
    monkeypatch.setenv(
        EnvironmentVariable.COMMAND_LOG, str(directory / FixturePath.COMMAND_LOG)
    )
    monkeypatch.setenv(
        EnvironmentVariable.DESKTOP_COOKIE, str(FixturePath.DESKTOP_COOKIE)
    )
    checkout_directory = tmp_path / FixturePath.CHECKOUT
    (checkout_directory / FixturePath.GIT).mkdir(parents=True)
    (checkout_directory / FixturePath.DOCKER_DIRECTORY).mkdir(parents=True)
    installer = checkout_directory / FileName.DEPENDENCY_INSTALLER
    installer.parent.mkdir(parents=True)
    (checkout_directory / FixturePath.DOCKER_DIRECTORY / FixturePath.DOCKERFILE).touch()
    installer.touch()
    robot_workspace_sources = tmp_path / FixturePath.ROBOT_WORKSPACE_SOURCES
    robot_workspace_sources.mkdir(parents=True)
    (robot_workspace_sources / FixturePath.EXISTING_PACKAGE).touch()
    monkeypatch.setenv(
        EnvironmentVariable.SOURCE_DIRECTORY, str(robot_workspace_sources)
    )
    return DevelopmentEnvironment(directory, docker_executable)


# %% Installation fixture
@pytest.fixture
def checkout_name() -> str:
    """
    Use a checkout directory independent of CI naming conventions.
    """
    return str(FixturePath.CHECKOUT)


@pytest.fixture
def checkout(tmp_path: Path, checkout_name: str) -> Path:
    """
    Provide a local installation script in an isolated checkout.
    """
    checkout_directory = tmp_path / checkout_name
    installer = checkout_directory / FileName.DEPENDENCY_INSTALLER
    installer.parent.mkdir(parents=True)
    shutil.copyfile(FixturePath.INSTALLATION, installer)
    return checkout_directory
