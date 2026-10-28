"""
Validate launcher settings before running image builds.
"""

from __future__ import annotations

# %% Imports
import subprocess
import sys

import pytest
from typing_extensions import TYPE_CHECKING

from scripts.cramployer.commands import EnvironmentSetup
from scripts.cramployer.settings import DeploymentSettings
from scripts.cramployer.launch import DeploymentLauncher
from scripts.cramployer.launcher_vocabulary import (
    CommandName,
    DockerArgument,
    EnvironmentVariable,
)
from scripts.cramployer.vocabulary import CommandArgument

from .dataset.paths import FixturePath

if TYPE_CHECKING:
    from pathlib import Path

    from .dataset.deployment import DevelopmentEnvironment


# %% Required build inputs
def test_setup_requires_the_installation_workflow(
    deployment: DevelopmentEnvironment, tmp_path: Path
) -> None:
    """
    Reject incomplete checkouts before building the base image.
    """
    source_checkout = tmp_path / FixturePath.CHECKOUT
    (source_checkout / FixturePath.WORKFLOW).unlink()
    settings = DeploymentSettings(
        deployment.deployment_directory,
        source_checkout,
        tmp_path / FixturePath.ROBOT_WORKSPACE_SOURCES,
    )
    with pytest.raises(FileNotFoundError):
        EnvironmentSetup(settings).build_base_image()


# %% Host bootstrap
def test_host_launcher_runs_without_site_packages(
    deployment: DevelopmentEnvironment, tmp_path: Path
) -> None:
    """
    Run the host launcher with third-party Python packages disabled.
    """
    settings = DeploymentSettings(
        deployment.deployment_directory,
        tmp_path / FixturePath.CHECKOUT,
        tmp_path / FixturePath.ROBOT_WORKSPACE_SOURCES,
    )
    environment = settings.process_environment.copy()
    environment.update(
        {
            EnvironmentVariable.SOURCE_CHECKOUT: str(settings.source_checkout),
            EnvironmentVariable.ROBOT_WORKSPACE_SOURCES: str(
                settings.robot_workspace_sources
            ),
            EnvironmentVariable.REPOSITORY_ADDRESS: settings.repository_address,
            EnvironmentVariable.REPOSITORY_BRANCH: settings.repository_branch,
            EnvironmentVariable.USER_IDENTIFIER: str(settings.user_identifier),
            EnvironmentVariable.GROUP_IDENTIFIER: str(settings.group_identifier),
            EnvironmentVariable.DEPENDENCY_SOURCE: settings.dependency_source,
        }
    )
    subprocess.run(
        [
            sys.executable,
            CommandArgument.NO_SITE_PACKAGES,
            CommandArgument.MODULE,
            DeploymentLauncher.__module__,
            str(settings.deployment_directory),
            CommandName.CONFIGURATION,
        ],
        cwd=FixturePath.REPOSITORY,
        env=environment,
        check=True,
    )
    assert deployment.read_commands()[-1][-1] == DockerArgument.CONFIGURATION
