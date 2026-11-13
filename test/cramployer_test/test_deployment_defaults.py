"""Check the commands and persistent storage offered by a fresh deployment."""

from __future__ import annotations

# %% Imports
from enum import StrEnum

import yaml
from typing_extensions import TYPE_CHECKING

from scripts.cramployer.launch import DeploymentLauncher
from scripts.cramployer.launcher_vocabulary import ExitStatus
from scripts.cramployer.settings import DeploymentSettings

from .dataset.paths import FixturePath

if TYPE_CHECKING:
    from pathlib import Path


# %% Deployment vocabulary
class DeploymentKey(StrEnum):
    """Compose storage settings and retired command names under test."""

    VOLUMES = "volumes"
    """Persistent volume declarations."""
    EXTERNAL = "external"
    """Require a volume managed outside this Compose project."""
    OFFLINE_SHELL = "shell-offline"
    """Retired alias for the development shell."""


# %% Command selection
def test_shell_offline_is_rejected(tmp_path: Path) -> None:
    """Reject the retired shell alias through ordinary command validation."""
    settings = DeploymentSettings(tmp_path, tmp_path, tmp_path)
    assert (
        DeploymentLauncher(settings).run((DeploymentKey.OFFLINE_SHELL,))
        == ExitStatus.INVALID_SELECTION
    )


# %% Persistent storage
def test_volumes_are_managed_by_compose() -> None:
    """Declare persistent volumes without external lifecycle configuration."""
    configuration = yaml.safe_load(
        (FixturePath.DEPLOYMENT / FixturePath.COMPOSE).read_text()
    )
    assert {
        volume_name
        for volume_name, definition in configuration[DeploymentKey.VOLUMES].items()
        if DeploymentKey.EXTERNAL in definition
    } == set()
