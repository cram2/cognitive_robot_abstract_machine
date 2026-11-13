"""Record behavior selected through deployment extension interfaces."""

from __future__ import annotations

# %% Imports
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path

from typing_extensions import TYPE_CHECKING

from scripts.cramployer.commands import DeploymentCommand
from scripts.cramployer.features import ContainerFeature

from scripts.cramployer.launcher_vocabulary import ExitStatus

if TYPE_CHECKING:
    from scripts.cramployer.settings import DeploymentSettings


# %% Extension inputs
class DispatchName(StrEnum):
    """Command and feature inputs supplied by extension tests."""

    RECORD = "record"
    """Command selecting the recording implementation."""
    OUTPUT = "selection"
    """File receiving the selected behavior's output."""
    ENABLED_FEATURE = "CUSTOM_CONTAINER_FEATURE"
    """Environment variable selecting the additional feature."""
    ENABLED = "true"
    """Enabled feature setting."""


# %% Command implementations
@dataclass
class SelectionRecording(DeploymentCommand):
    """Write the arguments supplied to an additional command."""

    def execute(self, arguments: tuple[str, ...]) -> int:
        (self.configuration.deployment_directory / DispatchName.OUTPUT).write_text(
            "\n".join(arguments)
        )
        return ExitStatus.SUCCESS


@dataclass
class ComposeFileRecording(DeploymentCommand):
    """Write the Compose files selected by container features."""

    def execute(self, arguments: tuple[str, ...]) -> int:
        (self.configuration.deployment_directory / DispatchName.OUTPUT).write_text(
            "\n".join(str(path) for path in self.configuration.compose_files)
        )
        return ExitStatus.SUCCESS


# %% Feature implementation
@dataclass
class AdditionalComposeFile(ContainerFeature):
    """Select an additional Compose override without desktop authorization."""

    compose_file: Path = Path("additional-compose.yaml")
    """Additional service configuration selected by the test."""

    def select_compose_file(self) -> Path:
        return self.compose_file

    def prepare_environment(
        self, configuration: DeploymentSettings, authorize_desktop: bool
    ) -> None:
        pass
