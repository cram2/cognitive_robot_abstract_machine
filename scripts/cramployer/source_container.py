"""
Temporary containers providing the base image's ROS package sources.
"""

from __future__ import annotations

# %% Imports
import subprocess
from contextlib import AbstractContextManager
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from scripts.cramployer.exceptions import SourceCopyFailed
from scripts.cramployer.launcher_vocabulary import (
    ArchiveArgument,
    DeploymentName,
    DockerArgument,
    Executable,
)

if TYPE_CHECKING:
    from types import TracebackType

    from scripts.cramployer.settings import DeploymentSettings


# %% Source container lifetime
@dataclass
class SourceContainer(AbstractContextManager[object]):
    """
    Supply base-image sources and remove the temporary container on exit.
    """

    configuration: DeploymentSettings
    """Host paths and environment for source extraction."""
    container_name: str = field(init=False)
    """
    Container created for the current extraction.
    """

    def __enter__(self) -> SourceContainer:
        """
        Create the temporary source container.
        """
        self.container_name = self.configuration.run(
            [
                Executable.DOCKER,
                DockerArgument.CREATE,
                DeploymentName.BASE_IMAGE,
                Executable.BASH,
            ],
            capture_output=True,
        ).stdout.strip()
        return self

    def __exit__(
        self,
        exception_type: type[BaseException] | None,
        exception: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        """
        Remove the source container after successful or failed extraction.
        """
        self.configuration.run(
            [Executable.DOCKER, DockerArgument.REMOVE, self.container_name],
            capture_output=True,
        )

    def copy_sources(self) -> None:
        """
        Stream the image's source archive into the host ROS source directory.
        """
        configuration = self.configuration
        with subprocess.Popen(
            [
                Executable.DOCKER,
                DockerArgument.COPY,
                f"{self.container_name}:{DeploymentName.SOURCE_ARCHIVE}",
                ArchiveArgument.STANDARD_INPUT,
            ],
            cwd=configuration.deployment_directory,
            env=configuration.process_environment,
            stdout=subprocess.PIPE,
        ) as archive:
            subprocess.run(
                [
                    Executable.ARCHIVE_EXTRACTOR,
                    ArchiveArgument.EXTRACT,
                    ArchiveArgument.HOST_OWNERSHIP,
                    ArchiveArgument.FILE,
                    ArchiveArgument.STANDARD_INPUT,
                    ArchiveArgument.DIRECTORY,
                    str(configuration.robot_workspace_sources),
                ],
                stdin=archive.stdout,
                cwd=configuration.deployment_directory,
                env=configuration.process_environment,
                check=True,
            )
            exit_status = archive.wait()
        if exit_status:
            raise SourceCopyFailed(self.container_name, exit_status)
