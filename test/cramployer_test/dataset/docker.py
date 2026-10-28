#!/usr/bin/env python3
"""Record Docker commands without starting containers."""

from __future__ import annotations

# %% Imports
import json
import os
import sys
import tarfile
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path


# %% Command vocabulary
class EnvironmentVariable(StrEnum):
    """Environment shared by command fixtures and their recording executable."""

    SOURCE_DIRECTORY = "DOCKER_SOURCE_DIRECTORY"
    """Host directory streamed by the source-container mimic."""
    DESKTOP_COOKIE = "DESKTOP_COOKIE"
    """File providing the desktop cookie to the authorization mimic."""
    COMMAND_LOG = "DOCKER_COMMAND_LOG"
    """File receiving recorded Docker arguments."""
    RUNNING_CONTAINERS = "DOCKER_RUNNING_CONTAINERS"
    """Container names returned by the Docker mimic."""
    TEST_ROOT = "DEPLOY_TEST_ROOT"
    """Root of the isolated source directories."""
    REPOSITORY = "CRAM_REPOSITORY"
    """Checkout enclosing the deployment scripts."""
    CRAM_SOURCE = "CRAM_SOURCE"
    """Editable source mounted in containers."""
    ROBOT_WORKSPACE_SOURCES = "ROS_SOURCE"
    """ROS sources mounted in containers."""
    PATH = "PATH"
    """Executable search directories."""
    MIDDLEWARE = "RMW_IMPLEMENTATION"
    """ROS middleware selection."""


class RecordedCommandName(StrEnum):
    """Docker commands whose output is supplied by the recorder."""

    CREATE = "create"
    """Create a source container."""
    COPY = "cp"
    """Stream an archive of source files."""
    LIST = "ps"
    """List running development containers."""


# %% Command recording
@dataclass
class CommandRecorder:
    """Record Docker arguments and supply configured running containers."""

    command_log: Path
    """File receiving each invocation's arguments."""
    running_containers: str
    """Container names exposed by the listing command."""

    source_directory: str = ""
    """Directory containing source files to archive."""
    container_name: str = "source-container"
    """Name returned when creating a source container."""

    def record(self, arguments: list[str]) -> None:
        """Append an invocation and answer container-listing requests."""
        with self.command_log.open("a") as stream:
            stream.write(json.dumps(arguments) + "\n")
        if arguments[:1] == [RecordedCommandName.LIST]:
            print(self.running_containers)
        if arguments[:1] == [RecordedCommandName.CREATE]:
            print(self.container_name)
        if arguments[:1] == [RecordedCommandName.COPY]:
            with tarfile.open(fileobj=sys.stdout.buffer, mode="w|") as archive:
                archive.add(self.source_directory, arcname=".")


# %% Executable
if __name__ == "__main__":
    CommandRecorder(
        Path(os.environ[EnvironmentVariable.COMMAND_LOG]),
        os.environ.get(EnvironmentVariable.RUNNING_CONTAINERS, ""),
        os.environ.get(EnvironmentVariable.SOURCE_DIRECTORY, ""),
    ).record(sys.argv[1:])
