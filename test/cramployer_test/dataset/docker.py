#!/usr/bin/env python3
"""Record Docker commands without starting containers."""

from __future__ import annotations

# %% Imports
import json
import os
import sys
import tarfile
from dataclasses import dataclass
from enum import IntEnum, StrEnum
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
    FAILURE_ARGUMENT = "DOCKER_FAILURE_ARGUMENT"
    """Argument selecting an invocation that should fail."""
    FAILURE_STATUS = "DOCKER_FAILURE_STATUS"
    """Exit status returned by the selected invocation."""


class RecordedExitStatus(IntEnum):
    """Process outcomes supplied by the Docker recorder."""

    SUCCESS = 0
    """Invocation completed successfully."""
    FAILURE = 23
    """Invocation failed with a distinguishable nonzero status."""


class RecordedCommandName(StrEnum):
    """Docker commands whose output is supplied by the recorder."""

    CREATE = "create"
    """Create a source container."""
    COPY = "cp"
    """Stream an archive of source files."""
    LIST = "ps"
    """List running development containers."""
    CONFIGURATION = "config"
    """Report the source checkout supplied to Compose."""


# %% Command recording
@dataclass
class CommandRecorder:
    """Record Docker arguments and supply container and checkout responses."""

    command_log: Path
    """File receiving each invocation's arguments."""
    running_containers: str
    """Container names exposed by the listing command."""

    source_directory: str = ""
    """Directory containing source files to archive."""
    container_name: str = "source-container"
    """Name returned when creating a source container."""
    failure_argument: str = ""
    """Argument identifying a failing invocation."""
    failure_status: int = RecordedExitStatus.FAILURE
    """Exit status returned by a failing invocation."""
    source_checkout: str = ""
    """Source checkout reported by the configuration command."""

    def record(self, arguments: list[str]) -> int:
        """Record an invocation, emit its response, and return its exit status."""
        with self.command_log.open("a") as stream:
            stream.write(json.dumps(arguments) + "\n")
        if self.failure_argument in arguments:
            return self.failure_status
        if arguments[-1:] == [RecordedCommandName.CONFIGURATION]:
            print(self.source_checkout)
        if arguments[:1] == [RecordedCommandName.LIST]:
            print(self.running_containers)
        if arguments[:1] == [RecordedCommandName.CREATE]:
            print(self.container_name)
        if arguments[:1] == [RecordedCommandName.COPY]:
            with tarfile.open(fileobj=sys.stdout.buffer, mode="w|") as archive:
                archive.add(self.source_directory, arcname=".")
        return RecordedExitStatus.SUCCESS


# %% Executable
if __name__ == "__main__":
    raise SystemExit(
        CommandRecorder(
            Path(os.environ[EnvironmentVariable.COMMAND_LOG]),
            os.environ.get(EnvironmentVariable.RUNNING_CONTAINERS, ""),
            os.environ.get(EnvironmentVariable.SOURCE_DIRECTORY, ""),
            failure_argument=os.environ.get(EnvironmentVariable.FAILURE_ARGUMENT, ""),
            failure_status=int(
                os.environ.get(
                    EnvironmentVariable.FAILURE_STATUS, str(RecordedExitStatus.FAILURE)
                )
            ),
            source_checkout=os.environ.get(EnvironmentVariable.CRAM_SOURCE, ""),
        ).record(sys.argv[1:])
    )
