#!/usr/bin/env python3
"""Record installation commands without changing the host or contacting services."""

from __future__ import annotations

# %% Imports
import json
import os
import subprocess
import sys
from dataclasses import dataclass
from enum import IntEnum, StrEnum
from pathlib import Path


# %% Recorder settings
class RecorderSetting(StrEnum):
    """Environment and executable names understood by the command recorder."""

    COMMAND_LOG = "INSTALL_COMMAND_LOG"
    """File receiving recorded command arguments."""
    FAILURE_COMMAND = "INSTALL_FAILURE_COMMAND"
    """Executable selected to fail."""
    PRIVILEGE_COMMAND = "sudo"
    """Privilege request forwarded to a local mimic."""


class RecorderStatus(IntEnum):
    """Successful process status returned by installation mimics."""

    SUCCESS = 0
    """The requested command completed successfully."""


# %% Installation recording
@dataclass
class InstallationRecorder:
    """Record external installation requests and simulate a selected failure."""

    command_log: Path
    """File receiving executable names and arguments."""
    failure_command: str
    """Executable whose invocation should fail."""
    failure_status: int = 23
    """Status returned by the selected failing executable."""

    def run(self, executable: str, arguments: list[str]) -> int:
        """Record a command, forwarding privilege requests to the local mimics."""
        if executable == RecorderSetting.PRIVILEGE_COMMAND:
            return subprocess.run(arguments, check=False).returncode
        with self.command_log.open("a") as stream:
            stream.write(json.dumps([executable, *arguments]) + "\n")
        return (
            self.failure_status
            if executable == self.failure_command
            else RecorderStatus.SUCCESS
        )


# %% Executable
if __name__ == "__main__":
    raise SystemExit(
        InstallationRecorder(
            Path(os.environ[RecorderSetting.COMMAND_LOG]),
            os.environ.get(RecorderSetting.FAILURE_COMMAND, ""),
        ).run(Path(sys.argv[0]).name, sys.argv[1:])
    )
