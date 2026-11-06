"""
An isolated development environment with recorded Docker commands.
"""

from __future__ import annotations

# %% Imports
import json
import subprocess
from dataclasses import dataclass

from typing_extensions import TYPE_CHECKING

from .paths import FixturePath

if TYPE_CHECKING:
    from pathlib import Path

    from scripts.cramployer.launcher_vocabulary import CommandName


# %% Command execution
@dataclass
class DevelopmentEnvironment:
    """
    Run launcher commands against temporary configuration and Docker mimics.
    """

    deployment_directory: Path
    """Directory containing the launcher and its configuration."""
    docker_executable: Path
    """
    Executable replacing Docker during the test.
    """

    def run(
        self,
        command: CommandName,
        *arguments: str,
        working_directory: Path | None = None,
    ) -> subprocess.CompletedProcess[str]:
        """
        Execute a launcher command and capture its output and exit status.
        """
        return subprocess.run(
            [
                str(self.deployment_directory / FixturePath.LAUNCHER),
                command,
                *arguments,
            ],
            cwd=working_directory,
            capture_output=True,
            text=True,
            check=False,
        )

    def read_commands(self) -> list[list[str]]:
        """
        Read recorded Docker arguments in invocation order.
        """
        return [
            json.loads(line)
            for line in (self.deployment_directory / FixturePath.COMMAND_LOG)
            .read_text()
            .splitlines()
        ]
