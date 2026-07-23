"""Source-code formatting helpers backed by external tools (Black, Ruff)."""

from __future__ import annotations

import subprocess
import sys
from typing import List

from krrood.exceptions import SubprocessExecutionError

# %%
# Formatting


def run_subprocess_on_file(command: List[str]) -> None:
    """Run a subprocess command and handle errors.

    :param command: The command to run as a list of arguments.
    :raises SubprocessExecutionError: If the subprocess command fails.
    """
    try:
        result = subprocess.run(command, check=True, capture_output=True, text=True)
    except subprocess.CalledProcessError as e:
        raise SubprocessExecutionError(command, e.returncode, e.stdout, e.stderr) from e


def run_black_on_file(filename: str) -> None:
    """Format *filename* with Black.

    :param filename: The name of the file to format.
    """
    command = [sys.executable, "-m", "black", filename]
    run_subprocess_on_file(command)


def run_ruff_check_on_file(filename: str) -> None:
    """Lint and fix *filename* with Ruff.

    :param filename: The name of the file to format.
    """
    command = ["ruff", "check", "--fix", filename]
    run_subprocess_on_file(command)


def run_ruff_format_on_file(filename: str) -> None:
    """
    Format *filename* with Ruff.

    :param filename: The name of the file to format.
    """
    command = ["ruff", "format", filename]
    run_subprocess_on_file(command)
