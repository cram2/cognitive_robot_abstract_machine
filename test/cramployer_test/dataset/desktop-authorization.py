#!/usr/bin/env python3
"""Supply and store desktop cookies without contacting an X server."""

from __future__ import annotations

# %% Imports
import os
import sys
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path


# %% Authorization vocabulary
class AuthorizationSetting(StrEnum):
    """Inputs understood by the desktop authorization mimic."""

    COOKIE_FILE = "DESKTOP_COOKIE"
    """File supplying the recorded authorization cookie."""
    LIST = "nlist"
    """Read the display cookie."""


# %% Authorization behavior
@dataclass
class DesktopAuthorization:
    """Read fixture cookies and store imported container authorization."""

    cookie_file: Path
    """File supplying the desktop authorization records."""

    def execute(self, arguments: list[str]) -> None:
        """Respond to cookie listing and file-import commands."""
        if arguments[0] == AuthorizationSetting.LIST:
            sys.stdout.write(self.cookie_file.read_text())
            return
        Path(arguments[1]).write_text(sys.stdin.read())


# %% Executable
if __name__ == "__main__":
    DesktopAuthorization(Path(os.environ[AuthorizationSetting.COOKIE_FILE])).execute(
        sys.argv[1:]
    )
