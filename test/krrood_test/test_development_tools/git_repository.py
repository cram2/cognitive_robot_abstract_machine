"""
Running git in the repositories the development tools tests create.
"""

from __future__ import annotations

import subprocess
from pathlib import Path

from krrood.development_tools.agents.checker import GitArgument

# %% git commands


def git(repository: Path, *arguments: str) -> None:
    """
    Run a git command in *repository*, which must succeed.
    """
    subprocess.run([GitArgument.GIT, *arguments], cwd=repository, check=True)
